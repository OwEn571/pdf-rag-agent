# 检索策略链路 — 当前阶段完整说明

> 写给外部专家评审。涵盖从用户查询到论文候选池的完整链路，标注了关键配置和消融依据。

---

## 概览

检索分为两级：**论文级**（paper，粗筛 6 篇）和**块级**（block，精排 14 条）。本文档聚焦论文级检索链路。

```
用户查询
  │
  ▼
┌─ 第1次LLM: Intent Router (tool-calling) ─┐
│  提取 targets + relation → QueryContract  │
└──────────────────────────────────────────┘
  │
  ▼
┌─ canonicalize_targets() ─────────────────┐
│  targets 匹配 paper title/aliases/       │
│  body_acronyms，规范化为库中实际名称       │
└──────────────────────────────────────────┘
  │
  ▼
┌─ paper_query_text(contract) ─────────────┐
│  条件化构建检索 query:                    │
│  单target+definition → 仅用 canonical    │
│  多target / 描述查询 → 拼接 targets+query │
└──────────────────────────────────────────┘
  │
  ▼
┌─ search_papers() [DualIndexRetriever] ───┐
│  ① Dense (Milvus, top-12)                │
│  ② Title Anchor fallback (条件触发)       │
│  ③ _paper_match_boost (post-retrieval)   │
│  ④ screen_papers (post-retrieval 重排)   │
│  → top-6 CandidatePaper                  │
└──────────────────────────────────────────┘
  │
  ▼
expand_evidence() → block 级检索 → Solver → Verifier → Composer
```

---

## 1. Intent Router — 第 1 次 LLM 调用

**文件**: `app/services/intents/router.py:155`

使用 Chat Model 的 tool-calling 模式，从 5 个工具中选择：

| 工具 | 用途 |
|------|------|
| `need_corpus_search` | 学术查询 → 检索论文库 |
| `answer_directly` | 元对话（问候/感谢/能力询问） |
| `need_clarify` | 歧义或低置信 |
| `need_web` | 外部搜索 |
| `need_conversation_tool` | 库状态/推荐/记忆追问 |

**System Prompt 核心规则**（已针对 GPT-4o 优化）:

> "用户询问任何术语、概念、方法、缩写、公式的含义或定义时，必须调用 need_corpus_search。调用时必须从用户问题中提取 targets（方法名、缩写、术语）。如果能识别出至少一个术语，targets 不能为空。"

**Router 同时顺带输出**（不增加额外 LLM 调用）:
- `targets`: 如 `["GRPO", "Group Relative Policy Optimization"]`
- `relation`: 如 `entity_definition`、`formula_lookup`
- `query`: 可能经过 LLM 语义扩展的查询文本
- `confidence`: 置信度

**产出**: `QueryContract` — 下游所有组件共享的语义契约。

---

## 2. Target Canonicalization

**文件**: `app/services/retrieval/core.py:1025-1052`

将 Router 输出的 targets 规范化为论文库中的实际名称。

**匹配范围**: 每篇论文的 title、aliases（标题缩写）、body_acronyms（正文提取的缩写词）。

**body_acronyms 提取**（入库时自动，非 LLM）: `indexing.py:346-379`
- 模式 1: `"Group Relative Policy Optimization (GRPO)"` → 提取 acronym + expansion
- 模式 2: `"GRPO algorithm/method/loss"` → 提取 acronym
- 模式 3: `L_{GRPO}` → 提取 acronym + LaTeX 形式

**当前限制**（已知问题）: 当 target 是短缩写（如 "GRPO"）且匹配到 body_acronyms 时，只返回缩写本身，不返回同篇论文中的对应全称（"Group Relative Policy Optimization"）。GPT-4o 等不知道此缩写的模型会因此丢失检索信号。计划改进：匹配到 body_acronyms 时自动补全 expansion。

---

## 3. 检索查询构建 (paper_query_text)

**文件**: `app/services/planning/query_shaping.py:83-97`

条件化构建 Dense 检索的查询文本——不是简单的 target 拼接：

```python
def paper_query_text_from_context(context):
    target_text = " ".join(context.targets).strip()
    if target_text and goals & {"definition", "entity_type", ...}:
        if len(context.targets) > 1:
            return f"{target_text} {clean_query}"  # 多target: 拼接
        return target_text                          # 单target definition: 仅用 canonical
    if target_text and target_text.lower() not in clean_query.lower():
        return f"{target_text} {clean_query}"       # 补充缺失术语
    return clean_query                               # 无需增强
```

**示例**:
- "LoRA是什么" + targets=["LoRA"] → 查询 `"LoRA"`（精确，避免噪声）
- "GRPO是什么" + targets=["GRPO", "Group Relative Policy Optimization"] → 查询 `"GRPO Group Relative Policy Optimization GRPO是什么"`
- "残差网络的核心思想" + targets=["ResNet", ...] → 查询 `"ResNet 残差网络的核心思想"`（补充英文术语）

---

## 4. search_papers — Dense + Title Anchor Fallback

**文件**: `app/services/retrieval/core.py:147-190`

### 4.1 主路径: Dense-only

`text-embedding-3-large` (3072-dim)，Milvus 向量检索，`paper_dense_top_k=12`。

**为何砍掉多路融合**：159题×12配置消融 (§11.5)，多路 WRRF (0.931) 在所有条件下不如 Pure Dense + QE (0.975)，且慢 6 倍。BM25 中文修复 (jieba, Hit@1: 0.176→0.748) 是过程产出，但融合不带来净增益。

### 4.2 Fallback: Title Anchor 注入

```python
anchor_docs = self.title_anchor(target_terms)  # 精确匹配 title/aliases/body_acronyms
dense_pids = {d.pid for d in dense_docs}
injected = [d for d in anchor_docs if d.pid not in dense_pids]  # 去重
fused = injected[:3] + dense_docs  # prepend，保证锚点论文进入候选池
```

**触发条件**: 仅当 Router 提取的 targets 在论文库的 body_acronyms 中有匹配时生效。不匹配 → 零开销（`title_anchor` 返回空列表）。

**解决的问题**: "缩写不在论文标题" 的盲区查询。如 "GRPO" → DeepSeekMath（标题不含 GRPO，但 body_acronyms 含）。Dense embedding 无法桥接此 gap（GRPO 被映射到 TRPO/PPO 的语义邻域），Title Anchor 的精确字符串匹配提供互补信号。

**与四路 WRRF 的区别**: 不是每问必跑的全路径，而是仅在 targets 命中 body_acronyms 时触发的安全网。不增加主流查询的延迟（<1ms，纯内存 dict 扫描）。

### 4.3 Post-retrieval 重排序

- `_paper_match_boost(doc, contract)`: targets 匹配 title (+0.6), content (+0.2), body_acronyms (+0.9) 加分
- `screen_papers(candidates)`: 综合 targets 匹配 + 年份优先 + 来源信号做最终排序
- 输出: top-6 `CandidatePaper`

---

## 5. 关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Embedding 模型 | text-embedding-3-large (3072-dim) | Qihai 网关 |
| Chat 模型 | deepseek-v4-flash (推荐) / gpt-4o | Router + Planner + Solver + Composer 共用 |
| paper_dense_top_k | 12 | Milvus 召回数 |
| paper_limit_default | 6 | 最终返回下游的论文数 |
| evidence_limit_default | 14 | 最终返回 solver 的证据块数 |
| max_agent_steps | 8 | Agent loop 步数上限 |
| max_calls_per_tool | 3 | 单工具调用上限 |
| pdf_hi_res_max_pages | 20 | 结构化提取页数上限（基于 3180 页评分分布分析） |

---

## 6. 消融数据摘要

159 题 × 12 配置 (3 策略 × 2 摘要 × 2 QE)，paper_query_text 作为 QE 机制:

| Strategy | Condition | Hit@1 | MRR | Lat(ms) |
|----------|-----------|-------|-----|---------|
| **Pure Dense** | **+Sum, +QE** | **0.9748** | 0.9845 | 827 |
| Pure Dense | +Sum, -QE | 0.9371 | 0.9646 | 1069 |
| BM25+Dense RRF | +Sum, +QE | 0.9308 | 0.9473 | 970 |

QE (paper_query_text): Dense +3.8pp, BM25 +11.3pp  
Summary: Dense +2.5pp, BM25 +0.6pp

---

## 7. 已知限制与改进方向

1. **canonicalize 不补全 abbreviation expansion**: 短缩写匹配到 body_acronyms 时不自动补全长全称。计划改进。
2. **GRPO vs PPO 对比查询**: 多 target 时 Title Anchor 对高频缩写（如 PPO 在数十篇论文中）的区分度不足。
3. **body_acronyms 提取噪声**: ~30-50% 的提取项是普通英文单词的全大写形式（LANGUAGE, THESE, LEARNING），非真正缩写。模式 2/3 过于宽松。
4. **大规模语料 (500+ 论文)**: 当前结论 "Pure Dense > 融合" 严格限于 113 篇。IR 理论 (BEIR/MTEB) 预测 BM25+Dense 在大规模下反超，但无实验数据。