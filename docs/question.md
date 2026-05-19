Q：一段话描述现在的正式链路
A：用户提交问题后，API 层先把请求交给 Agent。Agent 首先通过 intent router 和规则辅助逻辑把自然语言问题转换成 QueryContract，明确这是一个 research 请求、目标对象是什么、问题属于哪种 relation、答案需要哪些字段和证据类型。随后内置 planning 逻辑根据 QueryContract 生成 ResearchPlan，决定候选论文数量、证据数量和 solver 顺序。进入 tool loop 后，planner 安排 search_corpus：系统先用论文级索引召回多个 candidate_papers，再按目标和任务类型筛选成 screened_papers，接着在这些论文内部检索第一轮 evidence blocks。之后 read_memory 检查当前 session 是否已有可继承的研究上下文。进入 compose 阶段时，系统会先判断 evidence 是否暴露出实体/缩写歧义；如果存在多个候选解释，就调用 LLM-judge 做消歧，高置信度时自动绑定到最匹配论文，并在必要时对该论文做证据拓展。证据准备完成后，solver 根据 relation 和 answer_slots 生成 claim，例如公式 claim、起源 claim、总结 claim。verify_claim 再用 evidence 对 claim 做 grounding 校验；如果通过，answer composer 将 claim、证据和引用组织成最终 Markdown 回答；如果不足，则可能触发 retry、clarify 或给出证据不足说明。最后，Agent 将 final answer、citations、verification report 和 runtime trace 返回给前端展示。


Q：既然大改，那之前的状态机代码是否有意义？设计agent的过程是否需要先用状态机试试结果，再改为agent？
A：有价值。状态机阶段证明了 RAG 流水线可用、确立了数据契约（QueryContract/EvidenceBlock）、提供了回归基线。演化路径"状态机 → LLM 辅助 agent → 全 LLM 驱动"是正确的——每步验证后才迈下一步。

Q、该框架是否可以经过简单的改造直接变成其他领域的通用Agent？
A：可以。当前 20 个原子工具、LLM router、动态注册表、可配置规划表已是领域无关的。Zotero 特有部分在 retriever/library/specialized solvers——换掉这些就是通用 Agent 框架。

Q、本项目有没有办法处理跨页表格？
A、目前不支持跨页表格的自动合并。PDF 提取层使用两阶段策略——pypdf 全量扫描打分选出表格密集页，然后按配置选择高价值页面调 unstructured hi_res 做结构化提取。代码兜底默认是 6 页，但当前项目 `.env` 已覆盖为 `PDF_HI_RES_MAX_PAGES_PER_DOCUMENT=20`，现有入库产物也能看到单篇论文最多 20 个结构化页。跨页表格仍会被拆成两个独立文档分别入库，检索时可能只召回一半。解决方案是在 _elements_to_blocks() 之后加相邻页表格续接检测，比较行列模式匹配则合并。这是一个已知的架构限制，优先级取决于实际查询中跨页表格的命中频率。

Q：项目里怎么做意图识别，如何理解用户真正要解决的问题？
A：意图识别由 `LLMIntentRouter`（`intents/router.py`）驱动，分三层：

**第 1 层 — LLM Tool-Calling 粗分类**：Router 将 5 个路由工具（`answer_directly`、`need_conversation_tool`、`need_corpus_search`、`need_web`、`need_clarify`）绑定到 Chat Model，让 LLM 通过 tool-calling 选择最合适的行动。这一步在 ~5-15 秒内完成，核心是判断"用户问题是否需要检索本地 PDF"。

**第 2 层 — Contract 精细化**：`query_contract_from_router_decision()` 将 Router 的粗粒度选择结构化——提取 `targets`（如 "DPO"、"Transformer"）、推断 `relation`（如 `formula_lookup`、`paper_summary_results`、`origin_lookup` 等 20+ 种）、确定 `requested_fields` 和 `required_modalities`（如需要公式、表格、图表）。这一步不是 LLM 调用，而是规则+词典的组合。

**第 3 层 — 会话上下文消解**（`contract_extraction.py`）：拿到初始 contract 后，还要走多层加工——followup 关系继承（"第一篇"指代上一轮的哪个推荐？）、contextual resolver（从会话历史中解析实体引用）、conversation memory（从之前轮次的 target_bindings 中查找已消歧的实体）、clarification 处理（上轮 Agent 反问后用户的选择）。最终产出完整的 `QueryContract`。

**Router 还承担了 token 规划**：Router 输出的 `planned_actions`（如 `["read_memory", "search_corpus", "compose"]`）直接作为初始工具计划，跳过了单独的 Planner LLM 调用。一次 tool-calling 完成了意图分类 + target 提取 + 工具规划三件事。

Q：QueryContract 里面那些字段到底是什么意思？
A：`QueryContract` 可以理解为“自然语言问题被系统翻译后的任务单”。它不是答案，而是后续检索、规划、求解和校验共同遵守的结构化约定。

一个典型例子是用户问“Transformer架构最先由哪篇论文提出？”：

```json
{
  "clean_query": "Transformer架构最先由哪篇论文提出？",
  "interaction_mode": "research",
  "relation": "origin_lookup",
  "targets": ["Transformer"],
  "answer_slots": ["origin"],
  "requested_fields": ["paper_title", "year", "evidence"],
  "required_modalities": ["paper_card", "page_text"],
  "notes": ["router_planned_actions=search_corpus,compose", "intent_confidence=0.95"]
}
```

`interaction_mode` 决定走闲聊/工具型 conversation，还是走本地论文库 research；`relation` 是问题类型，例如 `origin_lookup`、`formula_lookup`、`entity_definition`；`targets` 是用户真正追问的对象；`answer_slots` 是最终答案必须填出来的槽位，比如 origin、formula、definition；`requested_fields` 是答案需要的字段；`required_modalities` 是检索时优先看的证据类型，例如论文卡片、正文块、表格、图片；`notes` 是路由和后续模块之间传递的附加信号，比如置信度、是否需要澄清、Router 顺手生成的工具计划。

Q、本项目混合检索方式
A、最初设计为 Weighted RRF 四路召回（Title Anchor 1.6 / Relation Anchor 1.3 / BM25 0.9 / Dense 0.8）。159 题 × 12 配置消融后，结论是 **Pure Dense + paper_query_text QE 在所有条件下均最优**（Hit@1=97.5%），多路融合不如 Dense 且慢 6 倍。当前默认检索为 Dense-only，BM25、Title/Relation Anchor 保留为可选模块。详见 §11.5。

面试策略：先介绍四路设计展示系统思维，再说明消融数据推动务实简化。

Q、如何入库的，数据来源是什么？
A、数据来源是用户本地 Zotero 论文库（zotero.sqlite + storage 目录）。入库由 `scripts/ingest_rebuild.py` 驱动，`IngestionService.rebuild()` 执行 5 步流程：读取 Zotero 元信息 → PDF 两阶段抽取 → 生成 paper_card（论文级）→ 生成 block 文档（块级，RecursiveCharacterTextSplitter 800/120）→ 写入 Milvus 向量索引。最终形成 paper + block 两级索引结构，JSONL 供 BM25/Title Anchor 检索（可选），Milvus 供 Dense 检索（默认）。详见项目文档 §4 数据流与索引设计。

Q、四路各自查什么？Relation Anchor 是怎么实现的？
A、四路全部查 paper_card，Weighted RRF（k=60）融合：

| 路 | 权重 | 方式 |
|---|------|------|
| Title Anchor | 1.6 | 精确匹配 target 是否出现在 title/aliases/body_acronyms |
| Relation Anchor | 1.3 | 从锚点论文提取标签/缩写词/作者/Zotero 分类作为指纹，遍历非锚点论文按共享信号打分 |
| BM25 | 0.9 | jieba 分词后 TF-IDF 匹配 |
| Dense | 0.8 | Milvus 向量语义检索 |

Relation Anchor 的信号权重：Zotero 分类 +2.5（用户手动整理，最强）、共享标签 +1.8、共享作者 +1.2、共享缩写词 +1.0（最多 5 个）。

> **当前状态**：消融数据证伪了多路融合——最优混合配置 Hit@1=0.931，不如 Pure Dense + paper_query_text QE（0.975），且慢 6 倍。当前默认 Dense-only。四路设计在面试中可展示架构能力，最终决策体现务实态度。详见 §11.5。

Q：检索的 topK 是怎么设计的？
A：

| 参数 | 默认值 | 当前状态 |
|------|--------|---------|
| paper_dense_top_k | 12 | 直接用 top-6（经 screen_papers 重排序） |
| paper_limit_default | 6 | 最终返回下游的论文数 |
| evidence_limit_default | 14 | 最终返回 solver 的证据块数 |

BM25 相关参数（paper_bm25_top_k=12 等）保留但默认不启用。最初设计为 12+12 双路融合后筛选到 6，已简化为 Dense-only。

Q：PDF解析是RAG的重灾区。你的"图文异构切块"具体是怎么做的？
A：采用**两阶段策略**（`pdf_extractor.py`）：

**阶段 1 — pypdf 全量快速扫描**：用 pypdf 提取全部页面的原始文本，对每页计算 11 维信号（caption 锚点密度、数值密度、短行比例、分隔符模式、图片数量等），合成 3 个得分：`table_like_score`（≥2.5 入选）、`figure_like_score`（≥2.5）、`scanned_like_score`（≥2.0，扫描版 PDF 文本量极少）。

**阶段 2 — unstructured hi_res 定向精析**：选取得分最高的最多 20 页（当前 `.env` 为 `PDF_HI_RES_MAX_PAGES_PER_DOCUMENT=20`；代码里的兜底默认值是 6），调 `unstructured.partition.pdf(strategy="hi_res", infer_table_structure=True)` 做结构化提取。输出 table/figure/caption 三类 ExtractedBlock，每个 block 带 bbox 坐标。caption 通过 bbox 空间距离自动匹配到最近的 table/figure。这里把上限提高到 20，是为了覆盖更多表格/图像密集页，代价是入库会明显更慢。

**切块策略**：hi_res 产出的结构化块（table/figure/caption）以完整内容单条入库，不做切分。page_text 用 `RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180, separators=["\n\n", "\n", "。", " ", ""])` 切分，保证中文句号处断句、段落边界优先。

**已知限制**：跨页表格不支持自动合并（ExtractedBlock 只有单页字段）；公式在 PDF 提取层不特殊处理，而是在检索层通过 `formula_hint` 正则（π/β/sigma/log σ/loss/objective 等）和 `retrieval_formula_token_weights` 做加权匹配。

Q：图像类问答引入了VLM，图像是如何入库的？
A：图像**不入库**——不单独做多模态 embedding（不用 CLIP/ColPali）。走的是"文本检索 + VLM 按需分析"的路径：

**入库阶段**：PDF 解析（unstructured hi_res）检测到 Image/Picture 时，存储为 `ExtractedBlock(block_type="figure", caption=<匹配到的最近标题>)`。只有 caption 文本参与 embedding 索引（JSONL + Milvus），图片本身不向量化。

**查询阶段**：Router 识别到 figure 查询 → `figure_solver` 被调度 → 用 pdf2image 将目标 PDF 页渲染为图片（`pdf_render_dpi=180`）→ 图片 + caption + query 作为 multimodal content 发给 VLM（`vlm_model=gpt-4.1-mini`）→ VLM 直接从图片中提取结论/数据，返回结构化 Claim。

**信息折损怎么解决**：折损主要在检索阶段——如果 caption 写得差（如 "Figure 1" 无实质描述），Dense 检索可能找不到该图。一旦检索命中，VLM 直接看原图，不存在信息折损。折损风险集中在 caption 质量上，可通过改进 PDF 解析时的 caption 匹配逻辑缓解。

Q：BM25 + Milvus 混合检索的得分是如何融合的？最终选择了哪种方案？
A：最初设计 Weighted RRF 四路融合（k=60, 权重 1.6/1.3/0.9/0.8）。消融后 Pure Dense + paper_query_text QE 最优（Hit@1=0.975），多路融合最优仅 0.931。关键工程发现：BM25 默认分词对中文完全失效（Hit@1=0.176），接入 jieba 后恢复到 0.748+。最终选择 Dense-only + paper_query_text QE。面试展示：设计 → 实验 → 数据驱动决策的完整闭环。

Q：你的检索策略最终选择了什么？经历了怎样的迭代过程？
A：**最终方案：Pure Dense + paper_query_text QE**（Milvus 向量检索 + Router targets 条件化查询构建）。

| 阶段 | 发现 | 行动 |
|------|------|------|
| 初始设计 | 四路 Weighted RRF | 实现并上线 |
| BM25 诊断 | 中文完全失效（Hit@1=0.176） | 接入 jieba → 恢复到 0.748 |
| 系统消融 | 12 配置：QE +3.8pp，摘要 +2.5pp | 简化为 Dense-only + paper_query_text |
| 最终决策 | Pure Dense + QE = 0.975 | 多路融合保留为可选 |

消融数据详见项目文档 §11.5。

Q：Query Enhancement的具体策略是什么？
A：核心是 **LLM-driven Semantic Target Extraction + Canonicalization**，三步完成：① Regex 初提取（零成本）→ ② LLM Router 在 tool-calling 时顺带输出 targets（不增加额外调用）→ ③ canonicalize_targets() 规范化为论文库中的实际名称。最终通过 `paper_query_text(contract)` 条件化拼入检索查询——对 definition 查询仅用 canonical target（如 "LoRA"），对描述性查询追加 Router 术语。消融数据：Dense +3.8pp，BM25 +11.3pp。详见项目文档 §11.5。

Q：你是怎么验证检索设计是正确的？做了哪些消融实验？
A：2 因子（摘要 × QE）× 3 策略 = 12 配置全矩阵，159 题评测集（easy/medium/hard 三层）。关键发现：Pure Dense +Summary +QE 最优（0.975），paper_query_text QE +3.8pp，摘要 +2.5pp，多路融合在所有条件下不如单路 Dense。决策：简化为 Dense-only + paper_query_text QE。详见项目文档 §11.4-11.5 和 docs/测试结果.md。

Q：目前对一次用户提问，LLM 调用了多少次？分别在做什么？
A：3-4 次（已通过 Router+Planner 合并优化，原为 4-5 次）：① **Intent Router（含 tool planning）**——tool-calling 一次完成意图分类 + target 提取 + 初始工具序列规划（Router 输出的 `planned_actions` 被 Planner 直接采用，跳过单独的 Planner LLM 调用）→ ② **Solvers**（从 evidence 提取 claims，1 次，schema 或 deterministic）→ ③ **Verification**（0-1 次，大部分是规则校验不走 LLM）→ ④ **Answer Composer**（汇总生成最终答案，1 次 streaming）。Router 提取的 targets 通过 `paper_query_text(contract)` 条件化拼入检索查询（QE）。详见项目文档 §8 Agent 主链路。

Q：Router + Planner 合并是怎么做的？
A：Router 的 `need_corpus_search` tool schema 新增了 `planned_actions` 字段，LLM 在做意图分类时顺带输出建议的工具序列（如 `["read_memory", "search_corpus", "compose"]`）。Planner 检查 contract notes 中是否有 `router_planned_actions=`，如果有且 actions 都在可用工具集中，直接采用，跳过 Planner LLM 调用。节省 1 次 LLM 调用（10-15s）。如果 Router 未提供或 actions 无效，回退到正常 Planner 流程。

Q：为什么自己写 Agent Loop，而不是用 LangGraph / LangChain AgentExecutor / AutoGen 等现成框架？
A：自建 Agent Loop（`execute_tool_loop` in `runtime_helpers.py`）而非依赖第三方框架，基于三个考量：

**1. 执行模型需要精确控制**：LangGraph 的 StateGraph 是 DAG 语义，适合预定义的分支/合并流程，但本项目的需求是"plan-then-loop"混合模型——先用 Planner 生成初始计划队列，执行过程中队列空了再调 LLM 动态重规划。这个"队列优先 + 队空时 LLM 决策"的模式用 LangGraph 表达反而更绕。自己写的 `execute_tool_loop` 不到 100 行，逻辑完全透明。

**2. 状态管理比框架更简单**：113 篇论文的封闭域不需要分布式状态。当前用 `dict` + `SessionContext` (Pydantic) 做内存状态，SQLite 持久化，比 LangGraph 的 Checkpointer 体系轻量得多。没有序列化/反序列化的心智负担。

**3. 工具系统已有自己的注册和执行机制**：`AgentToolSpec`（20 个内置工具）+ `RegisteredAgentTool`（handler + 依赖解析）+ `AgentToolExecutor`（并行执行 + 去重）。这套三层工具系统直接对接 LangChain 的 `ChatOpenAI.bind_tools()`，不需要 LangGraph 的 tool node 抽象。

简单说：**需求够简单（单一 Agent、少量工具、封闭域），自建 loop 比引入框架的收益更大**。如果未来扩展到多 Agent 协作或需要复杂的条件分支，LangGraph 的 `StateGraph` 和 `Command` 机制会更有价值。

Q：那项目现在完全不用 LangChain 了吗？
A：不是。准确说是**不用 LangChain/LangGraph 来编排 Agent 主循环**，但底层能力仍复用了 LangChain 生态。比如 `ChatOpenAI.bind_tools()` 用来做 Router tool-calling，`OpenAIEmbeddings` 用来接 embedding，`langchain_milvus` 负责 Milvus 向量库封装，`langchain_core.documents.Document` 是检索文档载体，`RecursiveCharacterTextSplitter` 用于文本切块，BM25 retriever 也来自 LangChain 社区包。

所以面试时不要说“完全不用 LangChain”，而应该说：**Agent orchestration、SessionStore、Tool Loop、Verification 是自研；模型客户端、Document 抽象、Milvus/BM25/TextSplitter 等基础组件复用了 LangChain。**

Q：Agent 会不会陷入"检索-校验失败-重试"的死循环？退出机制怎么设计？
A：有四层防护，确保不会死循环：

**第 1 层 — 步数上限**（`max_agent_steps=8`）：`execute_tool_loop` 的 `for` 循环硬上限，超过 8 步直接终止，不管是否完成。

**第 2 层 — 单工具调用上限**（`max_calls_per_tool=3`）：同一工具（如 `search_corpus`）最多被调用 3 次，防止 Planner 反复尝试同一个失败的检索。

**第 3 层 — 终止条件**（`stop_condition`）：Research 路径的终止条件是 `verification.status ∈ {pass, clarify}`。一旦 verifier 产出了通过或需澄清的判断，loop 就停。Conversation 路径的终止条件是 `answer` 字段非空。

**第 4 层 — 澄清次数上限**（`max_clarification_attempts=2`）：如果 verifier 反复返回 `clarify`，达到上限后走 `force_best_effort_after_clarification_limit`——直接用最可能的意图走一遍 `search_corpus → compose`，打上 `best_effort` 标签，**强制作答**。同时 verifier 被标记为 `pass`，loop 终止。

**此外**：Planner 的 `fallback_plan()` 在所有 LLM 规划失败时返回兜底序列 `["search_corpus", "compose"]`，确保最坏情况下也能走完检索→回答的基本流程。整个系统在任何异常路径上都保证有出口。

Q：证据校验（Evidence Verification）的具体逻辑是什么？如何保证校验步骤的可靠性？
A：校验不是简单把最终回答丢给 LLM 让它“感觉一下对不对”，而是分层处理：先做 evidence-id 白名单审计，防止 claim 引用不存在的证据；如果 claim 来自 schema solver，会尝试一次 schema LLM verifier；随后进入按 claim 类型分发的规则/半规则 verifier（`verifier_pipeline.py` + `type_verifiers.py`）。

**调度机制**：根据 `QueryContract` 的 goals（由 Router 确定），选择对应的 verifier check。例如 `origin` → `verify_origin_lookup_claims`，`definition` → `verify_entity_definition_claims`，`formula` → `verify_formula_lookup_claims`，通用兜底 → `verify_general_question_claims`。

**主要校验逻辑**：
- `origin`：检查 claim 是否包含 paper_ids + evidence_ids，且 claim 文本或其关联的 evidence 中是否包含目标术语的"引言/提出"模式（如 "we propose", "we introduce"）——纯字符串匹配 + 位置打分
- `metric`：检查 claims 是否非空即可
- `formula`：先检查 formula claim 是否存在、公式值是否真的像公式，再调用公式 LLM verifier 判断公式是否被 evidence 支撑、是否对应用户 targets；如果 LLM verifier 不可用，再退回目标对齐规则
- `concept`：检查 claims 是否包含 evidence citation（至少有一条 evidence 被引用）
- `general`：检查目标对象是否被当前 papers/evidence 支撑

**返回值**：verifier 返回 `VerificationReport`，status 为 `pass`（通过）、`retry`（重试）或 `clarify`（需澄清）。`retry` 触发 agent loop 重新执行（受步数上限约束），`clarify` 触发澄清流程（受澄清次数上限约束）。

**如何防止 LLM 幻觉**：关键是 claim 必须绑定真实 evidence_ids/paper_ids，引用还会过 citation whitelist；LLM verifier 只参与部分结构化判断，不允许凭空引入新证据。也就是说，系统不是相信 LLM 的“自我评价”，而是要求每个 claim 回到已召回证据里接受检查。

Q：目前代码里有 LLM-as-judge 吗？
A：有，但要分清用途。线上链路里真正稳定使用的是**消歧 judge**：当同一个 target 可能对应多篇论文或多个解释时，系统把候选 option、证据片段、用户问题交给 LLM，让它输出 `auto_resolve` 或 `ask_human`、`selected_option_id`、`confidence` 和 reason。只有置信度达到阈值时才自动消歧，否则应该反问用户。

另一个是**证据 verifier 里的 LLM 辅助校验**，例如 schema claim verifier 和 formula verifier。它不是通用“评分裁判”，而是检查 claim 是否被 evidence 覆盖、公式是否和目标一致。评测意义上的 LLM-as-judge 可以单独做离线评估，但当前主链路不是靠一个大而全的 LLM judge 给最终答案打分。

Q：前端 trace 里 `confidence=0.95` 是怎么来的？
A：要看它出现在哪个阶段。Router 的 intent confidence 来自 LLM tool-calling 输出，写进 contract notes，例如 `intent_confidence=0.95`；消歧阶段的 confidence 来自 LLM judge 对候选解释的选择置信度，达到阈值才会 `auto_resolve`；verification 阶段的 confidence 不是模型直接打分，而是 `confidence_from_verification_report()` 按 status 计算：`pass` 基础约 0.9，有 missing/unsupported/contradiction 会扣分，`retry` 约 0.35，`clarify` 为 0。answer logprobs/self-consistency 也有实现，但默认关闭。

Q：Solvers 和 Compose 具体在做什么？

A：**Solvers**（claim 提取层）和 **Answer Composer**（答案生成层）是 Agent 链路中靠后的两个阶段，职责完全不同：

**Solvers** — 从检索到的 evidence blocks 中提取结构化 Claim。这里的“提取”不是固定一种方式，而是按 `plan.solver_sequence` 调度不同 handler：简单事实、论文元数据、起源类问题尽量用规则生成；公式、定义、机制、图表理解这类语义抽取任务会调用 LLM/VLM。

- `formula_solver`：先按公式信号挑选 evidence，再调 LLM 从公式密集片段中提取目标函数、变量说明、公式 LaTeX；如果 LLM 不可用或没有返回合法公式/evidence_ids，则回退到规则窗口抽取。当前代码没有真正启用 formula VLM 分支
- `text_solver` / schema claim solver：调 LLM，从文本 evidence 中抽取概念定义、机制解释、结构化结论
- `concept_definition_solver`：LLM + 规则兜底，抽取概念定义、类别、支撑证据
- `figure_solver`：渲染 PDF 页为图片 → 调 VLM 直接从图中提取结论
- `table_solver` / metric solver：先按 table/caption/metric token 给 evidence 排序；如果开启 `enable_table_vlm` 且能把表格/caption 所在 PDF 页渲染成图片，就调用 VLM 结合表格文本和页面截图抽指标；如果 VLM 不可用、没有图片、或没有返回合法 claim，就回退到文本规则抽取指标行
- `origin_solver`：主要是规则匹配，在 evidence 中搜索 "we propose/introduce/present"、"本文提出" 等溯源线索，确定原始论文
- 论文摘要、推荐、元数据类 claim：多数是从 candidate paper 的 title/year/summary/doc_ids 中规则组装

每个 solver 产出一个或多个 `Claim`（包含 claim_type、text/value、evidence_ids、paper_ids、confidence 等字段）。重点是：LLM 可以参与 claim extraction，但不能随便当最终事实来源；claim 必须绑定真实 `evidence_ids` / `paper_ids`，后续 verifier 和 citation whitelist 会检查这些 ID 是否存在、是否和当前 evidence 对得上。

**Answer Composer** — 汇总所有 Claim + Evidence + Papers，调 LLM 生成最终答案和引用列表。流程：① 从 claims 中提取 evidence_ids → 构建 citations（`AssistantCitation`，含 title、page、snippet）→ ② 根据 claim 类型选 compose 策略（结构化研究答案 / Markdown 长文 / 简单回复）→ ③ 调 LLM 生成最终 answer 文本，支持 streaming callback 逐 token 输出。如果 verification 返回 `clarify`，composer 改为生成澄清问题而非答案。

Q：流式服务如何区分文本 Token 和节点状态/证据引用？用的 SSE 还是 WebSocket？

A：用 **SSE（Server-Sent Events）**（`/api/v1/chat/stream`），`FastAPI StreamingResponse` + `text/event-stream`。前端通过 SSE 的 `event` 字段区分数据类型，不需要解析消息内容来判断类型：

```
event: contract           ← 结构化意图
data: {"relation":"entity_definition","targets":["GRPO",...]}

event: agent_plan         ← 工具执行计划
data: {"actions":["search_corpus","solve","verify","compose"]}

event: answer_delta       ← LLM 生成的文本 token（逐字流式输出）
data: {"text":"GRPO"}

event: candidate_papers   ← 检索到的候选论文
event: screened_papers    ← 筛选后的论文
event: evidence           ← 召回的 evidence blocks
event: claims             ← solver 提取的 claims
event: verification       ← 校验结果
event: confidence         ← 置信度
event: thinking_delta     ← Agent 思考过程
event: observation        ← 工具执行结果摘要
```

**为什么是 SSE 而不是 WebSocket**：这个场景是单向推送（服务端→前端），不需要前端随时发消息。SSE 比 WebSocket 更轻量——原生 HTTP、自动重连、不需要心跳维护。前端通过 `EventSource` API 直接按 event type 注册不同 handler，流式文本渲染到回答区，结构化事件更新侧边栏状态面板。

Q：多用户并发时，会话状态在服务端如何管理？
A：不用 LangGraph Checkpointer。自建 `SessionStore`（`memory/session_store.py`），`SQLiteSessionStore` 单文件存储：每个 session 一行 `session_id` + `context_json`（`SessionContext.model_dump_json()`），`threading.RLock` 保证并发安全。前端每次请求带 `session_id`，服务端查 SQLite → 反序列化 → Agent 链路修改 → `commit_turn()` 写回。不同 session_id 完全隔离。SQLite WAL 模式足够应对个人论文库工具的并发量；如需水平扩展，接口不变，换成 Redis/PostgreSQL 即可。

Q：目前agent的记忆系统是怎么设计的，分为几个层次？分别用什么技术存放在哪里？
A：三层记忆，从短到长：

**第 1 层 — 会话短期记忆**（`SessionContext.turns` + `summary` + `turn_index`）：每次对话轮次存为 `SessionTurn`，`SessionStore.commit_turn()` 同时把 `turn_index` 加 1，作为这个 session 的绝对轮次计数。生产环境 `agent_history_max_turns=24`，接近上限时 `compress_session_history_if_needed()` 会调 LLM 把较早轮次压缩成 3-6 句摘要写入 `session.summary`，只保留最近一部分 turns；保存时 `_trim_context_history()` 还有非 LLM 的裁剪兜底。Router/Planner 看到的是近期 turns + 更早的压缩摘要。存储在 SQLite 的 `sessions` 表（JSON 序列化）。

**第 2 层 — 工作记忆**（`SessionContext.working_memory`）：跨轮次的知识绑定 dict。核心字段 `target_bindings`——每轮研究通过验证后，`remember_research_outcome()` 将 target → paper_id/title/year/evidence_ids/created_turn_index 的映射写入。后续轮次中 Router/Contract 层遇到同一 target 时可以直接从 binding 取已消歧论文，不需要重新从候选论文里猜。同存于 SQLite sessions 表。

**第 3 层 — 持久学习**（`data/learnings/*.md`）：Answer Composer 通过 `_load_assistant_self_knowledge()` → `load_learnings()` 加载所有 learnings 文件拼接内容（最多 6000 字符），注入对话类回复的 system prompt。Agent 的自我认知（身份、领域范围、回答规则、库状态处理）存储为 `assistant-self-knowledge.md`。用户可直接编辑或通过 API 追加新文件，无需改代码。存储为本地 markdown 文件。

简单说：SQLite 存会话状态，working_memory 跨轮复用检索结果，markdown 文件做持久知识积累。

Q：你简历上写的 LangGraph，但代码里根本不是——这不是造假吗？
A：不是造假，是迭代。项目经历了三个阶段：早期版本确实基于 LangGraph StateGraph 做 agent 调度（简历反映的是那个阶段）；后续消融实验发现当前的"plan-then-loop"自定义模型在灵活性和性能上都优于 LangGraph 方案，所以做了替换。简历还没来得及更新。

如果你追问"LangGraph 方案和现在哪个更好"，我会诚实说：**场景决定**。LangGraph 的优势是可视化 DAG、Checkpointer 开箱即用、条件分支声明式定义——适合多 Agent 协作或复杂分支流程。但本项目是单 Agent + 动态重规划："队列优先执行 → 队列空了 LLM 重规划"这个模式用 LangGraph 的 `add_conditional_edges` 需要绕弯（把队列状态编码进 graph state，用条件边模拟出队/重规划循环），而自建 loop 不到 100 行直接表达。当前方案更简单透明。如果未来扩展到多 Agent，我会重新评估 LangGraph 的 `Command` 机制。

Q：Caption 缺乏实质信息时，VLM 检索不到——这个漏洞怎么解决？
A：你说得对，这是现有方案的已知弱点。当前缓解措施：① hi_res 阶段 caption 通过 bbox 空间距离自动匹配到最近的 figure/table，且 page_text chunk 中如果紧邻出现图表的描述性文本（如 "As shown in Figure 3, the accuracy..."），检索时仍能命中；② VLM 看图时不仅依赖 caption，同时接收整页渲染图和用户查询，即使 caption 是 "Figure 1" 也能从图中直接提取信息。

如果加强：可以在入库阶段对每个 figure block 调一次轻量 VLM 生成描述性 alt-text（"该图展示了 XX 方法在 YY 数据集上的准确率对比..."），存为 caption 补充，成本是每张图一次 VLM 调用。目前未做是因为 figure 查询频率低，投入产出比不划算。

Q：在 Evidence Verification 阶段，你说为了防止 LLM 幻觉，你的 origin_solver 退回到了纯规则匹配，比如去匹配字符串 'we propose', 'we introduce'。
这种做法在工程上极其脆弱。如果这是一篇中文论文，写的是‘本文提出’呢？如果作者用的是被动语态 'is presented in this paper' 呢？面对多语种、多风格的学术语料，你的正则表达式能覆盖多少情况？你为了防 LLM 的幻觉，引入了硬编码的规则缺陷，这难道不是一种技术倒退吗？如果让你重新设计，如何在不使用死板正则的前提下，低成本且高鲁棒地完成溯源校验？”
A：这个批评对早期版本成立，但对当前版本要更精确地回答：`origin_solver` 仍然是规则驱动，所以不是无限鲁棒；但它已经不只是匹配英文 `we propose / we introduce / we present`。

当前实现集中在 `origin_target_intro_score()` 和 `origin_claim_has_intro_support()`：① 支持英文主动表达，如 `propose / introduce / present / define / construct / create / release` 及其三单、过去式；② 支持中文表达，如“提出 / 引入 / 介绍 / 定义 / 构建 / 发布”；③ 支持一部分英文被动语态，如 `is/was/has been introduced/proposed/presented/defined...`；④ 不只看 claim 文本，还会检查 candidate paper 的 `paper_card_text / generated_summary / abstract_note`、paper document 和关联 evidence；⑤ 判断 cue 是否出现在 target alias 附近，并结合年份、候选论文分数和 evidence 得分排序。

所以它不是“中文必漏”的脆弱正则，而是一个轻量符号溯源器。真正的局限在于：它依赖显式 cue 和 target alias，如果论文用非常隐晦的表达、跨句指代、非中英文、或者 source paper 没进候选池，规则仍然可能漏。相比让 LLM 全权判断，它的优点是可解释、低成本、不会凭空编证据；缺点是召回边界受规则覆盖范围限制。更合理的后续方向是继续扩展多语言 cue、加入 citation graph / 年份先验 / title anchor 召回，而不是把 origin 校验完全交给 LLM。

Q：你提到 Query Enhancement (QE) 给你带来了 3.8% 的召回率提升，这是通过 LLM Router 在 tool-calling 时顺带提取 Target 并拼接实现的。
但这意味着，用户发出的每一次查询，都必须先等 LLM 完成一次思考和输出，才能开始 Milvus 向量检索。LLM 的首字延迟（TTFT）加上生成 Target 的时间，通常在 500ms 到 1.5s 之间。为了这区区 3.8% 的提升，你让用户的‘检索前置延迟’翻了数倍。在真实的 C 端或高并发场景下，这样的延迟是不可接受的。
你有做过 Latency 的消融对比吗？有没有更轻量的 Target 提取方案？

A：做过的。消融数据显示 Dense+QE 延迟 827ms，Dense-QE 延迟 1069ms——QE 竟然还更快。因为 `paper_query_text` 对 definition 查询直接返回 canonical target（如 "LoRA"），检索词更短更精确，Milvus 更快。Router 调用本身是必经之路（要做 intent routing），target 提取不增加额外 LLM 调用。

如果追求更低延迟：① Regex 初提取（`extract_targets()`）本身已经覆盖了引号内容和全大写缩写，对于 "PPO是什么" 这类查询 regex 就能拿到 "PPO"，不需要等 LLM；② 只有中文描述性查询（如 "残差网络的核心思想"）才依赖 LLM 提取英文 target。可以做成**渐进式**：regex 先提取 → 如果是中文且 regex 为空，再调 LLM——大部分查询被 regex 拦截，延迟不受影响。

Q：你通过 159 题的评测集得出了一个结论：Pure Dense (Hit@1=0.975) 吊打所有融合方案，所以你把 BM25 默认关掉了。
我对这个结论持怀疑态度。你用的是哪款 Embedding 模型？学术论文中充满了极其生僻的专有名词、新造的缩写（例如某个冷门蛋白的名字，或者非常规的算法缩写）。稠密向量（Dense）在处理 OOV（未登录词）和强符号逻辑串时天生存在平滑缺陷，而这恰恰是 BM25 精确匹配的绝对强项。
你真的确信 Dense 在所有长尾硬核查询上都表现最好吗？还是过拟合了 159 题？

A：我不确信。你说得对，159 题评测集可能存在过拟合风险。`text-embedding-3-large` 的 subword tokenization 对 OOV 有一定鲁棒性（"GRPO" 被拆成 subword tokens 仍能部分匹配），但 BM25 对精确字符串匹配（如 "L_GRPO"、"π_θ"、"β·KL"）的区分度是 Dense 无法替代的。

当前结论"Pure Dense > 融合"严格限定在**113 篇论文 + 3072 维 embedding** 的条件下成立。扩展到更大规模时情况可能反转——IR 领域 BEIR 基准和 MTEB 都表明，hybrid（Dense + BM25）在大规模语料上优于纯 Dense。原因是 Dense 存在"hubness"问题：语料增大后 embedding 空间拥挤，相似论文互相干扰；而 BM25 的 IDF 权重天然压制高频词、放大罕见词，对于长尾术语的区分度不随语料增长而退化。

如果有 500+ 甚至 1000+ 篇论文，我会预期 BM25 + Dense RRF 反超 Pure Dense。但目前没有这个规模的实验数据——这是诚实的局限性，也是后续工作的方向。

Q：整个开发过程中排查时间最长的 Bug 是什么？怎么解决的？

A：**”Transformer”起源查询总是返回 ViT 或 BERT，而不是 Attention Is All You Need。** 这个问题折腾了最久，因为它不是”代码写错”——是 Dense embedding 的语义特性与起源类问题的需求发生了根本冲突。

**现象**：问”Transformer架构最早在哪篇论文提出”，Dense 检索返回的第一名永远是 ViT（An Image is Worth 16x16 Words）或 BERT，而不是正确答案 Attention Is All You Need。因为 ViT 把 Transformer 用到了图像、BERT 是 Transformer 的经典应用——在 3072 维语义空间中，这些论文的 embedding 都和 “Transformer” 高度相关。Dense 无法区分”使用了 Transformer 的论文”和”提出了 Transformer 的论文”。

**根因**：起源类问题不只是一个检索问题，它需要**溯源信号**——哪篇论文在正文中”提出/引入/定义”了这个概念。Dense embedding 捕捉了语义相关性，但完全丢失了这个溯源信号。

**解决方案**：专门为 origin 类 query 写了 `origin_solver`（`origin_selection.py`），不走 LLM，纯规则+打分：
1. 在候选论文的 paper_card 和 evidence 中搜索 `origin_cue` 正则（覆盖中英文 30+ 种引言模式：”we propose/introduce/present”、”本文提出/引入/定义”）
2. 计算 `origin_target_intro_score`：检查 target term（如 “Transformer”）是否出现在 introduction cue 的 ±200 字符范围内
3. 结合 title 匹配度、年份优先级、snippet 证据质量做加权排序
4. 如果没有任何论文命中 intro 模式，退回到 Dense 排序但标记 “low confidence”

这本质上是把”语义检索”和”符号溯源”解耦——Dense 负责召回候选池，origin_solver 负责在池内找到真正的提出者。现在问”Transformer最早提出”，Attention Is All You Need 排第一，ViT 和 BERT 被正确压在后面。

---

**Bug 2：SSE 流式连接频繁中断——根因是 AI 生成的 Caddy 配置误加了压缩**。这个 bug 排查了整整一天，涉及后端代码、网络层、反向代理、CDN 四个层面，最终通过 git blame 锁死到一个 AI 写的 commit。

**症状**：前端提问”DPO的公式是什么？”，SSE 流式连接 ~33 秒处断开，前端显示”流式中断，切换普通请求...”→ fallback 非流式请求也失败→”Failed to fetch”。偶发能返回结果时答案也经常错误。简单查询（<5s）完全正常。

**排查过程**采用严格的控制变量法：

第一轮修了 6 个后端 bug（precision_requirement 校验崩溃、SSE payload 含 63KB 的 embedding 向量、LLM planner 选劣化工具、LLM 调用无超时保护、答案无逐字效果），每个都让系统更健壮，但流式中断依然存在。

第二轮做三路对比测试矩阵——Agent 直连✅、Caddy:8080 反代❌、Cloudflare HTTPS❌，直接锁定故障在 Caddy 层。curl verbose 输出的关键线索：`(18) transfer closed with outstanding read data remaining`——Caddy 提前关闭了 chunked transfer。

第三轮 git 考古——`git log -- deploy/caddy/Caddyfile` 发现 commit `731ecb1`（2026-05-02，author: `ubuntu@localhost.localdomain`，AI session）新增了 `pdf_rag_v4_routes` 片段。AI 从已有的 `pdf_rag_v3_routes` 复制了模式，照搬了 `encode zstd gzip` 到所有 v4 API 路由上。这是典型的 AI 代码生成中的”模式盲从”——复制粘贴者不知道每条路由的语义差异：静态资源压缩是好优化，但 SSE 流式响应走压缩，压缩器的内部缓冲会破坏 chunked transfer 的分块边界，导致 Caddy 在 ~33 秒处发送不完整的终止 chunk。

**解决方案**：给 `/chat/stream` 端点单独路由，去掉压缩，保留 `flush_interval -1`：

```caddy
handle /api/v1/chat/stream {
    reverse_proxy 127.0.0.1:8001 {
        flush_interval -1
    }
}
```

**教训**：① SSE 流式响应不能走 HTTP 压缩；② 控制变量法是排查网络层 bug 最有效的手段；③ `git blame` + `git diff` 考古能快速定位引入 bug 的 commit；④ curl `-v` 输出的错误码是排查 HTTP 层问题的关键线索；⑤ AI 生成的配置代码（Caddyfile/nginx/Dockerfile）需要人工审查——AI 擅长复制模式但不理解语义差异。


Q：你的 origin_solver 本质上是一个 Post-retrieval（检索后） 的处理逻辑。它发挥作用的前提，是 Attention Is All You Need 这篇论文已经存在于 Dense 召回的 Top-K 候选池里。
如果我的论文库有 10 万篇，关于 Transformer 的衍生论文就有 5000 篇。当用户搜‘Transformer 首次提出’时，可能排在 Dense Top-20 的全是最近的高引衍生论文，而真正的源头论文因为年代久远或标题不够‘夺人眼球’，被挤到了第 50 名，根本没进候选池。这时候你的 origin_solver 连看都看不到它，这个问题怎么解？”

A：这个质疑对“只做 Dense Top-K 后处理”的方案成立，但当前实现已经不完全是这个结构。`origin_solver` 的入口确实会接收 Dense 检索得到的 `candidate_papers` 和 evidence；不过 `select_origin_paper()` 还会额外调用 `origin_candidates_from_corpus()`，遍历 `retriever.paper_documents()` 里的 paper-card / generated_summary / abstract_note，用 target alias + origin cue 再补一批候选。因此，如果 Attention Is All You Need 没进 Dense Top-K，但它的 paper-card 或摘要中有 “Transformer is introduced / 提出 Transformer” 之类线索，仍然有机会被补进候选池，再由 origin score、年份和证据分数排序选中。

所以更准确地说：它不是完全受限于 Dense Top-K，但也不是无限召回。它当前补召回扫的是论文级文档，不会实时扫全库所有 block；如果源头论文的 paper-card/summary/abstract 没有显式 origin cue，或者 target alias 没写准，仍可能漏。10 万篇规模下，每次 origin 查询线性遍历 paper documents 也会变成性能问题。

改进方向有三个层面：① 检索层——对 origin 类 query 加 title anchor + BM25 / paper-card cue 检索，避免只依赖 Dense 语义相似度；② 索引层——把 `target alias + origin cue` 离线建成一个轻量倒排表或 SQLite FTS 表，而不是每次线性扫描；③ 离线知识层——在入库阶段结合 citation graph、年份、标题和 introduction cue，预先标注“首次提出 X”的候选关系。这样既保留 origin_solver 的可解释性，又能支撑更大规模语料。

Q：2026-05 专家架构审视发现了哪些关键问题？做了哪些修复？
A：专家对 pdf-rag-agent-v4 进行了全面代码审查，识别出 21 个问题（P0-P2 三级），全部已修复。最关键的 4 项：

- **引用幻觉防护**（P0-1/2/3）：新增 citation_whitelist.py 后置过滤器——LLM 生成的回答中所有《XXX》引用、斜体标题、方括号编号必须通过 evidence/citations/screened_papers 白名单校验，违规触发 retry。claim evidence-id 审计改为严格子集校验（issubset），伪造 id 不再被放行。

- **复合任务 integrity**（P0-4/5）：comparison_synthesis 之前丢弃子任务 evidence 导致比较 LLM 只能瞎猜，现已传入 top-4 evidence。子任务 clarify 不再全局回退——收集所有 blocked subtasks 后统一返回，已完成结果持久化不丢失。

- **Router 输出 sanitization**（P1-9）：LLM router 经常输出无效的 continuation_mode 值（如 context_continuation），此前 _string_value 直接透传导致 Pydantic ValidationError。现新增 _sanitize_continuation_mode 做规范化映射。

- **Prompt 注入防护**（P1-3/2-6）：entity、topology、concept_definition、formula 四个 LLM 路径此前对 untrusted evidence 无保护。现统一用 wrap_untrusted_document_text 包裹 evidence snippet，并注入 DOCUMENT_SAFETY_INSTRUCTION。

全部修复清单详见 docs/expert-review-2026-05-04.md。

Q：模型的调用是怎么做容错和超时的？
A：`ModelClients` 的 httpx 客户端统一配置了 `timeout=Timeout(60.0, connect=10.0)`。ChatOpenAI 实例额外设置了 `request_timeout=90.0` + `max_retries=1`。Agent 总超时通过 `asyncio.wait_for(worker, timeout=110.0)` 兜底，超时后返回"请求超时"提示而非挂死。所有 LLM 调用（invoke_text / invoke_json / invoke_tool_plan）在 model_clients 内部有 try/except 包裹，失败时返回 fallback 值。stream_text 有 2 次重试——首次失败且已有部分 chunks 时直接使用已有结果，避免回退到非流式。embedding 调用在 vector_index.py 中失败时自动禁用 dense 搜索 10 分钟，到期自动恢复。

Q：为什么代码中到处有 v4/v5 版本号？后来怎么处理的？
A：早期开发阶段沿用的版本标记。后来做了一次全量清理：API 路由 `/api/v1/v4/*` → `/api/v1/*`，前端页面 `/v4`/`/v5 ` → `/`，静态文件 `v4.html` → `index.html`，Caddy 配置 `pdf_rag_v4_routes` → `pdf_rag_routes`。内部类名（如 `ResearchAssistantAgentV4`）和数据文件名（如 `v4_papers.jsonl`）保留不变——类名改名需改动 100+ 引用风险太高，数据文件名是实际存储路径不应改动。

Q：Redis 缓存是怎么用的？为什么不缓存最终答案？
A：当前 Redis 只缓存两个可复用中间结果，不缓存最终 LLM 回答。第一类是 query embedding，key 形如 `zprag:v4:emb:query:<embedding_model>:<query_hash>`，TTL 30 天；第二类是 dense retrieval 结果，key 形如 `zprag:v4:retr:dense:<index_version>:<request_hash>`，TTL 6 小时。`query_hash` 来自规范化后的 query，`request_hash` 会把 collection、embedding_model、query、limit、filter 等参数纳入哈希；dense retrieval key 还带 `index_version`，当 JSONL、collection、embedding model 等索引依赖变化时自动换 key。

不缓存最终答案是刻意设计。最终回答依赖 session history、active research、QueryContract、用户消歧选择、证据集合、模型输出、web search 开关等上下文。直接用原始 query 当 key 缓存 final answer，容易把上一轮语境、旧索引或错误消歧结果带到下一次请求里。Redis 在这里是性能优化层，负责减少 embedding API 和 Milvus dense search 的重复成本；会话记忆仍由 SQLite SessionStore 保存。

Q：长查询（50-180s）的用户体验怎么改善？
A：四个层面：① **TTFT（首字节感知）**——Router 调用前立即 emit "正在分析问题意图..." thinking 事件，用户无需等 10-15s 才看到第一个反馈；② **逐字流式**——答案阶段后端 drip-feed 10 字符小块输出，前端实时逐字渲染；③ **心跳保活**——独立 asyncio task 每 10s 发 heartbeat SSE 事件，防止代理/CDN 因静默超时掐断连接；④ **Redis 中间结果缓存**——重复或相同检索请求可以跳过 query embedding / dense retrieval 的部分成本，但仍要走 Router、solver、verification、composer，不承诺同 query 30ms 返回最终答案。

Q：Prometheus 的 `/metrics` 是什么？里面的数据从哪来？
A：`/metrics` 是 Prometheus scrape endpoint，由 `prometheus_fastapi_instrumentator` 在 FastAPI app 上暴露。Prometheus 定时访问这个 HTTP 接口，把文本格式的指标拉走存入时序数据库。指标来源有三类：① FastAPI HTTP 指标，例如请求总数、状态码分布、请求耗时；② `prometheus_client` 默认运行时指标，例如 Python GC、进程 CPU/内存、线程等；③ 项目自定义 Agent 指标，目前主要是 `tool_calls_total{name, ok}` 和 `tool_latency_seconds{name}`，由 agent tool 执行器和 observation event 记录，用来观察哪个工具调用多、失败多、耗时长。

Q：为什么会有多个候选论文？证据拓展又是什么？
A：用户问题往往先指向一个概念，而不是唯一论文。比如“Transformer 最早由哪篇论文提出”会召回 Attention Is All You Need、BERT、ViT 等多篇相关论文；Dense 检索只能说明“相关”，不能直接说明“谁是源头”。所以系统先在论文级索引召回多个 `candidate_papers`，再用 relation、target、年份、标题、证据线索筛成 `screened_papers`。

证据拓展是指：当 compose/消歧阶段发现某个候选论文更可能是正确对象，但第一轮 evidence 不够支撑最终回答时，系统围绕这个已选论文继续补查更多 block，例如正文、公式、图表或引言段。它不是扩大到全库乱搜，而是在“已确认/高概率确认的论文”内部补证据，让后续 solver 和 verifier 有足够材料生成 claim。

Q：target_bindings 的 TTL 是怎么设计的？
A：P0-8 和 P1-7 联合修复了 target_bindings 的两个缺陷。P0-8 防止 best_effort（澄清耗尽后的妥协回答）污染永久绑定——新增 `VerificationReport.original_status` 字段标记来源，best_effort 结果不写入 `target_bindings`，仅写入临时 `_temp_best_effort_bindings`。P1-7 给 `SessionContext` 增加单调递增的 `turn_index`，每次 `commit_turn()` 递增；`remember_research_outcome()` 写 binding 时记录本轮的 `created_turn_index`；`_expire_old_bindings()` 用当前 `turn_index - created_turn_index > 20` 判断过期。这样即使 `turns` 列表被压缩或裁剪，TTL 仍按真实会话轮次生效。

Q：这次 API Key 泄漏后，项目做了哪些安全修复？
A：这次事故的根因有两层：第一，早期调试脚本里曾经硬编码过供应商 API key，虽然当前文件后来改掉了，但 Git 历史仍能看到；第二，公网 Caddy 把 `/api/v1/chat` 和 `/api/v1/chat/stream` 直接反代给后端，而 chat 接口当时没有鉴权，外部用户即使拿不到供应商 key，也可以让后端代为调用模型。

修复措施按“止血、封入口、清历史”三层完成：

1. **止血**：废弃泄漏过的旧 key，新 Qihai key 只写入本地 `.env` 的 `EMBEDDING_API_KEY` 和 `VLM_API_KEY`，没有写入代码、文档或 Git；`.env` 继续由 `.gitignore` 忽略。
2. **供应商 key 与访问 key 分离**：新增 `CHAT_API_KEY`，它只是访问本项目后端 chat API 的本地门禁 key，和 Qihai / DeepSeek 供应商 key 分开。前端只需要填写 `CHAT_API_KEY` 或 `ADMIN_API_KEY`，绝不能填写供应商 key。
3. **后端鉴权**：新增 `require_chat_access()`，`POST /api/v1/chat` 和 `POST /api/v1/chat/stream` 都必须带 `X-API-Key` 或 `Authorization: Bearer ...`，且 key 必须匹配 `CHAT_API_KEY` 或 `ADMIN_API_KEY`。
4. **限流**：新增 `CHAT_RATE_LIMIT_PER_WINDOW`，默认每个请求来源 60 秒内最多 8 次 chat 请求，避免公网接口被批量刷。
5. **前端配合**：前端输入框改为 `CHAT / ADMIN_API_KEY`，发送 chat 请求时自动把该值放进 `X-API-Key`。这个值会存在浏览器 localStorage，仍然只是项目后端访问 key，不是模型供应商 key。
6. **错误信息收敛**：上游模型服务临时失败时，对用户返回“请等待 10 秒后重试”这类短提示，避免把过多上游错误细节透出给前端。
7. **Git 历史清理**：使用临时仓库重写远端 `main` 历史，把历史中所有 `sk-...` 格式密钥替换为占位符，并 force push 到远端；本地删除含泄漏历史的备份分支，过期 reflog 后做 GC。

验证结果：本地和公网无 key 调用 `/api/v1/chat`、`/api/v1/chat/stream` 都返回 `401 invalid api key`；安全相关测试 `tests/test_security.py` 和 `tests/test_api_routes.py` 通过。后续仍应在供应商平台继续设置额度上限、日限额、模型白名单或 IP 白名单，因为 Git 历史清理不能让已经泄漏过的旧 key 重新变安全。
