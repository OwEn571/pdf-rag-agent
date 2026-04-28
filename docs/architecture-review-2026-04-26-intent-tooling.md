# Agent 意图理解与工具调度专项评审

日期：2026-04-26

范围：`pdf-rag-agent-v4` 当前 agent 改造后的意图理解、工具选择、运行时调度链路。

## 结论摘要

改造方向是对的：拆分 Planner / Runtime / Tool Registry，并引入 OpenAI tool calling。但“agent 不智能”的根因仍在：系统本质上仍是“22 路 relation 分类 + 固定工具流水线”，LLM 只承担路由器和文本生成器角色，真正的工具选择被硬编码序列夺走了。

要让 Agent 自己理解意图、自己选工具、自己决定何时回答，需要三步：

1. 意图识别从“22 选 1”改成“少量正交维度的结构化 intent”。
2. 工具调度从“relation -> 固定序列查表”改成 observation 驱动的 ReAct loop。
3. solver / verifier / composer 的 if-relation 分支合并为 schema 化 LLM 调用。

## 一、为什么“你好”会被当成论文召回

当前走读路径位于 `agent.py:_extract_query_contract`：

```text
clean_query
  -> _llm_route_query
  -> _fallback_query_contract
  -> _normalize_conversation_tool_contract
  -> _augment_contextual_acronym_contract
  -> _refine_followup_contract
  -> _normalize_training_component_question_contract
  -> _normalize_conversation_tool_contract
  -> _inherit_followup_relationship_contract
  -> _normalize_followup_direction_contract
  -> _apply_conversation_memory_to_contract
```

这条链有多层 normalize / refine / augment / inherit，每一层都可能改写 `relation`、`targets`、`continuation_mode`。

### 1. 22-relation 大 prompt 不是可靠分类器

`agent.py` 中的 system prompt 要求 LLM 在 22 个 relation 之间单选：

- `greeting`
- `self_identity`
- `capability`
- `library_status`
- `library_recommendation`
- `clarify_user_intent`
- `correction_without_context`
- `memory_followup`
- `memory_synthesis`
- `library_citation_ranking`
- `origin_lookup`
- `formula_lookup`
- `followup_research`
- `entity_definition`
- `topology_discovery`
- `topology_recommendation`
- `figure_question`
- `paper_summary_results`
- `metric_value_lookup`
- `concept_definition`
- `paper_recommendation`
- `general_question`

这些 relation 之间语义重叠，例如 `concept_definition` / `entity_definition` / `general_question`，`library_recommendation` / `paper_recommendation`，`memory_followup` / `memory_synthesis`。边界模糊时，简单寒暄也可能被分到 `clarify_user_intent` 或 `general_question`。一旦落到默认 research 模式，就会进入 `search_papers`。

### 2. Greeting 兜底正则写成了字符类

现有逻辑：

```python
if re.fullmatch(r"[\s,.!?。！？你好哈嗨hello hi]+", lowered):
```

这是字符类，不是词集合。结果是：

- `hi` 会匹配；
- `你好吗` 不匹配，因为 `吗` 不在字符类中；
- `helloo`、`hihi` 甚至单字符 `o` 都能匹配；
- 它只在 LLM router 返回 `None` 时使用，通常挡不住 router 的错误分类。

### 3. 后置层会把 conversation 改回 research

`_apply_conversation_memory_to_contract` 在 session 中存在 `active_targets` 时会把 contract 强行视作 follow-up。随后 `_inherit_followup_relationship_contract` 又可能用 working memory 覆盖目标。只要上一轮做过 research，下一句“你好”就可能被 memory 命中改成 `research + followup`，再被 research 工具链塞入检索。

## 二、“假 Tool Calling”：计划层的最大漏洞

`invoke_tool_plan` 和 `AgentPlanner.plan_with_tool_calls` 看起来已经接入了真正的 tool calling，但 runtime 会把 LLM 输出重新并入固定序列：

```python
actions = research_tool_sequence(
    planned_actions=raw_actions,
    use_web_search=web_enabled,
    needs_reflection=...
)
```

`research_tool_sequence` 强制加入：

```text
understand_user_intent
reflect_previous_answer?
LLM planned actions
search_papers
search_evidence
web_search?
solve_claims
verify_grounding
compose_or_ask_human
```

这意味着只要进入 research 通路，无论 LLM 选择了什么工具，都会被并集成固定流水线。conversation 通路也类似：`conversation_tool_sequence` 使用 relation 到固定 list 的映射。LLM 选错工具会被丢弃，选对工具也只是被合并进硬编码序列。

结论：当前仍是“router 选 relation -> 查表得到固定流水线”。LLM 不能主动跳过 search，不能调换工具顺序，不能在 observation 后决定下一步。

此外，`RegisteredAgentTool.requires` 会自动补依赖。例如调用 `solve_claims` 会自动级联 `search_papers` + `search_evidence`，即使未来 runtime 放开，LLM 也无法表达“只 verify 不 search”。

## 三、仍残留的硬编码

| 位置 | 残留 | 影响 |
| --- | --- | --- |
| `agent.py` fallback query contract | 多个关键词分支决定 relation | LLM router 返回 `None` 时仍走硬编码分类 |
| `_normalize_training_component_question_contract` | `reward model` / `critic` / `value model` 命中后覆盖 relation | LLM 已分类还会被覆盖 |
| `_normalize_followup_direction_contract` | 正则推断“A 是 B 的后续吗” | 与模型分类重复，并可能重写 |
| `retrieval.py` | `PPO` / `DPO` / `LDPO` 等硬编码加分 | 召回阶段引入论文偏置 |
| `CONVERSATION_RELATION_TOOL_SEQUENCES` | relation 到固定 list | conversation 通路 LLM 无决定权 |
| `research_tool_sequence` | 必塞 search / solve / verify / compose | research 通路 LLM 无决定权 |
| 旧 planner prompt 与新 planner prompt | 业务规则重复 | prompt 成为第二事实源，规则会漂移 |

## 四、建议解法

### Step 1：用“小维度 + 槽位”代替“22-relation 单选”

不要让模型在扁平大 enum 里单选，改为正交维度：

```python
class Intent(BaseModel):
    intent_kind: Literal[
        "smalltalk",
        "meta_library",
        "research",
        "memory_op",
    ]
    needs_local_corpus: bool
    needs_web: bool
    refers_previous_turn: bool
    target_entities: list[str]
    user_goal: str
    confidence: float
    ambiguous_slots: list[str]
```

所有旧 relation 可以从这些字段映射出来，但模型不再被迫做 22 选 1。实现上使用 structured output 或 tool calling 强制 schema，温度 0。返回后用纯函数 `Intent -> ToolPlan`，逐步删除 `_normalize_*` 后置层。

### Step 2：真正的 ReAct Loop

删除固定工具序列，让 runtime 基于 observation 循环决策：

```python
state = ToolState(intent=intent, evidence=[], claims=[], ...)
for step in range(MAX_STEPS):
    decision = llm.tool_call(
        system=TOOL_LOOP_SYSTEM,
        observations=state.observations,
        intent=intent,
        budget=remaining_budget,
    )
    if decision.tool == "finalize":
        break
    obs = TOOLS[decision.tool](decision.args, state)
    state.observations.append(obs)
answer = compose(state)
```

工具应压缩为少量原子能力：

| Tool | 作用 |
| --- | --- |
| `search_corpus(query, k, modalities=[])` | 统一走 Milvus + BM25 |
| `web_search(query, k, kind=...)` | 外部搜索 |
| `read_memory(scope=...)` | 读取 session memory |
| `compose(answer_shape, evidence_ids=[])` | 写答案 |
| `ask_human(question, options=[])` | 澄清问题 |

### Step 3：删掉 relation 分支式 solver / verifier / composer

`solver_pipeline.py` 的多个 `_solve_*_text` 可以合并为一个“given evidence, produce structured claims”调用。

`claim_verifier.py` 的多个 `_verify_*_claims` 可以合并为一个“given claims + evidence, return covered / missing_slots / contradictions”调用。

`answer_composer.py` 的 if-relation 分支应退化成 `answer_shape + required slots`，由 schema 化 prompt 约束。

### Step 4：让 ask_human 由置信度驱动

当前 `ask_human` 触发仍主要靠关键词推断。应由 intent 模型输出：

```python
confidence: float
ambiguous_slots: list[str]
```

当 `confidence < 0.6` 或 `ambiguous_slots` 非空时直接触发澄清。

### Step 5：意图层与工具层共享同一 prompt-as-code 源

不要在 router prompt、planner prompt、工具序列里重复写“何时调用 X”。规则应只出现在工具 description 和 args schema 中。模型通过工具描述理解能力边界。

## 五、最小 PR 顺序

| PR | 改动 | 文件 | 工作量 |
| --- | --- | --- | --- |
| PR-1 | 替换意图层：结构化 Intent schema；删除或旁路危险 `_normalize_*_contract` 后置层 | `app/services/intent.py` 新增，`agent.py` 精简 | 1 天 |
| PR-2 | 22 relation -> 少量工具，Runtime 改 ReAct loop | `agent_runtime.py`，`agent_tools.py` | 1.5 天 |
| PR-3 | solver / verifier / composer 合并为 schema 化 LLM 调用 | `agent_mixins/*.py` | 2 天 |
| PR-4 | `ask_human` 走 confidence 而不是关键词 | `agent.py` | 0.5 天 |
| PR-5 | retrieval 去硬编码加分，boost 改 config | `retrieval.py` | 0.5 天 |

## 2026-04-26 落地状态

本轮已按该蓝图推进到 PR-1 / PR-2 / PR-3 主路径，并接入 PR-5 的配置化召回改造：

- 新增 `app/services/intent.py`，主路由改为结构化 `Intent`：`intent_kind / needs_local_corpus / needs_web / refers_previous_turn / target_entities / user_goal / answer_slots / confidence / ambiguous_slots`。
- `agent.py:_extract_query_contract` 不再调用 22-relation `_llm_route_query`；旧 router 与关键词 fallback 已物理删除，新链路通过 `IntentRecognizer -> QueryContract adapter` 接入下游。
- 本地 protected intent 修复了 greeting 正则问题，“你好 / 你好吗 / hi”等寒暄不会被 session memory 改写成 research。
- `agent_tools.py` 暴露给 planner 的工具从 18 个业务步骤收敛为 5 个原子工具：`read_memory / search_corpus / web_search / compose / ask_human`。
- `agent_runtime.py` 不再使用“planned actions + required sequence”的并集流水线，改为预算内 tool loop：先执行 planner 选择，随后基于 observation/state 选择下一步，旧工具名仅作为兼容别名。
- `conversation_tool_sequence` / `research_tool_sequence` 不再按 relation 兜底生成固定队列；当 planner 没有给出 actions 时，runtime 的 next-action policy 根据当前 state 决定是否读记忆、搜语料、web、compose 或 ask_human。
- planner 的 JSON / tool-call 输出只接受 5 个 canonical tools；旧工具名只在 legacy adapter 和历史测试中保留，不再作为模型可选能力面暴露。
- `agent_tools.py` 的 allowed/executable tool set 已收缩为 canonical 5 tools；`build_*_tool_registry` 对 runtime 只返回 `read_memory / search_corpus / web_search / compose / ask_human`。
- `_record_agent_observation` / `_emit_agent_tool_call` 增加 canonical event gate：内部兼容函数即使仍调用 `search_papers`、`solve_claims` 等旧实现，SSE、`execution_steps` 和 `runtime_summary` 对外也只暴露 5 个工具名。
- 旧 `_fallback_query_contract` 和 22-relation `_llm_route_query` 已从 `agent.py` 物理删除；意图入口只保留 `IntentRecognizer -> QueryContract adapter`。
- research registry 移除了 `solve_claims -> search_papers/search_evidence` 的自动依赖；`compose` 内部暂时承接 solve/verify 兼容层。
- `solver_pipeline.py` 新增通用 schema claim extractor 主路径。生产环境有 chat model 且 evidence 足够时，先走“given evidence -> structured claims”；离线 fallback 也已由 claim goals 驱动。
- `claim_verifier.py` 新增通用 schema verifier 主路径。schema solver 产出的 claims 先走“given claims + evidence -> pass/retry/clarify”；离线 fallback 也已由 verification goals 驱动。
- `answer_composer.py` 对 schema solver 产出的 claims 优先走统一 LLM composer；旧结构化 composer 保留为离线/确定性 fallback。
- `solver_pipeline.py` 的主 fallback 调度已从 `solver_sequence -> relation handler` 改为 `required_claims / requested_fields / answer_slot / modality -> claim goals`，旧 `_solve_*` helper 只作为 deterministic implementation detail 使用。
- `claim_verifier.py` 的主 fallback 调度已从 `contract.relation -> verifier` 改为 verification goals；不再维护 relation verifier 分发表。
- `answer_composer.py` 的结构化 fallback 已从 relation if-chain 改为 claim_type 驱动，LLM evidence budget 也改为 requested fields / modalities 驱动。
- `agent.py:_build_research_plan` 已移除 `sequence_map / required_map / paper_limit_map / evidence_limit_map` 这组 relation 查表；现在统一从 `requested_fields / answer_slot / required_modalities / query cues` 推导 research goals，再生成 recall budget、solver metadata 和 required claims。
- `retrieval.py` 的 origin / followup / formula / figure / metric / summary 加权已改为 goal/modality 驱动，不再直接按 `contract.relation` 给 block type 或年份加分。
- planner 的 JSON/tool-call payload 不再把 relation 当作模型决策输入，改传 intent kind、answer slots、requested fields、modalities 和 canonical tool manifest。

- 复合问题分解 prompt 不再要求 LLM 输出 relation enum；新 schema 使用 `answer_slots / requested_fields / required_modalities` 描述子任务，旧 relation payload 仅作为历史测试和兼容 adapter 输入。
- research composer 的 LLM prompt 已从“如果 relation 是 X”改为“如果 claim_types / requested_fields 包含 X”，避免回答模板成为另一套 relation 业务规则。
- `retrieval.py` 去掉代码层 `PPO` 等特殊论文 boost；paper match boost 与公式 token 权重迁移到 `Settings`：`retrieval_paper_match_boosts`、`retrieval_formula_token_weights`、`retrieval_target_formula_token_weights`、`solver_metric_token_weights`。
- API 响应新增 `runtime_summary`，把结构化 intent、planned/observed canonical tools、verification、claim source 统计压成前端友好的摘要。
- `/api/v1/v4/health` 现在返回 `runtime_profile=structured-intent-react-loop`、`runtime_summary_supported=true` 和 canonical tools，用于识别线上服务是否已加载新版后端。
- `app/static/v4.html` 右侧 Run 面板同步为 `Intent & Tool Loop`：展示 intent kind/confidence/targets/slots、planned tools、observed tools、grounding 状态；Debug 面板新增 `Runtime Summary`，顶部 health 现在显示 `ok · react-loop` 并暴露 canonical tools，避免前端继续用旧 relation pipeline 叙事解释新后端。
- 测试已更新为新架构契约，并通过全量测试。

## 2026-04-26 行为回归修复

用户实测暴露了两类“架构改完但答案仍错”的问题：

- `AlignX 数据集有后续工作吗？` 被通用 schema claim solver 抢先生成了过强的 followup claim，绕过更保守的关系验证器。
- `PBA 公式是什么` 在 verification 已经返回 `retry_formula_target_alignment` 后，composer 仍把未通过 grounding 的 PPO/CLIP 公式渲染成最终答案。

已修复：

- `formula / followup / strict_followup` 这类高精度任务不再走通用 schema claim solver，改回专门 deterministic solver + verifier。
- formula canonicalizer 只有在用户目标明确是 `PPO` 时才允许返回 PPO clipped surrogate；目标是 `PBA` 时不会因为 evidence 里出现 PPO/clip 就套公式。
- composer 遇到 `verification.status=retry` 时不再继续组织 claim 答案；`target_aligned_formula` 缺失会明确说明“不能确认公式”，并拒绝把 PPO 公式当作 PBA 公式。
- pending ambiguity 只接受明确选择（序号、精确含义、选择类措辞）；“这不是 PPO 的公式”这类纠错句不会被误当成选择第一个 PBA 含义。
- 新增回归测试覆盖：PBA 不渲染 PPO 公式、formula/followup 禁用 schema solver、纠错句不消费 pending disambiguation。

## 2026-04-26 多轮 PBA / AlignX 公式修复

继续实测发现上一节只是挡住了错答，并没有真正接住用户的定位信息：

- `PBA 的公式是什么` 失败后，系统仍把失败检索里的候选论文写进 `target_bindings`，导致下一句 `有 PBA，就在 AlignX 的论文里` 被错误绑定到 PersonaDual。
- 用户给出精确标题 `From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment` 时，router 把标题片段误抽成 `User-level` 概念解释，丢失上一轮 `PBA + formula` 任务。
- AlignX 原 PDF 第 6 页存在 PBA Eq. (2)，但公式抽取器没有 PBA canonicalizer，不能稳定输出 `L_PBA` 及 `\tilde{P}` 变量解释。

已修复：

- 只有 `verification.status == pass` 的研究结果才会写入 `target_bindings`；`retry` 不再污染会话记忆和 active titles。
- 新增公式任务的论文定位 resolver：当上一轮是公式任务，用户说“就在 AlignX 那篇 / 在某标题中”时，直接生成 `formula_lookup + selected_paper_id`，保留上一轮目标 `PBA`。
- 新增 active paper context binding：如果当前活跃研究上下文是 AlignX，用户追问 `PBA 公式`，且该论文中存在目标相关公式块，会自动限定到 AlignX 原文。
- 短缩写公式查询已接入 ambiguity detection；当证据里能稳定构造多个缩写含义时会先澄清，不再假装只有一个含义。
- pending ambiguity 现在保留原始任务类型；用户选择某个公式含义后仍继续 `formula_lookup`，不会退化成 `entity_definition`。
- PBA canonical formula 支持 AlignX Eq. (2)，输出 `L_{\mathrm{PBA}}`，并补充 `\pi_\phi`、`\tilde{P}` 的变量解释。
- retry 文案改为按目标和指定论文动态说明，不再硬编码 PBA 或 PPO。
- 新增回归测试覆盖：`PBA -> 就在 AlignX 那篇`、`PBA -> 在 From 1,000,000 Users... 中`、活跃 AlignX 上下文下直接问 `PBA 公式`。

## 2026-04-26 AlignX 来源任务修复

继续实测 `ALignX 是哪篇论文提出的？` 暴露了 origin/source 任务的另一处硬伤：

- intent 已能识别为 `origin_lookup`，但 solver 仍把普通相关性和年份排序当作来源判断依据。
- verifier 只检查 claim 是否有 `paper_title` 和 evidence id，没有确认证据是否真的表达“提出 / 引入 / introduce / propose”。
- 因此后续使用或评测 AlignX 的论文，例如 PersonaDual 或 Text as a Universal Interface，也可能被误写成“提出 AlignX”的论文。

已修复：

- `origin / paper_title / year` 已列为高精度任务，不再走通用 schema claim solver。
- origin solver 增加全库 paper-card / abstract 扫描；真实来源论文即使别名中没有写 `AlignX`，也能通过 `we introduce AlignX` 一类语义证据被召回。
- origin 排序必须有目标附近的 introduction/proposal cue；“使用 AlignX / 在 AlignX 上评测 / benchmark 包含 AlignX”只算相关，不再能支撑来源 claim。
- origin verifier 现在会检查 claim paper 和 citation evidence 是否存在目标附近的 `introduce / propose / 提出 / 引入 / 构建 / 发布` 证据；缺失则返回 `retry_origin`。
- 新增回归测试覆盖：大小写异常的 `ALignX`、高分但仅使用 AlignX 的噪声论文、以及 verifier 拒绝无来源语义的 claim。

仍待后续处理：

- `QueryContract.relation` 仍作为 API、历史测试、session memory 和若干 conversation handler 的兼容字段保留；新 planner/runtime/research plan/retrieval/composer 主路径已不再把它作为工具选择事实源。
- `solver_pipeline.py`、`claim_verifier.py`、`answer_composer.py` 仍保留一些命名为 `_solve_*`、`_verify_*`、`_compose_*` 的 deterministic helper；它们已不再由 relation 分发表调度，后续主要是命名和重复逻辑削减。
- `followup_routing.py` 仍承担少量历史追问修复和 relation adapter 兼容；主入口已由 structured intent 接管，后续可把它进一步并入 memory resolver。

## 2026-04-26 Claude Code follow-up 诊断

本次 follow-up 走读范围包括：`intent.py`、新版 `agent_runtime.py`、`agent_planner.py`、收敛到 5 个工具的 `agent_tools.py`，以及仍承担大量兼容逻辑的 `agent.py` 与 6 个 mixin。结论是：结构层面的改造已经有明显进步，但“多轮注意力涣散”和“PBA 公式找不到”不是单点 bug，而是几条剩余设计错误叠加的结果。

### A. 多轮注意力涣散的根因

1. 意图识别、planner、composer 仍主要把历史压缩成 JSON 塞进 prompt，而不是把真实 user / assistant turns 作为原生 chat messages 传给 LLM。`_session_llm_history_messages` 已存在，但覆盖面不足。
2. `Intent.answer_slots` 仍会经 `_research_relation` / `_research_requirements` 压回单一 relation。多槽问题会被硬序优先级截断，例如 “DPO 公式是什么，PPO 公式又是什么” 仍容易只解一半。
3. `_apply_conversation_memory_to_contract`、`_inherit_followup_relationship_contract`、`_normalize_followup_direction_contract` 会在意图识别后根据 session memory 强行覆盖 contract。缺少 `continue / switch / new` 三态话题判断，只靠 `refers_previous_turn: bool` 分不清追问和换题。
4. `SessionContext` 虽已引入 `ActiveResearch`，但仍保留 9 个 legacy active 字段，且多处写入仍可能造成字段错位。后续应继续把写入收敛到 `SessionStore.commit_turn`。

### B. PBA 公式找不到的根因

1. 召回层 anchor 只看 title 和 title aliases，正文中定义的 acronym 不会成为 paper anchor。PBA 常见于正文定义和公式附近，不一定出现在标题或 paper card 中。
2. formula solver 仍存在单论文假设：`_solve_formula` 只用 `papers[0]`，即使多篇论文都召回，也只能产出一篇论文的一条 claim。
3. `_canonical_formula_text` 对 DPO / PPO / PBA 仍有硬编码白名单。PBA 分支依赖固定短语和固定变量集合，遇到不同展开或符号变体会直接失败。
4. `_best_formula_window` 对 PDF 抽取后的断行、unicode 数学符号和 narrative 段落鲁棒性不足，容易挑到邻近但不属于目标的公式。
5. verifier 中 “非 PPO/DPO 且 evidence 含 clip + advantage/ratio 就 reject” 的黑名单会误杀真实 PBA 证据，因为 PBA 论文常在公式附近讨论 PPO/DPO 差异。
6. acronym 目标匹配过于依赖英文词边界，`L_{PBA}`、`Lpba`、大小写或下标变体可能匹配失败。

### C. 新的最小 PR 顺序

| PR | 改动 | 目标 |
| --- | --- | --- |
| PR-1 | 所有 LLM 路由 / planner / composer 输入改用原生 chat messages，非语言状态只放 system prompt 尾部 | 修复指代消解、短追问、纠错 follow-up 的注意力问题 |
| PR-2 | `refers_previous_turn` 升级为 `topic_state = continue / switch / new`，并输出 `active_topic` 与 `target_aliases` | 只有 continue 才允许 session memory 覆盖，阻断话题漂移 |
| PR-3 | 删除 `_research_relation` 单选 if 链，contract 直接持有 slots，下游按 slot 和 paper 维度分发 | 支持多槽、多论文并行 solver |
| PR-4 | ingestion 抽取正文 acronym / alias 并写入 block 和 paper metadata；retrieval 用 body acronym 做 anchor | 修复 PBA、DeepSeek-R1、AlignX 等正文 acronym 召回 |
| PR-5 | 删除 formula canonical 白名单与 verifier token 黑名单，改为 evidence -> structured formula claims + LLM verifier | 修复 PBA 公式和多定义公式 |
| PR-6 | ask_human 由 intent confidence / ambiguous_slots 驱动 | 让澄清早于错误召回和错误 composer |
| PR-7 | `ActiveResearch` 成为唯一 active context，写入收敛到 commit_turn，增加 topic signature | 降低多轮字段错位和状态漂移 |

### D. 本轮已执行

1. PR-1 / PR-2 第一阶段：意图识别、planner tool-call、planner JSON fallback 和 research composer 已优先使用原生 chat messages；非语言状态放入 system prompt 尾部。`Intent` 新增 `topic_state`、`active_topic`、`target_aliases`，并对旧 `refers_previous_turn` 保持兼容。
2. session memory 覆盖已加 topic gate：`_apply_conversation_memory_to_contract` 和 `_inherit_followup_relationship_contract` 只有在 `topic_state=continue` 时才允许继承 active context。
3. PR-4 第一阶段：ingestion 写入 `body_acronyms` 和 block 级 `mentioned_acronyms`；retrieval 的 paper anchor、paper boost、evidence boost 和 acronym target matching 已支持正文 acronym、`L_{PBA}` / `LPBA` 形式。
4. PR-5 第一阶段：formula solver 已移除 `_canonical_formula_text` 的 DPO / PPO / PBA 白名单；公式抽取顺序改为 evidence -> LLM structured formula payload（可用时）-> generic formula window fallback。fallback 只做通用 LaTeX / unicode / label 归一化，不再按目标模板补公式。
5. PR-3 第一阶段：`QueryContract.answer_slots` 已成为一等字段，`SessionTurn` 会持久化该字段；planner、runtime summary、retrieval goals、solver goals、verifier goals、composer goals 和 tool registries 都优先读 `contract.answer_slots`，notes 中的 `answer_slot=...` 仅作为兼容 fallback。
6. PR-3 第一阶段还把 intent 的 research requirements 从“命中第一个 slot 就返回”改为按多个 slot 合并 requested fields / modalities / precision。`relation` 仍保留为 API/session/test 兼容字段，但下游主路径不再只靠它理解任务。
7. PR-3 第二阶段：`_solve_formula` 已从 `papers[0]` 单论文假设改为按候选论文循环产出 formula claims；多目标公式查询会按 claim 实际匹配到的目标标注 entity；formula verifier 会逐条检查公式 claim 的目标对齐；formula composer 支持同一问题展示多篇论文里的多个公式定义。
8. PR-3 第三阶段：topology discovery、paper summary、metric context、default text answer 不再默认取第一篇论文；它们会按证据覆盖和候选分数聚合多篇 paper ids / evidence ids。entity definition 的无目标 fallback 也从 `papers[0]` 改为证据分数优先的候选选择。
9. PR-5 第二阶段：formula verifier 新增 LLM evidence verifier；有 chat model 时由 “formula claim 是否被 evidence 支撑、是否对齐用户 targets” 决定 pass / retry / clarify。旧的 “非 PPO/DPO + clip/advantage/ratio” 上下文黑名单已退出主路径，离线 fallback 只做 claim/evidence 目标匹配。
10. PR-6 第一阶段：`intent.confidence < 0.6` 或 `ambiguous_slots` 非空会直接生成 `clarify_user_intent` contract，并带上 `intent_needs_clarification`。planner / runtime fallback 会优先执行 `ask_human`；conversation 路径现在也会返回 `needs_human`、`clarification_question` 和 options，不再只给一段普通回答。
11. PR-6 第二阶段：删除未被调用的 `_augment_contextual_acronym_contract`；保留 PBA 这类短缩写保护，但入口改为 `_disambiguation_options_from_evidence`，由 contract goals / `ambiguous_slot` 和 evidence 中的多个候选含义触发，而不是继续挂在“entity keyword”补丁上。
12. PR-7 第一阶段：`ActiveResearch` 新增 `last_topic_signature`，`SessionContext.set_active_research` 会自动生成签名；intent / planner / clarification / followup prompt 都会看到该签名。`SessionStore.commit_turn` 已支持 `active=ActiveResearch`。
13. PR-7 第二阶段：`agent.py` 的 5 个 `commit_turn` 主路径（conversation、research、citation ranking、memory synthesis、compound）已全部改为通过 `commit_turn(active=...)` 写入 active context；旧 `_set_active_research` helper 已删除，active 写入边界进一步收口。
14. 回归修复：topic gate 仍默认阻止 active context 强覆盖，但新增两个窄口：显式 followup relationship 复查可以继承上一轮 seed/candidate；用户显式提到且 memory 中已有绑定的目标（例如 DPO）可以使用对应 `selected_paper_id`。ingestion 的 acronym alias 抽取补上 hyphenated expansion 和 `L_{PBA}` 下标形式。
15. PBA fresh formula 修复：`PBA 的公式是什么` 这类裸公式查询不再因为 session 中残留的 active paper、旧 `target_bindings`，或 LLM 把重复追问标成 `topic_state=continue`，被限定到 PersonaDual 等上一轮候选论文。active paper 公式绑定现在必须满足“明确上下文追问”（如“那 PBA 公式呢 / 这篇里”）或用户直接提到 active paper；裸 formula lookup 不复用旧 memory binding。
16. PR-7 读路径收口：`ActiveResearch` 新增 `context_payload()`，`SessionContext` 新增 `effective_active_research()` 和 `active_research_context_payload()`。intent / planner / followup refine / clarification prompt / memory observation / runtime summary 不再手写 9 个 legacy active 字段，而是通过统一 payload 读取。`effective_active_research()` 会兼容仍直接写 `active_targets` 等 legacy 字段的旧测试和旧调用方。
17. 前端运行态优化：`app/static/v4.html` 的 Run 面板新增 Context 区块，展示 `topic_state`、active research、selected paper / binding source、clarification reasons；Intent 区块补充 topic、relation、aliases 和 mode。最终响应的 `needs_human` 会直接反映到顶部 Status，Debug 仍保留完整 `runtime_summary`。
18. PR-5 收尾第一段：公式抽取 payload 进一步标准化。LLM 公式抽取器现在支持根级公式或 `formulas[]`，会筛掉没有合法 evidence id 的候选；输出统一写入 claim 的 `structured_data.formula_text / formula_latex / formula_format / variables / terms / paper_id / paper_title / evidence_ids / source`。变量 schema 统一为 `{symbol, description}`，并归一化 `πϕ`、`P̃`、`β` 等符号。
19. PR-5 收尾第二段：公式 composer 优先渲染结构化 `variables` 中的解释，不再把 LLM 给出的变量说明压平成 term 名称；只有没有变量 schema 时才回退到旧 term fallback。formula retry 文案也移除了 PPO / CLIP 关键词特判，改为展示 verifier 的 unsupported claims 或通用待复查说明。
20. PR-6 收尾第一段：clarification options 标准化为 `clarification_option.v1`，统一包含 `option_id / kind / target / label / description / source_relation / source_requested_fields / source_answer_slots`，同时保留 `paper_id / meaning / title / year` 等旧字段。后端 pending memory、contract notes、前端按钮 payload 都走同一 schema；用户只传 `option_id` 也能稳定恢复选择。
21. PR-6 收尾第二段：前端澄清按钮会展示 label、论文元信息和 evidence snippet，并把 `option_id / kind / target / label` 回传给后端。澄清文案移除残留的 PPO / CLIP 专门分支，改成通用“上一条候选公式不匹配当前目标”的表述。
22. PR-7 收尾第三段：`SessionContext.normalize_active_research()` 会把 legacy `active_*` 字段或 `active_research` 中较新的表示提升为 canonical `ActiveResearch`，并反向同步兼容字段。`SessionStore.upsert / commit_turn` 在持久化前自动归一，旧调用方即使直接写 `active_targets` 等字段，也不会把 SQLite 中的 `active_research` 留成陈旧状态。
23. PR-7 收尾第四段：业务回归测试中的 active context 构造已迁到 `set_active_research()`，只保留专门的 legacy 兼容测试继续直接写 `active_*`。这让新入口成为默认约束，同时仍覆盖旧 session JSON / 旧调用方的恢复路径。
24. 验收修复：`PDF-Agent / multi-agent topology` 这类新会话工程建议不再被强制要求是 topology discovery 的 follow-up；verifier 允许有 topology evidence 的 fresh recommendation 直接通过。Composer 会把论文证据边界和工程推断分开写，并过滤 LLM 产出的英文“不支持 / impossible to determine”类 rationale，避免答案同时给建议又自我否定。
25. 验收修复补丁：用户在旧会话里先被问澄清、再回答“需要解析 PDF、交互式问答、智能体频繁通信”等规格时，即使第二句话没有“拓扑”二字，也会被 deterministic intent 识别为 `topology_recommendation`，不再落到普通问答或随机论文摘要。`topology_recommendation` 的最终回答改为优先走结构化 composer，不再交给通用 LLM composer 自由改写，避免生成 “Based on the evidence ... / The provided evidence ...” 这类英文解释段。
26. 验收修复：compound query 中任一研究子任务返回 `clarify` 时，整轮会立即停止后续子任务和比较 synthesis，返回 `needs_human=true` 与该子任务的 `clarification_option.v1` 选项。这样 `PPO 和 DPO 有什么区别` 不会一边提示 PPO/DPO 歧义、一边继续写比较结论。
27. 多模态策略收口：figure/page 已有 VLM 路径；table solver 现在也会在 `enable_table_vlm=true` 且页面图像可渲染时先把表格文本和页面图片交给 VLM 解析，失败或不可渲染时回退到原文本表格解析。`enable_figure_vlm` 控制图像/figure，`enable_table_vlm` 控制表格视觉读表。
28. 本地 PDF 预览收口：`require_pdf_access` 新增 `allow_local_pdf_without_api_key` 配置。未配置 `LIBRARY_API_KEY / ADMIN_API_KEY` 时，仅允许 loopback / localhost 的开发预览继续打开 PDF；远程请求仍返回 503，已配置 key 时本机请求也必须带 key。
29. 多轮 active paper 绑定修复：用户显式说“这篇论文 / 该论文 / 文中 / 论文里”时，research contract 会从 `ActiveResearch.titles` 解析当前论文并写入 `selected_paper_id` / `memory_title`，检索与 evidence 会被限定在该论文。这样“这篇论文中 PBA 和 ICA 的具体效果如何”不会被 PBA/ICA 关键词重新带到 PersonaDual。
30. 纠错型 paper scope 修复：用户说“我问的是 AlignX 最初论文中的”这类 scope correction 时，会继承上一轮研究目标与字段，只替换论文作用域为用户点名的论文。`followup_routing` 同时修复了多目标 follow-up 被压成第一个 target 的问题，`PBA + ICA` 不再悄悄变成只剩 `PBA`。
31. topic_state prompt 收紧：意图路由器现在明确以 `new` 为默认；只有当前消息含明确指代词或省略式追问时才设为 `continue`，如果用户提到与活跃 targets 完全不同的新实体名必须设为 `switch`，并明确禁止因为存在 `active_research_context` 就默认 continue。
32. target_alias acronym anchor 加权：`target_alias=` notes 之前已进入 `_contract_target_terms()`，但当无关论文标题含 `L_PBA` 时仍可能压过正文 acronym。现在显式 alias 命中 `body_acronyms` 会得到额外 boost，用于保护 `L_PBA -> PBA` 这类查询。
33. 多论文 composer guard：最终回答整理器 prompt 新增约束：当 claims 来自不同 `paper_id` 时必须按论文分节，每节标题用论文标题，并标注该节公式、数值或结论仅来自该论文，避免把不同论文定义揉成一句。
34. normalize 链减负：删除 `_normalize_training_component_question_contract` 及其调用。`reward model / critic / value model` 已由 `IntentRecognizer` 的 `training_component` slot 覆盖；`_normalize_followup_direction_contract` 暂保留，因为它仍负责中文“A 是 B 的后续吗”语序中的 seed/candidate 解析，现有回归测试仍依赖该窄口。

验证：使用本机 conda 环境 `zotero-paper-rag` 运行 `python -m py_compile ...` 和 `python -m pytest -q`，全量 135 个测试通过；前端内联脚本在上一轮 UI 改动后已通过 `node --check`，本轮未改前端。

PR-5 剩余工作主要是把多定义聚合的 UI 表达和 verifier 输出进一步结构化；PR-6 剩余工作是继续把 intent confidence / ambiguous_slots 的澄清理由结构化给 UI；PR-7 剩余工作是逐步隐藏 legacy active 字段的公开使用面，等旧 session JSON 迁移期结束后再考虑删除字段。
