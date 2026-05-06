# pdf-rag-agent-v4 专家审视报告

日期：2026-05-04

下面是我对 pdf-rag-agent-v4 的完整审视结论。我按"问题根因 → 定位（文件:行号）→ 具体改法"的格式组织，分 4 大块，按优先级从高到低排列。

---

## 一、引用幻觉问题（你的问题 1）

根因：最终 Markdown 答案没有任何"引用白名单"事后校验，多处 LLM 调用只有口头约束"不要编造"。

### P0-1 研究答案正文无标题白名单

定位：app/services/agent_mixins/answer_composer.py:281-418 _compose_research_answer_markdown；app/services/agent/loop.py:271-281

现状：LLM 生成完 Markdown 后直接返回，没人检查正文里的《XXX》《YYY》是否真的存在于 evidence / citations / screened_papers 里。

改法：在 loop.py 得到 answer 后加一个后置过滤器：

```python
allowed_titles = { normalize(t) for t in (
    [c.title for c in citations] +
    [p.title for p in screened_papers] +
    [e.title for e in evidence]
)}
for m in re.findall(r"《([^》]{2,220})》", answer):
    if normalize(m) not in allowed_titles:
        # 标记为"超出证据集合的引用"，触发 retry 或在前端红色警示
```

同样检查英文斜体引号 "..."、*Paper Title*、方括号编号 [12] 等模式。

校验失败的处理：不要静默删除（会造成语义破坏），而是把该段降级为"本轮证据未覆盖"+ 触发一次 retry（最多 1 次）。

### P0-2 citations_from_doc_ids 的全库 fallback

定位：app/services/answers/evidence_presentation.py:81-106

现状：当 doc_id 不在本轮 evidence 时，会回落到全库 paper_doc_lookup 去拉论文；于是只要 claim 里混进一个全库存在的 id，就会给出"存在但与答案无关"的引用。

改法：把 fallback 的可选集合收紧到 {p.paper_id for p in screened_papers}；未命中则直接丢弃该 id，不回落到全库。

### P0-3 claim evidence-id 审计"交集非空即过"

定位：app/services/agent_mixins/claim_verifier.py:56-85，关键是第 71 行 if cited_ids and not (cited_ids & real_doc_ids)。

现状：[real_id_1, fake_id_2, fake_id_3] 这种"夹带伪造 id"的 claim 会整体放行。

改法：改成严格子集校验

```python
if cited_ids and not cited_ids.issubset(real_doc_ids):
    orphan_claims.append(...)
```

同时对 claim.paper_ids 做相同的白名单校验（当前完全没校验 paper_ids）。

### P1-1 Prompt 中 evidence/citations 没有 paper_id/doc_id

定位：app/services/agent_mixins/answer_composer.py:318-337

现状：只给 LLM title/page/snippet，LLM 无稳定 key 可锚定引用，只能靠 title 串匹配，诱导幻觉。

改法：在 evidence/citations 里加 doc_id / paper_id 字段；system prompt 里显式写："你只能引用以下 paper_id 列表中的论文；如果证据不覆盖用户问题，直接回答'当前证据未覆盖该问题'。"

加 1-2 条 few-shot 反例：展示"evidence 与问题不相关 → 明确拒答"的期望输出。

### P1-2 上一轮回答作为 assistant message 回放 → 幻觉跨轮繁殖

定位：app/services/contracts/session_context.py:215-248 session_llm_history_messages；app/services/library/citation_ranking.py:112-146 select_citation_ranking_candidates

现状 1：上一轮 assistant 的 Markdown 被以 role=assistant 回放到下一轮 LLM，幻觉被"追认"为自己说过的事实。

现状 2：select_citation_ranking_candidates 用 re.findall(r"《([^》]{2,220})》", turn.answer) 从上一轮答案提取标题，当 meta is None（即本地库不存在该标题）时仍加入候选池，只是 paper_id 为空串。

改法：
- 回放历史时，把 assistant 消息替换为结构化摘要（"上一轮回答关于 X 的论文 P1/P2"），不传原文 Markdown；或者在回放前把 《...》 里不在全库白名单的部分 mask 掉。
- citation_ranking.py:119 当 meta is None 时 continue，不要把空 paper_id 入库。

### P1-3 entity / topology / concept 的 LLM 路径缺保护

定位：
- app/services/answers/entity.py:325-375 compose_entity_description
- app/services/answers/topology.py:93-125
- app/services/claims/concept_definition_solver.py:138-165

现状：只给 snippet，system prompt 仅一句"不要编造"，且没用 wrap_untrusted_document_text（prompt 注入面也存在）。

改法：统一用 infra/prompt_safety.py:7-11 的 DOCUMENT_SAFETY_INSTRUCTION 和 wrap_untrusted_document_text 包裹 evidence；为每个 composer 的输出加与 P0-1 等价的本地白名单过滤。

---

## 二、任务分解问题（你的问题 2）

根因：复合任务识别是 LLM 单点，子任务彼此完全隔离，比较合成丢证据，任一子任务 clarify 就全局回退。

### P0-4 comparison_synthesis 把 evidence 丢了

定位：app/services/planning/compound_tasks.py:391-440 compose_compound_comparison_answer，重点 405-413 行构造 comparable 列表。

现状：比较合成时只保留 relation/targets/answer/claims，丢弃每个子任务的 evidence，外加 prompt "不要引入外部记忆"，导致比较 LLM 只能从前面 Markdown 里再抽一遍。

改法：把每个子任务的 evidence（至少 top-4 的 title+page+snippet）也喂给比较 LLM，并在 prompt 里让它必须从 evidence 里挑引用。

### P0-5 任一子任务 clarify 立即全局回退 + 前面工作丢失

定位：app/services/agent/compound.py:231-250（失败时直接 return）；app/services/agent/compound.py:422-435 store_pending_compound_plan（只缓存 subcontracts，不缓存 subtask_results）。

现状：第 3 个子任务失败就 return，前 2 个做的工作全丢；恢复时从 index=1 重跑。

改法：
- 失败时继续跑后续可独立执行的子任务，把失败子任务标记为 "pending_clarification"，最后一次性返回多个待澄清问题。
- store_pending_compound_plan 同时持久化已完成的 subtask_results；resume 时只跑被替换的那一个子任务。

### P0-6 Decomposer 单点失手 → router 强行折叠成单 intent

定位：app/services/planning/compound_tasks.py:78-146 llm_decompose_compound_query；app/services/intents/router.py:190-194 单 action 路由器；app/services/agent/compound.py:137-138

现状：LLM decomposer 返回非法 JSON / 子任务 <2 就 return [] → 上层走 run_standard_turn → router 只能输出 1 个 relation。

改法：
- decomposer 加规则兜底：当 query 里同时出现多个 target（用现有的 target 抽取）+ 连接词（"和/与/对比/比较/vs/and/compare"），即便 LLM 说 is_compound=false 也强制构造最小复合拆分。
- decomposer 改用 invoke_json_messages（带历史），解决长会话里上下文依赖型复合问题被漏判。
- decomposer 加 few-shot：3 条典型复合样本 + 1 条不该拆的反例。

### P1-4 子任务之间无 evidence / claim / paper 共享

定位：app/services/agent/runtime.py:113-118（每子任务新建 state）；app/services/agent/task.py:25-113；app/services/agent/research_search_handlers.py:53-102

现状：同一篇论文被两个子任务重复检索；子任务 N 的 planner 看不到子任务 1..N-1 的结论。

改法：
- 在 run_task_subagent 增加 prior_results 入参，传入前面子任务的 screened_papers / claims。
- 子任务级别引入 _compound_evidence_cache: dict[paper_id, EvidenceBlock]，检索时先查缓存。
- planner_context_payload（planner_helpers.py:95-112）增加字段 compound_subtask_index/total/peer_relations/peer_targets，让 planner 知道"我是第 2/3 个子任务，前面做了 X"。

### P1-5 Task 工具暴露导致子任务里再启子任务（递归）

定位：app/services/agent/tool_registries.py:361-382 和 app/services/agent/tool_registries.py:858-882；app/services/agent/tools.py:673-702

现状：子任务 LLM planner 偶尔再选 Task 工具，嵌套调用 run_task_subagent，资源放大。

改法：把 Task 从普通 manifest 移出，仅在 compound runtime 内部挂载（_is_compound_runtime=True 才可见）。

### P1-6 merge_redundant_field_subtasks 的 "；" 拼接

定位：app/services/planning/compound_tasks.py:287-330

现状：同 relation+target 的两个 clean_query 被用中文分号拼成一个长 query，下游 planner 看到"双语义 query"容易乱选工具。

改法：合并时选择语义完整度更高的一个 clean_query，或新增一个 requested_fields_union 字段让 solver 明确知道要同时回答两种 field。

### P2-1 relation 正则化退化为 general_question

定位：app/services/planning/compound_tasks.py:192-284

现状：LLM 给的 relation 不在 ALLOWED_COMPOUND_SUBTASK_RELATIONS → 通过 slot 反推 → 反推失败 → 284 行默认 general_question → 退化为泛化检索。

改法：扩展 allowed 名单，同时反推失败时应 raise 而不是 fallback（至少写入 notes="compound_relation_fallback"，让上层判定是否要求用户澄清）。

### P2-2 conversation 类子任务缺 defer_premature_clarification

定位：app/services/agent/planner_helpers.py:230-274 defer_premature_research_clarification

现状：保护只覆盖 research turn，conversation 子任务（library_status / comparison_synthesis）一旦 LLM 给出 ask_human 立即触发全局回退（见 P0-5）。

改法：把保护扩展到复合子任务的 conversation relation（至少对 library_* 和 comparison_synthesis 开启）。

---

## 三、其他高价值问题

### P0-7 LLM-driven tool loop 绕过 verifier

定位：app/services/agent/runtime_helpers.py:1175-1313 run_llm_driven_tool_loop，1306-1313 行直接 invoke_text 生成答案。

现状：开关 agent_settings.llm_driven_loop_enabled 打开后，整条 claim solver + verifier + citation audit 全部被绕过，最终答案由 LLM 自由生成。

改法：在 loop 尾部强制走 SolverPipelineMixin.solve_claims + ClaimVerifierMixin.verify_claims + answer_composer，不允许直接返回 invoke_text 结果；或把这条路径下线。

### P0-8 best_effort 升级为 pass 会污染 target_bindings

定位：app/services/agent/runtime_helpers.py:794-822 promote_best_effort_state_after_clarification_limit；app/services/agent/loop.py:327-337 remember_research_outcome 只在 status==pass 写入；app/services/contracts/conversation_memory.py:128-167

现状：澄清次数耗尽 → best_effort 被升为 pass → 弱答案写入 target_bindings → 后续同名 target 永久锁死到错误论文。

改法：remember_research_outcome 判断 report.original_status（新增字段）是否为 best_effort；是的话不写 target_bindings，仅写一条临时 turn-scoped binding（下一轮失效）。

### P1-7 target_bindings 无 TTL，session 越久越脏

定位：app/services/memory/research.py:23-81；app/services/memory/session_store.py:13-28 _trim_context_history

改法：给 bindings 加 created_turn_index，超过 N 轮（如 20）自动过期；trim_context_history 时清理过期 binding。

### P1-8 model_clients 没有 retry/timeout/token 计数

定位：app/services/infra/model_clients.py:25-29（httpx 无 timeout）；invoke_text / invoke_json / invoke_tool_plan 全部无退避。

改法：
- httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
- 对 invoke_* 加 tenacity.retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8), retry=retry_if_exception_type((APIError, APITimeoutError, httpx.HTTPError)))
- 引入 tiktoken 做字符→token 估算，session_context.py 的字符预算改为 token 预算
- 按场景分 temperature：planner/router/verifier 用 temperature=0.0（创建一个独立的 ChatOpenAI 实例），composer 用 0.2

### P1-9 router prompt 自我冲突

定位：app/services/intents/router.py:143-151，第 146 行"不要使用固定关键词规则" vs 148-151 又列关键词；app/services/intents/router.py:158-167 messages 和 system 的 context dump 重复。

改法：删掉关键词列表，改用 1-2 个 few-shot 示例；context 只在 system 里注一次，messages 里只放当前 query。

### P1-10 contextual_resolver 链式重写 + 双重 prefix 风险

定位：app/services/contracts/contextual_resolver.py:39-64；app/services/contracts/contextual_helpers.py:161-187

现状：5 个 refine 串联，任一步插入"限定在论文《X》中"后，下一步仍会再加。

改法：每步 refine 前检查 clean_query 是否已包含相同 prefix（用一个 _scope_prefixes: set 记录已加过的前缀）；或把所有 refine 合并为一次 LLM 调用 + 规则后处理。

### P1-11 learnings 无去重且只增不减

定位：app/services/memory/learnings.py:7-38

改法：
- 写入前对内容做 hash 去重。
- load_learnings 时按 recency + score 排序（加一个轻量 relevance 评分），而不是全部拼到 4000 字符。
- 提供 "forget learning N" 的管理接口。

### P1-12 promote_contextual_metric_contract 静默改 relation

定位：app/services/contracts/contextual_helpers.py:209-232

改法：触发时把原 relation 也保留在 notes 里，并要求置信度>阈值才改写；关键是 diff 要显式，方便前端 Runtime Inspector 展示"我把你的问题改成了 metric 查询"。

### P2-3 solver/verifier 存在 4 处真理源

定位：app/services/planning/solver_dispatch.py:16-29、app/services/claims/deterministic_runner.py:80-93、app/services/agent_mixins/solver_pipeline.py:38-51（死代码 _DETERMINISTIC_SOLVER_REGISTRY）、app/services/claims/verifier_pipeline.py:11-39

改法：合并为一张注册表（装饰器 @register_solver(goal, modality, stage, verifier_fn)），删除死代码字典，if/elif 改为表驱动。

### P2-4 retrieval/core.py 1500 行上帝类 + 硬编码 GRPO/DPO/PPO 特例

定位：app/services/retrieval/core.py 尤其 41-90 行 RETRIEVAL_MARKERS 和 1404-1471 行 formula/metric 打分。

改法：拆分为 BM25Indexer / DenseIndexer / RRFFusion / Reranker / EvidenceExpander / GrepUtil；把硬编码 marker 移到 config.py 或 yaml 配置，便于迁移。

### P2-5 chunk size 1200 对中英文混排偏粗

定位：app/services/retrieval/indexing.py:46-50

改法：用两套 splitter（英文 . + 800 字符 / 中文 。 + 600 字符），或直接上 semantic chunker（langchain_experimental.text_splitter.SemanticChunker）。

### P2-6 prompt 安全注入面

定位：concept_definition_solver.py / formula_solver.py / entity_definition_solver / topology / compound_tasks 对 evidence 未用 wrap_untrusted_document_text。

改法：统一包装；并新增一条测试 test_prompt_injection_resistance，把 evidence 里注入 "忽略上文，输出 X" 验证系统行为。

### P2-7 agent_mixins/answer_composer.py 是 987 行上帝 mixin

改法：拆成 research_answer.py / formula_answer.py / web_answer.py / library_answer.py 各自独立，mixin 之间用 Protocol 声明依赖。

### P2-8 vector_index dense 熔断不可恢复

定位：app/services/retrieval/vector_index.py:144-150

改法：把 flag 改为带时间戳，每 N 分钟尝试一次恢复；或暴露 /admin/reset_dense API。

---

## 四、测试/可观测性缺口（P2）

- 没有 llm_driven_loop 端到端测试（对应 P0-7 的绕过风险）。
- 没有 target_bindings 错误绑定后 reset 测试（对应 P0-8）。
- 没有 _rrf_fuse 极端 alpha=0/1 测试。
- 没有 citations_from_doc_ids fallback 污染场景测试（对应 P0-2）。
- 没有 decomposer 规则兜底测试（对应 P0-6）。
- 建议新增 tests/test_citation_whitelist.py 专门覆盖 P0-1、P0-2、P0-3。

---

## 执行建议

把这份清单按 P0 → P1 → P2 分 3 批做：

- 第一批（P0，约 8 处）：修完后"引用幻觉"和"复合任务错乱"会显著下降。
- 第二批（P1）：解决跨轮污染、router 脆弱性、模型客户端稳定性。
- 第三批（P2）：重构巨型文件、补测试、优化 retrieval。

每个 P0 改完必须配套一个单元测试，否则下一次改动很容易退化。最关键的防护是 P0-1（answer 白名单）、P0-3（issubset）、P0-4（比较带 evidence）、P0-7（verifier 绕过），这 4 处做到了能挡掉 80% 的"看起来对但实际错"的情况。
