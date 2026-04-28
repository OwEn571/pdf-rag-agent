# V4 Milestones

## Goal

- 在新目录 `pdf-rag-agent-v4` 中落地 Workflow-first 研究型 Agent V4。
- 验收以 `evals/cases_test_md.yaml` 为准，完整覆盖 `docs/test.md` 的所有问题。
- 目标延迟优先压到单 turn `<120s`，硬上限 `<180s`。

## Locked Decisions

- 新项目独立目录，不覆盖 V2/V3。
- 首版交付为 `API + SSE + benchmark/eval + 里程碑文档`。
- ingestion 明确重做，建立 `paper index + block index` 双索引。
- 摘要策略为“Zotero 原生摘要优先，缺失时 LLM 补全”。
- `conversation` 问题零检索，precision task 必须引用证据。
- ingestion 过滤 `tag=书籍`，并仅保留论文型条目或论文型网页来源。
- 向量入库增加 timeout / retry / resume，避免最后批次卡死后整轮重来。
- 概念/术语类问题新增独立 `concept_definition` 主干，不再复用普通 paper-summary 路径。
- 对 target-bearing 的通用问题新增 relevance guardrail：找不到稳定证据时澄清，而不是返回弱相关论文摘要。
- 运行时配置优先级固定为 `环境变量 > V4 .env > 旧仓库 .env > 默认值`，避免 V4 被旧项目的 `EMBEDDING_MODEL=text-embedding-3-small` 反向覆盖。
- V4 当前固定使用 `EMBEDDING_MODEL=text-embedding-3-large`，与现有 `zprag_v4_*` collection 的向量维度保持一致。
- 研究类最终回答不再使用硬编码 fallback 模板；线上主路径固定为 `claims/evidence -> LLM final composer -> Markdown answer`，若 LLM 不可用则直接报错。
- V4 前端默认走 `/api/v1/v4/chat/stream`，显式展示 workflow 节点进度，并将最终结果按 Markdown 渲染。

## Current Risks

- Zotero `abstractNote` 在不同库里的覆盖率可能不稳定，需要 fallback summary。
- `--resume` 目前只在向量入库阶段续跑，若前置 PDF 抽取中断仍会重新抽取。
- 通用概念题目前已具备 acronym expansion + safe clarify，但更复杂的开放式百科问答仍需后续继续扩充 intent/router。
- 线上仍有少量旧页面在请求 `/query`，需要和 V4 前端调试结果区分开，不应混淆为当前 `/api/v1/v4/chat` 的行为。
- 前端现在默认采用 `Ctrl/Cmd+Enter` 发送、`Enter` 换行，以避免中文输入法下的回车误发；如后续要支持可配置快捷键，需要额外产品层决策。

## Milestone Status

- `M0`: completed
- `M1`: completed
- `M2`: completed
- `M3`: completed
- `M4`: completed
- `M5`: completed

## Latest Benchmark

- 2026-04-23 本地完整入库后 benchmark：`9 / 9` case 通过。
- 最新 ingest 结果：`paper_records=116`，`papers_indexed=114`，`paper_docs=114`，`block_docs=11128`，`vectors_upserted=11242`，`papers_with_generated_summary=3`。
- 最新 eval 结果：`avg_latency_ms=1944.86`，`p95_latency_ms=5137.11`。
- 2026-04-23 新增通用概念 smoke check：`PPO / RLHF / DPO / GRPO / RAG` 可走 `concept_definition`，未知术语会返回澄清而非随机摘要。
- 2026-04-23 修复 V4 配置覆盖链路后，线上服务已按 `text-embedding-3-large` 重启；本地 smoke check 中 `你是谁 / AlignX是什么 / DeepSeek R1 Figure 1` 均返回正确路由与正常回答。
- 2026-04-23 新的流式前端已接入真实 SSE 事件链，stream smoke check 已确认可依次收到 `session -> contract -> plan -> candidate_papers -> screened_papers -> evidence -> claims -> verification -> answer_delta -> final`。
- 2026-04-23 V4 运行时 `CHAT_MODEL` 已切换为 `gpt-4o` 并重启生效，用于优先测试更快的在线回复速度。
- 2026-04-26 收口本地 PDF preview 访问控制、active paper 指代绑定、paper scope 纠错继承和多目标 follow-up 保留；本地全量 `pytest -q` 通过 133 个测试。
- 2026-04-26 加硬 `topic_state` 判别 prompt，强化 `target_alias -> body_acronyms` 召回加权，要求多论文 claims 按论文分节输出，并删除冗余 training component normalize；本地全量 `pytest -q` 通过 135 个测试。
- 2026-04-26 前端 UI refresh：顶部叙事改为 structured intent / active memory / grounded multimodal RAG；聊天区新增 runtime pipeline；Run 面板新增 Intent / Memory / Evidence / Verify 结构图，右侧 summary 改为 Intent / Tools / Grounding。
- 2026-04-26 前端 Markdown/LaTeX 渲染加固：数学公式在进入 Markdown parser 前用私有占位符保护；KaTeX 未加载时不再泄漏占位符，而是显示可读 fallback，覆盖 DPO 这类长公式场景。
- 2026-04-26 公式输出 prompt 收口：公式抽取器要求产出可直接放入 `$$...$$` 的标准 LaTeX，变量 symbol 也要求可放入 `$...$`；新增通用数学符号规范化，将 `∇θLDPO / πθ / πref / logσ` 等紧凑 OCR/LLM 输出转成 KaTeX 友好的 LaTeX。
- 2026-04-26 对话体验收口：`怎么理解这个公式` 走上一轮公式记忆解释，不再重复检索；`不是这个公式` 保留上一轮论文范围并偏向重新找目标函数/损失；`我要中文` 作为回答语言修正处理，不再触发随机论文检索。前端新会话现在会清空聊天流、预览区和 runtime 状态。本地全量 `pytest -q` 通过 139 个测试。
- 2026-04-27 运行时检索增加历史书籍过滤，避免已入库的书籍/教程污染论文 origin 问题；origin 类召回新增“提出/引入目标词”排序信号，`Transformer` 起源真实库 smoke check 中《Attention Is All You Need》排第一。PDF 预览允许无 key 的同源浏览器页面访问但仍阻断裸远程请求。前端改为更现代的三栏 Agent 工作台，新增夜间模式切换。本地全量 `pytest -q` 通过 141 个测试。
- 2026-04-27 前端重新设计为 GPT/DeepSeek 风格的聊天主界面：左侧抽屉/侧栏承载本地历史会话与论文库切换，中间只保留主聊天和输入框，右侧承载 Evidence / PDF / Run 三个 inspector 面板；浅色/夜间模式改为单套 token 驱动，兼容桌面三栏与移动端抽屉布局。本地全量 `pytest -q` 通过 141 个测试。
- 2026-04-27 前端交互修补：回答完成后强制隐藏进度圈；恢复 clarification options 点击选择；用户消息右对齐以区别助手；会话历史支持本地删除；SSE `answer_delta` 兼容 `text/delta/content` 字段。`/chat/stream` smoke check 已确认 `session -> contract -> answer_delta -> final` 事件链正常，本地全量 `pytest -q` 通过 141 个测试。
- 2026-04-27 AlignX 路由收口：显式目标的“主要结论/数据支持”问题不再被 LLM 扩写到 `AlignXplore`；“第一个/第一篇/首次提出 X”本地保护为 `origin_lookup`，并禁止 origin 查询继承陈旧 `selected_paper_id`。paper summary solver 优先精确命中或引入目标的论文。前端移除输入栏 `Transformer/DPO` 固定快捷按钮；线上服务已重启，smoke check 中 `AlignX中主要结论...` 和 `我问的是AlignX的第一篇论文` 均命中《From 1,000,000 Users to Every User...》。本地全量 `pytest -q` 通过 145 个测试。
- 2026-04-28 按 `docs/review-2026-04-28.md` 推进 M1/M2：现有 Agent 工具补齐真实 `input_schema`，`ModelClients._openai_tool_definitions` 改为透传 schema，不再把所有工具参数强制压成 `{reason}`；runtime 开始保存并透传 `tool_call_args`，`search_corpus / web_search / query_library_metadata` 可消费结构化 `query/top_k/max_results`；新增 `app/core/agent_settings.py` 集中 `max_agent_steps / confidence_floor / clarification / disambiguation` 阈值，并让 runtime 读取配置化步数。SSE payload 增加兼容式 `type/name/input/output/ok` 协议字段；新增 `todo_write` 工具，可维护前端可见 TODO 并写入 session working memory。继续接入 `Task` 子任务工具，先复用现有 contract/planner/runtime 执行独立子任务并汇总回答；新增 `fetch_url` 原子工具与 SSRF 安全校验层，仅允许 HTTPS 且拒绝 localhost/private/link-local/reserved 地址。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 158 个测试。
- 2026-04-28 提交首个本地 git baseline：`a67007b chore: baseline pdf rag agent v4`，`.env / data / caches / pyc` 已通过 `.gitignore` 排除。随后继续按 review 推进可观测性与 learnings：每轮 Agent 事件写入 `data/traces/<session>/<turn>.jsonl`，final 只保留 answer preview/chars 以便 eval trace diff；新增 `remember` 工具和 `data/learnings/*.md` 持久化，`_session_conversation_context` 会注入 `persistent_learnings`。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 160 个测试。
- 2026-04-28 按 review 继续拆分检索原子工具：`DualIndexRetriever` 新增 block BM25、`bm25_search / vector_search / hybrid_search / rerank_evidence`；Agent 工具 manifest 与 research registry 接入 `bm25_search / vector_search / hybrid_search / rerank`，让 LLM 可以选择具体检索策略而不是只能调用粗粒度 `search_corpus`。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 162 个测试。
- 2026-04-28 继续按 review §5 补齐本地语料原子工具：`read_pdf_page` 可按 `paper_id + page range` 读取已索引 PDF page/table/caption blocks，`grep_corpus` 可对 paper cards / PDF blocks 做限长正则精确查找；research registry 会把结果并入 evidence/candidate paper state。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 163 个测试。
- 2026-04-28 继续按 review §5 补齐推理原子工具：新增 `summarize` / `verify_claim` 的可测试本地实现，支持对文本、EvidenceBlock、fetch_url payload 和 inline evidence 做摘要/claim 覆盖校验；conversation/research registry 均可执行并记录 `summaries / tool_verifications`。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 168 个测试。
- 2026-04-28 按 review §5 删除旧 solver/retrieval 工具别名入口：`search_papers / search_evidence / solve_claims / verify_grounding` 不再被 canonical 工具映射接纳，内部阶段改用真实工具名 `search_corpus / compose / verify_claim` 上报并保留 `stage` 字段。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 169 个测试。
- 2026-04-28 按 review §6 起步动态工具注册：新增 `propose_tool` schema 与危险标记，conversation/research registry 可把新工具提案写入 `data/tools_proposed/*.json` 待人工审核，当前不执行 `python_code`，为后续 sandbox/approve 流程留口。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 171 个测试。
- 2026-04-28 按 review §7 起步不确定性机制：新增 `Confidence` / `confidence_from_contract` / `should_ask_human`，runtime 与 planner 的澄清触发统一走 `agent_settings.confidence_floor`，保留 `ambiguous_slot` / `low_intent_confidence` 作为置信度来源而不是散落硬判断。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 174 个测试。
- 2026-04-28 继续推进 review §7：新增 `confidence_from_self_consistency` 和 `confidence_from_verification_report`，runtime 在最终 verification 后输出统一 `confidence` state/SSE 事件，runtime summary 的 grounding 增加 verifier confidence，为后续 N=3 采样与 cheap verifier 接入留接口。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 177 个测试。
- 2026-04-28 按 review §8 起步去硬编码 intent：新增旁路 `LLMIntentRouter` 与 `answer_directly / need_corpus_search / need_web / need_clarify` 四个真实 schema router tools，可把 tool-call payload 归一化为 `RouterDecision`，暂不替换旧 `IntentRecognizer`，为逐段迁移 marker 规则留稳定接口。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 181 个测试。
- 2026-04-28 按 review §9 起步检索层工具化：新增 `query_rewrite` 原子工具和本地 deterministic rewrite fallback，支持 `multi_query / hyde / step_back` 模式；research registry 会记录 `query_rewrites / rewritten_queries`，atomic search/grep 可复用首个改写查询。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 185 个测试。
- 2026-04-28 继续推进 review §9：`rerank` schema 扩展为 `query, candidates[], top_k, focus[]`，research registry 会优先重排显式 candidates，没有传入时仍兼容当前 state evidence，向“screen/expand evidence 都变成纯 rerank 工具输入”的方向收口。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 186 个测试。
- 2026-04-28 继续推进 review §9：`search_corpus` 增加可选 `strategy:auto|legacy|bm25|vector|hybrid`，显式传入原子策略时会委托对应 `*_search` 工具路径，默认仍保留旧 paper+evidence 兼容流程，便于逐步迁出粗粒度检索。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 187 个测试。
- 2026-04-28 继续推进 review §9：新增 `retrieval_filter_formula_heavy_non_formula` 配置项，默认保持非公式类实体检索中过滤 formula-heavy snippet 的旧行为，但可关闭该 heuristic，避免检索层硬规则不可控。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 188 个测试。
- 2026-04-28 按 review §11 收口 subprocess 安全：确认 `agent.py` 中 `subprocess` 用于 PDF page 渲染，保留但增加 `ALLOWED_SUBPROCESS_COMMANDS={"pdftoppm"}` 白名单，只允许 bare `pdftoppm` 命令，阻断路径形式或其他命令。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 189 个测试。
- 2026-04-28 按 review §10 补 tool 维度 Prometheus 指标：新增 `tool_calls_total{name,ok}` 与 `tool_latency_seconds{name}`，Agent 工具执行器负责真实 latency/成功失败计数，事件归一化层覆盖旧 observation 路径并避免执行器内双计数。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 193 个测试。
- 2026-04-28 按 review §10/M3 起步 eval trace 回归：新增 `agent_trace_diff` 稳定签名比较与 `scripts/diff_agent_traces.py` CLI，可对 `data/traces/*.jsonl` 做事件序列、工具名、ok/count/status、final execution nodes 的回归 diff，同时忽略 id/took_ms/answer_preview 等易变字段。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 195 个测试。
- 2026-04-28 按 review §11 加固 prompt injection 边界：新增 `prompt_safety.wrap_untrusted_document_text`，将核心 LLM 证据路径中的 PDF/Web snippet 包进 `<document>...</document>` 并 XML escape，同时在 claim 抽取、claim 验证、最终回答和 Web 证据整理 system prompt 中声明 document 内指令不得执行。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 197 个测试。
- 2026-04-28 按 review M1 起步拆分 `agent.py` emit/trace 边界：新增 `agent_emit.AgentEventRecorder` 统一事件归一化、缓存和 SSE callback 转发，新增 `write_turn_trace_safe` 收口 trace 写入与异常隔离，`ResearchAssistantAgentV4._run` 不再内联事件 recorder。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 201 个测试。
- 2026-04-28 继续按 review M1 拆分运行上下文：新增 `agent_context.AgentRunContext` 收口单 turn 的 `session_id/session/events/execution_steps/emit`，`ResearchAssistantAgentV4._run` 通过上下文对象传递运行时状态，为后续拆 `agent_loop.py` 降低参数散落。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 202 个测试。
- 2026-04-28 继续按 review M1 拆 `agent_loop.py`：新增 `run_conversation_turn`，把 `_run` 中 conversation 工具执行、clarification/session commit、response payload 组装迁出 `agent.py`，保留现有 runtime 与私有 helper 行为不变。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 204 个测试。
- 2026-04-28 继续按 review M1 拆 `agent_loop.py`：新增 `run_research_turn`，把 `_run` 中 research tool loop、best-effort clarification limit、compose、active research memory、session commit 和 response payload 组装迁出 `agent.py`。`_run` 现在主要负责 contract/plan、conversation/research 分流和 trace 收尾。使用 `zotero-paper-rag` conda 环境全量 `pytest` 通过 205 个测试。
