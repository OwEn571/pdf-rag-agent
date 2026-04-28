# pdf-rag-agent-v4 Architecture Review

Date: 2026-04-25

Source: Claude architecture review pasted into the Codex session.

Status: accepted as a hardening and V5 refactor backlog.

Raw original: [architecture-review-2026-04-25-original-raw.md](architecture-review-2026-04-25-original-raw.md)

Important: the raw original above is the canonical, non-overwritable critique. This file can keep the formatted review and remediation tracker, but the original text must remain preserved separately.

Note: the review below is preserved in full as the original critique text. Some line numbers and implementation details refer to the code snapshot at review time and may differ after remediation work. See "Current Remediation Status" at the end for items already addressed or started.

## Original Review

已经把 `pdf-rag-agent-v4` 的 entry / 路由 / 依赖注入 / 配置 / agent 主循环 + 6 个 mixin / dual-index 检索 / Milvus 索引 / model_clients / session_store / domain models / ingestion / 部署单元都通读了一遍。下面是以严苛 Agent 架构师视角给出的体检报告，分四部分：架构判断 / 致命问题 / 严重问题 / 改造建议。

## 一、整体架构判断

你说 "Workflow-first Zotero paper agent V4"，但实际代码呈现的是一个伪 Agent + 巨型规则引擎：

- 真正的 LLM 决策只有两处：`_llm_route_query`（意图路由）与 `_plan_agent_actions`（计划生成）。两者输出都被 `allowed = {...}` 集合强裁剪回固定枚举（`agent.py:4019-4045`、`agent.py:570-597`）。
- 真正的"工具调用"并不存在。`_execute_conversation_tools` 是一个 `for action in actions: if action == "...": ...` 的 if-else 串（`agent.py:393-557`），ReAct loop 是 `_run_research_agent_loop` 内部固定阶段：`search_papers -> screen -> search_evidence -> (web) -> solve -> verify -> retry`。模型从未真正"选择并发出 tool call"。
- 业务规则全部硬编码在一个 5685 行的 `agent.py` 加 6 个 mixin（一共 9465 行的 `ResearchAssistantAgentV4` 一类），relation 数约 22 种，每种有自己的 contract / planner / solver / verifier / composer 分支。这不是 agent，这是一个领域 DSL 解释器，且解释器跟它的 prompt 是分裂的。
- LLM 在 prompt 里被告知"必须先 reflect_previous_answer"，但即使它不写，`agent.py:3963` 的 fallback 也会塞回去。
- Prompt 里规定的 `library_status / library_recommendation / memory_followup / library_citation_ranking` 映射，几乎 1:1 复刻在 `_conversation_tool_actions` 的硬编码 dict 里。LLM 实际上只是给硬编码工作流贴了一层装饰。

结论：当前是 rule-based RAG + LLM 兜底改写，不是 Agent。把它叫 "ReAct/Planning 控制器" 是对该术语的误用。

## 二、致命问题（生产风险）

### 1. 单类多继承怪兽，违反开闭原则

`ResearchAssistantAgentV4` 同时继承 6 个 Mixin，全部共享 `self`，互相调用 `self._normalize_text / self._matches_target / self.retriever / self._normalize_lookup_text / self._claim_focus_titles / self._is_short_acronym` 等。

Mixin 之间没有显式接口契约，任何重命名都会爆。Verifier 还反向访问 `self.retriever._normalize_entity_text`（`claim_verifier.py:278-284`），直接突破封装去调用 retriever 的私有方法。

### 2. `chat()` 是同步阻塞，`achat()` 假异步

`achat` 用 `asyncio.to_thread(self.chat, ...)` 包同步实现；`stream_chat_events` 启动一个线程，主线程 `await asyncio.wait_for(queue.get(), timeout=0.1)` 轮询（`agent.py:139-146`）。

- LangChain 的 `ChatOpenAI` 已支持 `ainvoke / astream`，全部被绕开。
- 每个并发请求都会占一条 worker thread，FastAPI 默认 thread pool 40，QPS 几乎上来就会堵死。
- 0.1s 轮询既增加延迟又烧 CPU，正确做法是 `await queue.get()`，直到 worker 完成后 put 一个 sentinel。

### 3. 全局单例 + 进程内状态，0 持久化

`InMemorySessionStore`：所有 session 在进程内 dict。`systemd Restart=always` + 一行未捕获异常 = 全部用户上下文丢失。

`get_agent / get_retriever / get_sessions` 都 `@lru_cache`，等于全局单例；多 worker（`uvicorn --workers N`）时 session 不共享，行为随机。

当前 `--host 127.0.0.1 --port 8001` 单进程跑，意味着它压根不能水平扩展。

### 4. 检索层在内存里全表扫描

`DualIndexRetriever` 启动时把整份 `v4_blocks.jsonl`（26 MB）读进内存为 `Document` 列表，每次 `search_concept_evidence / search_entity_evidence / expand_evidence` 都 `for doc in self._block_docs` 全量遍历跑正则评分（`retrieval.py:87-156, 186-252, 382-446`）。

Milvus 已经接好了，结果只在 BM25 / 向量召回阶段用一次，evidence 阶段完全不走 Milvus。这是把向量库当摆设。

- `expand_evidence` 在 `len(candidates) < limit` 时才补一次 dense（`retrieval.py:422`），这等于 dense 永远是兜底。
- 排序里硬编码 `if normalized.upper() == "PPO" and "proximal policy optimization algorithms" in title: score += 2.2`（`retrieval.py:577`）和 `weighted_tokens["ldpo"] = 3.0` 等（`retrieval.py:820-869`）：针对单个论文的硬编码加分，eval 集肯定能过，泛化是负的。

### 5. 配置层把 Settings 当可变全局

`config.py:113-163` 用 `lru_cache` 拿到 `Settings()` 后直接 mutate，例如 `settings.openai_api_key = ...`。这和 `pydantic-settings` 的 `BaseSettings` 不可变契约相反。

同时 `model_config.env_file=PROJECT_ENV_FILE` 已经会读 `.env`，紧接着又用 `dotenv_values(...)` 重新读了一遍，再读历史目录 `LEGACY_ENV_FILE`：三重来源，优先级隐式，下次切环境必踩坑。

还有 `zotero_root` 默认 `/Users/owen/Zotero`（macOS 路径）跑在 Linux 服务器上。

### 6. 错误处理"全吞掉"

`routes.py:87,107,142` 全部 `except Exception -> HTTPException(400)`：5xx 和 4xx 不分，前端无法做策略性重试。

`model_clients.invoke_text/json` 任何异常都返回 fallback（`pdf-rag-agent-v4/app/services/model_clients.py#L75-L77`、`L108-L113`）。LLM 限流、超时、key 失效，agent 会看似工作但所有 LLM 决策静默退化为硬编码，且无任何 metric/log 区分两种路径。

`vector_index.search_documents` 一遇 dimension mismatch 就 `self._dense_search_disabled = True` 永久关闭：单实例运行期降级且不可恢复，需要重启服务。

### 7. 安全/合规

静态文件 `app/static/v4.html` 3034 行直出，路由 `/v4` 主动加 `Cache-Control: no-store`（`main.py:42-45`），但没有 CORS / 没有鉴权 / 没有 rate limit。

`POST /api/v1/v4/ingest/rebuild` 是删库重建级别动作，任何人都能调。

`paper_pdf` 直接把 Zotero PDF 文件原样下发，无访问控制；`paper_id` 来自用户输入到 `LibraryBrowserService.pdf_path`，需要确认是否做了路径白名单（建议立即审）。

## 三、严重问题（架构腐化信号）

### 8. `agent.py` 5685 行 + `answer_composer.py` 1062 行 + `entity_definition.py` 951 行

这是典型的"上帝对象"。每加一种 relation，至少要在 5 个文件里同步：路由 prompt -> planner allowed-set -> conversation_tool_actions -> solver dict -> verifier dict -> composer dict。这套不是开闭，是堆砌。

### 9. Prompt 与代码的双重事实源

`agent.py:3978-3992` 与 `agent.py:4119-4290` 的 prompt 重复声明了一长串"必须按 X -> Y -> Z 调用"的规则，这些规则同时在 `_conversation_tool_actions / _agent_actions_for_execution / _plan_agent_actions` 里硬编码。

中文 prompt 里写 `library_citation_ranking`，代码 enum 里也写一遍。prompt 漂移 = 静默 bug。

### 10. 工具清单是假的

`_agent_tool_manifest` 把工具列表传给 LLM，但执行根本不用 LLM 选择的工具。这是"manifest 给模型看，actions 给硬编码用"的双轨制，一旦未来真要切到 OpenAI tool calling / LangGraph / LangChain agents，要重写整条主链。

### 11. Verifier / Solver 的语义颗粒度过细

22 种 relation 乘以各自专属 verifier，每个 verifier 都做 `if claim is None or not evidence_ids: status="retry"` 的相似检查。

完全可以抽 `EvidenceCoverageRule / TargetSupportRule / IdentityMatchRule` 三类规则组合，目前每个 relation 都重写一遍。

### 12. 重复方法、命名碰撞

`_normalize_title_key` 在 `agent.py` 里出现两次（`3933`、`1787`），后者覆盖前者。

`_is_initialism_alias_match / _is_identity_alias_match` 既在 `claim_verifier` 又被 `agent.py` 调用。

### 13. 测试代码是巨石

`tests/test_agent_v4.py` 3033 行单文件，一旦失败定位极难；`evals` 只有 184 行 yaml，覆盖远远不够支撑这么多 relation。

### 14. 数据/索引耦合

索引 `doc_id` 用 `sha1(text)[:16]` + 再 `sha1`（`indexing.py:356-359`）：稳定但不可变；文本一改 doc_id 全变，Milvus 里旧向量孤儿。

`_persist_jsonl` 直接覆盖写整个 `v4_blocks.jsonl`，没有 atomic rename，写入中途崩溃 = 索引损坏。

同时 `state["papers"]` 全量 JSON 一次 dump，规模大了会卡，没有分块/事务。

### 15. 流式接口实现细节漏

SSE 在异常路径 yield 一条 `error` 后不发 `final`，前端会卡在"等待结尾"。

`astream_chat_events` 用 `events: list[dict]` 收集所有事件等任务结束后再统计 `answer_was_streamed`（`agent.py:147-153`）。`events` 在 worker 线程被 append、主协程读，没有锁：目前因 GIL 可用，未来并行/free-threading 会出问题。

### 16. `httpx.Client(trust_env=False)` 多处复用

`model_clients.py:27` 和 `vector_index.py:51` 各自创建独立 `httpx.Client`，没有显式 close，进程退出时连接池靠 GC。

embeddings 失败时 `_reset_embedding_client` 仅清自己那份。Tavily web client 见 `web_search.py` 同病。

### 17. Compound query 与 memory_synthesis 路径双重实现

有 `_run_compound_query_if_needed`、`_run_memory_synthesis_if_needed`、`_run_library_citation_ranking_if_needed` 三个 `*_if_needed` 短路，每个都自己写一份 emit / contract / turn 持久化逻辑。和后面统一的研究主循环重复实现。

任何修改 `SessionTurn` 字段，都要同步 4 个写入点。

### 18. 研究 turn 的 "Active research context" 是 7 个字段散在 `SessionContext`

`SessionContext` 里 `active_targets / active_titles / answered_titles / active_research_relation / active_requested_fields / active_required_modalities / active_answer_shape / active_precision_requirement / active_clean_query`。

这些应该聚合成 `ActiveResearch` 子模型。当前每个写入点要拷 9 个字段，漏一个就是上下文错位。

## 四、改造路径建议（按优先级）

| Pri | 工作 | 收益 |
| --- | --- | --- |
| P0 | 接入真正的 tool calling / LangGraph：把 22 个 relation 折成"tools + 状态机"，让 LLM 真正选工具，而不是 prompt 里写规矩然后硬编码兜底。 | 让"agent"名实相符；删除约 30% 代码 |
| P0 | Session 持久化 -> Redis / SQLite，并把 `lru_cache` 单例改 DI 工厂；服务可水平扩展。 | 重启不丢上下文；多 worker 可用 |
| P0 | 拆 `agent.py`：按 intent_router / planner / executor / verifier / composer 五层独立模块，Mixin 全删，依赖注入。 | 可读、可测、可扩展 |
| P0 | Evidence 检索走 Milvus：移除内存全表扫描，全量召回交给 BM25（BM25 索引也应外置 ES/Lucene）+ Milvus 混合，过滤逻辑下沉到 metadata filter。 | 去掉 26MB JSONL 全表扫；可用论文规模提升 100 倍 |
| P0 | 保护管理面：`/v4/ingest/rebuild`、`/v4/library/papers/{id}/pdf` 必须鉴权 + rate limit；前端加 CORS allowlist。 | 防数据泄漏与误删 |
| P1 | 真正异步：`ChatOpenAI.ainvoke / astream`、`httpx.AsyncClient`、SSE 用 `asyncio.Queue + sentinel` 而非 100ms 轮询。 | 吞吐提升、延迟下降 |
| P1 | 统一异常体系：自定义 `AgentError(code, http_status)`，路由层根据 code 决定 4xx/5xx；LLM 退化路径必须 metric 化。 | 可观测、可重试 |
| P1 | 模型/检索热点参数外移：删 `if "PPO"` 这类硬编码评分；改成可配置 boost map / per-relation feature weight，evals 跑 offline tuning。 | 真正泛化 |
| P1 | Settings 不可变：去掉 `.env` 三处来源、mutate settings 的写法；`get_settings()` 之前先合并环境，构造一次。 | 排查成本下降 |
| P2 | 测试拆分：3033 行单文件 -> 按 relation / contract 分；为每条 evals case 跑 fixture。 | 回归速度 |
| P2 | `SessionContext` 重构：`ActiveResearch` 聚合子模型；`SessionTurn` 写入集中到 `SessionStore.commit_turn(...)`。 | 字段不再漂移 |
| P2 | 观测：`prometheus_fastapi_instrumentator` 已挂，但缺 LLM token / 工具调用 / verification status / cache hit 自定义指标。 | 上线必备 |

## 一句话总结

这是一个被 LLM 包装过的硬编码 RAG，不是 Agent。

真要变成 V5 / Agent 化，必须做 "prompt + 控制流" 的真正合一：让模型挑工具、让代码只管执行；同时把检索、状态、配置、异常这些工程基底彻底正交化。否则每加一个 relation，复杂度就以 5 文件乘以 N 函数的速度继续堆。

## Current Remediation Status

第一轮修复已在当前代码中开始落地，主要对应原文 P0/P1 的工程硬化项：

- SQLite session store 已接入，默认 `get_sessions()` 使用 `SQLiteSessionStore`，保留 `InMemorySessionStore` 给测试用。
- `/api/v1/v4/ingest/rebuild` 已加管理员 API key 与限流依赖。
- `/api/v1/v4/library/papers/{paper_id}/pdf` 已加 API key、限流和 Zotero root/storage 路径白名单。
- `Settings` 已改为 frozen 构造，`.env` 读取由 `pydantic-settings` 负责，不再在 `get_settings()` 后手动 mutate 字段。
- CORS allowlist 已通过 `cors_allow_origins` 配置接入。
- SSE stream 已改为 sentinel queue 模式，异常路径会发 `error` 后再发 `final`。
- `prometheus_fastapi_instrumentator` 已挂载基础 HTTP metrics，但还缺 LLM/tool/retrieval/verification 等业务 metrics。
- Ingestion JSONL 与 state 文件已改为同目录临时文件 + atomic replace，避免写入中途崩溃留下半截索引文件。
- Block `doc_id` 已去掉文本 hash 依赖，改为基于 attachment/page/block_type/chunk_index 的稳定 ID，减少文本微调导致 Milvus 旧向量孤儿的问题。
- `DualIndexRetriever` 已建立 paper/block lookup index，`expand_evidence` 和 scoped `search_concept_evidence` 先按 `paper_id` 定位候选 block，减少已选论文证据扩展路径的全量扫描。
- FastAPI lifespan 已接入 cached resource shutdown，应用关闭时会释放 `ModelClients` 与 retriever/vector index 的 httpx 连接池。
- 测试运行统一使用 conda 环境 `zotero-paper-rag`，并已补装 `pytest`。
- systemd service 模板与当前 `/etc/systemd/system/pdf-rag-agent-v4.service` 已切到 `/home/ubuntu/miniconda3/envs/zotero-paper-rag/bin/python`，线上服务已重启并通过 `/api/v1/v4/health`。
- 原始批评已作为 canonical raw original 单独持久化到 `docs/architecture-review-2026-04-25-original-raw.md`，本文件只保留格式化阅读版和推进记录。
- Agent tool manifest / allowed action set 已抽到 `app/services/agent_tools.py`，planner 和 research executor 不再各自维护一份硬编码工具清单，开始收敛 prompt 与代码的双重事实源。
- Research 主循环与 conversation 工具循环已移除 `if action == ...` 调度串，改为 `RegisteredAgentTool` 注册式 executor；工具依赖由 `requires` 声明补齐，`compose_or_ask_human` 作为 terminal tool。
- 通用 `AgentToolExecutor` 已从 `agent.py` 抽到 `app/services/agent_tools.py`，统一管理依赖补齐、已执行工具集合和 dependency-cycle 检测；两个主循环只负责构建 registry 和调用 executor。
- Conversation/research 两套工具 registry 构建已从 `agent.py` 抽到 `app/services/agent_tool_registries.py`，主类不再内嵌大段工具闭包。
- Conversation relation -> tool sequence、research fallback sequence、plan action normalization 已集中到 `app/services/agent_tools.py`，`agent.py` 不再维护 relation action dict 或手写 research action 补齐规则。
- Planner 已抽到 `app/services/agent_planner.py`：优先走 LangChain/OpenAI tool-calling 风格，`ModelClients.invoke_tool_plan()` 会把 `agent_tools` registry 转成 tool schema，让模型返回 tool calls；无 tool calls、无模型或调用失败时再回退到 JSON planner/fallback。`ResearchAssistantAgentV4` 当前只保留薄 wrapper 调用 `self.planner.plan_actions(...)`。
- `_conversation_tool_actions` / `_agent_actions_for_execution` 两个主类 wrapper 已删除，执行动作序列直接使用 `app/services/agent_tools.py` 的集中策略函数；`agent.py` 当前从原始 5685 行降到约 5250 行。
- Conversation/research 执行 loop 已抽到 `app/services/agent_runtime.py` 的 `AgentRuntime`，`agent.py` 不再直接创建工具 registry 或驱动 `AgentToolExecutor`；主类当前降到约 5110 行。
- `SessionContext` 已新增 `ActiveResearch` 聚合子模型，并保持旧字段兼容同步；主研究回答、引用排序、记忆综合、compound query 的活跃研究上下文写入已开始改为集中 API。
- `SessionStore.commit_turn(context, turn)` 已新增并迁移 agent 主要写入路径，提交的是当前已被修改过的 `SessionContext`，避免 SQLite `append_turn(session_id, ...)` 重新读取旧 session 导致 active research / working memory / clarification state 丢失。
- `SessionTurn.from_contract(...)` 已新增，主流程、引用排序、记忆综合、compound query 不再手抄 relation / targets / requested_fields / modalities / precision 等 turn 字段；`agent.py` 当前降到约 5082 行。
- 前端 `/v4` 已同步当前架构口径：标题、顶部状态、Run 面板和流式进度文案从旧 `Workflow-first` 口径改为 tool-calling planner / registered runtime / grounded verification；FastAPI description 也已同步更新。
- Compound comparison 已修复上下文记忆注入：例如先问 `DPO` 公式，再问“那 `PPO` 呢，两者有什么区别”，比较器会从 `working_memory.target_bindings` 补回上一轮 DPO 结果，不再把上一轮对象当作“证据不足”。
- Compound / citation ranking 的“计划”不再写入最终答案正文，计划改由前端 Run 面板的简洁 plan list 展示；compound 输出改为按 chunk 发 `answer_delta`，改善流式体感。
- `/v4` 页面做了密度优化：主工作区改为固定视口高度的应用布局，聊天区获得更多宽度和高度，Run 面板的执行阶段改为双列紧凑展示，减少拥挤感。
- 路由器、compound 任务分解器和研究追问合同修复器已支持把最近多轮对话作为真实 chat messages 传给 LLM，同时继续携带结构化 `conversation_context`；这减少了“这个/那篇/是吗”这类追问被当成孤立新问题的概率。
- 最终研究回答整理器已接入 `conversation_context`，用于解析指代、继承上一轮 seed / comparison 对象并避免重复回答；事实边界仍以当前 claims / evidence / citations 为主。回归用例覆盖了 `AlignX 数据集有后续工作吗？` 后追问 `Extended Inductive Reasoning ... 是吗`，回答会说明它是强相关延续候选而非直接严格后续。
- Compound 分解增加同一论文/同一实体多字段合并：例如 `POPI 的核心结论是什么，实验结果如何？` 会合并为一个 `paper_summary_results` 任务，不再拆成两个重复步骤。
- Query contract 前链路已瘦身：compound 任务分解不再每轮无条件调用 LLM，只在明显多任务、库数量+推荐、或“新目标 + 历史比较对象”时触发；研究追问合同修复器也只在 router 明确需要上下文修复或纠错/澄清时调用。
- `followup_research` 增加显式方向归一化：`A 是 B 的后续工作吗` 会解析为 `targets=[B]` 且 `notes=candidate_title=A`，solver 只验证该候选相对 seed 的关系，避免反向把 A 当 seed 去找 A 的后续。
- `followup_research` 的候选验证路径已开始从“硬加分排序”收敛到证据驱动 verifier：当问题是 `A 是否是 B 的严格后续` 时，系统会持久化 seed/candidate relationship context，后续“仔细确认是不是严格后续”直接复用该关系；候选验证优先走 LLM relationship verifier，明确区分 `strict_followup / direct_use_or_evaluation / related_continuation / not_enough_evidence / unrelated`，只在无模型或无结果时回退本地启发式。
- Relationship verifier 已接入 seed/candidate 双论文取证：`followup_solver` 现在把 `search_evidence` 得到的 evidence 传入关系验证路径；显式候选验证会针对 seed 与 candidate 的 `page_text` 证据块补取 snippets，LLM verifier 只允许基于这些 evidence 判断严格后续。最终回答会显示验证分类与证据范围，citations 会包含 seed/candidate 两侧的 evidence block，而不再只引用 paper_card。

仍待继续推进的重点：

- 将 tool-calling planner 继续推进为多步 tool-call/observation 状态机，或接入 LangGraph。
- 拆分 `agent.py` 和 mixin 依赖边界。
- Evidence 检索移出全量内存扫描，更多使用 Milvus/BM25 metadata filter。
- 统一 `AgentError` 异常体系和 LLM fallback 可观测性。
- 抽离硬编码检索 boost。
- 继续迁移剩余散落 active research 字段写入，并逐步让 compound / memory / citation ranking 等短路路径复用统一 runtime。
- 拆分巨型测试文件。
