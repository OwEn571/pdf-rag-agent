# PDF-RAG-Agent V5

面向 Zotero 个人论文库的智能研究助手。将论文检索、证据抽取、歧义消解、grounding 校验和答案组合全部接入 Agent Loop，支持多轮对话、公式/图表/指标精确查询、LLM-judge 自动消歧和流式可视化。

**当前运行时版本 V5**，基于 FastAPI + SSE + Milvus + BM25，Chat Model 部署 `deepseek-v4-flash`。

## 架构

```
app/
├── api/              FastAPI 路由 (health, library, chat, stream, ingest, tool proposals)
├── core/             配置、依赖注入、安全、日志
├── domain/           数据模型 (QueryContract, SessionContext, ResearchPlan, Claim 等)
├── schemas/          API 请求/响应模型
├── static/           前端单页 v4.html
├── prompts/          Agent 自述 prompt
└── services/         16 个子包、140+ 模块
    ├── infra/        模型客户端封装 (deepseek-v4-flash + gpt-4.1-mini VLM + Qihai embedding)
    ├── retrieval/    双路检索 (BM25 + Milvus)、PDF 抽取、向量索引、Web 搜索
    ├── library/      Zotero SQLite 读取、元信息查询、引用排序
    ├── memory/       会话持久化 (SQLite)、学习记忆
    ├── intents/      LLMIntentRouter (tool-calling 意图路由, 20+ 种 relation)
    ├── planning/     研究计划生成、查询改写、复合查询分解
    ├── contracts/    会话上下文、合约规范化、追问关系
    ├── claims/       ★ 23 模块: 13 种 deterministic solver + verifier pipeline
    ├── answers/      答案组合 (公式/论文/实体/指标/推荐等)
    ├── entities/     实体定义与类型推断
    ├── followup/     追问候选管理
    ├── clarification/ 澄清问题生成与限流
    ├── agent/        ★ 26 模块: 编排核心 (loop, planner, runtime, tool registries)
    ├── agent_mixins/ 5 个 Mixin (AnswerComposer, ClaimVerifier, SolverPipeline 等)
    └── tools/        动态工具提案系统
```

## 核心链路

```
用户问题
→ LLMIntentRouter (tool-calling 路由, 5 tool choice → 20+ relation)
→ extract_agent_query_contract (多层加工: followup继承 → normalize → contextual resolve)
→ AgentPlanner (tool-calling → JSON → fallback 三级 plan 生成)
→ AgentRuntime (conversation 12工具 / research 18工具 两条 tool loop)
→ Claim Solver (13 deterministic solvers + schema solver + shadow mode)
→ Claim Verifier (三层: 证据ID审计 → type-specific → LLM fallback)
→ Answer Composer (按 relation 分发, 输出带引用的 Markdown)
→ SSE 流式推送到前端 (18 种事件类型, Runtime Inspector 实时可视化)
```

## 快速开始

```bash
# 配置环境
cp env.template .env
# 编辑 .env 填入 API key 和路径

# 安装依赖
pip install -r requirements.txt

# 入库论文 (离线)
python scripts/ingest_rebuild.py

# 启动服务
uvicorn app.main:app --host 127.0.0.1 --port 8001

# 访问
open http://127.0.0.1:8001/v5
```

## 环境变量

| 变量 | 说明 | 当前值 |
|------|------|--------|
| `CHAT_MODEL` | Chat 模型 | `deepseek-v4-flash` |
| `OPENAI_BASE_URL` | Chat/VLM API 地址 | `api.deepseek.com/v1` |
| `VLM_MODEL` | Vision 模型 | `gpt-4.1-mini` |
| `EMBEDDING_MODEL` | Embedding 模型 | `text-embedding-3-large` |
| `EMBEDDING_BASE_URL` | Embedding API 地址 | `api.qhaigc.net/v1` |
| `MILVUS_URI` | Milvus 地址 | `localhost:19530` |
| `TAVILY_API_KEY` | Web Search API key | - |

完整文档见 [docs/项目文档.md](docs/项目文档.md)。

## License

MIT
