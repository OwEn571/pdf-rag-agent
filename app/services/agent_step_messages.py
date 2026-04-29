from __future__ import annotations

from app.domain.models import QueryContract


def agent_step_message(*, action: str, contract: QueryContract) -> str:
    target_text = " / ".join(contract.targets) if contract.targets else "当前问题"
    messages = {
        "read_memory": "读取会话工作记忆，确认上一轮目标、选择和工具结果。",
        "search_corpus": f"从本地论文库检索与 {target_text} 相关的论文和证据块。",
        "bm25_search": f"对 {target_text} 相关内容做关键词检索，优先召回精确术语、公式和标题。",
        "vector_search": f"对 {target_text} 相关内容做语义向量检索，补足改写表达。",
        "hybrid_search": f"对 {target_text} 相关内容做混合检索，融合关键词和语义召回。",
        "rerank": "按当前问题重新排序已收集证据，优先保留最相关片段。",
        "read_pdf_page": "读取本地论文 PDF 索引中的指定页文本、表格或图注块。",
        "grep_corpus": "用精确字符串或正则在本地论文库中查找公式、术语和片段。",
        "query_rewrite": "改写当前问题，生成多路本地检索查询。",
        "summarize": "压缩文本或当前证据，生成面向后续推理的短摘要。",
        "verify_claim": "检查具体 claim 是否被当前或传入证据支持。",
        "compose": "基于当前记忆或证据进入最终整理；研究问题会先完成内部求解和校验。",
        "todo_write": "更新可见任务列表，让多步检索/验证过程可以被前端追踪。",
        "remember": "把可复用的学习或用户偏好持久化，供后续轮次读取。",
        "propose_tool": "记录一个待人工审核的新工具提案，不执行其中的代码。",
        "Task": "派发一个独立子任务，通过同一套工具循环收集结果。",
        "understand_user_intent": f"先确认任务类型：{contract.relation}，目标是 {target_text}。",
        "reflect_previous_answer": "先反思上一轮回答，排除已经被用户否定的解释。",
        "answer_conversation": "调用对话工具处理普通交流，不从主流程直接回答。",
        "get_library_status": "调用论文库状态工具读取当前索引、分类和文章预览。",
        "query_library_metadata": "调用只读库元信息 SQL 工具，按当前问题查询论文标题、作者、年份、分类、标签等索引字段。",
        "get_library_recommendation": "调用库内推荐工具，基于当前论文库挑出值得优先读的论文。",
        "answer_from_memory": "调用通用记忆问答工具，回答用户对上一轮工具结果的追问。",
        "read_conversation_memory": "读取会话工作记忆，继承上一轮工具结果和目标绑定。",
        "synthesize_previous_results": "基于已保留的工具结果做综合，不重新猜测。",
        "recover_previous_recommendation_candidates": "先恢复上一轮推荐候选，避免凭空换一批论文。",
        "web_citation_lookup": "引用数是外部动态指标，逐篇调用 Web citation 检索工具。",
        "rank_by_verified_citation_count": "只用抽取得到的 citation count 做排序，并说明证据边界。",
        "web_search": "本地证据不够或问题需要外部动态信息，补充 Web 检索。",
        "fetch_url": "读取一个已知 HTTPS URL 的正文，并执行 SSRF 安全校验。",
        "ask_human": "当前存在实质歧义，需要用户选择后再继续。",
        "compose_or_ask_human": "证据链检查完成，整理最终回答或交互选项。",
    }
    return messages.get(action, action)
