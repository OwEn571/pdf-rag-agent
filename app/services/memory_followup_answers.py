from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from app.domain.models import QueryContract, SessionContext
from app.services.memory_artifact_helpers import answer_from_recent_tool_artifact_reference


ConversationContextFn = Callable[..., dict[str, Any]]
CleanTextFn = Callable[[str], str]


def compose_memory_synthesis_answer(
    *,
    query: str,
    session: SessionContext,
    contract: QueryContract,
    clients: Any,
    conversation_context: ConversationContextFn,
    clean_text: CleanTextFn,
) -> str:
    if getattr(clients, "chat", None) is not None:
        text = clients.invoke_text(
            system_prompt=(
                "你是论文研究 Agent 的会话记忆综合器。"
                "只基于 conversation_context 中已经发生的工具结果、回答、claims/引用线索来回答当前追问；"
                "不要重新检索，不要编造未出现过的新事实。"
                "如果用户问比较/区别，先给一句总览，再用简洁表格或要点比较。"
                "如果记忆里的某一项证据不足，要明确说证据不足。"
                "输出中文 Markdown。"
            ),
            human_prompt=json.dumps(
                {
                    "current_query": query,
                    "current_contract": contract.model_dump(),
                    "conversation_context": conversation_context(session),
                },
                ensure_ascii=False,
            ),
            fallback="",
        ).strip()
        if text:
            return clean_text(text)
    last_compound = dict((session.working_memory or {}).get("last_compound_query", {}) or {})
    rows = []
    for item in list(last_compound.get("subtasks", []) or [])[:4]:
        targets = item.get("targets") or []
        target = str(targets[0]) if targets else str(item.get("clean_query", "对象"))
        preview = " ".join(str(item.get("answer_preview", "")).split())
        rows.append(f"- **{target}**：{preview[:260] if preview else '上一轮没有留下足够细节。'}")
    if rows:
        return "基于上一轮已经完成的检索结果，先做保守比较：\n\n" + "\n".join(rows)
    return "这轮追问看起来是在比较上一轮对象，但当前会话记忆里没有足够可综合的工具结果。"


def compose_memory_followup_answer(
    *,
    query: str,
    session: SessionContext,
    contract: QueryContract,
    clients: Any,
    conversation_context: ConversationContextFn,
    clean_text: CleanTextFn,
) -> str:
    requested = {str(item) for item in list(contract.requested_fields or [])}
    if "formula_interpretation" in requested:
        return compose_formula_interpretation_followup_answer(
            query=query,
            session=session,
            contract=contract,
            clients=clients,
            conversation_context=conversation_context,
            clean_text=clean_text,
        )
    if "answer_language_preference" in requested:
        return compose_language_preference_followup_answer(
            query=query,
            session=session,
            contract=contract,
            clients=clients,
            conversation_context=conversation_context,
            clean_text=clean_text,
        )
    artifact_answer = answer_from_recent_tool_artifact_reference(query=query, session=session)
    if artifact_answer:
        return artifact_answer
    if getattr(clients, "chat", None) is not None:
        text = clients.invoke_text(
            system_prompt=(
                "你是论文研究 Agent 的通用会话记忆问答工具。"
                "你的输入是完整 conversation_context，其中包含历史用户问题、助手回答、工具结果摘要、working_memory。"
                "请只基于这些记忆回答当前追问；不要重新推荐、不要重新检索、不要编造未出现过的新事实。"
                "只适用于回答上一轮工具输出本身的依据、选择理由、排序理由、措辞解释。"
                "如果用户是在问某篇论文/模型/方法本身的正文内容、核心结论、实验结果、方法细节，"
                "不要在这里编答案，应说明需要进入论文检索/阅读工具。"
                "如果记忆不足以回答，就说清楚缺什么，并建议下一步调用哪个工具，而不是假装知道。"
                "输出简洁中文 Markdown，像在继续聊天，不要复读整份上一轮答案。"
            ),
            human_prompt=json.dumps(
                {
                    "current_query": query,
                    "current_contract": contract.model_dump(),
                    "conversation_context": conversation_context(session),
                },
                ensure_ascii=False,
            ),
            fallback="",
        ).strip()
        if text:
            return clean_text(text)
    previous = session.turns[-1].answer if session.turns else ""
    if previous:
        compact = " ".join(previous.split())
        return f"我根据上一轮结果回答：{compact[:420]}"
    return "当前会话记忆里没有足够的上一轮工具结果来回答这个追问。"


def compose_formula_interpretation_followup_answer(
    *,
    query: str,
    session: SessionContext,
    contract: QueryContract,
    clients: Any,
    conversation_context: ConversationContextFn,
    clean_text: CleanTextFn,
) -> str:
    previous_turns = [
        turn
        for turn in reversed(session.turns)
        if turn.relation == "formula_lookup"
        or "formula" in {str(item) for item in list(turn.requested_fields or [])}
        or "formula" in {str(item) for item in list(turn.answer_slots or [])}
    ]
    previous = previous_turns[0] if previous_turns else (session.turns[-1] if session.turns else None)
    previous_answer = previous.answer if previous is not None else ""
    if getattr(clients, "chat", None) is not None and previous_answer:
        text = clients.invoke_text(
            system_prompt=(
                "你是论文公式讲解器。用户当前是在追问上一轮已经给出的公式应该如何理解。"
                "只能基于上一轮回答、变量解释和会话记忆来解释，不要重新检索、不要引入新论文事实。"
                "不要完整重抄公式，不要重新列一遍变量表；最多引用 1-2 个关键符号。"
                "用简洁中文 Markdown 输出，重点讲：这个式子在优化什么、正负样本如何影响方向、"
                "参考策略/温度系数/sigmoid 或 log-ratio 的直觉，以及最容易误解的边界。"
                "所有数学符号必须用 KaTeX 可渲染的标准 LaTeX 并包在 $...$ 中，例如 $\\pi_{\\theta}$、"
                "$\\pi_{\\mathrm{ref}}$、$y_w$、$y_l$、$\\log \\sigma$。"
                "不要输出 $pi_{theta}$、$pi_mathrmref$、$frac...$ 这类缺少反斜杠或大括号的裸符号；"
                "如果不确定 LaTeX 写法，就改用中文描述，不要写半截公式。"
            ),
            human_prompt=json.dumps(
                {
                    "current_query": query,
                    "current_contract": contract.model_dump(),
                    "previous_formula_query": previous.query if previous is not None else "",
                    "previous_formula_answer": previous_answer,
                    "conversation_context": conversation_context(session, max_chars=12000),
                },
                ensure_ascii=False,
            ),
            fallback="",
        ).strip()
        if text:
            return clean_text(text)
    if previous_answer:
        compact = " ".join(previous_answer.split())
        return (
            "## 怎么读\n\n"
            "这条公式先看优化方向：它想让偏好回答相对参考策略更可能，让劣选回答相对参考策略更不可能。"
            "再看缩放项：温度系数控制偏好信号强度，sigmoid/log-ratio 把“偏好回答是否已经明显强于劣选回答”变成训练权重。\n\n"
            f"上一轮公式摘要：{compact[:360]}"
        )
    return "我需要上一轮已经定位到的公式，才能继续解释它的直觉。"


def compose_language_preference_followup_answer(
    *,
    query: str,
    session: SessionContext,
    contract: QueryContract,
    clients: Any,
    conversation_context: ConversationContextFn,
    clean_text: CleanTextFn,
) -> str:
    previous = session.turns[-1].answer if session.turns else ""
    if getattr(clients, "chat", None) is not None and previous:
        text = clients.invoke_text(
            system_prompt=(
                "你是论文研究 Agent 的回答语言修正器。用户指出上一条回答中英文混杂，要求中文。"
                "请只基于上一条回答做中文化改写或简短确认，不要检索论文，不要新增引用，不要编造新事实。"
                "公式、变量符号、论文标题和不可翻译专名可以保留英文；变量解释、句子说明必须使用中文。"
                "输出简洁中文 Markdown。"
            ),
            human_prompt=json.dumps(
                {
                    "current_query": query,
                    "current_contract": contract.model_dump(),
                    "previous_answer": previous,
                    "conversation_context": conversation_context(session, max_chars=8000),
                },
                ensure_ascii=False,
            ),
            fallback="",
        ).strip()
        if text:
            return clean_text(text)
    return "好的，后续我会用中文说明；公式符号和论文标题会保留原样，变量解释和推理过程用中文。"
