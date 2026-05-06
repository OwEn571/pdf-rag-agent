from __future__ import annotations

import json
import logging
from typing import Any

from app.domain.models import ActiveResearch, SessionContext, SessionTurn
from app.services.contracts.context import conversation_relation_updates_research_context
from app.services.memory.learnings import load_learnings


logger = logging.getLogger(__name__)


def session_conversation_context(
    session: SessionContext,
    *,
    persistent_learnings: str = "",
    max_chars: int = 24000,
) -> dict[str, Any]:
    payload = {
        "summary_of_compressed_older_turns": session.summary,
        "active_research_context": session.active_research_context_payload(),
        "pending_clarification": {
            "type": session.pending_clarification_type,
            "target": session.pending_clarification_target,
            "options": session.pending_clarification_options,
        },
        "working_memory": session.working_memory,
        "persistent_learnings": persistent_learnings,
        "turns": [turn_context_payload(turn, answer_limit=1800) for turn in session.turns],
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    if len(serialized) <= max_chars:
        return payload

    compact_turns: list[dict[str, Any]] = []
    for index, turn in enumerate(session.turns):
        limit = 900 if index >= max(0, len(session.turns) - 4) else 280
        compact_turns.append(turn_context_payload(turn, answer_limit=limit))
    payload["turns"] = compact_turns
    payload["context_compression_note"] = "Older answers were shortened because the raw conversation context was near the prompt budget."
    return payload


def agent_session_conversation_context(
    session: SessionContext,
    *,
    settings: Any,
    max_chars: int = 24000,
    learnings_max_chars: int = 4000,
) -> dict[str, Any]:
    """Build the Agent-facing conversation context with persistent learnings."""

    try:
        persistent_learnings = load_learnings(data_dir=settings.data_dir, max_chars=learnings_max_chars)
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to load persistent learnings: %s", exc)
        persistent_learnings = ""
    return session_conversation_context(
        session,
        persistent_learnings=persistent_learnings,
        max_chars=max_chars,
    )


def make_active_research(
    *,
    relation: str,
    targets: list[str],
    titles: list[str],
    requested_fields: list[str],
    required_modalities: list[str],
    answer_shape: str,
    precision_requirement: str,
    clean_query: str,
) -> ActiveResearch:
    precision = precision_requirement if precision_requirement in {"exact", "high", "normal"} else "normal"
    active = ActiveResearch(
        relation=relation,
        targets=targets,
        titles=titles,
        requested_fields=requested_fields,
        required_modalities=required_modalities,
        answer_shape=answer_shape,
        precision_requirement=precision,  # type: ignore[arg-type]
        clean_query=clean_query,
    )
    if not active.last_topic_signature:
        active.last_topic_signature = active.topic_signature()
    return active


def active_research_from_contract(contract: Any, *, titles: list[str]) -> ActiveResearch:
    return make_active_research(
        relation=str(getattr(contract, "relation", "") or ""),
        targets=list(getattr(contract, "targets", []) or []),
        titles=list(titles),
        requested_fields=list(getattr(contract, "requested_fields", []) or []),
        required_modalities=list(getattr(contract, "required_modalities", []) or []),
        answer_shape=str(getattr(contract, "answer_shape", "") or ""),
        precision_requirement=str(getattr(contract, "precision_requirement", "normal") or "normal"),
        clean_query=str(getattr(contract, "clean_query", "") or ""),
    )


def conversation_active_research_from_contract(contract: Any, *, titles: list[str]) -> ActiveResearch | None:
    relation = str(getattr(contract, "relation", "") or "")
    if not conversation_relation_updates_research_context(relation):
        return None
    return active_research_from_contract(contract, titles=titles)


def session_history_compression_window(
    session: SessionContext,
    *,
    max_turns: int,
) -> tuple[int, list[SessionTurn]]:
    if len(session.turns) < max(6, max_turns - 1):
        return 0, []
    retained_turns = max(4, max_turns // 2)
    older_turns = session.turns[:-retained_turns]
    if not older_turns:
        return retained_turns, []
    return retained_turns, older_turns


def session_history_compression_system_prompt() -> str:
    return (
        "你是研究助手的会话记忆压缩器。"
        "请把较早的对话压缩成简洁中文摘要，保留："
        "1. 主要研究主题和实体；"
        "2. 已经回答过的问题类型（如公式、定义、实验结果、图表）；"
        "3. 仍然可能被继续追问的开放上下文。"
        "不要编造。输出 3-6 句纯文本摘要。"
    )


def session_history_compression_payload(session: SessionContext, *, older_turns: list[SessionTurn]) -> dict[str, Any]:
    return {
        "existing_summary": session.summary,
        "older_turns": [
            {
                "query": turn.query,
                "relation": turn.relation,
                "interaction_mode": turn.interaction_mode,
                "targets": turn.targets,
                "requested_fields": turn.requested_fields,
                "answer_shape": turn.answer_shape,
                "answer": turn.answer[:320],
            }
            for turn in older_turns
        ],
    }


def apply_session_history_compression(
    session: SessionContext,
    *,
    compressed: str,
    retained_turns: int,
) -> None:
    if compressed:
        session.summary = compressed
    session.turns = session.turns[-retained_turns:]


def compress_session_history_if_needed(
    *,
    session: SessionContext,
    clients: Any,
    settings: Any,
    sessions: Any,
) -> bool:
    if getattr(clients, "chat", None) is None:
        return False
    retained_turns, older_turns = session_history_compression_window(
        session,
        max_turns=int(getattr(settings, "agent_history_max_turns", 16)),
    )
    if not older_turns:
        return False
    compressed = clients.invoke_text(
        system_prompt=session_history_compression_system_prompt(),
        human_prompt=json.dumps(
            session_history_compression_payload(session, older_turns=older_turns),
            ensure_ascii=False,
        ),
        fallback=session.summary,
    ).strip()
    apply_session_history_compression(session, compressed=compressed, retained_turns=retained_turns)
    sessions.upsert(session)
    return True


def turn_context_payload(turn: SessionTurn, *, answer_limit: int) -> dict[str, Any]:
    return {
        "user_query": turn.query,
        "assistant_answer": truncate_context_text(turn.answer, limit=answer_limit),
        "query_contract": {
            "relation": turn.relation,
            "interaction_mode": turn.interaction_mode,
            "clean_query": turn.clean_query,
            "targets": turn.targets,
            "answer_slots": turn.answer_slots,
            "requested_fields": turn.requested_fields,
            "required_modalities": turn.required_modalities,
            "answer_shape": turn.answer_shape,
            "precision_requirement": turn.precision_requirement,
        },
        "citation_titles": turn.titles,
    }


def session_llm_history_messages(
    session: SessionContext,
    *,
    max_turns: int = 4,
    answer_limit: int = 700,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if session.summary:
        messages.append(
            {
                "role": "human",
                "content": "以下是更早对话的压缩摘要，请用于解析后续指代，不要把摘要当成新问题：\n" + session.summary,
            }
        )
    for turn in session.turns[-max_turns:]:
        user_query = str(turn.query or "").strip()
        if user_query:
            messages.append({"role": "user", "content": user_query})
        # P1-2: Replace raw assistant Markdown with structured summary to prevent
        # hallucination cross-turn propagation. LLM-confabulated titles in
        # previous answers would otherwise be "confirmed" as fact in later turns.
        answer_preview = truncate_context_text(turn.answer, limit=answer_limit)
        summary_parts = [f"上一轮回答了关于 {', '.join(turn.targets[:3]) or '未知主题'} 的问题"]
        if turn.titles:
            summary_parts.append(f"涉及论文：{'、'.join(turn.titles[:3])}")
        summary_parts.append(f"关系类型：{turn.relation}")
        metadata = {
            "relation": turn.relation,
            "interaction_mode": turn.interaction_mode,
            "targets": turn.targets,
            "answer_slots": turn.answer_slots,
            "requested_fields": turn.requested_fields,
            "citation_titles": turn.titles,
        }
        messages.append(
            {
                "role": "assistant",
                "content": "；".join(summary_parts) + "。\n\n[上一轮工具上下文]\n" + json.dumps(metadata, ensure_ascii=False),
            }
        )
    return messages


def truncate_context_text(text: str, *, limit: int) -> str:
    compact = str(text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."
