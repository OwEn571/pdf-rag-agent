from __future__ import annotations

import json
from typing import Any

from app.domain.models import SessionContext, SessionTurn


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
        answer = truncate_context_text(turn.answer, limit=answer_limit)
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
                "content": answer + "\n\n[上一轮工具上下文]\n" + json.dumps(metadata, ensure_ascii=False),
            }
        )
    return messages


def truncate_context_text(text: str, *, limit: int) -> str:
    compact = str(text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."
