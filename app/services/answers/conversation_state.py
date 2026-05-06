from __future__ import annotations

from typing import Any, Callable

from app.services.answers.evidence_presentation import chunk_text


def set_conversation_answer(
    *,
    state: dict[str, Any],
    answer: str,
    emit: Callable[[str, dict[str, Any]], None],
    chunk_size: int = 96,
) -> None:
    state["answer"] = answer
    for chunk in chunk_text(str(answer or ""), size=chunk_size):
        emit("answer_delta", {"text": chunk})
