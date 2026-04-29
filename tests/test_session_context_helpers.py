from __future__ import annotations

from app.domain.models import ActiveResearch, SessionContext, SessionTurn
from app.services.session_context_helpers import (
    session_conversation_context,
    session_llm_history_messages,
    truncate_context_text,
)


def _turn(index: int, *, answer: str = "answer") -> SessionTurn:
    return SessionTurn(
        query=f"q{index}",
        answer=answer,
        relation="formula_lookup",
        clean_query=f"clean {index}",
        targets=["DPO"],
        answer_slots=["formula"],
        titles=[f"Paper {index}"],
        requested_fields=["formula"],
        required_modalities=["page_text"],
    )


def test_session_conversation_context_renders_active_memory_and_learnings() -> None:
    session = SessionContext(
        session_id="ctx",
        active_research=ActiveResearch(relation="formula_lookup", targets=["DPO"], titles=["Paper"]),
        pending_clarification_type="ambiguity",
        pending_clarification_target="DPO",
        pending_clarification_options=[{"title": "Paper"}],
        working_memory={"tool_results": [{"tool": "search"}]},
        turns=[_turn(1, answer="A" * 2000)],
    )

    payload = session_conversation_context(session, persistent_learnings="prefer Chinese")

    assert payload["active_research_context"]["targets"] == ["DPO"]
    assert payload["pending_clarification"]["target"] == "DPO"
    assert payload["persistent_learnings"] == "prefer Chinese"
    assert payload["turns"][0]["assistant_answer"].endswith("...")
    assert payload["turns"][0]["query_contract"]["answer_slots"] == ["formula"]


def test_session_conversation_context_compacts_when_prompt_budget_is_small() -> None:
    session = SessionContext(session_id="ctx", turns=[_turn(index, answer="x" * 1000) for index in range(6)])

    payload = session_conversation_context(session, max_chars=200)

    assert payload["context_compression_note"]
    assert len(payload["turns"][0]["assistant_answer"]) <= 280
    assert len(payload["turns"][-1]["assistant_answer"]) <= 900


def test_session_llm_history_messages_include_summary_and_metadata() -> None:
    session = SessionContext(session_id="history", summary="older", turns=[_turn(1), _turn(2)])

    messages = session_llm_history_messages(session, max_turns=1, answer_limit=5)

    assert messages[0]["role"] == "human"
    assert messages[1] == {"role": "user", "content": "q2"}
    assert messages[2]["role"] == "assistant"
    assert "an..." in messages[2]["content"]
    assert '"relation": "formula_lookup"' in messages[2]["content"]


def test_truncate_context_text_strips_and_limits() -> None:
    assert truncate_context_text("  abc  ", limit=10) == "abc"
    assert truncate_context_text("abcdef", limit=5) == "ab..."
