from __future__ import annotations

from app.domain.models import ActiveResearch, QueryContract, SessionContext, SessionTurn
from app.services.memory.learnings import remember_learning
from app.services.contracts.session_context import (
    agent_session_conversation_context,
    active_research_from_contract,
    apply_session_history_compression,
    compress_session_history_if_needed,
    conversation_active_research_from_contract,
    make_active_research,
    session_conversation_context,
    session_history_compression_payload,
    session_history_compression_system_prompt,
    session_history_compression_window,
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


def test_agent_session_conversation_context_loads_persistent_learnings(tmp_path) -> None:
    remember_learning(data_dir=tmp_path, key="style", content="prefer Chinese")
    session = SessionContext(session_id="ctx")

    payload = agent_session_conversation_context(
        session,
        settings=type("Settings", (), {"data_dir": tmp_path})(),
    )

    assert "prefer Chinese" in payload["persistent_learnings"]


def test_make_active_research_normalizes_precision_and_signature() -> None:
    active = make_active_research(
        relation="formula_lookup",
        targets=["DPO"],
        titles=["Direct Preference Optimization"],
        requested_fields=["formula"],
        required_modalities=["page_text"],
        answer_shape="bullets",
        precision_requirement="unsupported",
        clean_query="DPO公式是什么",
    )

    assert active.precision_requirement == "normal"
    assert active.last_topic_signature == active.topic_signature()
    assert active.targets == ["DPO"]


def test_active_research_from_contract_centralizes_contract_mapping() -> None:
    contract = QueryContract(
        clean_query="DPO公式是什么",
        relation="formula_lookup",
        targets=["DPO"],
        requested_fields=["formula"],
        required_modalities=["page_text"],
        answer_shape="bullets",
        precision_requirement="exact",
    )

    active = active_research_from_contract(contract, titles=["Direct Preference Optimization"])
    skipped = conversation_active_research_from_contract(
        QueryContract(clean_query="hello", interaction_mode="conversation", relation="smalltalk"),
        titles=[],
    )

    assert active.relation == "formula_lookup"
    assert active.titles == ["Direct Preference Optimization"]
    assert active.precision_requirement == "exact"
    assert skipped is None


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


def test_session_history_compression_helpers_build_payload_and_apply() -> None:
    session = SessionContext(session_id="history", summary="old summary", turns=[_turn(index, answer="x" * 500) for index in range(8)])

    retained_turns, older_turns = session_history_compression_window(session, max_turns=8)
    payload = session_history_compression_payload(session, older_turns=older_turns)

    assert retained_turns == 4
    assert len(older_turns) == 4
    assert "会话记忆压缩器" in session_history_compression_system_prompt()
    assert payload["existing_summary"] == "old summary"
    assert payload["older_turns"][0]["answer"] == "x" * 320

    apply_session_history_compression(session, compressed="new summary", retained_turns=retained_turns)
    assert session.summary == "new summary"
    assert [turn.query for turn in session.turns] == ["q4", "q5", "q6", "q7"]


def test_compress_session_history_if_needed_invokes_client_and_persists() -> None:
    class _Clients:
        chat = object()

        def __init__(self) -> None:
            self.called = False

        def invoke_text(self, **_: object) -> str:
            self.called = True
            return "compressed"

    class _Sessions:
        def __init__(self) -> None:
            self.saved: list[SessionContext] = []

        def upsert(self, session: SessionContext) -> None:
            self.saved.append(session)

    session = SessionContext(session_id="history", summary="old", turns=[_turn(index) for index in range(8)])
    clients = _Clients()
    sessions = _Sessions()

    compressed = compress_session_history_if_needed(
        session=session,
        clients=clients,
        settings=type("Settings", (), {"agent_history_max_turns": 8})(),
        sessions=sessions,
    )

    assert compressed is True
    assert clients.called is True
    assert session.summary == "compressed"
    assert [turn.query for turn in session.turns] == ["q4", "q5", "q6", "q7"]
    assert sessions.saved == [session]


def test_truncate_context_text_strips_and_limits() -> None:
    assert truncate_context_text("  abc  ", limit=10) == "abc"
    assert truncate_context_text("abcdef", limit=5) == "ab..."
