from __future__ import annotations

from types import SimpleNamespace

from app.domain.models import QueryContract, ResearchPlan
from app.services.agent_runtime_helpers import (
    agent_loop_summary,
    configured_max_steps,
    contract_needs_human_clarification,
    conversation_runtime_actions,
    conversation_runtime_state,
    dequeue_action,
    finalize_research_verification,
    next_conversation_action,
    next_research_action,
    research_runtime_actions,
    research_runtime_state,
    tool_loop_ready_tool,
    verification_execution_step,
)


def test_runtime_helpers_configure_max_steps() -> None:
    assert configured_max_steps(None, fallback=8) == 8
    assert configured_max_steps(SimpleNamespace(max_agent_steps="3"), fallback=8) == 3
    assert configured_max_steps(SimpleNamespace(max_agent_steps="bad"), fallback=8) == 8
    assert configured_max_steps(SimpleNamespace(max_agent_steps=0), fallback=8) == 1


def test_runtime_helpers_build_initial_conversation_and_research_state() -> None:
    contract = QueryContract(clean_query="x")
    agent_plan = {"tool_call_args": [{"name": "fetch_url", "args": {"url": "https://example.com"}}]}
    plan = ResearchPlan(solver_sequence=["origin_lookup"])

    conversation_state = conversation_runtime_state(contract=contract, agent_plan=agent_plan)
    research_state = research_runtime_state(
        contract=contract,
        plan=plan,
        excluded_titles={"A"},
        agent_plan=agent_plan,
    )

    assert conversation_state["contract"] == contract
    assert conversation_state["answer"] == ""
    assert conversation_state["verification_report"] == {
        "status": "pass",
        "recommended_action": "conversation_tool_answer",
    }
    assert conversation_state["tool_inputs"] == {"fetch_url": {"url": "https://example.com"}}
    assert research_state["plan"] == plan
    assert research_state["excluded_titles"] == {"A"}
    assert research_state["tool_inputs"] == {"fetch_url": {"url": "https://example.com"}}
    assert research_state["verification"] is None
    assert agent_loop_summary(["read_memory", "compose"]) == "read_memory -> compose"
    assert tool_loop_ready_tool(["search_corpus", "compose"]) == "search_corpus"
    assert tool_loop_ready_tool(["compose"]) == "compose"


def test_runtime_helpers_build_action_sequences_and_dequeue_actions() -> None:
    conversation_contract = QueryContract(clean_query="x", interaction_mode="conversation", relation="greeting")
    research_contract = QueryContract(clean_query="DPO 是什么", relation="entity_definition")
    queue = ["read_memory", "compose"]

    assert conversation_runtime_actions(
        contract=conversation_contract,
        agent_plan={"actions": ["read_memory", "not_a_tool", "compose"]},
    ) == ["read_memory", "compose"]
    assert conversation_runtime_actions(contract=conversation_contract, agent_plan={"actions": "bad"}) == []
    assert research_runtime_actions(
        contract=research_contract,
        agent_plan={"actions": ["search_corpus", "not_a_tool", "compose"]},
        web_enabled=False,
        is_negative_correction_query=lambda _: False,
    ) == ["search_corpus", "compose"]
    assert research_runtime_actions(
        contract=research_contract,
        agent_plan={"actions": "bad"},
        web_enabled=True,
        is_negative_correction_query=lambda _: True,
    ) == []
    assert dequeue_action(queue=queue, executed={"read_memory"}) == "compose"
    assert queue == []
    assert dequeue_action(queue=[], executed=set()) is None


def test_runtime_helpers_detect_clarification_need_from_contract_confidence() -> None:
    settings = SimpleNamespace(confidence_floor=0.6)

    assert contract_needs_human_clarification(QueryContract(clean_query="x"), settings) is False
    assert contract_needs_human_clarification(
        QueryContract(clean_query="x", notes=["ambiguous_slot=paper_title"]),
        settings,
    ) is True


def test_runtime_helpers_finalize_research_verification_and_confidence() -> None:
    missing_state: dict[str, object] = {}
    verification, confidence = finalize_research_verification(missing_state)

    assert verification.status == "clarify"
    assert verification.missing_fields == ["verified_claims"]
    assert missing_state["verification"] == verification
    assert missing_state["confidence"] == confidence
    assert confidence["basis"] == "verifier"
    assert confidence["score"] == 0.0
    assert verification_execution_step(verification) == {
        "node": "agent_tool:verify_claim",
        "summary": "clarify",
    }

    pass_state = {"verification": verification.model_copy(update={"status": "pass", "missing_fields": []})}
    passed, passed_confidence = finalize_research_verification(pass_state)

    assert passed.status == "pass"
    assert passed_confidence["score"] > 0.8
    assert pass_state["verification"] == passed


def test_runtime_helpers_choose_next_conversation_action() -> None:
    settings = SimpleNamespace(confidence_floor=0.6)

    assert (
        next_conversation_action(
            contract=QueryContract(clean_query="x", notes=["ambiguous_slot=paper_title"]),
            state={},
            executed=set(),
            agent_settings=settings,
        )
        == "ask_human"
    )
    assert (
        next_conversation_action(
            contract=QueryContract(clean_query="x", notes=["intent_kind=memory_op"]),
            state={},
            executed=set(),
            agent_settings=settings,
        )
        == "read_memory"
    )
    assert (
        next_conversation_action(
            contract=QueryContract(clean_query="x", requested_fields=["citation_count_ranking"]),
            state={},
            executed={"read_memory"},
            agent_settings=settings,
        )
        == "web_search"
    )
    assert (
        next_conversation_action(
            contract=QueryContract(clean_query="x", relation="library_status"),
            state={},
            executed=set(),
            agent_settings=settings,
        )
        == "query_library_metadata"
    )
    assert (
        next_conversation_action(
            contract=QueryContract(clean_query="x"),
            state={"answer": ""},
            executed=set(),
            agent_settings=settings,
        )
        == "compose"
    )
    assert (
        next_conversation_action(
            contract=QueryContract(clean_query="x"),
            state={"answer": "done"},
            executed={"compose"},
            agent_settings=settings,
        )
        is None
    )


def test_runtime_helpers_choose_next_research_action() -> None:
    settings = SimpleNamespace(confidence_floor=0.6)

    assert (
        next_research_action(
            contract=QueryContract(clean_query="x", notes=["low_intent_confidence"]),
            state={},
            executed=set(),
            web_enabled=False,
            agent_settings=settings,
        )
        == "ask_human"
    )
    assert (
        next_research_action(
            contract=QueryContract(clean_query="x", continuation_mode="followup"),
            state={},
            executed=set(),
            web_enabled=False,
            agent_settings=settings,
        )
        == "read_memory"
    )
    assert (
        next_research_action(
            contract=QueryContract(clean_query="x"),
            state={"evidence": [], "screened_papers": []},
            executed=set(),
            web_enabled=False,
            agent_settings=settings,
        )
        == "search_corpus"
    )
    assert (
        next_research_action(
            contract=QueryContract(clean_query="x"),
            state={"evidence": [object()], "screened_papers": [object()]},
            executed={"search_corpus"},
            web_enabled=True,
            agent_settings=settings,
        )
        == "web_search"
    )
    assert (
        next_research_action(
            contract=QueryContract(clean_query="x"),
            state={"evidence": [object()], "screened_papers": [object()]},
            executed={"search_corpus", "web_search"},
            web_enabled=True,
            agent_settings=settings,
        )
        == "compose"
    )
    assert (
        next_research_action(
            contract=QueryContract(clean_query="x"),
            state={"evidence": [object()], "screened_papers": [object()]},
            executed={"search_corpus", "web_search", "compose"},
            web_enabled=True,
            agent_settings=settings,
        )
        is None
    )
