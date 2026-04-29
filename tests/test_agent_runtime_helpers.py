from __future__ import annotations

from types import SimpleNamespace

from app.domain.models import QueryContract, ResearchPlan
from app.services.agent_runtime_helpers import (
    agent_loop_summary,
    configured_max_steps,
    contract_needs_human_clarification,
    conversation_runtime_state,
    next_conversation_action,
    next_research_action,
    research_runtime_state,
    tool_loop_ready_tool,
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


def test_runtime_helpers_detect_clarification_need_from_contract_confidence() -> None:
    settings = SimpleNamespace(confidence_floor=0.6)

    assert contract_needs_human_clarification(QueryContract(clean_query="x"), settings) is False
    assert contract_needs_human_clarification(
        QueryContract(clean_query="x", notes=["ambiguous_slot=paper_title"]),
        settings,
    ) is True


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
