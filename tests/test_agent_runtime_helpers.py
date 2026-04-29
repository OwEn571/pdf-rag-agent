from __future__ import annotations

from types import SimpleNamespace

from app.domain.models import ActiveResearch, CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan, SessionContext, SessionTurn
from app.services.agent_runtime_helpers import (
    agent_loop_summary,
    agent_loop_execution_step,
    claim_focus_titles,
    configured_max_steps,
    contract_needs_human_clarification,
    conversation_runtime_actions,
    conversation_runtime_state,
    dequeue_action,
    entity_evidence_limit,
    execute_tool_loop,
    excluded_focus_titles,
    finalize_research_runtime,
    finalize_research_verification,
    filter_candidate_papers_by_excluded_titles,
    filter_evidence_by_excluded_titles,
    next_conversation_action,
    next_research_action,
    record_tool_loop_ready,
    planner_next_action,
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
    assert agent_loop_execution_step(["read_memory", "compose"]) == {
        "node": "agent_loop",
        "summary": "read_memory -> compose",
    }
    assert tool_loop_ready_tool(["search_corpus", "compose"]) == "search_corpus"
    assert tool_loop_ready_tool(["compose"]) == "compose"


def test_runtime_helpers_record_tool_loop_ready_event_and_step() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    execution_steps: list[dict[str, object]] = []

    record_tool_loop_ready(
        emit=lambda event, payload: events.append((event, payload)),
        execution_steps=execution_steps,
        tool="search_corpus",
        actions=["search_corpus", "compose"],
        tool_inputs={"search_corpus": {"query": "DPO"}},
    )

    assert events[0][0] == "observation"
    assert events[0][1]["tool"] == "search_corpus"
    assert events[0][1]["payload"] == {
        "actions": ["search_corpus", "compose"],
        "tool_inputs": {"search_corpus": {"query": "DPO"}},
    }
    assert execution_steps == [{"node": "agent_loop", "summary": "search_corpus -> compose"}]


def test_runtime_helpers_filter_excluded_focus_titles_and_limits() -> None:
    session = SessionContext(
        session_id="s1",
        active_research=ActiveResearch(titles=["Old Focus Paper"]),
        turns=[SessionTurn(query="q", answer="a", titles=["Recent Paper"])],
    )
    contract = QueryContract(clean_query="不是这篇", notes=["exclude_previous_focus"], targets=["PBA"])

    excluded = excluded_focus_titles(
        session=session,
        contract=contract,
        is_negative_correction_query=lambda _: False,
    )

    assert excluded == {"old focus paper", "recent paper"}
    candidates = [
        CandidatePaper(paper_id="old", title="Old Focus Paper"),
        CandidatePaper(paper_id="new", title="New Paper"),
    ]
    evidence = [
        EvidenceBlock(doc_id="1", paper_id="old", title="Recent Paper", file_path="", page=1, block_type="page_text", snippet="old"),
        EvidenceBlock(doc_id="2", paper_id="new", title="New Paper", file_path="", page=1, block_type="page_text", snippet="new"),
    ]

    assert [item.paper_id for item in filter_candidate_papers_by_excluded_titles(candidates, excluded_titles=excluded)] == ["new"]
    assert [item.doc_id for item in filter_evidence_by_excluded_titles(evidence, excluded_titles=excluded)] == ["2"]
    assert entity_evidence_limit(
        contract=QueryContract(clean_query="PBA是什么", targets=["PBA"], requested_fields=["role_in_context"]),
        plan=ResearchPlan(evidence_limit=14),
        excluded_titles=set(),
    ) == 72
    assert entity_evidence_limit(
        contract=QueryContract(clean_query="PBA是什么", targets=["PBA"], requested_fields=["role_in_context"]),
        plan=ResearchPlan(evidence_limit=14),
        excluded_titles={"old focus paper"},
    ) == 96


def test_runtime_helpers_claim_focus_titles_falls_back_to_lookup_and_candidates() -> None:
    papers = [CandidatePaper(paper_id="p1", title="Known Paper"), CandidatePaper(paper_id="p3", title="Fallback Paper")]
    claims = [Claim(claim_type="definition", paper_ids=["p1", "p2"])]

    assert claim_focus_titles(
        claims=claims,
        papers=papers,
        paper_title_lookup=lambda paper_id: "Looked Up Paper" if paper_id == "p2" else None,
    ) == ["Known Paper", "Looked Up Paper"]
    assert claim_focus_titles(claims=[], papers=papers, paper_title_lookup=lambda _: None) == ["Known Paper", "Fallback Paper"]


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


def test_runtime_helpers_choose_planner_next_action() -> None:
    base_contract = QueryContract(clean_query="base")
    override_contract = QueryContract(clean_query="override")
    session = SimpleNamespace()
    calls: list[dict[str, object]] = []

    class Planner:
        def choose_next_action(self, **kwargs: object) -> str:
            calls.append(kwargs)
            return "compose"

    action = planner_next_action(
        agent=SimpleNamespace(planner=Planner()),
        contract=base_contract,
        session=session,
        state={"contract": override_contract},
        executed_actions=["read_memory"],
        allowed_tools={"read_memory", "compose"},
    )

    assert action == "compose"
    assert calls == [
        {
            "contract": override_contract,
            "session": session,
            "state": {"contract": override_contract},
            "executed_actions": ["read_memory"],
            "allowed_tools": {"read_memory", "compose"},
        }
    ]
    assert (
        planner_next_action(
            agent=SimpleNamespace(),
            contract=base_contract,
            session=session,
            state={},
            executed_actions=[],
            allowed_tools={"compose"},
        )
        is None
    )


def test_runtime_helpers_execute_tool_loop_runs_planned_and_fallback_actions() -> None:
    contract = QueryContract(clean_query="x")
    session = SimpleNamespace()
    events: list[tuple[str, dict[str, object]]] = []
    steps: list[dict[str, object]] = []

    class Agent:
        agent_settings = SimpleNamespace(max_agent_steps=4)

        def _emit_agent_step(self, **kwargs: object) -> None:
            steps.append(kwargs)

    class Executor:
        def __init__(self) -> None:
            self.executed: set[str] = set()
            self.runs: list[str] = []

        def run(self, action: str) -> bool:
            self.executed.add(action)
            self.runs.append(action)
            return False

    executor = Executor()
    state = {
        "contract": contract,
        "tool_inputs": {"read_memory": {"reason": "context"}, "compose": {"style": "short"}},
    }

    execute_tool_loop(
        agent=Agent(),
        contract=contract,
        session=session,
        state=state,
        executor=executor,
        planned_actions=["read_memory"],
        allowed_tools={"read_memory", "compose"},
        emit=lambda event, payload: events.append((event, payload)),
        fallback_next=lambda executed: "compose" if "compose" not in executed else None,
        stop_condition=lambda executed: "compose" in executed,
    )

    assert executor.runs == ["read_memory", "compose"]
    assert state["current_tool_input"] == {"style": "short"}
    assert [step["action"] for step in steps] == ["read_memory", "compose"]
    assert steps[0]["arguments"] == {"reason": "context"}
    assert steps[1]["arguments"] == {"style": "short"}


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


def test_runtime_helpers_finalize_research_runtime_emits_verification_and_confidence() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    execution_steps: list[dict[str, object]] = []
    session = SimpleNamespace()

    class Agent:
        def _agent_reflect(self, **kwargs: object) -> None:
            kwargs["state"]["reflected"] = True  # type: ignore[index]
            kwargs["execution_steps"].append({"node": "reflect", "summary": "ok"})  # type: ignore[index, union-attr]

    state: dict[str, object] = {}
    finalize_research_runtime(
        agent=Agent(),
        state=state,
        session=session,
        emit=lambda event, payload: events.append((event, payload)),
        execution_steps=execution_steps,
    )

    assert state["reflected"] is True
    assert events[0][0] == "verification"
    assert events[0][1]["status"] == "clarify"
    assert events[1][0] == "confidence"
    assert execution_steps[-1] == {"node": "agent_tool:verify_claim", "summary": "clarify"}


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
