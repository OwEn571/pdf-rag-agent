from __future__ import annotations

from types import SimpleNamespace

from app.domain.models import (
    ActiveResearch,
    CandidatePaper,
    Claim,
    EvidenceBlock,
    QueryContract,
    ResearchPlan,
    SessionContext,
    SessionTurn,
    VerificationReport,
)
from app.services.agent_runtime_helpers import (
    agent_loop_summary,
    agent_loop_execution_step,
    claim_focus_titles,
    clarification_limit_decision,
    clarify_retry_verification_if_needed,
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
    prepare_retry_research_materials,
    promote_best_effort_state_after_clarification_limit,
    record_tool_loop_ready,
    planner_next_action,
    prefer_selected_clarification_paper,
    refresh_selected_ambiguity_materials,
    research_runtime_actions,
    research_runtime_state,
    retry_research_limits,
    screen_agent_papers,
    search_agent_candidate_papers,
    search_agent_evidence,
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


def test_runtime_helpers_prefer_selected_clarification_paper() -> None:
    candidates = [
        CandidatePaper(paper_id="p1", title="First"),
        CandidatePaper(paper_id="p2", title="Second"),
    ]

    selected = prefer_selected_clarification_paper(
        candidates,
        contract=QueryContract(clean_query="选第二个", notes=["selected_paper_id=p2"]),
        paper_lookup=lambda _: None,
    )

    assert [item.paper_id for item in selected] == ["p2"]

    fallback = prefer_selected_clarification_paper(
        candidates,
        contract=QueryContract(clean_query="选第三个", notes=["selected_paper_id=p3"]),
        paper_lookup=lambda paper_id: CandidatePaper(paper_id=paper_id, title="Third"),
    )

    assert [item.paper_id for item in fallback] == ["p3"]
    assert prefer_selected_clarification_paper(
        candidates,
        contract=QueryContract(clean_query="普通查询"),
        paper_lookup=lambda _: None,
    ) == candidates


def test_runtime_helpers_screen_agent_papers_entity_path_filters_selected_evidence() -> None:
    candidates = [
        CandidatePaper(paper_id="p1", title="First"),
        CandidatePaper(paper_id="p2", title="Second"),
    ]
    evidence = [
        EvidenceBlock(doc_id="e1", paper_id="p1", title="First", file_path="", page=1, block_type="page_text", snippet="PBA one"),
        EvidenceBlock(doc_id="e2", paper_id="p2", title="Second", file_path="", page=1, block_type="page_text", snippet="PBA two"),
    ]
    calls: dict[str, object] = {}

    def search_entity(query: str, contract: QueryContract, limit: int) -> list[EvidenceBlock]:
        calls["args"] = (query, contract, limit)
        return evidence

    screened, precomputed = screen_agent_papers(
        contract=QueryContract(
            clean_query="PBA是什么",
            targets=["PBA"],
            requested_fields=["role_in_context"],
            notes=["selected_paper_id=p2"],
        ),
        plan=ResearchPlan(paper_limit=4, evidence_limit=8),
        candidate_papers=candidates,
        excluded_titles=set(),
        paper_lookup=lambda _: None,
        paper_summary_text=lambda _: "",
        prefer_identity_matching_papers=lambda papers, _: papers,
        search_entity_evidence=search_entity,
        ground_entity_papers=lambda papers, found_evidence, limit: [
            paper for paper in papers[:limit] if paper.paper_id in {item.paper_id for item in found_evidence}
        ],
    )

    assert [item.paper_id for item in screened] == ["p2"]
    assert [item.doc_id for item in precomputed or []] == ["e2"]
    query, _, limit = calls["args"]
    assert "PBA" in str(query)
    assert limit == 72


def test_runtime_helpers_screen_agent_papers_formula_path_uses_identity_matcher() -> None:
    candidates = [
        CandidatePaper(paper_id="p1", title="Wrong"),
        CandidatePaper(paper_id="p2", title="DPO"),
    ]

    screened, precomputed = screen_agent_papers(
        contract=QueryContract(clean_query="DPO公式", targets=["DPO"], requested_fields=["formula"]),
        plan=ResearchPlan(paper_limit=4, evidence_limit=8),
        candidate_papers=candidates,
        excluded_titles=set(),
        paper_lookup=lambda _: None,
        paper_summary_text=lambda _: "",
        prefer_identity_matching_papers=lambda papers, targets: [paper for paper in papers if paper.title in targets],
        search_entity_evidence=lambda *_: [],
        ground_entity_papers=lambda papers, *_: papers,
    )

    assert [item.paper_id for item in screened] == ["p2"]
    assert precomputed is None


def test_runtime_helpers_search_agent_evidence_uses_precomputed_when_available() -> None:
    papers = [CandidatePaper(paper_id="p1", title="Paper")]
    precomputed = [
        EvidenceBlock(doc_id="e1", paper_id="p1", title="Paper", file_path="", page=1, block_type="page_text", snippet="hit")
    ]
    calls: list[str] = []

    result = search_agent_evidence(
        contract=QueryContract(clean_query="DPO", targets=["DPO"]),
        plan=ResearchPlan(evidence_limit=8),
        tool_input={"query": "custom DPO", "top_k": 3},
        screened_papers=papers,
        precomputed_evidence=precomputed,
        excluded_titles=set(),
        search_concept_evidence=lambda *_: [],
        expand_evidence=lambda *_: calls.append("expand") or [],
    )

    assert result.evidence == precomputed
    assert result.query == "custom DPO"
    assert result.limit == 3
    assert calls == []


def test_runtime_helpers_search_agent_candidate_papers_filters_and_falls_back_to_active_targets() -> None:
    old = CandidatePaper(paper_id="old", title="Old Focus")
    fallback = CandidatePaper(paper_id="p2", title="Active Target Paper")
    calls: list[tuple[str, QueryContract, int]] = []

    def search(query: str, contract: QueryContract, limit: int) -> list[CandidatePaper]:
        calls.append((query, contract, limit))
        if len(calls) == 1:
            return [old]
        return [fallback]

    result = search_agent_candidate_papers(
        contract=QueryContract(clean_query="继续", continuation_mode="followup", targets=[]),
        paper_query="继续",
        paper_limit=5,
        active_targets=["Active Target"],
        excluded_titles={"old focus"},
        search_papers=search,
        paper_lookup=lambda _: None,
    )

    assert result.contract.targets == ["Active Target"]
    assert result.candidate_papers == [fallback]
    assert len(calls) == 2
    assert calls[1][1].targets == ["Active Target"]
    assert calls[1][2] == 5


def test_runtime_helpers_search_agent_candidate_papers_prefers_selected_lookup() -> None:
    selected = CandidatePaper(paper_id="p2", title="Selected")

    result = search_agent_candidate_papers(
        contract=QueryContract(clean_query="选第二个", notes=["selected_paper_id=p2"]),
        paper_query="query",
        paper_limit=5,
        active_targets=[],
        excluded_titles=set(),
        search_papers=lambda *_: [],
        paper_lookup=lambda paper_id: selected if paper_id == "p2" else None,
    )

    assert result.contract.notes == ["selected_paper_id=p2"]
    assert result.candidate_papers == [selected]


def test_runtime_helpers_prepare_retry_research_materials_expands_limits_and_filters_selection() -> None:
    candidates = [
        CandidatePaper(paper_id="p1", title="Old Focus"),
        CandidatePaper(paper_id="p2", title="Selected"),
    ]
    evidence = [
        EvidenceBlock(doc_id="old", paper_id="p1", title="Old Focus", file_path="", page=1, block_type="page_text", snippet="old"),
        EvidenceBlock(doc_id="selected", paper_id="p2", title="Selected", file_path="", page=1, block_type="page_text", snippet="selected"),
    ]
    search_calls: list[tuple[str, int]] = []
    expand_calls: list[tuple[list[str], int]] = []
    plan = ResearchPlan(paper_limit=4, evidence_limit=8)

    materials = prepare_retry_research_materials(
        contract=QueryContract(clean_query="DPO", targets=["DPO"], notes=["selected_paper_id=p2"]),
        plan=plan,
        excluded_titles={"old focus"},
        search_papers=lambda query, _contract, limit: search_calls.append((query, limit)) or candidates,
        paper_lookup=lambda _: None,
        search_concept_evidence=lambda *_: [],
        search_entity_evidence=lambda *_: [],
        expand_evidence=lambda paper_ids, _query, _contract, limit: expand_calls.append((paper_ids, limit)) or evidence,
        ground_entity_papers=lambda papers, *_: papers,
    )

    assert retry_research_limits(plan).paper_limit == 10
    assert retry_research_limits(plan).evidence_limit == 20
    assert [item.paper_id for item in materials.candidate_papers] == ["p2"]
    assert [item.doc_id for item in materials.evidence] == ["selected"]
    assert search_calls == [("DPO", 10)]
    assert expand_calls == [(["p2"], 20)]


def test_runtime_helpers_prepare_retry_research_materials_entity_path_grounds_candidates() -> None:
    candidates = [
        CandidatePaper(paper_id="p1", title="First"),
        CandidatePaper(paper_id="p2", title="Second"),
    ]
    evidence = [
        EvidenceBlock(doc_id="e2", paper_id="p2", title="Second", file_path="", page=1, block_type="page_text", snippet="PBA")
    ]
    entity_limits: list[int] = []

    materials = prepare_retry_research_materials(
        contract=QueryContract(clean_query="PBA是什么", targets=["PBA"], requested_fields=["role_in_context"]),
        plan=ResearchPlan(paper_limit=4, evidence_limit=8),
        excluded_titles=set(),
        search_papers=lambda _query, _contract, _limit: candidates,
        paper_lookup=lambda _: None,
        search_concept_evidence=lambda *_: [],
        search_entity_evidence=lambda _query, _contract, limit: entity_limits.append(limit) or evidence,
        expand_evidence=lambda *_: [],
        ground_entity_papers=lambda papers, found_evidence, limit: [
            paper for paper in papers[:limit] if paper.paper_id in {item.paper_id for item in found_evidence}
        ],
    )

    assert [item.paper_id for item in materials.candidate_papers] == ["p2"]
    assert materials.evidence == evidence
    assert entity_limits == [72]
    assert "role_in_context" in materials.goals


def test_runtime_helpers_refresh_selected_ambiguity_materials_reuses_existing_evidence() -> None:
    paper = CandidatePaper(paper_id="p2", title="Selected")
    evidence = [
        EvidenceBlock(doc_id="e1", paper_id="p1", title="Other", file_path="", page=1, block_type="page_text", snippet="other"),
        EvidenceBlock(doc_id="e2", paper_id="p2", title="Selected", file_path="", page=1, block_type="page_text", snippet="selected"),
    ]

    refresh = refresh_selected_ambiguity_materials(
        selected={"paper_id": "p2"},
        contract=QueryContract(clean_query="DPO", targets=["DPO"]),
        plan=ResearchPlan(evidence_limit=8),
        candidate_papers=[paper],
        existing_evidence=evidence,
        excluded_titles=set(),
        paper_lookup=lambda _: None,
        search_concept_evidence=lambda *_: [],
        expand_evidence=lambda *_: [],
    )

    assert refresh is not None
    assert refresh.selected_papers == [paper]
    assert [item.doc_id for item in refresh.evidence] == ["e2"]
    assert refresh.evidence_refreshed is False


def test_runtime_helpers_refresh_selected_ambiguity_materials_fetches_missing_evidence() -> None:
    fetched = [
        EvidenceBlock(doc_id="e2", paper_id="p2", title="Selected", file_path="", page=1, block_type="page_text", snippet="selected")
    ]
    calls: list[tuple[list[str], int]] = []

    refresh = refresh_selected_ambiguity_materials(
        selected={"paper_id": "p2"},
        contract=QueryContract(clean_query="DPO", targets=["DPO"], requested_fields=["definition"]),
        plan=ResearchPlan(evidence_limit=8),
        candidate_papers=[],
        existing_evidence=[],
        excluded_titles=set(),
        paper_lookup=lambda paper_id: CandidatePaper(paper_id=paper_id, title="Selected"),
        search_concept_evidence=lambda _query, _contract, paper_ids, limit: calls.append((paper_ids, limit)) or fetched,
        expand_evidence=lambda *_: [],
    )

    assert refresh is not None
    assert [item.paper_id for item in refresh.selected_papers] == ["p2"]
    assert refresh.evidence == fetched
    assert refresh.evidence_refreshed is True
    assert calls == [(["p2"], 8)]


def test_runtime_helpers_search_agent_evidence_concept_fallback_filters_selection_and_excluded() -> None:
    papers = [
        CandidatePaper(paper_id="p1", title="Old Focus"),
        CandidatePaper(paper_id="p2", title="Selected"),
    ]
    expanded = [
        EvidenceBlock(doc_id="old", paper_id="p1", title="Old Focus", file_path="", page=1, block_type="page_text", snippet="old"),
        EvidenceBlock(doc_id="selected", paper_id="p2", title="Selected", file_path="", page=1, block_type="page_text", snippet="selected"),
    ]
    calls: list[tuple[str, int]] = []

    result = search_agent_evidence(
        contract=QueryContract(
            clean_query="PBA是什么",
            relation="concept_definition",
            targets=["PBA"],
            requested_fields=["definition"],
            notes=["selected_paper_id=p2"],
        ),
        plan=ResearchPlan(evidence_limit=8),
        tool_input={},
        screened_papers=papers,
        precomputed_evidence=None,
        excluded_titles={"old focus"},
        search_concept_evidence=lambda query, _contract, _paper_ids, limit: calls.append((query, limit)) or [],
        expand_evidence=lambda _paper_ids, _query, _contract, _limit: expanded,
    )

    assert calls and "PBA" in calls[0][0]
    assert calls[0][1] == 8
    assert [item.doc_id for item in result.evidence] == ["selected"]


def test_runtime_helpers_claim_focus_titles_falls_back_to_lookup_and_candidates() -> None:
    papers = [CandidatePaper(paper_id="p1", title="Known Paper"), CandidatePaper(paper_id="p3", title="Fallback Paper")]
    claims = [Claim(claim_type="definition", paper_ids=["p1", "p2"])]

    assert claim_focus_titles(
        claims=claims,
        papers=papers,
        paper_title_lookup=lambda paper_id: "Looked Up Paper" if paper_id == "p2" else None,
    ) == ["Known Paper", "Looked Up Paper"]
    assert claim_focus_titles(claims=[], papers=papers, paper_title_lookup=lambda _: None) == ["Known Paper", "Fallback Paper"]


def test_runtime_helpers_clarify_retry_verification_for_targeted_research_goals() -> None:
    retry = VerificationReport(status="retry", missing_fields=["evidence"], recommended_action="expand")

    clarified = clarify_retry_verification_if_needed(
        contract=QueryContract(clean_query="DPO是什么", targets=["DPO"], requested_fields=["definition"]),
        verification=retry,
    )

    assert clarified.status == "clarify"
    assert clarified.missing_fields == ["relevant_evidence"]
    assert clarify_retry_verification_if_needed(
        contract=QueryContract(clean_query="继续", targets=[]),
        verification=retry,
    ) is retry


def test_runtime_helpers_clarification_limit_decision_uses_first_option() -> None:
    contract = QueryContract(
        clean_query="PBA是什么",
        targets=["PBA"],
        notes=["ambiguity_option=1"],
    )
    verification = VerificationReport(status="clarify", recommended_action="clarify_ambiguous_entity")

    decision = clarification_limit_decision(
        contract=contract,
        verification=verification,
        next_attempt=2,
        max_attempts=2,
        options=[
            {
                "target": "PBA",
                "paper_id": "p1",
                "title": "AlignX",
                "meaning": "Preference Bridged Alignment",
            }
        ],
    )

    assert decision is not None
    assert decision.summary == "selected=Preference Bridged Alignment"
    assert decision.forced_plan["actions"] == ["search_corpus", "compose"]
    assert "clarification_limit_reached" in decision.forced_contract.notes
    assert "assumed_most_likely_intent" in decision.forced_contract.notes
    assert decision.observation_payload == {
        "max_attempts": 2,
        "attempt": 2,
        "assumption": "selected=Preference Bridged Alignment",
    }


def test_runtime_helpers_clarification_limit_decision_requires_limit() -> None:
    contract = QueryContract(clean_query="PBA是什么", targets=["PBA"])
    verification = VerificationReport(status="clarify", recommended_action="clarify_ambiguous_entity")

    assert clarification_limit_decision(
        contract=contract,
        verification=verification,
        next_attempt=1,
        max_attempts=2,
        options=[],
    ) is None
    assert clarification_limit_decision(
        contract=contract,
        verification=VerificationReport(status="pass"),
        next_attempt=2,
        max_attempts=2,
        options=[],
    ) is None


def test_runtime_helpers_clarification_limit_decision_without_options_marks_best_effort() -> None:
    contract = QueryContract(clean_query="PBA是什么", targets=["PBA"], notes=["existing"])
    verification = VerificationReport(status="clarify", recommended_action="clarify_target")

    decision = clarification_limit_decision(
        contract=contract,
        verification=verification,
        next_attempt=3,
        max_attempts=2,
        options=[],
    )

    assert decision is not None
    assert decision.summary == "clarify_target"
    assert decision.forced_contract.notes == ["existing", "clarification_limit_reached", "best_effort_answer"]


def test_runtime_helpers_promote_best_effort_state_after_clarification_limit() -> None:
    contract = QueryContract(clean_query="PBA是什么", notes=["clarification_limit_reached"])
    state = {
        "contract": contract,
        "verification": VerificationReport(status="clarify", recommended_action="clarify_target"),
        "claims": [Claim(claim_type="definition", text="PBA is a method.")],
    }

    promoted = promote_best_effort_state_after_clarification_limit(state)

    assert promoted is not state
    assert promoted["verification"].status == "pass"
    assert promoted["verification"].recommended_action == "best_effort_after_clarification_limit"
    assert promoted["contract"].notes == [
        "clarification_limit_reached",
        "best_effort_after_clarification_limit",
    ]
    unchanged = {"verification": VerificationReport(status="clarify")}
    assert promote_best_effort_state_after_clarification_limit(unchanged) is unchanged


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
