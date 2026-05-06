from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan, SessionContext, VerificationReport
from app.services.agent.emit import emit_agent_step
from app.services.agent.tools import conversation_tool_sequence, research_tool_sequence
from app.services.agent.planner_helpers import research_contract_should_try_tools_before_human
from app.services.clarification.intents import (
    ambiguity_options_from_notes,
    contract_from_selected_clarification_option,
    contract_needs_evidence_disambiguation,
    disambiguation_missing_fields,
    selected_clarification_paper_id,
)
from app.services.infra.confidence import (
    confidence_from_contract,
    confidence_from_verification_report,
    confidence_payload,
    should_ask_human,
)
from app.services.contracts.context import contract_has_note, contract_notes
from app.services.contracts.normalization import normalize_lookup_text
from app.services.followup.candidates import filter_followup_candidates
from app.services.planning.query_shaping import evidence_query_text, is_short_acronym, paper_query_text, should_use_concept_evidence
from app.services.planning.research import research_plan_context_from_contract
from app.services.tools.registry_helpers import (
    tool_input_from_state,
    tool_inputs_by_name,
    tool_loop_ready_observation,
    coerce_int,
)
from app.services.retrieval.web_evidence import solve_claims_with_web_research

NegativeCorrectionFn = Callable[[str], bool]
EmitFn = Callable[[str, dict[str, Any]], None]
FallbackNextFn = Callable[[set[str]], str | None]
StopConditionFn = Callable[[set[str]], bool]
PaperTitleLookupFn = Callable[[str], str | None]
PaperLookupFn = Callable[[str], CandidatePaper | None]
CandidatePaperSearchFn = Callable[[str, QueryContract, int], list[CandidatePaper]]
PaperSummaryFn = Callable[[str], str]
PaperIdentityMatcherFn = Callable[[list[CandidatePaper], list[str]], list[CandidatePaper]]
EntityEvidenceSearchFn = Callable[[str, QueryContract, int], list[EvidenceBlock]]
GroundEntityPapersFn = Callable[[list[CandidatePaper], list[EvidenceBlock], int], list[CandidatePaper]]
ConceptEvidenceSearchFn = Callable[[str, QueryContract, list[str], int], list[EvidenceBlock]]
ExpandEvidenceFn = Callable[[list[str], str, QueryContract, int], list[EvidenceBlock]]
AmbiguityOptionCountFn = Callable[[], int]
AgentClaimSolverFn = Callable[[QueryContract, ResearchPlan, list[CandidatePaper], list[EvidenceBlock]], list[Claim]]
AgentWebClaimBuilderFn = Callable[[QueryContract, list[EvidenceBlock]], Claim]
RetryClaimSolverFn = Callable[[ResearchPlan, list[CandidatePaper], list[EvidenceBlock]], list[Claim]]
RetryClaimVerifierFn = Callable[
    [ResearchPlan, list[Claim], list[CandidatePaper], list[EvidenceBlock]],
    VerificationReport,
]
ScreenAgentPapersFn = Callable[
    [QueryContract, ResearchPlan, list[CandidatePaper], set[str]],
    tuple[list[CandidatePaper], list[EvidenceBlock] | None],
]

PAPER_DISCOVERY_STAGE = "paper_discovery"
EVIDENCE_RETRIEVAL_STAGE = "evidence_retrieval"
CLAIM_COMPOSITION_STAGE = "claim_composition"
GROUNDING_VERIFICATION_STAGE = "grounding_verification"
RESEARCH_RETRY_STAGE = "research_retry"


@dataclass(frozen=True)
class AgentEvidenceSearchResult:
    evidence: list[EvidenceBlock]
    query: str
    limit: int
    tool_call_arguments: dict[str, Any]
    observation_summary: str
    observation_payload: dict[str, Any]


@dataclass(frozen=True)
class AgentCandidatePaperSearchResult:
    contract: QueryContract
    candidate_papers: list[CandidatePaper]


@dataclass(frozen=True)
class AgentPaperSearchRun:
    contract: QueryContract
    query: str
    candidate_papers: list[CandidatePaper]
    screened_papers: list[CandidatePaper]
    precomputed_evidence: list[EvidenceBlock] | None
    tool_call_arguments: dict[str, Any]
    observation_summary: str
    observation_payload: dict[str, Any]


@dataclass(frozen=True)
class RetryResearchLimits:
    paper_limit: int
    evidence_limit: int


@dataclass(frozen=True)
class RetryResearchMaterials:
    candidate_papers: list[CandidatePaper]
    evidence: list[EvidenceBlock]
    limits: RetryResearchLimits
    goals: set[str]


@dataclass(frozen=True)
class RetryVerificationResult:
    candidate_papers: list[CandidatePaper]
    evidence: list[EvidenceBlock]
    claims: list[Claim]
    verification: VerificationReport
    should_replace_materials: bool
    observation_summary: str
    observation_payload: dict[str, Any]


@dataclass(frozen=True)
class SelectedAmbiguityRefresh:
    paper_id: str
    selected_papers: list[CandidatePaper]
    evidence: list[EvidenceBlock]
    evidence_refreshed: bool


@dataclass(frozen=True)
class ClarificationLimitDecision:
    forced_contract: QueryContract
    forced_plan: dict[str, Any]
    summary: str
    observation_payload: dict[str, Any]


def conversation_runtime_state(*, contract: QueryContract, agent_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": contract,
        "answer": "",
        "citations": [],
        "verification_report": {"status": "pass", "recommended_action": "conversation_tool_answer"},
        "citation_candidates": [],
        "citation_lookup": {},
        "tool_inputs": tool_inputs_by_name(agent_plan),
        "current_tool_input": {},
    }


def conversation_runtime_actions(
    *,
    contract: QueryContract,
    agent_plan: dict[str, Any],
    extra_allowed_tools: set[str] | None = None,
) -> list[str]:
    raw_actions = agent_plan.get("actions", []) if isinstance(agent_plan, dict) else []
    planned_actions = [str(item) for item in list(raw_actions or [])]
    return conversation_tool_sequence(
        planned_actions=planned_actions,
        extra_allowed=extra_allowed_tools,
    )


def research_runtime_state(
    *,
    contract: QueryContract,
    plan: Any,
    excluded_titles: set[str],
    agent_plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "contract": contract,
        "plan": plan,
        "candidate_papers": [],
        "screened_papers": [],
        "precomputed_evidence": None,
        "evidence": [],
        "web_evidence": [],
        "claims": [],
        "verification": None,
        "reflection": {},
        "excluded_titles": excluded_titles,
        "tool_inputs": tool_inputs_by_name(agent_plan),
        "current_tool_input": {},
    }


def research_runtime_actions(
    *,
    contract: QueryContract,
    agent_plan: dict[str, Any],
    web_enabled: bool,
    is_negative_correction_query: NegativeCorrectionFn,
    extra_allowed_tools: set[str] | None = None,
) -> list[str]:
    raw_actions = agent_plan.get("actions", []) if isinstance(agent_plan, dict) else []
    return research_tool_sequence(
        planned_actions=raw_actions if isinstance(raw_actions, list) else [],
        extra_allowed=extra_allowed_tools,
    )


def agent_loop_summary(actions: list[str]) -> str:
    return " -> ".join(actions)


def agent_loop_execution_step(actions: list[str]) -> dict[str, str]:
    return {"node": "agent_loop", "summary": agent_loop_summary(actions)}


def tool_loop_ready_tool(actions: list[str]) -> str:
    return "search_corpus" if "search_corpus" in actions else "compose"


def record_tool_loop_ready(
    *,
    emit: EmitFn,
    execution_steps: list[dict[str, Any]],
    tool: str,
    actions: list[str],
    tool_inputs: dict[str, Any],
) -> None:
    emit(
        "observation",
        tool_loop_ready_observation(tool=tool, actions=actions, tool_inputs=tool_inputs),
    )
    execution_steps.append(agent_loop_execution_step(actions))


def excluded_focus_titles(
    *,
    session: SessionContext,
    contract: QueryContract,
    is_negative_correction_query: NegativeCorrectionFn,
) -> set[str]:
    if not contract_has_note(contract, "exclude_previous_focus") and not is_negative_correction_query(contract.clean_query):
        return set()
    titles: list[str] = []
    titles.extend(session.effective_active_research().titles)
    if session.turns:
        titles.extend(session.turns[-1].titles)
    return {normalize_lookup_text(title) for title in titles if normalize_lookup_text(title)}


def filter_candidate_papers_by_excluded_titles(
    candidates: list[CandidatePaper],
    *,
    excluded_titles: set[str],
) -> list[CandidatePaper]:
    if not excluded_titles:
        return candidates
    return [item for item in candidates if normalize_lookup_text(item.title) not in excluded_titles]


def filter_evidence_by_excluded_titles(
    evidence: list[EvidenceBlock],
    *,
    excluded_titles: set[str],
) -> list[EvidenceBlock]:
    if not excluded_titles:
        return evidence
    return [item for item in evidence if normalize_lookup_text(item.title) not in excluded_titles]


def prefer_selected_clarification_paper(
    candidates: list[CandidatePaper],
    *,
    contract: QueryContract,
    paper_lookup: PaperLookupFn,
) -> list[CandidatePaper]:
    selected_paper_id = selected_clarification_paper_id(contract)
    if not selected_paper_id:
        return candidates
    selected = [item for item in candidates if item.paper_id == selected_paper_id]
    if not selected:
        paper = paper_lookup(selected_paper_id)
        selected = [paper] if paper is not None else []
    return selected or candidates


def entity_evidence_limit(*, contract: QueryContract, plan: ResearchPlan, excluded_titles: set[str]) -> int:
    goals = set(research_plan_context_from_contract(contract).goals)
    if goals & {"entity_type", "role_in_context"} and contract.targets and is_short_acronym(contract.targets[0]):
        return max(plan.evidence_limit, 96 if excluded_titles else 72)
    return plan.evidence_limit


def screen_agent_papers(
    *,
    contract: QueryContract,
    plan: ResearchPlan,
    candidate_papers: list[CandidatePaper],
    excluded_titles: set[str],
    paper_lookup: PaperLookupFn,
    paper_summary_text: PaperSummaryFn,
    prefer_identity_matching_papers: PaperIdentityMatcherFn,
    search_entity_evidence: EntityEvidenceSearchFn,
    ground_entity_papers: GroundEntityPapersFn,
) -> tuple[list[CandidatePaper], list[EvidenceBlock] | None]:
    selected_paper_id = selected_clarification_paper_id(contract)
    candidate_papers = prefer_selected_clarification_paper(
        candidate_papers,
        contract=contract,
        paper_lookup=paper_lookup,
    )
    screened_papers = candidate_papers
    precomputed_evidence: list[EvidenceBlock] | None = None
    goals = set(research_plan_context_from_contract(contract).goals)
    if goals & {"followup_papers", "candidate_relationship", "strict_followup"}:
        screened_papers = filter_followup_candidates(
            contract=contract,
            candidates=candidate_papers,
            paper_summary_text=paper_summary_text,
        )
    elif "formula" in goals and contract.targets:
        screened_papers = prefer_identity_matching_papers(candidate_papers, contract.targets) or candidate_papers
    elif "figure_conclusion" in goals and contract.targets:
        screened_papers = prefer_identity_matching_papers(candidate_papers, contract.targets)
    elif goals & {"entity_type", "role_in_context"}:
        limit = entity_evidence_limit(
            contract=contract,
            plan=plan,
            excluded_titles=excluded_titles,
        )
        precomputed_evidence = search_entity_evidence(
            evidence_query_text(contract),
            contract,
            limit,
        )
        if selected_paper_id:
            precomputed_evidence = [item for item in precomputed_evidence if item.paper_id == selected_paper_id]
        if excluded_titles:
            precomputed_evidence = filter_evidence_by_excluded_titles(
                precomputed_evidence,
                excluded_titles=excluded_titles,
            )
        grounded_papers = ground_entity_papers(candidate_papers, precomputed_evidence, plan.paper_limit)
        if grounded_papers:
            screened_papers = grounded_papers
    # Apply paper_limit across all query types — not just entity queries
    if len(screened_papers) > plan.paper_limit:
        screened_papers = screened_papers[: max(1, plan.paper_limit)]
    return screened_papers, precomputed_evidence


def search_agent_evidence(
    *,
    contract: QueryContract,
    plan: ResearchPlan,
    tool_input: dict[str, Any],
    screened_papers: list[CandidatePaper],
    precomputed_evidence: list[EvidenceBlock] | None,
    excluded_titles: set[str],
    search_concept_evidence: ConceptEvidenceSearchFn,
    expand_evidence: ExpandEvidenceFn,
) -> AgentEvidenceSearchResult:
    evidence_limit = coerce_int(
        tool_input.get("top_k", plan.evidence_limit),
        default=plan.evidence_limit,
        minimum=1,
        maximum=50,
    )
    evidence_query = str(tool_input.get("query", "") or "").strip() or evidence_query_text(contract)
    paper_ids = [item.paper_id for item in screened_papers]
    if should_use_concept_evidence(contract):
        evidence = search_concept_evidence(
            evidence_query,
            contract,
            paper_ids,
            evidence_limit,
        )
        if not evidence:
            evidence = expand_evidence(
                paper_ids,
                evidence_query,
                contract,
                evidence_limit,
            )
    else:
        evidence = precomputed_evidence or expand_evidence(
            paper_ids,
            evidence_query,
            contract,
            evidence_limit,
        )
    if excluded_titles:
        evidence = filter_evidence_by_excluded_titles(evidence, excluded_titles=excluded_titles)
    selected_paper_id = selected_clarification_paper_id(contract)
    if selected_paper_id:
        evidence = [item for item in evidence if item.paper_id == selected_paper_id]
    return AgentEvidenceSearchResult(
        evidence=evidence,
        query=evidence_query,
        limit=evidence_limit,
        tool_call_arguments={
            "stage": EVIDENCE_RETRIEVAL_STAGE,
            "query": evidence_query,
            "paper_ids": paper_ids,
            "limit": evidence_limit,
            "modalities": contract.required_modalities,
        },
        observation_summary=f"evidence={len(evidence)}",
        observation_payload={
            "stage": EVIDENCE_RETRIEVAL_STAGE,
            "evidence_count": len(evidence),
            "block_types": list(dict.fromkeys(item.block_type for item in evidence[:12])),
        },
    )


def search_agent_candidate_papers(
    *,
    contract: QueryContract,
    paper_query: str,
    paper_limit: int,
    active_targets: list[str],
    excluded_titles: set[str],
    search_papers: CandidatePaperSearchFn,
    paper_lookup: PaperLookupFn,
) -> AgentCandidatePaperSearchResult:
    candidate_papers = search_papers(paper_query, contract, paper_limit)
    if excluded_titles:
        candidate_papers = filter_candidate_papers_by_excluded_titles(
            candidate_papers,
            excluded_titles=excluded_titles,
        )
    effective_contract = contract
    if not candidate_papers and contract.continuation_mode == "followup" and active_targets:
        fallback_contract = contract.model_copy(update={"targets": list(active_targets)})
        candidate_papers = search_papers(
            paper_query_text(fallback_contract),
            fallback_contract,
            paper_limit,
        )
        if excluded_titles:
            candidate_papers = filter_candidate_papers_by_excluded_titles(
                candidate_papers,
                excluded_titles=excluded_titles,
            )
        effective_contract = fallback_contract
    candidate_papers = prefer_selected_clarification_paper(
        candidate_papers,
        contract=effective_contract,
        paper_lookup=paper_lookup,
    )
    return AgentCandidatePaperSearchResult(contract=effective_contract, candidate_papers=candidate_papers)


def run_agent_paper_search(
    *,
    contract: QueryContract,
    plan: ResearchPlan,
    tool_input: dict[str, Any],
    active_targets: list[str],
    excluded_titles: set[str],
    search_papers: CandidatePaperSearchFn,
    paper_lookup: PaperLookupFn,
    screen_papers: ScreenAgentPapersFn,
) -> AgentPaperSearchRun:
    paper_limit = coerce_int(
        tool_input.get("top_k", plan.paper_limit),
        default=plan.paper_limit,
        minimum=1,
        maximum=50,
    )
    # Ensure we don't feed the solver more papers than the plan allows
    paper_limit = min(paper_limit, plan.paper_limit * 2)
    search_plan = plan.model_copy(update={"paper_limit": paper_limit}) if paper_limit != plan.paper_limit else plan
    paper_query = str(tool_input.get("query", "") or "").strip() or paper_query_text(contract)
    paper_result = search_agent_candidate_papers(
        contract=contract,
        paper_query=paper_query,
        paper_limit=search_plan.paper_limit,
        active_targets=active_targets,
        excluded_titles=excluded_titles,
        search_papers=search_papers,
        paper_lookup=paper_lookup,
    )
    screened_papers, precomputed_evidence = screen_papers(
        paper_result.contract,
        search_plan,
        paper_result.candidate_papers,
        excluded_titles,
    )
    return AgentPaperSearchRun(
        contract=paper_result.contract,
        query=paper_query,
        candidate_papers=paper_result.candidate_papers,
        screened_papers=screened_papers,
        precomputed_evidence=precomputed_evidence,
        tool_call_arguments={
            "stage": PAPER_DISCOVERY_STAGE,
            "query": paper_query,
            "limit": search_plan.paper_limit,
            "requested_fields": paper_result.contract.requested_fields,
            "modalities": paper_result.contract.required_modalities,
        },
        observation_summary=f"candidates={len(paper_result.candidate_papers)}, selected={len(screened_papers)}",
        observation_payload={
            "stage": PAPER_DISCOVERY_STAGE,
            "candidate_count": len(paper_result.candidate_papers),
            "selected_count": len(screened_papers),
            "selected_titles": [item.title for item in screened_papers[:5]],
        },
    )


def solve_agent_state_claims(
    *,
    state: dict[str, Any],
    explicit_web: bool,
    solve_claims: AgentClaimSolverFn,
    build_claim: AgentWebClaimBuilderFn,
) -> list[Claim]:
    contract: QueryContract = state["contract"]
    plan: ResearchPlan = state["plan"]
    papers: list[CandidatePaper] = state["screened_papers"]
    evidence: list[EvidenceBlock] = state["evidence"]
    return solve_claims_with_web_research(
        contract=contract,
        web_evidence=state["web_evidence"],
        explicit_web=explicit_web,
        solve_claims=lambda: solve_claims(contract, plan, papers, evidence),
        build_claim=build_claim,
    )


def retry_research_limits(plan: ResearchPlan) -> RetryResearchLimits:
    return RetryResearchLimits(
        paper_limit=max(plan.paper_limit + 4, 10),
        evidence_limit=max(plan.evidence_limit + 12, int(plan.evidence_limit * 1.5)),
    )


def prepare_retry_research_materials(
    *,
    contract: QueryContract,
    plan: ResearchPlan,
    excluded_titles: set[str],
    search_papers: CandidatePaperSearchFn,
    paper_lookup: PaperLookupFn,
    search_concept_evidence: ConceptEvidenceSearchFn,
    search_entity_evidence: EntityEvidenceSearchFn,
    expand_evidence: ExpandEvidenceFn,
    ground_entity_papers: GroundEntityPapersFn,
) -> RetryResearchMaterials:
    limits = retry_research_limits(plan)
    broader_candidates = search_papers(
        paper_query_text(contract),
        contract,
        limits.paper_limit,
    )
    if excluded_titles:
        broader_candidates = filter_candidate_papers_by_excluded_titles(
            broader_candidates,
            excluded_titles=excluded_titles,
        )
    selected_paper_id = selected_clarification_paper_id(contract)
    broader_candidates = prefer_selected_clarification_paper(
        broader_candidates,
        contract=contract,
        paper_lookup=paper_lookup,
    )
    goals = set(research_plan_context_from_contract(contract).goals)
    evidence_query = evidence_query_text(contract)
    if should_use_concept_evidence(contract):
        broader_evidence = search_concept_evidence(
            evidence_query,
            contract,
            [item.paper_id for item in broader_candidates],
            limits.evidence_limit,
        )
    elif goals & {"entity_type", "role_in_context"}:
        broader_evidence = search_entity_evidence(
            evidence_query,
            contract,
            max(
                entity_evidence_limit(contract=contract, plan=plan, excluded_titles=excluded_titles),
                limits.evidence_limit,
            ),
        )
        if excluded_titles:
            broader_evidence = filter_evidence_by_excluded_titles(
                broader_evidence,
                excluded_titles=excluded_titles,
            )
        broader_candidates = ground_entity_papers(
            broader_candidates,
            broader_evidence,
            limits.paper_limit,
        )
    else:
        broader_evidence = expand_evidence(
            [item.paper_id for item in broader_candidates],
            evidence_query,
            contract,
            limits.evidence_limit,
        )
    if excluded_titles:
        broader_evidence = filter_evidence_by_excluded_titles(
            broader_evidence,
            excluded_titles=excluded_titles,
        )
    if selected_paper_id:
        broader_evidence = [item for item in broader_evidence if item.paper_id == selected_paper_id]
    return RetryResearchMaterials(
        candidate_papers=broader_candidates,
        evidence=broader_evidence,
        limits=limits,
        goals=goals,
    )


def run_retry_verification_from_materials(
    *,
    contract: QueryContract,
    plan: ResearchPlan,
    materials: RetryResearchMaterials,
    solve_claims: RetryClaimSolverFn,
    verify_claims: RetryClaimVerifierFn,
    prefer_identity_matching_papers: PaperIdentityMatcherFn,
) -> RetryVerificationResult:
    retry_plan = plan.model_copy(update={"retry_budget": 0})
    retry_claims = solve_claims(retry_plan, materials.candidate_papers, materials.evidence)
    retry_report = verify_claims(retry_plan, retry_claims, materials.candidate_papers, materials.evidence)
    candidate_papers = materials.candidate_papers
    if retry_report.status == "pass" and "figure_conclusion" in materials.goals and contract.targets:
        candidate_papers = prefer_identity_matching_papers(materials.candidate_papers, contract.targets)
    return RetryVerificationResult(
        candidate_papers=candidate_papers,
        evidence=materials.evidence,
        claims=retry_claims,
        verification=retry_report,
        should_replace_materials=retry_report.status == "pass",
        observation_summary=f"retry_status={retry_report.status}",
        observation_payload={
            "candidate_count": len(materials.candidate_papers),
            "evidence_count": len(materials.evidence),
            "claim_count": len(retry_claims),
            "status": retry_report.status,
        },
    )


def refresh_selected_ambiguity_materials(
    *,
    selected: dict[str, Any],
    contract: QueryContract,
    plan: ResearchPlan,
    candidate_papers: list[CandidatePaper],
    existing_evidence: list[EvidenceBlock],
    excluded_titles: set[str],
    paper_lookup: PaperLookupFn,
    search_concept_evidence: ConceptEvidenceSearchFn,
    expand_evidence: ExpandEvidenceFn,
) -> SelectedAmbiguityRefresh | None:
    paper_id = str(selected.get("paper_id", "") or "").strip()
    if not paper_id:
        return None
    selected_papers = [paper for paper in candidate_papers if paper.paper_id == paper_id]
    if not selected_papers:
        paper = paper_lookup(paper_id)
        selected_papers = [paper] if paper is not None else []
    evidence = [item for item in existing_evidence if item.paper_id == paper_id]
    evidence_refreshed = False
    if not evidence:
        evidence_query = evidence_query_text(contract)
        if should_use_concept_evidence(contract):
            evidence = search_concept_evidence(
                evidence_query,
                contract,
                [paper_id],
                plan.evidence_limit,
            )
            if not evidence:
                evidence = expand_evidence(
                    [paper_id],
                    evidence_query,
                    contract,
                    plan.evidence_limit,
                )
        else:
            evidence = expand_evidence(
                [paper_id],
                evidence_query,
                contract,
                plan.evidence_limit,
            )
        if excluded_titles:
            evidence = filter_evidence_by_excluded_titles(evidence, excluded_titles=excluded_titles)
        evidence_refreshed = True
    return SelectedAmbiguityRefresh(
        paper_id=paper_id,
        selected_papers=selected_papers[:1],
        evidence=evidence,
        evidence_refreshed=evidence_refreshed,
    )


def claim_focus_titles(
    *,
    claims: list[Claim],
    papers: list[CandidatePaper],
    paper_title_lookup: PaperTitleLookupFn,
) -> list[str]:
    titles: list[str] = []
    by_id = {item.paper_id: item.title for item in papers}
    for claim in claims:
        for paper_id in claim.paper_ids:
            title = by_id.get(paper_id)
            if not title:
                title = str(paper_title_lookup(paper_id) or "")
            if title and title not in titles:
                titles.append(title)
    return titles[:3] or [item.title for item in papers[:3]]


def finalize_research_verification(state: dict[str, Any]) -> tuple[VerificationReport, dict[str, Any]]:
    verification = state.get("verification")
    if not isinstance(verification, VerificationReport):
        verification = VerificationReport(
            status="clarify",
            missing_fields=["verified_claims"],
            recommended_action="clarify_after_reflection",
        )
        state["verification"] = verification
    confidence = confidence_payload(confidence_from_verification_report(verification))
    state["confidence"] = confidence
    return verification, confidence


def clarify_retry_verification_if_needed(*, contract: QueryContract, verification: VerificationReport) -> VerificationReport:
    goals = set(research_plan_context_from_contract(contract).goals)
    if verification.status == "retry" and contract.targets and (
        goals & {"definition", "mechanism", "examples", "figure_conclusion", "answer", "general_answer"}
    ):
        return VerificationReport(
            status="clarify",
            missing_fields=["relevant_evidence"],
            recommended_action="clarify_target",
        )
    return verification


def verify_grounding_tool_call_arguments(*, plan: ResearchPlan, claims: list[Claim]) -> dict[str, Any]:
    return {
        "stage": GROUNDING_VERIFICATION_STAGE,
        "claim_count": len(claims),
        "required_claims": plan.required_claims,
    }


def verification_observation_payload(verification: VerificationReport) -> dict[str, Any]:
    return {"stage": GROUNDING_VERIFICATION_STAGE, **verification.model_dump()}


def clarification_limit_decision(
    *,
    contract: QueryContract,
    verification: Any,
    next_attempt: int,
    max_attempts: int,
    options: list[dict[str, Any]],
) -> ClarificationLimitDecision | None:
    if not isinstance(verification, VerificationReport) or verification.status != "clarify":
        return None
    if next_attempt < max_attempts:
        return None
    if options:
        selected = options[0]
        forced_contract = contract_from_selected_clarification_option(
            clean_query=contract.clean_query,
            target=contract.targets[0] if contract.targets else str(selected.get("target", "") or ""),
            selected=selected,
            notes_extra=["clarification_limit_reached", "assumed_most_likely_intent"],
        )
        summary = f"selected={selected.get('meaning') or selected.get('title') or 'first_option'}"
    else:
        notes = list(dict.fromkeys([*contract_notes(contract), "clarification_limit_reached", "best_effort_answer"]))
        forced_contract = contract.model_copy(update={"notes": notes})
        summary = verification.recommended_action or "best_effort_answer"
    return ClarificationLimitDecision(
        forced_contract=forced_contract,
        forced_plan={
            "thought": "Clarification limit reached; proceed with the most likely intent and provide a grounded best-effort answer.",
            "actions": ["search_corpus", "compose"],
            "stop_conditions": ["best_effort_answer"],
        },
        summary=summary,
        observation_payload={
            "max_attempts": max_attempts,
            "attempt": next_attempt,
            "assumption": summary,
        },
    )


def promote_best_effort_state_after_clarification_limit(state: dict[str, Any]) -> dict[str, Any]:
    verification = state.get("verification")
    if not (
        isinstance(verification, VerificationReport)
        and verification.status == "clarify"
        and state.get("claims")
    ):
        return state
    promoted = dict(state)
    promoted["verification"] = VerificationReport(
        status="pass",
        recommended_action="best_effort_after_clarification_limit",
        original_status="best_effort",  # P0-8: flag to prevent target_binding pollution
    )
    contract = promoted.get("contract")
    if isinstance(contract, QueryContract):
        promoted["contract"] = contract.model_copy(
            update={
                "notes": list(
                    dict.fromkeys(
                        [
                            *contract_notes(contract),
                            "clarification_limit_reached",
                            "best_effort_after_clarification_limit",
                        ]
                    )
                )
            }
        )
    return promoted


def reflect_agent_state_decision(
    *,
    contract: QueryContract,
    claims: list[Claim],
    focus_titles: list[str],
    verification: VerificationReport,
    excluded_titles: set[str],
    target_binding_exists: bool,
    ambiguity_option_count: AmbiguityOptionCountFn,
) -> dict[str, Any]:
    repeated_excluded = bool(excluded_titles & {normalize_lookup_text(title) for title in focus_titles})
    if repeated_excluded:
        return {
            "decision": "clarify",
            "reason": "The candidate answer still points to a paper the user just rejected.",
            "missing_fields": ["different_interpretation"],
            "recommended_action": "clarify_or_search_alternative",
            "focus_titles": focus_titles,
        }
    if verification.status == "clarify":
        return {
            "decision": "clarify",
            "reason": verification.recommended_action or "human clarification required",
            "missing_fields": verification.missing_fields,
            "recommended_action": verification.recommended_action,
            "focus_titles": focus_titles,
        }
    if contract_needs_evidence_disambiguation(contract):
        if target_binding_exists and not contract_has_note(contract, "exclude_previous_focus"):
            option_count = 1
        else:
            option_count = ambiguity_option_count()
        if option_count > 1 and not claims and not ambiguity_options_from_notes(contract_notes(contract)):
            return {
                "decision": "clarify",
                "reason": "Multiple acronym meanings remain unresolved.",
                "missing_fields": disambiguation_missing_fields(contract),
                "recommended_action": "clarify_ambiguous_entity",
                "focus_titles": focus_titles,
            }
    return {
        "decision": verification.status,
        "reason": "grounding verified" if verification.status == "pass" else verification.recommended_action,
        "focus_titles": focus_titles,
    }


def verification_execution_step(verification: VerificationReport) -> dict[str, str]:
    return {"node": "agent_tool:verify_claim", "summary": verification.status}


def finalize_research_runtime(
    *,
    agent: Any,
    state: dict[str, Any],
    session: SessionContext,
    emit: EmitFn,
    execution_steps: list[dict[str, Any]],
) -> None:
    from app.services.agent.research_reflection_handlers import agent_reflect

    agent_reflect(
        agent=agent,
        state=state,
        session=session,
        emit=emit,
        execution_steps=execution_steps,
    )
    verification, confidence = finalize_research_verification(state)
    emit("verification", verification.model_dump())
    emit("confidence", confidence)
    execution_steps.append(verification_execution_step(verification))


def configured_max_steps(agent_settings: Any, *, fallback: int) -> int:
    value = getattr(agent_settings, "max_agent_steps", fallback)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(1, parsed)


def dequeue_action(*, queue: list[str], executed: set[str]) -> str | None:
    while queue:
        action = queue.pop(0)
        if action not in executed:
            return action
    return None


def planner_next_action(
    *,
    agent: Any,
    contract: QueryContract,
    session: SessionContext,
    state: dict[str, Any],
    executed_actions: list[str],
    allowed_tools: set[str],
) -> str | None:
    planner = getattr(agent, "planner", None)
    choose_next = getattr(planner, "choose_next_action", None)
    if not callable(choose_next):
        return None
    return choose_next(
        contract=state.get("contract", contract),
        session=session,
        state=state,
        executed_actions=executed_actions,
        allowed_tools=allowed_tools,
    )


def planner_next_actions(
    *,
    agent: Any,
    contract: QueryContract,
    session: SessionContext,
    state: dict[str, Any],
    executed_actions: list[str],
    allowed_tools: set[str],
) -> list[tuple[str, dict[str, Any]]]:
    """Return (action_name, arguments) pairs from the LLM planner."""
    planner = getattr(agent, "planner", None)
    choose_next = getattr(planner, "choose_next_actions", None)
    if not callable(choose_next):
        action = planner_next_action(
            agent=agent,
            contract=contract,
            session=session,
            state=state,
            executed_actions=executed_actions,
            allowed_tools=allowed_tools,
        )
        return [(action, {})] if action else []
    return choose_next(
        contract=state.get("contract", contract),
        session=session,
        state=state,
        executed_actions=executed_actions,
        allowed_tools=allowed_tools,
    )


def _run_tool_batch(
    *,
    executor: Any,
    actions: list[str],
    tool_inputs: dict[str, dict[str, Any]],
    state: dict[str, Any],
    max_workers: int,
    emit: EmitFn,
) -> bool:
    """Run a batch of tools, using parallel execution when available."""
    parallel = getattr(executor, "run_parallel", None)
    if callable(parallel) and len(actions) > 1:
        try:
            return parallel(
                actions,
                arguments=tool_inputs,
                argument_provider=lambda name: tool_input_from_state(state, name),
                max_workers=max_workers,
                emit=emit,
            )
        except TypeError:
            # Graceful fallback for mock executors
            pass
    # Sequential fallback
    stopped = False
    for action in actions:
        try:
            ok = executor.run(
                action,
                arguments=tool_inputs.get(action),
                argument_provider=lambda name: tool_input_from_state(state, name),
                emit=emit,
            )
        except TypeError:
            # Fallback for mock executors that don't accept emit
            ok = executor.run(
                action,
                arguments=tool_inputs.get(action),
                argument_provider=lambda name: tool_input_from_state(state, name),
            )
        if ok:
            stopped = True
    return stopped


def _capture_tool_result_previews(*, state: dict[str, Any], actions: list[str]) -> None:
    """Append brief human-readable result summaries for the LLM's next decision."""
    previews = state.setdefault("_tool_result_previews", [])
    for action in actions:
        summary = _tool_result_snapshot(state=state, action=action)
        if summary:
            previews.append(summary)


def _tool_result_snapshot(*, state: dict[str, Any], action: str) -> str:
    """Return a one-line snapshot of what the tool produced."""
    if action in {"search_corpus", "bm25_search", "vector_search", "hybrid_search"}:
        evidence = list(state.get("evidence", []) or [])
        if evidence:
            paper_ids = list(dict.fromkeys(item.paper_id for item in evidence if getattr(item, "paper_id", None)))
            return f"{action}: {len(evidence)} evidence blocks across {len(paper_ids)} papers"
        papers = list(state.get("screened_papers", []) or [])
        return f"{action}: {len(papers)} candidate papers screened"
    if action == "read_memory":
        return "read_memory: session context loaded"
    if action == "grep_corpus":
        evidence = list(state.get("evidence", []) or [])
        return f"grep_corpus: {len(evidence)} exact matches"
    if action == "read_pdf_page":
        evidence = list(state.get("evidence", []) or [])
        return f"read_pdf_page: {len(evidence)} blocks from target pages"
    if action == "web_search":
        web = list(state.get("web_evidence", []) or [])
        return f"web_search: {len(web)} external results"
    if action == "fetch_url":
        return "fetch_url: page content fetched"
    if action == "query_rewrite":
        return "query_rewrite: alternate queries generated"
    if action == "summarize":
        return "summarize: text compressed"
    if action == "verify_claim":
        checks = list(state.get("claim_checks", []) or [])
        return f"verify_claim: {len(checks)} checks recorded"
    if action == "todo_write":
        return "todo_write: task list updated"
    if action == "remember":
        return "remember: learning persisted"
    if action == "propose_tool":
        return "propose_tool: proposal recorded for review"
    if action == "Task":
        tasks = list(state.get("task_results", []) or [])
        return f"Task: subtask completed ({len(tasks)} total)"
    if action == "ask_human":
        return "ask_human: clarification requested"
    if action == "compose":
        claims = list(state.get("claims", []) or [])
        verification = state.get("verification")
        has_verify = getattr(verification, "status", None) if verification is not None else None
        return f"compose: {len(claims)} claims, verification={has_verify or 'pending'}"
    return ""


def execute_tool_loop(
    *,
    agent: Any,
    contract: QueryContract,
    session: SessionContext,
    state: dict[str, Any],
    executor: Any,
    planned_actions: list[str],
    allowed_tools: set[str],
    emit: EmitFn,
    fallback_next: FallbackNextFn,
    stop_condition: StopConditionFn,
    max_steps: int = 8,
) -> None:
    # When LLM-driven mode is enabled, the LLM autonomously decides what tools
    # to call based on full message history.  Fall back to Python-driven loop
    # if the LLM-driven loop is unavailable or disabled.
    if bool(getattr(getattr(agent, "agent_settings", None), "llm_driven_loop_enabled", False)):
        run_llm_driven_tool_loop(
            agent=agent,
            contract=contract,
            session=session,
            state=state,
            executor=executor,
            allowed_tools=allowed_tools,
            emit=emit,
            stop_condition=stop_condition,
            max_steps=max_steps,
        )
        return

    queue = [action for action in planned_actions if action in allowed_tools]
    executed_order: list[str] = []
    max_step_count = configured_max_steps(
        getattr(agent, "agent_settings", None),
        fallback=max_steps,
    )
    max_calls_per_tool = max(1, int(getattr(getattr(agent, "agent_settings", None), "max_calls_per_tool", 3)))
    max_parallel = max(1, int(getattr(getattr(agent, "agent_settings", None), "max_parallel_tools", 4)))
    for index in range(1, max_step_count + 1):
        # Collect (action_name, arguments) for this step.
        action_args: list[tuple[str, dict[str, Any]]] = []
        action = dequeue_action(queue=queue, executed=executor.executed)
        if action is not None:
            action_args = [(action, tool_input_from_state(state, action))]
        else:
            action_args = planner_next_actions(
                agent=agent,
                contract=contract,
                session=session,
                state=state,
                executed_actions=executed_order,
                allowed_tools=allowed_tools,
            )
        if not action_args:
            action = fallback_next(executor.executed)
            if action:
                action_args = [(action, tool_input_from_state(state, action))]
        if not action_args:
            break
        # Filter by call count limit and allowed tools
        action_args = [
            (name, args) for name, args in action_args
            if name in allowed_tools
            and sum(1 for entry in getattr(executor, "execution_log", []) if entry.get("tool") == name) < max_calls_per_tool
        ]
        if not action_args:
            break
        # Limit parallel width
        action_args = action_args[:max_parallel]
        # Build tool_inputs dict: merge LLM-provided args with state fallback
        tool_inputs: dict[str, dict[str, Any]] = {}
        for name, args in action_args:
            # LLM args take priority; state provides fallback for missing keys
            merged = dict(tool_input_from_state(state, name))
            merged.update(args)
            tool_inputs[name] = merged
        state["current_tool_input"] = tool_inputs.get(action_args[0][0]) if action_args else {}
        state["current_tool_inputs"] = tool_inputs
        actions = [name for name, _ in action_args]
        state["_execution_log"] = getattr(executor, "execution_log", [])
        for action in actions:
            emit_agent_step(
                emit=emit,
                index=index,
                action=action,
                contract=state.get("contract", contract),
                arguments=tool_inputs.get(action),
            )
        should_stop = _run_tool_batch(
            executor=executor,
            actions=actions,
            tool_inputs=tool_inputs,
            state=state,
            max_workers=max_parallel,
            emit=emit,
        )
        # Capture result previews so the LLM can see what each tool returned.
        _capture_tool_result_previews(state=state, actions=actions)
        executed_order.extend(actions)
        if should_stop or stop_condition(executor.executed):
            break


def run_llm_driven_tool_loop(
    *,
    agent: Any,
    contract: QueryContract,
    session: SessionContext,
    state: dict[str, Any],
    executor: Any,
    allowed_tools: set[str],
    emit: EmitFn,
    stop_condition: StopConditionFn,
    max_steps: int = 8,
) -> None:
    """LLM-driven tool-use loop — the LLM sees full message history including
    every tool_use and tool_result, and autonomously decides what to call next.

    This replaces the Python-driven sequential loop when
    agent_settings.llm_driven_loop_enabled is True.
    """
    import json as _json

    planner_fn = getattr(getattr(agent, "clients", None), "invoke_tool_plan_messages", None)
    if not callable(planner_fn):
        # Fall back: the caller will use execute_tool_loop
        return

    from app.services.agent.tools import agent_tool_manifest_for_names
    from app.services.agent.planner_helpers import planner_intent_payload, planner_context_json
    from app.services.contracts.context import contract_notes

    available_tools = agent_tool_manifest_for_names(allowed_tools)
    if not available_tools:
        return

    max_step_count = configured_max_steps(
        getattr(agent, "agent_settings", None),
        fallback=max_steps,
    )
    max_calls_per_tool = max(1, int(getattr(getattr(agent, "agent_settings", None), "max_calls_per_tool", 3)))

    # Build the initial message list: system context + user query
    system_prompt = (
        "你是论文研究助手，通过与工具交互来回答用户问题。\n"
        "每一步你可以调用零个或多个工具，工具会返回观察结果。\n"
        "当你有足够信息回答时，不要再调用工具，直接给出最终答案。\n"
        "如果证据不足或用户意图不明，调用 ask_human 工具。\n"
        "用中文回答，引用来源时使用 [doc:paper_id#page] 格式。"
    )
    context = {
        "intent": planner_intent_payload(contract),
        "targets": contract.targets,
        "notes": contract_notes(contract),
    }
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"{contract.clean_query}\n\n[上下文]\n{planner_context_json(context)}"},
    ]

    executed_order: list[str] = []
    for _step in range(1, max_step_count + 1):
        # Ask LLM: what tools to call next?
        payload = planner_fn(
            system_prompt=system_prompt,
            messages=messages,
            tools=available_tools,
            fallback={},
        )
        tool_calls = payload.get("tool_call_args", [])
        if not isinstance(tool_calls, list) or not tool_calls:
            # LLM decided to stop — no more tool calls
            break

        # Filter to allowed & within per-tool call limits
        filtered_calls: list[dict[str, Any]] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name", "") or "").strip()
            if name not in allowed_tools:
                continue
            call_count = sum(1 for entry in getattr(executor, "execution_log", []) if entry.get("tool") == name)
            if call_count >= max_calls_per_tool:
                continue
            filtered_calls.append(call)

        if not filtered_calls:
            break

        # Execute each tool call and capture results
        tool_results: list[dict[str, Any]] = []
        for call in filtered_calls:
            name = str(call.get("name", "") or "").strip()
            args = call.get("args", {})
            if isinstance(args, str):
                try:
                    args = _json.loads(args)
                except (_json.JSONDecodeError, TypeError):
                    args = {}

            tool_input = dict(args)
            emit_agent_step(
                emit=emit,
                index=_step,
                action=name,
                contract=state.get("contract", contract),
                arguments=tool_input,
            )
            executor.run(
                name,
                arguments=tool_input,
                argument_provider=lambda n: tool_input_from_state(state, n),
                emit=emit,
            )
            executed_order.append(name)

            # Build a human-readable result message for the LLM
            result_snapshot = _tool_result_snapshot(state=state, action=name)
            result_text = result_snapshot or f"{name}: executed"
            tool_results.append({
                "role": "human",
                "content": f"[tool_result: {name}]\n{result_text}",
            })

        _capture_tool_result_previews(state=state, actions=[c.get("name", "") for c in filtered_calls if isinstance(c, dict)])

        # Append all tool results to the message history
        messages.extend(tool_results)

        # Check stop condition
        if stop_condition(executor.executed):
            break

    # P0-7: Do NOT generate final answer via raw invoke_text — it bypasses
    # claim verifier + citation audit + answer_composer. The caller must
    # route through the normal SolverPipelineMixin.solve_claims +
    # ClaimVerifierMixin.verify_claims + answer_composer path.
    # Mark state so the caller knows the tool loop collected evidence
    # but still needs to run verification + composition.
    state["_llm_driven_loop_collected_evidence"] = True


def contract_needs_human_clarification(contract: QueryContract, agent_settings: Any) -> bool:
    if research_contract_should_try_tools_before_human(contract):
        return False
    return should_ask_human(confidence_from_contract(contract), agent_settings)


def _walk_fallback_chain(*, chain: list[str], executed: set[str]) -> str | None:
    """Return the first action in *chain* that hasn't been executed yet."""
    for action in chain:
        if action not in executed:
            return action
    return None


# Ordered fallback chains: walked top-to-bottom when the LLM planner fails
# to suggest a next action.  Append to these lists to add new fallback steps.
_CONVERSATION_FALLBACK_CHAIN: list[str] = [
    "compose",
]

_RESEARCH_FALLBACK_CHAIN: list[str] = [
    "search_corpus",
    "compose",
]


def next_conversation_action(
    *,
    contract: QueryContract,
    state: dict[str, Any],
    executed: set[str],
    agent_settings: Any,
) -> str | None:
    # Early bail: clarify with user when confidence is low
    if contract_needs_human_clarification(contract, agent_settings) and "ask_human" not in executed:
        return "ask_human"

    # Citation-ranking turns need web lookup for counts
    notes = set(contract_notes(contract))
    fields = {str(item) for item in contract.requested_fields}
    is_citation_turn = "citation_count_ranking" in fields or "citation_count_requires_web" in notes
    if is_citation_turn:
        if "web_search" not in executed:
            return "web_search"

    # Memory / followup turns should read context first
    is_memory_turn = (
        "intent_kind=memory_op" in notes
        or bool(fields & {"comparison", "synthesis", "previous_tool_basis"})
        or contract.continuation_mode == "followup"
    )
    if (is_memory_turn or is_citation_turn) and "read_memory" not in executed:
        return "read_memory"

    # Library status works better with metadata first
    if contract.relation == "library_status" and "query_library_metadata" not in executed:
        return "query_library_metadata"

    return _walk_fallback_chain(chain=_CONVERSATION_FALLBACK_CHAIN, executed=executed)


def next_research_action(
    *,
    contract: QueryContract,
    state: dict[str, Any],
    executed: set[str],
    web_enabled: bool,
    agent_settings: Any,
) -> str | None:
    # Early bail: clarify with user when confidence is low
    if contract_needs_human_clarification(contract, agent_settings):
        return "ask_human"

    # Followup / context-switch turns should read memory first
    if (
        contract.continuation_mode == "followup"
        or contract_has_note(contract, "memory_resolved_research")
        or contract_has_note(contract, "resolved_from_conversation_memory")
        or contract_has_note(contract, "exclude_previous_focus")
    ) and "read_memory" not in executed:
        return "read_memory"

    # Web search gets a dedicated slot before composing
    if web_enabled and "web_search" not in executed:
        return "web_search"

    # If individual search tools ran but the comprehensive hybrid search hasn't,
    # insert search_corpus before compose for better evidence quality.
    _individual_searchers = {"bm25_search", "vector_search", "grep_corpus"}
    if (executed & _individual_searchers) and "search_corpus" not in executed:
        return "search_corpus"

    return _walk_fallback_chain(chain=_RESEARCH_FALLBACK_CHAIN, executed=executed)
