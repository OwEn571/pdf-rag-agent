from __future__ import annotations

from typing import Any, Callable

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan, SessionContext, VerificationReport
from app.services.agent.runtime_helpers import (
    RESEARCH_RETRY_STAGE,
    clarify_retry_verification_if_needed,
    prepare_retry_research_materials,
    retry_research_limits,
    run_retry_verification_from_materials,
    verification_observation_payload,
    verify_grounding_tool_call_arguments,
)
from app.services.agent.tool_events import (
    emit_agent_tool_call as emit_agent_tool_call_event,
    record_agent_observation as record_agent_observation_event,
)


EmitFn = Callable[[str, dict[str, Any]], None]


def agent_retry_after_verification(
    *,
    agent: Any,
    state: dict[str, Any],
    session: SessionContext,
    emit: EmitFn,
    execution_steps: list[dict[str, Any]],
) -> None:
    contract: QueryContract = state["contract"]
    plan: ResearchPlan = state["plan"]
    excluded_titles: set[str] = state["excluded_titles"]
    retry_limits = retry_research_limits(plan)
    emit_agent_tool_call_event(
        emit=emit,
        tool="search_corpus",
        arguments={
            "stage": RESEARCH_RETRY_STAGE,
            "reason": state["verification"].recommended_action if state.get("verification") else "",
            "paper_limit": retry_limits.paper_limit,
            "evidence_limit": retry_limits.evidence_limit,
        },
    )
    retry_materials = prepare_retry_research_materials(
        contract=contract,
        plan=plan,
        excluded_titles=excluded_titles,
        search_papers=lambda query, search_contract, limit: agent.retriever.search_papers(
            query=query,
            contract=search_contract,
            limit=limit,
        ),
        paper_lookup=agent._candidate_from_paper_id,
        search_concept_evidence=lambda query, search_contract, paper_ids, limit: agent.retriever.search_concept_evidence(
            query=query,
            contract=search_contract,
            paper_ids=paper_ids,
            limit=limit,
        ),
        search_entity_evidence=lambda query, search_contract, limit: agent.retriever.search_entity_evidence(
            query=query,
            contract=search_contract,
            limit=limit,
        ),
        expand_evidence=lambda paper_ids, query, search_contract, limit: agent.retriever.expand_evidence(
            paper_ids=paper_ids,
            query=query,
            contract=search_contract,
            limit=limit,
        ),
        ground_entity_papers=lambda candidates, evidence, limit: agent._ground_entity_papers(
            candidates=candidates,
            evidence=evidence,
            limit=limit,
        ),
    )
    retry_result = run_retry_verification_from_materials(
        contract=contract,
        plan=plan,
        materials=retry_materials,
        solve_claims=lambda retry_plan, retry_papers, retry_evidence: agent._run_solvers(
            contract=contract,
            plan=retry_plan,
            papers=retry_papers,
            evidence=retry_evidence,
            session=session,
        ),
        verify_claims=lambda retry_plan, retry_claims, retry_papers, retry_evidence: agent._verify_claims(
            contract=contract,
            plan=retry_plan,
            claims=retry_claims,
            papers=retry_papers,
            evidence=retry_evidence,
        ),
        prefer_identity_matching_papers=lambda candidates, targets: [
            item for item in candidates if agent._paper_identity_matches_targets(paper=item, targets=targets)
        ],
    )
    if retry_result.should_replace_materials:
        state["screened_papers"] = retry_result.candidate_papers
        state["evidence"] = retry_result.evidence
        state["claims"] = retry_result.claims
    state["verification"] = retry_result.verification
    record_agent_observation_event(
        emit=emit,
        execution_steps=execution_steps,
        tool="search_corpus",
        summary=retry_result.observation_summary,
        payload={"stage": RESEARCH_RETRY_STAGE, **retry_result.observation_payload},
    )


def agent_verify_grounding(
    *,
    agent: Any,
    state: dict[str, Any],
    session: SessionContext,
    emit: EmitFn,
    execution_steps: list[dict[str, Any]],
) -> None:
    verification = state.get("verification")
    if isinstance(verification, VerificationReport) and verification.status == "clarify":
        record_agent_observation_event(
            emit=emit,
            execution_steps=execution_steps,
            tool="verify_claim",
            summary=verification.status,
            payload=verification_observation_payload(verification),
        )
        return
    contract: QueryContract = state["contract"]
    plan: ResearchPlan = state["plan"]
    claims: list[Claim] = state["claims"]
    screened_papers: list[CandidatePaper] = state["screened_papers"]
    evidence: list[EvidenceBlock] = state["evidence"]
    emit_agent_tool_call_event(
        emit=emit,
        tool="verify_claim",
        arguments=verify_grounding_tool_call_arguments(plan=plan, claims=claims),
    )
    verification = agent._verify_claims(
        contract=contract,
        plan=plan,
        claims=claims,
        papers=screened_papers,
        evidence=evidence,
    )
    state["verification"] = verification
    if verification.status == "retry" and plan.retry_budget > 0:
        agent_retry_after_verification(
            agent=agent,
            state=state,
            session=session,
            emit=emit,
            execution_steps=execution_steps,
        )
        verification = state["verification"]
    verification = clarify_retry_verification_if_needed(contract=contract, verification=verification)
    state["verification"] = verification
    record_agent_observation_event(
        emit=emit,
        execution_steps=execution_steps,
        tool="verify_claim",
        summary=verification.status,
        payload=verification_observation_payload(verification),
    )
