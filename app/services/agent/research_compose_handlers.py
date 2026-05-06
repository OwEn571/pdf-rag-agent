from __future__ import annotations

from typing import Any, Callable

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract, ResearchPlan, SessionContext
from app.services.agent.disambiguation_runtime import (
    disambiguation_options_from_evidence,
    judge_disambiguation_options,
    refresh_state_for_selected_ambiguity,
)
from app.services.agent.runtime_helpers import CLAIM_COMPOSITION_STAGE, solve_agent_state_claims
from app.services.agent.tool_events import (
    emit_agent_tool_call as emit_agent_tool_call_event,
    record_agent_observation as record_agent_observation_event,
)
from app.services.clarification.intents import resolve_disambiguation_judge_decision
from app.services.retrieval.web_evidence import build_web_research_claim


EmitFn = Callable[[str, dict[str, Any]], None]


def agent_solve_claims(
    *,
    agent: Any,
    state: dict[str, Any],
    session: SessionContext,
    explicit_web_search: bool,
    emit: EmitFn,
    execution_steps: list[dict[str, Any]],
) -> None:
    contract: QueryContract = state["contract"]
    plan: ResearchPlan = state["plan"]
    screened_papers: list[CandidatePaper] = state["screened_papers"]
    evidence: list[EvidenceBlock] = state["evidence"]
    emit_agent_tool_call_event(
        emit=emit,
        tool="compose",
        arguments={
            "stage": CLAIM_COMPOSITION_STAGE,
            "solver_sequence": plan.solver_sequence,
            "evidence_count": len(evidence),
        },
    )

    def solve_claims_with_shadow_trace(
        item_contract: QueryContract,
        item_plan: ResearchPlan,
        item_papers: list[CandidatePaper],
        item_evidence: list[EvidenceBlock],
    ):
        claims = agent._run_solvers(
            contract=item_contract,
            plan=item_plan,
            papers=item_papers,
            evidence=item_evidence,
            session=session,
        )
        shadow = dict(getattr(agent, "_last_generic_claim_solver_shadow", {}) or {})
        selected = str(shadow.get("selected", "") or "")
        has_comparison = isinstance(shadow.get("schema"), dict) or isinstance(shadow.get("deterministic"), dict)
        if selected:
            state["claim_solver_selection"] = selected
            if has_comparison:
                state["claim_solver_shadow"] = shadow
                emit("solver_shadow", shadow)
                schema_summary = shadow.get("schema") if isinstance(shadow.get("schema"), dict) else {}
                deterministic_summary = shadow.get("deterministic") if isinstance(shadow.get("deterministic"), dict) else {}
                record_agent_observation_event(
                    emit=emit,
                    execution_steps=execution_steps,
                    tool="compose",
                    summary=(
                        "generic_claim_solver_shadow="
                        f"selected:{selected};"
                        f"schema:{schema_summary.get('count', 0)};"
                        f"deterministic:{deterministic_summary.get('count', 0)}"
                    ),
                    payload={"stage": "generic_claim_solver_shadow", **shadow},
                )
            else:
                emit("solver_selection", {"selected": selected})
                record_agent_observation_event(
                    emit=emit,
                    execution_steps=execution_steps,
                    tool="compose",
                    summary=f"claim_solver={selected}",
                    payload={"stage": "claim_solver_selection", "selected": selected},
                )
            setattr(agent, "_last_generic_claim_solver_shadow", {})
        return claims

    ambiguity_options = disambiguation_options_from_evidence(
        contract=contract,
        session=session,
        papers=screened_papers,
        evidence=evidence,
        paper_lookup=agent._candidate_from_paper_id,
        search_concept_evidence=lambda query, contract, limit: agent.retriever.search_concept_evidence(
            query=query,
            contract=contract,
            limit=limit,
        ),
        evidence_limit_default=int(agent.settings.evidence_limit_default),
        paper_documents=lambda: agent.retriever.paper_documents(),
        block_documents_for_paper=lambda paper_id, block_limit: agent.retriever.block_documents_for_paper(
            paper_id,
            limit=block_limit,
        ),
    )
    if ambiguity_options:
        judge_decision = judge_disambiguation_options(
            contract=contract,
            options=ambiguity_options,
            clients=agent.clients,
            paper_lookup=agent._candidate_from_paper_id,
        )
        resolution = resolve_disambiguation_judge_decision(
            contract=contract,
            options=ambiguity_options,
            judge_decision=judge_decision,
            auto_resolve_threshold=agent.agent_settings.disambiguation_auto_resolve_threshold,
            recommend_threshold=agent.agent_settings.disambiguation_recommend_threshold,
        )
        record_agent_observation_event(
            emit=emit,
            execution_steps=execution_steps,
            tool=resolution.observation_tool,
            summary=resolution.observation_summary,
            payload=resolution.observation_payload,
        )
        if resolution.auto_resolve and resolution.selected_option is not None:
            contract = resolution.contract
            state["contract"] = contract
            refresh_state_for_selected_ambiguity(
                state=state,
                selected=resolution.selected_option,
                emit=emit,
                execution_steps=execution_steps,
                paper_lookup=agent._candidate_from_paper_id,
                search_concept_evidence=lambda query, search_contract, paper_ids, limit: agent.retriever.search_concept_evidence(
                    query=query,
                    contract=search_contract,
                    paper_ids=paper_ids,
                    limit=limit,
                ),
                expand_evidence=lambda paper_ids, query, search_contract, limit: agent.retriever.expand_evidence(
                    paper_ids=paper_ids,
                    query=query,
                    contract=search_contract,
                    limit=limit,
                ),
            )
            claims = solve_agent_state_claims(
                state=state,
                explicit_web=explicit_web_search,
                solve_claims=solve_claims_with_shadow_trace,
                build_claim=lambda item_contract, item_evidence: build_web_research_claim(
                    contract=item_contract,
                    web_evidence=item_evidence,
                ),
            )
            state["claims"] = claims
        else:
            state["contract"] = resolution.contract
            state["claims"] = []
            state["verification"] = resolution.verification
    else:
        claims = solve_agent_state_claims(
            state=state,
            explicit_web=explicit_web_search,
            solve_claims=solve_claims_with_shadow_trace,
            build_claim=lambda item_contract, item_evidence: build_web_research_claim(
                contract=item_contract,
                web_evidence=item_evidence,
            ),
        )
        state["claims"] = claims
    claims = state["claims"]
    emit("claims", {"count": len(claims), "items": [item.model_dump() for item in claims]})
    record_agent_observation_event(
        emit=emit,
        execution_steps=execution_steps,
        tool="compose",
        summary=f"claims={len(claims)}",
        payload={
            "stage": CLAIM_COMPOSITION_STAGE,
            "claim_count": len(claims),
            "claim_types": [item.claim_type for item in claims],
        },
    )
