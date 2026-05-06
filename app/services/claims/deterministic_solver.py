from __future__ import annotations

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan, SessionContext
from app.services.claims.deterministic_runner import DeterministicSolverHandlers, run_deterministic_solver_stage
from app.services.planning.solver_dispatch import SolverDispatchContext, deterministic_solver_stages_for_context
from app.services.planning.solver_goals import (
    append_unique_claims,
    claim_goal_context_from_contract_plan,
    claim_goals_for_context,
)


def solve_claims_with_deterministic_fallback(
    *,
    handlers: DeterministicSolverHandlers,
    contract: QueryContract,
    plan: ResearchPlan,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    session: SessionContext,
) -> list[Claim]:
    goal_context = claim_goal_context_from_contract_plan(contract=contract, plan=plan)
    goals = claim_goals_for_context(goal_context)
    claims: list[Claim] = []

    dispatch_context = SolverDispatchContext(
        goals=frozenset(goals),
        required_modalities=tuple(contract.required_modalities or []),
    )
    for stage in deterministic_solver_stages_for_context(dispatch_context):
        append_unique_claims(
            claims,
            run_deterministic_solver_stage(
                handlers=handlers,
                stage=stage,
                contract=contract,
                papers=papers,
                evidence=evidence,
                session=session,
                claims=claims,
            ),
        )
    if not claims and handlers.default_text is not None:
        append_unique_claims(
            claims,
            handlers.default_text(contract=contract, papers=papers, evidence=evidence, session=session),
        )
    return claims
