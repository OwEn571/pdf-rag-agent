from __future__ import annotations

from collections.abc import Callable

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan, VerificationReport

VerifierCheck = Callable[..., VerificationReport | None]


def claim_verifier_checks_for_goals(
    goals: set[str],
    *,
    checks: dict[str, VerifierCheck],
) -> list[VerifierCheck]:
    selected: list[VerifierCheck] = []
    if goals & {"paper_title", "year", "origin"}:
        selected.append(checks["origin"])
    entity_like = bool(goals & {"entity_type", "role_in_context"})
    if entity_like:
        selected.append(checks["entity"])
    if goals & {"followup_papers", "candidate_relationship", "strict_followup"}:
        selected.append(checks["followup"])
    if "recommended_papers" in goals:
        selected.append(checks["paper_recommendation"])
    if goals & {"best_topology", "langgraph_recommendation"}:
        selected.append(checks["topology"])
    if "figure_conclusion" in goals:
        selected.append(checks["figure"])
    if "metric_value" in goals:
        selected.append(checks["metric"])
    if "formula" in goals:
        selected.append(checks["formula"])
    if "reward_model_requirement" in goals:
        selected.append(checks["general"])
    elif not entity_like and goals & {"definition", "examples"}:
        selected.append(checks["concept"])
    if not selected:
        selected.append(checks["general"])
    return selected


def verify_claims_with_generic_fallback(
    *,
    contract: QueryContract,
    plan: ResearchPlan,
    claims: list[Claim],
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    goals: set[str],
    checks: dict[str, VerifierCheck],
) -> VerificationReport | None:
    for check in claim_verifier_checks_for_goals(goals, checks=checks):
        report = check(contract=contract, plan=plan, claims=claims, papers=papers, evidence=evidence)
        if report is not None:
            return report
    return None
