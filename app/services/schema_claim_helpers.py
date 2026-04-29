from __future__ import annotations

from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan
from app.services.confidence import coerce_confidence_value
from app.services.solver_goal_helpers import claim_goals


SCHEMA_CLAIM_SOLVER_BLOCKED_GOALS = {
    "formula",
    "origin",
    "paper_title",
    "year",
    "variable_explanation",
    "followup_papers",
    "candidate_relationship",
    "strict_followup",
    "best_topology",
    "langgraph_recommendation",
}


def should_use_schema_claim_solver(*, contract: QueryContract, plan: ResearchPlan) -> bool:
    goals = claim_goals(contract=contract, plan=plan)
    return not bool(goals & SCHEMA_CLAIM_SOLVER_BLOCKED_GOALS)


def claims_from_schema_payload(
    payload: Any,
    *,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    max_claims: int = 12,
) -> list[Claim]:
    if not isinstance(payload, dict):
        return []
    raw_claims = payload.get("claims", [])
    if not isinstance(raw_claims, list):
        return []
    evidence_ids = {item.doc_id for item in evidence}
    paper_ids = {item.paper_id for item in papers} | {item.paper_id for item in evidence}
    claims: list[Claim] = []
    for item in raw_claims[:max_claims]:
        if not isinstance(item, dict):
            continue
        selected_evidence_ids = _selected_ids(item.get("evidence_ids", []), allowed=evidence_ids)
        if not selected_evidence_ids:
            continue
        selected_paper_ids = _selected_ids(item.get("paper_ids", []), allowed=paper_ids) or list(
            dict.fromkeys(block.paper_id for block in evidence if block.doc_id in selected_evidence_ids)
        )
        structured_data = item.get("structured_data", {})
        if not isinstance(structured_data, dict):
            structured_data = {}
        structured_data = dict(structured_data)
        structured_data["source"] = "schema_claim_solver"
        claims.append(
            Claim(
                claim_type=str(item.get("claim_type", "") or "general_answer"),
                entity=str(item.get("entity", "") or (contract.targets[0] if contract.targets else "")),
                value=str(item.get("value", "") or ""),
                structured_data=structured_data,
                evidence_ids=selected_evidence_ids,
                paper_ids=list(dict.fromkeys(selected_paper_ids)),
                confidence=coerce_confidence_value(item.get("confidence", 0.72), default=0.82),
                required=bool(item.get("required", True)),
            )
        )
    return claims


def _selected_ids(value: Any, *, allowed: set[str]) -> list[str]:
    raw_ids = [value] if isinstance(value, str) else value
    if not isinstance(raw_ids, list):
        return []
    return [str(item).strip() for item in raw_ids if str(item).strip() in allowed]
