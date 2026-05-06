from __future__ import annotations

from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan
from app.services.planning.schema_claims import (
    claims_from_schema_payload,
    schema_claim_human_prompt,
    schema_claim_system_prompt,
)


def solve_claims_with_generic_schema(
    *,
    clients: Any,
    contract: QueryContract,
    plan: ResearchPlan,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    conversation_context: dict[str, Any],
) -> list[Claim]:
    if (
        clients is None
        or not hasattr(clients, "invoke_json")
        or getattr(clients, "chat", None) is None
        or not evidence
    ):
        return []
    payload = clients.invoke_json(
        system_prompt=schema_claim_system_prompt(),
        human_prompt=schema_claim_human_prompt(
            contract=contract,
            plan=plan,
            papers=papers,
            evidence=evidence,
            conversation_context=conversation_context,
        ),
        fallback={},
    )
    return claims_from_schema_payload(payload, contract=contract, papers=papers, evidence=evidence)


def claim_solver_shadow_summary(
    *,
    selected: str,
    schema_claims: list[Claim],
    deterministic_claims: list[Claim],
) -> dict[str, Any]:
    return {
        "mode": "generic_claim_solver_shadow",
        "selected": selected,
        "schema": claim_summary(schema_claims),
        "deterministic": claim_summary(deterministic_claims),
    }


def claim_summary(claims: list[Claim]) -> dict[str, Any]:
    sources: dict[str, int] = {}
    for claim in claims:
        source = str(dict(claim.structured_data or {}).get("source") or "deterministic_solver")
        sources[source] = sources.get(source, 0) + 1
    return {
        "count": len(claims),
        "types": [claim.claim_type for claim in claims[:12]],
        "paper_ids": list(dict.fromkeys(paper_id for claim in claims for paper_id in claim.paper_ids))[:12],
        "evidence_ids": list(dict.fromkeys(evidence_id for claim in claims for evidence_id in claim.evidence_ids))[:16],
        "sources": sources,
    }
