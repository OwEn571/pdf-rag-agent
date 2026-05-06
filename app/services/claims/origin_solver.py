from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract
from app.services.claims import origin_selection as origin_helpers
from app.services.answers.evidence_presentation import evidence_ids_for_paper
from app.services.planning.query_shaping import matches_target


def solve_origin_lookup_claims(
    *,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    paper_documents: Iterable[Any],
    candidate_from_paper_id: Callable[[str], CandidatePaper | None],
    paper_identity_matches_targets: Callable[[CandidatePaper, list[str]], bool],
) -> list[Claim]:
    selected = origin_helpers.select_origin_paper(
        contract=contract,
        papers=papers,
        evidence=evidence,
        paper_documents=paper_documents,
        candidate_from_paper_id=candidate_from_paper_id,
        paper_identity_matches_targets=paper_identity_matches_targets,
        target_matcher=matches_target,
    )
    if selected is None:
        return []
    supporting_ids = evidence_ids_for_paper(evidence, selected.paper_id, limit=2)
    return [origin_helpers.origin_lookup_claim(contract=contract, paper=selected, evidence_ids=supporting_ids)]
