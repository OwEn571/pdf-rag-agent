from __future__ import annotations

from collections.abc import Callable

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract

TargetMatcher = Callable[[str, str], bool]


def entity_definition_evidence_ids(
    *,
    contract: QueryContract,
    paper: CandidatePaper,
    evidence: list[EvidenceBlock],
    target_matcher: TargetMatcher,
) -> list[str]:
    evidence_ids = [item.doc_id for item in evidence[:3]]
    if evidence_ids:
        return evidence_ids
    paper_text = "\n".join(
        [
            paper.title,
            str(paper.metadata.get("paper_card_text", "")),
            str(paper.metadata.get("generated_summary", "")),
            str(paper.metadata.get("abstract_note", "")),
        ]
    )
    if paper.doc_ids and (
        not contract.targets
        or any(target_matcher(paper_text, target) for target in contract.targets if target)
    ):
        return list(paper.doc_ids[:1])
    return []


def entity_definition_claim(
    *,
    contract: QueryContract,
    paper: CandidatePaper,
    label: str,
    evidence_ids: list[str],
    definition_lines: list[str],
    mechanism_lines: list[str],
    application_lines: list[str],
) -> Claim:
    return Claim(
        claim_type="entity_definition",
        entity=contract.targets[0] if contract.targets else "",
        value=label,
        structured_data={
            "paper_title": paper.title,
            "description": "",
            "definition_lines": definition_lines,
            "mechanism_lines": mechanism_lines,
            "application_lines": application_lines,
        },
        evidence_ids=evidence_ids,
        paper_ids=[paper.paper_id],
        confidence=0.9,
    )
