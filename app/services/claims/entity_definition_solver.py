from __future__ import annotations

from collections.abc import Callable

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract
from app.services.entities.definition_helpers import entity_definition_claim, entity_definition_evidence_ids
from app.services.planning.query_shaping import matches_target

SelectEntitySupportingPaper = Callable[
    [QueryContract, list[CandidatePaper], list[EvidenceBlock]],
    tuple[CandidatePaper | None, list[EvidenceBlock]],
]
InferEntityType = Callable[[QueryContract, list[CandidatePaper], list[EvidenceBlock]], str]
EntitySupportingLines = Callable[[list[EvidenceBlock], str], list[str]]


def solve_entity_definition_claims(
    *,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    select_supporting_paper: SelectEntitySupportingPaper,
    infer_entity_type: InferEntityType,
    entity_supporting_lines: EntitySupportingLines,
) -> list[Claim]:
    paper, supporting_evidence = select_supporting_paper(contract, papers, evidence)
    if paper is None:
        return []
    relevant_evidence = supporting_evidence or [item for item in evidence if item.paper_id == paper.paper_id][:4]
    label = infer_entity_type(contract, [paper], relevant_evidence)
    return [
        entity_definition_claim(
            contract=contract,
            paper=paper,
            label=label,
            evidence_ids=entity_definition_evidence_ids(
                contract=contract,
                paper=paper,
                evidence=relevant_evidence,
                target_matcher=matches_target,
            ),
            definition_lines=entity_supporting_lines(relevant_evidence, "definition"),
            mechanism_lines=entity_supporting_lines(relevant_evidence, "mechanism"),
            application_lines=entity_supporting_lines(relevant_evidence, "application"),
        )
    ]
