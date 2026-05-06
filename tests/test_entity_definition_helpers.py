from __future__ import annotations

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services.entities.definition_helpers import entity_definition_claim, entity_definition_evidence_ids


def _evidence(doc_id: str) -> EvidenceBlock:
    return EvidenceBlock(
        doc_id=doc_id,
        paper_id="paper-1",
        title="AlignX",
        file_path="/tmp/alignx.pdf",
        page=1,
        block_type="page_text",
        snippet="AlignX is a personalized preference alignment dataset.",
    )


def test_entity_definition_evidence_ids_prefers_supporting_evidence() -> None:
    paper = CandidatePaper(paper_id="paper-1", title="AlignX", doc_ids=["paper-card"])
    contract = QueryContract(clean_query="AlignX 是什么", targets=["AlignX"])

    ids = entity_definition_evidence_ids(
        contract=contract,
        paper=paper,
        evidence=[_evidence("ev-1"), _evidence("ev-2"), _evidence("ev-3"), _evidence("ev-4")],
        target_matcher=lambda text, target: target in text,
    )

    assert ids == ["ev-1", "ev-2", "ev-3"]


def test_entity_definition_evidence_ids_falls_back_to_doc_when_target_matches_paper_text() -> None:
    paper = CandidatePaper(
        paper_id="paper-1",
        title="From 1,000,000 Users to Every User",
        doc_ids=["paper-card"],
        metadata={"generated_summary": "This paper introduces AlignX."},
    )
    contract = QueryContract(clean_query="AlignX 是什么", targets=["AlignX"])

    ids = entity_definition_evidence_ids(
        contract=contract,
        paper=paper,
        evidence=[],
        target_matcher=lambda text, target: target in text,
    )

    assert ids == ["paper-card"]


def test_entity_definition_claim_preserves_structured_support_lines() -> None:
    paper = CandidatePaper(paper_id="paper-1", title="AlignX Paper")
    contract = QueryContract(clean_query="AlignX 是什么", targets=["AlignX"])

    claim = entity_definition_claim(
        contract=contract,
        paper=paper,
        label="dataset",
        evidence_ids=["ev-1"],
        definition_lines=["AlignX is a dataset."],
        mechanism_lines=["It models user preferences."],
        application_lines=["It supports personalized alignment."],
    )

    assert claim.claim_type == "entity_definition"
    assert claim.entity == "AlignX"
    assert claim.value == "dataset"
    assert claim.structured_data["paper_title"] == "AlignX Paper"
    assert claim.structured_data["definition_lines"] == ["AlignX is a dataset."]
    assert claim.structured_data["mechanism_lines"] == ["It models user preferences."]
    assert claim.structured_data["application_lines"] == ["It supports personalized alignment."]
    assert claim.evidence_ids == ["ev-1"]
    assert claim.paper_ids == ["paper-1"]
    assert claim.confidence == 0.9
