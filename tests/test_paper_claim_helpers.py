from __future__ import annotations

from app.domain.models import CandidatePaper
from app.services.claims.paper_helpers import (
    default_text_claim,
    default_text_claims,
    paper_recommendation_claim,
    paper_summary_claim,
    paper_summary_claims,
)


def test_paper_summary_claim_filters_metric_lines_by_title_then_falls_back() -> None:
    paper = CandidatePaper(paper_id="p1", title="AlignX", year="2025", doc_ids=["paper-doc"])

    claim = paper_summary_claim(
        entity="AlignX",
        paper=paper,
        summary_text="summary",
        metric_lines=["AlignX accuracy 59.6", "Other score 50"],
        evidence_ids=[],
    )

    assert claim.claim_type == "paper_summary"
    assert claim.structured_data["metric_lines"] == ["AlignX accuracy 59.6"]
    assert claim.evidence_ids == ["paper-doc"]


def test_paper_summary_claims_skips_empty_papers_and_uses_fallback_entity() -> None:
    papers = [
        CandidatePaper(paper_id="p1", title="Paper One", year="2024"),
        CandidatePaper(paper_id="p2", title="Paper Two", year="2025"),
    ]

    claims = paper_summary_claims(
        entity="",
        papers=papers,
        metric_lines=["Paper Two accuracy 90"],
        summary_for_paper=lambda paper_id: "summary two" if paper_id == "p2" else "",
        evidence_ids_for_paper=lambda paper_id, limit: ["ev-2"] if paper_id == "p2" else [],
    )

    assert len(claims) == 1
    assert claims[0].entity == "Paper Two"
    assert claims[0].value == "summary two"
    assert claims[0].evidence_ids == ["ev-2"]


def test_paper_recommendation_claim_builds_reason_rows_and_dedupes_evidence() -> None:
    papers = [
        CandidatePaper(paper_id="p1", title="Paper One", year="2024", doc_ids=["doc-1"]),
        CandidatePaper(paper_id="p2", title="Paper Two", year="2025", doc_ids=["doc-1", "doc-2"]),
    ]

    claim = paper_recommendation_claim(entity="RAG", papers=papers, reason_for_paper=lambda paper: f"reason {paper.paper_id}")

    assert claim is not None
    assert claim.value == "Paper One (2024); Paper Two (2025)"
    assert claim.evidence_ids == ["doc-1"]
    assert claim.paper_ids == ["p1", "p2"]
    assert claim.structured_data["recommended_papers"][1]["reason"] == "reason p2"


def test_default_text_claim_uses_summary_or_title_and_fallback_doc_id() -> None:
    paper = CandidatePaper(paper_id="p1", title="Paper One", year="2024", doc_ids=["paper-doc"])

    claim = default_text_claim(entity="Paper One", paper=paper, summary="", evidence_ids=[])

    assert claim.claim_type == "text_answer"
    assert claim.value == "Paper One"
    assert claim.evidence_ids == ["paper-doc"]


def test_default_text_claims_builds_claims_from_summaries_or_evidence() -> None:
    papers = [
        CandidatePaper(paper_id="p1", title="Paper One", year="2024"),
        CandidatePaper(paper_id="p2", title="Paper Two", year="2025"),
    ]

    claims = default_text_claims(
        entity="RAG",
        papers=papers,
        summary_for_paper=lambda paper_id: "summary one" if paper_id == "p1" else "",
        evidence_ids_for_paper=lambda paper_id, limit: ["ev-2"] if paper_id == "p2" else [],
    )

    assert [claim.value for claim in claims] == ["summary one", "Paper Two"]
    assert [claim.entity for claim in claims] == ["RAG", "RAG"]
    assert claims[1].evidence_ids == ["ev-2"]
