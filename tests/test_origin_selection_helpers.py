from __future__ import annotations

from app.domain.models import CandidatePaper, QueryContract
from app.services.claims.origin_selection import (
    origin_lookup_claim,
    origin_display_entity,
    origin_paper_text,
    origin_target_aliases,
    origin_target_definition_score,
    origin_target_intro_score,
    paper_has_origin_intro_support,
    pick_origin_paper,
)


def test_origin_target_aliases_strip_domain_suffixes_and_split_camel_case() -> None:
    aliases = origin_target_aliases(["pdfRagAgent", "AlignX 方法"])

    assert "pdfRagAgent" in aliases
    assert "pdf Rag Agent" in aliases
    assert "AlignX" in aliases


def test_origin_intro_score_rewards_direct_intro_cues() -> None:
    aliases = ["AlignX"]

    assert origin_target_intro_score("In this paper, we propose AlignX for user-level alignment.", aliases) >= 6.0
    assert origin_target_intro_score("Prior work compares against AlignX in experiments.", aliases) == 0.0


def test_origin_definition_score_rewards_entity_definition_context() -> None:
    aliases = ["AlignX"]

    score = origin_target_definition_score("AlignX is a benchmark for personalized preference alignment.", aliases)

    assert score >= 3.0


def test_origin_display_entity_uses_paper_text_casing_before_fallback() -> None:
    paper = CandidatePaper(
        paper_id="p1",
        title="From 1,000,000 Users to Every User",
        metadata={"generated_summary": "This paper introduced AlignX for user-level alignment."},
    )

    assert origin_display_entity(targets=["alignx"], paper=paper) == "AlignX"


def test_origin_lookup_claim_uses_contract_target_and_doc_fallback() -> None:
    contract = QueryContract(clean_query="AlignX 最早是哪篇论文提出的", targets=["alignx"])
    paper = CandidatePaper(
        paper_id="p1",
        title="From 1,000,000 Users to Every User",
        year="2025",
        doc_ids=["doc-1"],
        metadata={"generated_summary": "This paper introduced AlignX for user-level alignment."},
    )

    claim = origin_lookup_claim(contract=contract, paper=paper, evidence_ids=[])

    assert claim.claim_type == "origin"
    assert claim.entity == "AlignX"
    assert claim.value == "From 1,000,000 Users to Every User"
    assert claim.structured_data == {"year": "2025", "paper_title": "From 1,000,000 Users to Every User"}
    assert claim.evidence_ids == ["doc-1"]
    assert claim.paper_ids == ["p1"]
    assert claim.confidence == 0.94


def test_paper_has_origin_intro_support_uses_combined_paper_text() -> None:
    paper = CandidatePaper(
        paper_id="p1",
        title="AlignX",
        metadata={"abstract_note": "We introduce AlignX, a personalized preference alignment dataset."},
    )

    assert "We introduce AlignX" in origin_paper_text(paper)
    assert paper_has_origin_intro_support(paper=paper, targets=["AlignX"])


def test_pick_origin_paper_prefers_earliest_year_then_score() -> None:
    early = CandidatePaper(paper_id="early", title="Early", year="2020", score=0.2)
    stronger_same_year = CandidatePaper(paper_id="strong", title="Strong", year="2020", score=0.9)
    recent = CandidatePaper(paper_id="recent", title="Recent", year="2024", score=2.0)

    assert pick_origin_paper([recent, early, stronger_same_year]) == stronger_same_year
    assert pick_origin_paper([]) is None
