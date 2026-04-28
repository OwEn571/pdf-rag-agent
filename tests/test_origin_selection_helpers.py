from __future__ import annotations

from app.domain.models import CandidatePaper
from app.services.origin_selection_helpers import (
    origin_display_entity,
    origin_paper_text,
    origin_target_aliases,
    origin_target_definition_score,
    origin_target_intro_score,
    paper_has_origin_intro_support,
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


def test_paper_has_origin_intro_support_uses_combined_paper_text() -> None:
    paper = CandidatePaper(
        paper_id="p1",
        title="AlignX",
        metadata={"abstract_note": "We introduce AlignX, a personalized preference alignment dataset."},
    )

    assert "We introduce AlignX" in origin_paper_text(paper)
    assert paper_has_origin_intro_support(paper=paper, targets=["AlignX"])
