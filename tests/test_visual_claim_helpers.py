from __future__ import annotations

from app.services.visual_claim_helpers import (
    figure_conclusion_claim_from_vlm_payload,
    table_metric_claim_from_vlm_payload,
)


def test_table_metric_claim_from_vlm_payload_uses_claim_or_draft_answer() -> None:
    claim = table_metric_claim_from_vlm_payload(
        {"claims": [{"claim": "", "metric_lines": [], "confidence": "high"}], "draft_answer": "PBA accuracy is 59.66"},
        entity="PBA",
        evidence_ids=["ev-1"],
        paper_ids=["paper-1"],
    )

    assert claim is not None
    assert claim.claim_type == "metric_value"
    assert claim.value == "PBA accuracy is 59.66"
    assert claim.structured_data["metric_lines"] == ["PBA accuracy is 59.66"]
    assert claim.confidence == 0.88


def test_table_metric_claim_from_vlm_payload_preserves_metric_lines() -> None:
    claim = table_metric_claim_from_vlm_payload(
        {"claims": [{"claim": "reported metrics", "metric_lines": ["ICA 57.80", "PBA 59.66"]}]},
        entity="AlignX",
        evidence_ids=["ev-1"],
        paper_ids=["paper-1"],
    )

    assert claim is not None
    assert claim.structured_data["mode"] == "vlm_table"
    assert claim.structured_data["metric_lines"] == ["ICA 57.80", "PBA 59.66"]


def test_figure_conclusion_claim_requires_signal_above_fallback() -> None:
    weak = figure_conclusion_claim_from_vlm_payload(
        {"claims": [{"claim": "short"}]},
        entity="Figure 1",
        evidence_ids=["fig-1"],
        paper_id="paper-1",
        fallback_text="strong fallback",
        signal_score=lambda text: 5 if "strong" in text else 1,
    )
    strong = figure_conclusion_claim_from_vlm_payload(
        {"claims": [{"claim": "strong visual conclusion", "confidence": "medium"}]},
        entity="Figure 1",
        evidence_ids=["fig-1"],
        paper_id="paper-1",
        fallback_text="plain",
        signal_score=lambda text: 5 if "strong" in text else 1,
    )

    assert weak is None
    assert strong is not None
    assert strong.structured_data["mode"] == "vlm"
    assert strong.confidence == 0.72
