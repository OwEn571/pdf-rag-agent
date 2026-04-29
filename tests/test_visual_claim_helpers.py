from __future__ import annotations

from app.services.visual_claim_helpers import (
    figure_conclusion_claim_from_vlm_payload,
    figure_vlm_human_content,
    figure_vlm_system_prompt,
    table_metric_claim_from_vlm_payload,
    table_vlm_human_content,
    table_vlm_system_prompt,
)
from app.domain.models import EvidenceBlock, QueryContract


def _evidence(block_type: str = "table") -> EvidenceBlock:
    return EvidenceBlock(
        doc_id="ev-1",
        paper_id="paper-1",
        title="Paper One",
        file_path="/tmp/paper.pdf",
        page=3,
        block_type=block_type,
        caption="Table caption",
        snippet="ICA 57.80 PBA 59.66",
    )


def test_table_vlm_content_includes_text_and_deduplicated_images() -> None:
    contract = QueryContract(clean_query="PBA 准确率多少？")
    content = table_vlm_human_content(
        contract=contract,
        ranked_blocks=[_evidence(), _evidence()],
        render_page_image=lambda file_path, page: f"data:{file_path}:{page}",
    )

    assert table_vlm_system_prompt() == "你是论文表格视觉理解求解器。只输出 JSON。"
    assert content[0]["type"] == "text"
    assert "PBA 准确率多少？" in content[0]["text"]
    assert sum(1 for item in content if item.get("type") == "image_url") == 1
    assert "ICA 57.80" in content[1]["text"]


def test_figure_vlm_content_includes_context_and_images() -> None:
    contract = QueryContract(clean_query="Figure 1 说明什么？")
    content = figure_vlm_human_content(
        contract=contract,
        figure_contexts=[
            {
                "title": "Paper One",
                "page": 1,
                "caption": "Figure 1",
                "figure_text": "benchmark results",
                "page_text": "AIME and MATH",
                "file_path": "/tmp/paper.pdf",
            }
        ],
        render_page_image=lambda file_path, page: f"data:{file_path}:{page}",
    )

    assert figure_vlm_system_prompt() == "你是论文图像理解求解器。只输出 JSON。"
    assert "Figure 1 说明什么？" in content[0]["text"]
    assert "benchmark results" in content[1]["text"]
    assert content[2]["type"] == "image_url"


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
