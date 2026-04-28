from __future__ import annotations

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services.metric_text_helpers import (
    extract_metric_lines,
    metric_block_score,
    metric_context_claim,
    metric_line_score,
    text_table_metric_claim,
)


METRIC_WEIGHTS = {"win rate": 3.0, "accuracy": 2.0, "acc": 1.0}


def _evidence(
    doc_id: str,
    *,
    snippet: str,
    block_type: str = "page_text",
    paper_id: str = "paper-1",
    score: float = 0.0,
) -> EvidenceBlock:
    return EvidenceBlock(
        doc_id=doc_id,
        paper_id=paper_id,
        title="AlignX",
        file_path="/tmp/a.pdf",
        page=1,
        block_type=block_type,
        snippet=snippet,
        score=score,
    )


def test_metric_line_score_uses_configured_weights() -> None:
    assert metric_line_score("PBA win rate and accuracy", token_weights=METRIC_WEIGHTS) == 6
    assert metric_line_score("plain summary", token_weights=METRIC_WEIGHTS) == 0


def test_extract_metric_lines_deduplicates_and_sorts_by_score() -> None:
    lines = extract_metric_lines(
        [
            _evidence("a", snippet="PBA accuracy is 57.8"),
            _evidence("b", snippet="PBA win rate and accuracy are reported"),
            _evidence("c", snippet="PBA accuracy is 57.8"),
        ],
        token_weights=METRIC_WEIGHTS,
    )

    assert lines == ["PBA win rate and accuracy are reported", "PBA accuracy is 57.8"]


def test_metric_block_score_rewards_table_target_and_primary_paper_match() -> None:
    evidence = _evidence("a", snippet="PBA win rate accuracy", block_type="table", score=0.4)
    contract = QueryContract(clean_query="PBA win rate 是多少？", targets=["PBA"])
    paper_by_id = {"paper-1": CandidatePaper(paper_id="paper-1", title="PBA: Preference Bridged Alignment")}

    base = metric_block_score(
        item=evidence,
        contract=contract,
        paper_by_id=paper_by_id,
        token_weights=METRIC_WEIGHTS,
        target_paper_match=False,
    )
    matched = metric_block_score(
        item=evidence,
        contract=contract,
        paper_by_id=paper_by_id,
        token_weights=METRIC_WEIGHTS,
        target_paper_match=True,
    )

    assert matched == base + 6.0


def test_metric_context_claim_uses_metric_evidence_or_fallback_ids() -> None:
    paper = CandidatePaper(paper_id="paper-1", title="AlignX", year="2025")
    evidence = [_evidence("ev-1", snippet="PBA accuracy", block_type="table")]

    claim = metric_context_claim(
        entity="PBA",
        selected_paper=paper,
        selected_papers=[paper],
        metric_lines=["PBA accuracy 59.66"],
        metric_evidence=evidence,
        fallback_evidence_ids=["fallback"],
        paper_ids=[],
    )

    assert claim.claim_type == "metric_context"
    assert claim.evidence_ids == ["ev-1"]
    assert claim.paper_ids == ["paper-1"]
    assert claim.structured_data["paper_titles"] == ["AlignX"]


def test_text_table_metric_claim_uses_first_line_or_default_text() -> None:
    paper = CandidatePaper(paper_id="paper-1", title="AlignX")

    claim = text_table_metric_claim(
        entity="PBA",
        metric_lines=[],
        evidence_ids=["ev-1"],
        paper_ids=[],
        selected_paper=paper,
    )

    assert claim.claim_type == "metric_value"
    assert claim.value == "已定位到表格指标证据。"
    assert claim.structured_data["mode"] == "text_table"
    assert claim.paper_ids == ["paper-1"]
