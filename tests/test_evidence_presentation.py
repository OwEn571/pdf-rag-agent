from __future__ import annotations

from types import SimpleNamespace

from app.domain.models import Claim, EvidenceBlock
from app.services.evidence_presentation import (
    build_figure_contexts,
    chunk_text,
    citations_from_doc_ids,
    claim_evidence_ids,
    evidence_ids_for_paper,
    extract_topology_terms,
    figure_fallback_summary,
    formula_terms,
    join_unique_text,
    paper_recommendation_reason,
    safe_year,
    top_evidence_ids,
)


def _evidence(
    doc_id: str,
    *,
    paper_id: str = "P1",
    page: int = 1,
    block_type: str = "page_text",
    snippet: str = "snippet",
) -> EvidenceBlock:
    return EvidenceBlock(
        doc_id=doc_id,
        paper_id=paper_id,
        title="Paper One",
        file_path="/tmp/paper.pdf",
        page=page,
        block_type=block_type,
        snippet=snippet,
        metadata={"authors": "A. Author", "year": "2024", "tags": "rag||agent"},
    )


def test_evidence_presentation_extracts_terms_and_ids() -> None:
    evidence = [
        _evidence("a", snippet="DAG topology with a random controller."),
        _evidence("b", paper_id="P2", snippet="Chain and tree variants."),
    ]
    claims = [
        Claim(claim_type="summary", evidence_ids=["a", "b"]),
        Claim(claim_type="result", evidence_ids=["b", "c"]),
    ]

    assert extract_topology_terms(evidence)[:3] == ["dag", "irregular/random", "chain"]
    assert {"pi_theta", "beta", "preferred"} <= set(formula_terms("pi_theta beta preferred y_w"))
    assert top_evidence_ids(evidence, limit=1) == ["a"]
    assert evidence_ids_for_paper(evidence, "P2", limit=2) == ["b"]
    assert claim_evidence_ids(claims) == ["a", "b", "c"]


def test_evidence_presentation_builds_citations_from_evidence_and_lookup() -> None:
    evidence = [_evidence("doc-a", snippet="A" * 300)]
    fallback_doc = SimpleNamespace(
        page_content="fallback paper card text",
        metadata={
            "doc_id": "paper::P2",
            "paper_id": "P2",
            "title": "Paper Two",
            "authors": "B. Author",
            "year": "2025",
            "file_path": "/tmp/p2.pdf",
        },
    )

    citations = citations_from_doc_ids(
        ["doc-a", "doc-a", "paper::P2"],
        evidence,
        paper_doc_lookup=lambda paper_id: fallback_doc if paper_id == "P2" else None,
    )

    assert [item.doc_id for item in citations] == ["doc-a", "paper::P2"]
    assert citations[0].snippet == "A" * 220
    assert citations[0].tags == ["rag", "agent"]
    assert citations[1].title == "Paper Two"


def test_evidence_presentation_ranks_figure_contexts_and_fallback_summary() -> None:
    evidence = [
        _evidence(
            "summary-page",
            page=4,
            snippet="Summary of AIME 2024, MATH-500, GPQA and MMLU evaluation results.",
        ),
        _evidence(
            "figure-page",
            page=1,
            snippet="Figure 1 | Benchmark performance on AIME 2024, GPQA, MATH-500, MMLU and SWE-bench.",
        ),
        _evidence(
            "caption-page",
            page=1,
            block_type="caption",
            snippet="Figure 1 caption: benchmark performance across AIME 2024 and Codeforces.",
        ),
    ]

    contexts = build_figure_contexts(evidence)
    summary = figure_fallback_summary(contexts)

    assert contexts[0]["page"] == 1
    assert "figure-page" in contexts[0]["doc_ids"]
    assert "Benchmark performance" in summary
    assert "benchmark 包括" in summary


def test_evidence_presentation_small_formatting_helpers() -> None:
    long_summary = "x" * 140

    assert paper_recommendation_reason("") == "与当前主题直接相关。"
    assert paper_recommendation_reason(long_summary).endswith("...")
    assert join_unique_text([" alpha ", "alpha", "beta-gamma"], limit=14) == "alpha\nbeta-..."
    assert safe_year("2024") == 2024
    assert safe_year("unknown") == 9999
    assert chunk_text("abcdef", size=2) == ["ab", "cd", "ef"]
