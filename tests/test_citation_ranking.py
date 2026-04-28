from __future__ import annotations

from app.domain.models import EvidenceBlock
from app.services.citation_ranking import (
    extract_citation_count_from_evidence,
    format_citation_ranking_answer,
    parse_citation_count,
    title_token_overlap,
)


def test_citation_ranking_extracts_count_from_matching_evidence() -> None:
    evidence = [
        EvidenceBlock(
            doc_id="web-1",
            paper_id="web-1",
            title="Example Paper | Semantic Scholar",
            file_path="https://www.semanticscholar.org/paper/example",
            page=0,
            block_type="web",
            snippet="Semantic Scholar citationCount: 12,345. Matched paper title: Example Paper.",
        )
    ]

    extracted = extract_citation_count_from_evidence(title="Example Paper", evidence=evidence)

    assert parse_citation_count("12,345 citations") == 12345
    assert title_token_overlap("Example Paper", "Example Paper | Semantic Scholar") >= 1.0
    assert extracted["citation_count"] == 12345
    assert extracted["doc_id"] == "web-1"


def test_citation_ranking_format_refuses_local_heuristic_without_web() -> None:
    answer = format_citation_ranking_answer(
        candidates=[{"title": "Example Paper", "year": "2024"}],
        citation_results=[],
        web_enabled=False,
    )

    assert "不能只靠本地 PDF 摘要推断" in answer
    assert "Example Paper" in answer


def test_citation_ranking_format_sorts_counted_results() -> None:
    answer = format_citation_ranking_answer(
        candidates=[{"title": "A", "year": "2024"}, {"title": "B", "year": "2023"}],
        citation_results=[
            {"title": "A", "year": "2024", "citation_count": 10, "source_url": "https://a.example"},
            {"title": "B", "year": "2023", "citation_count": 20, "source_url": "https://b.example"},
        ],
        web_enabled=True,
    )

    assert answer.index("《B》") < answer.index("《A》")
    assert "20" in answer
