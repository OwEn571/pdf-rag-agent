from __future__ import annotations

from types import SimpleNamespace

from app.domain.models import EvidenceBlock, SessionContext, SessionTurn
from app.services.citation_ranking import (
    extract_citation_count_from_evidence,
    format_citation_ranking_answer,
    parse_citation_count,
    select_citation_ranking_candidates,
    semantic_scholar_citation_evidence,
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


def test_citation_ranking_selects_previous_candidates_when_requested() -> None:
    paper_documents = [
        SimpleNamespace(metadata={"paper_id": "p1", "title": "Paper A", "year": "2024"}),
        SimpleNamespace(metadata={"paper_id": "p2", "title": "Paper B", "year": "2025"}),
    ]
    session = SessionContext(
        session_id="s1",
        turns=[
            SessionTurn(
                query="推荐几篇论文",
                answer="可以看《Paper B》（2025）和《Missing Paper》（2023）。",
                relation="library_recommendation",
            )
        ],
    )

    selected = select_citation_ranking_candidates(
        paper_documents=paper_documents,
        session=session,
        query="把刚才那些按引用数排序",
        limit=3,
        rank_library_papers_for_recommendation=lambda **_: [{"title": "Paper A"}],
    )

    assert selected == [
        {"title": "Paper B", "year": "2025", "paper_id": "p2", "reason": ""},
        {"title": "Missing Paper", "year": "2023", "paper_id": "", "reason": ""},
    ]


def test_citation_ranking_selects_ranker_candidates_without_previous_context() -> None:
    paper_documents = [
        SimpleNamespace(metadata={"paper_id": "p1", "title": "Paper A", "year": "2024", "generated_summary": "summary"}),
        SimpleNamespace(metadata={"paper_id": "p1", "title": "Duplicate Paper A", "year": "2024"}),
        SimpleNamespace(metadata={"paper_id": "p2", "title": "Paper B", "year": "2025"}),
    ]

    selected = select_citation_ranking_candidates(
        paper_documents=paper_documents,
        session=SessionContext(session_id="s1"),
        query="按引用数推荐",
        limit=1,
        rank_library_papers_for_recommendation=lambda **kwargs: [
            {"title": kwargs["docs"][0]["title"], "year": kwargs["docs"][0].get("year", ""), "reason": "ranked"}
        ],
    )

    assert selected == [{"title": "Paper A", "year": "2024", "paper_id": "p1", "reason": "ranked"}]


def test_semantic_scholar_citation_evidence_builds_evidence_block() -> None:
    class TavilyWebSearchClient:
        pass

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": [
                    {"title": "Different Paper", "citationCount": 1, "url": "https://example.com/different"},
                    {
                        "title": "Example Paper",
                        "year": 2024,
                        "citationCount": 1234,
                        "url": "https://www.semanticscholar.org/paper/example",
                    },
                ]
            }

    calls: list[dict[str, object]] = []

    def fake_get(*args: object, **kwargs: object) -> FakeResponse:
        calls.append({"args": args, **kwargs})
        return FakeResponse()

    evidence = semantic_scholar_citation_evidence(
        title="Example Paper",
        web_search=TavilyWebSearchClient(),
        timeout_seconds=9.0,
        http_get=fake_get,
    )

    assert evidence is not None
    assert evidence.metadata["source"] == "semantic_scholar"
    assert evidence.metadata["citation_count"] == 1234
    assert evidence.score >= 1.0
    assert calls[0]["timeout"] == 5.0


def test_semantic_scholar_citation_evidence_requires_tavily_client() -> None:
    assert semantic_scholar_citation_evidence(
        title="Example Paper",
        web_search=object(),
        timeout_seconds=3.0,
        http_get=lambda **_: None,
    ) is None


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
