from __future__ import annotations

import json
from types import SimpleNamespace

from app.domain.models import SessionContext, SessionTurn
from app.services.answers.library_recommendations import (
    clean_library_recommendation_criteria_note,
    compose_library_status_markdown,
    diversify_library_recommendations,
    library_paper_preview_lines,
    library_status_query_wants_listing,
    library_status_query_wants_recommendation,
    library_unique_paper_metadata,
    llm_select_library_recommendations,
    rank_library_papers_for_recommendation,
    recent_library_recommendation_titles,
    select_library_recommendations,
    split_library_authors,
)


class _FakeRecommendationClients:
    chat = object()

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.last_human_prompt = ""

    def invoke_json(self, *, system_prompt: str, human_prompt: str, fallback: object) -> object:
        _ = system_prompt, fallback
        self.last_human_prompt = human_prompt
        return self.payload


def test_library_status_query_helpers_detect_listing_and_recommendation() -> None:
    assert library_status_query_wants_listing("我的库里有哪些论文？")
    assert library_status_query_wants_recommendation("哪几篇最值得一读？")


def test_rank_library_papers_prefers_survey_when_requested() -> None:
    ranked = rank_library_papers_for_recommendation(
        docs=[
            {"paper_id": "p1", "title": "Narrow Method", "year": "2024", "tags": "method", "generated_summary": "A narrow method."},
            {"paper_id": "p2", "title": "A Survey of RAG", "year": "2021", "tags": "survey||rag", "generated_summary": "A comprehensive survey."},
        ],
        query="给我推荐一篇综述入门",
        limit=2,
    )

    assert ranked[0]["paper_id"] == "p2"
    assert "comprehensive survey" in ranked[0]["reason"]


def test_library_preview_and_recent_recommendation_helpers() -> None:
    preview = library_paper_preview_lines(
        docs=[
            {"paper_id": "old", "title": "Old Paper", "year": "2019", "tags": "tag"},
            {"paper_id": "new", "title": "New Paper", "year": "2025", "tags": "tag"},
        ],
        collection_paths={"new": ["Top/Recent"]},
        limit=2,
    )
    assert preview[0].startswith("- 《New Paper》")

    session = SessionContext(session_id="s")
    session.turns.append(SessionTurn(query="推荐", answer="先读《Paper A》，再读《Paper B》。", relation="library_recommendation"))
    assert recent_library_recommendation_titles(session) == ["Paper A", "Paper B"]


def test_library_unique_metadata_and_status_markdown() -> None:
    docs = library_unique_paper_metadata(
        paper_documents=[
            SimpleNamespace(
                metadata={
                    "paper_id": "p1",
                    "title": "Foundational RAG Survey",
                    "year": "2022",
                    "tags": "survey||rag",
                    "generated_summary": "A comprehensive survey for RAG.",
                    "file_path": "/tmp/p1.pdf",
                }
            ),
            SimpleNamespace(metadata={"paper_id": "p1", "title": "Duplicate"}),
            SimpleNamespace(
                metadata={
                    "paper_id": "p2",
                    "title": "New Method",
                    "year": "2025",
                    "tags": "method",
                    "generated_summary": "A method paper.",
                    "file_path": "/tmp/p2.txt",
                }
            ),
            SimpleNamespace(metadata={"title": "Missing id"}),
        ]
    )

    assert [item["paper_id"] for item in docs] == ["p1", "p2"]

    answer = compose_library_status_markdown(
        query="我的库里有哪些论文，哪篇值得一读？",
        docs=docs,
        collection_paths={"p1": ["RAG/Survey"]},
    )

    assert "共有 **2 篇论文**" in answer
    assert "PDF 路径的记录是 **1 篇**" in answer
    assert "年份范围大约是 **2022–2025**" in answer
    assert "RAG/Survey（1）" in answer
    assert "## 文章预览" in answer
    assert "## 默认推荐" in answer
    assert "top-k 候选数" in answer


def test_diversify_and_note_cleanup_and_author_split() -> None:
    diversified = diversify_library_recommendations(
        candidates=[
            {"title": "Paper A"},
            {"title": "Paper B"},
        ],
        recent_titles=["Paper A"],
        query="换一篇",
        limit=2,
    )
    assert [item["title"] for item in diversified] == ["Paper B", "Paper A"]
    assert clean_library_recommendation_criteria_note("This is a very English heavy explanation", has_recent_recommendations=False).startswith("我会按")
    assert split_library_authors("Alice, Bob and Carol") == ["Alice", "Bob", "Carol"]


def test_llm_select_library_recommendations_maps_titles_and_reasons() -> None:
    clients = _FakeRecommendationClients(
        {
            "criteria_note": "按问题主题和摘要证据挑选",
            "recommendations": [
                {"title": "Unknown Paper", "reason": "ignore"},
                {"title": "Paper B", "reason": "更贴合当前问题"},
                {"title": "Paper A", "reason": ""},
            ],
        }
    )
    candidates = [
        {"title": "Paper A", "year": "2023", "reason": "fallback A"},
        {"title": "Paper B", "year": "2024", "reason": "fallback B"},
    ]

    selected, note = llm_select_library_recommendations(
        query="推荐 personalization 方向",
        candidates=candidates,
        session=None,
        recent_titles=["Paper A"],
        limit=2,
        clients=clients,
        settings=SimpleNamespace(data_dir="/tmp"),
    )

    assert note == "按问题主题和摘要证据挑选"
    assert [item["title"] for item in selected] == ["Paper B", "Paper A"]
    assert selected[0]["reason"] == "更贴合当前问题"
    assert selected[1]["reason"] == "fallback A"
    prompt_payload = json.loads(clients.last_human_prompt)
    assert prompt_payload["current_query"] == "推荐 personalization 方向"
    assert prompt_payload["recently_recommended_titles"] == ["Paper A"]


def test_select_library_recommendations_falls_back_to_diversified_candidates() -> None:
    clients = SimpleNamespace(chat=None)
    session = SessionContext(session_id="s")
    session.turns.append(SessionTurn(query="推荐", answer="先读《Paper A》。", relation="library_recommendation"))

    selected, note = select_library_recommendations(
        query="换一篇推荐",
        candidates=[
            {"title": "Paper A", "reason": "A"},
            {"title": "Paper B", "reason": "B"},
        ],
        session=session,
        limit=2,
        clients=clients,
        settings=SimpleNamespace(data_dir="/tmp"),
    )

    assert [item["title"] for item in selected] == ["Paper B", "Paper A"]
    assert "避开" in note
