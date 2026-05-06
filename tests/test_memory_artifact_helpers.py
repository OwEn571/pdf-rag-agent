from __future__ import annotations

from app.domain.models import QueryContract, SessionContext
from app.services.memory.artifacts import (
    answer_from_recent_tool_artifact_reference,
    chinese_ordinal_value,
    conversation_tool_result_artifact,
    latest_list_tool_artifact,
    remember_conversation_tool_result,
    referenced_list_item_index,
)


def test_referenced_list_item_index_reads_common_ordinals() -> None:
    assert referenced_list_item_index("上一轮第一篇是什么") == 0
    assert referenced_list_item_index("show me the 3rd item") == 2
    assert referenced_list_item_index("第十二条") == 11
    assert referenced_list_item_index("没有序号") is None
    assert chinese_ordinal_value("两") == 2
    assert chinese_ordinal_value("十") == 10


def test_latest_list_tool_artifact_prefers_direct_memory() -> None:
    session = SessionContext(
        session_id="artifact",
        working_memory={
            "last_displayed_list": {"items": [{"row": {"title": "Direct"}}], "query": "direct", "tool": "direct_tool"},
            "tool_results": [
                {"artifact": {"items": [{"row": {"title": "Nested"}}]}, "query": "nested", "tool": "nested_tool"}
            ],
        },
    )

    assert latest_list_tool_artifact(session)["query"] == "direct"


def test_answer_from_recent_tool_artifact_reference_formats_selected_row() -> None:
    session = SessionContext(
        session_id="artifact-answer",
        working_memory={
            "tool_results": [
                {
                    "query": "推荐论文",
                    "tool": "get_library_recommendation",
                    "artifact": {
                        "items": [
                            {
                                "ordinal": 1,
                                "row": {"title": "First Paper", "year": "2025", "authors": "A; B"},
                            },
                            {
                                "ordinal": 2,
                                "row": {"title": "Second Paper", "paper_id": "p2"},
                            },
                        ]
                    },
                }
            ]
        },
    )

    answer = answer_from_recent_tool_artifact_reference(query="上一轮第二篇是什么", session=session)

    assert "推荐论文" in answer
    assert "Second Paper" in answer
    assert "paper_id：p2" in answer
    assert "第 2 条" in answer


def test_answer_from_recent_tool_artifact_reference_reports_out_of_range() -> None:
    session = SessionContext(
        session_id="artifact-miss",
        working_memory={"last_displayed_list": {"query": "推荐论文", "items": [{"row": {"title": "Only"}}]}},
    )

    assert "找不到第 3 条" in answer_from_recent_tool_artifact_reference(query="第三条", session=session)


def test_conversation_tool_result_artifact_compacts_tabular_sql_rows() -> None:
    artifact = conversation_tool_result_artifact(
        tool="query_library_metadata",
        result={
            "sql": "select title, year from papers order by year desc",
            "columns": ["title", "year", "score"],
            "row_count": 2,
            "rows": [
                {"paper_id": "p1", "title": "First Paper", "year": "2025", "score": 0.9},
                {"paper_id": "p2", "title": "Second " * 200, "year_int": 2024, "score": None},
            ],
        },
    )

    assert artifact["type"] == "tabular_sql_result"
    assert artifact["row_count"] == 2
    assert artifact["columns"] == ["title", "year", "score"]
    assert artifact["items"][0]["ordinal"] == 1
    assert artifact["items"][0]["paper_id"] == "p1"
    assert artifact["items"][0]["row"]["score"] == 0.9
    assert artifact["items"][1]["year_int"] == 2024
    assert len(artifact["items"][1]["row"]["title"]) <= 900


def test_conversation_tool_result_artifact_ignores_other_tools() -> None:
    assert conversation_tool_result_artifact(tool="read_memory", result={"rows": [{"title": "Nope"}]}) == {}


def test_remember_conversation_tool_result_stores_recent_list_artifact() -> None:
    session = SessionContext(session_id="remember-artifact")
    contract = QueryContract(
        clean_query="推荐论文",
        relation="library_recommendation",
        targets=["alignment"],
        requested_fields=["recommendation"],
    )

    remember_conversation_tool_result(
        session=session,
        contract=contract,
        tool="query_library_metadata",
        query="推荐论文",
        answer="hello",
        artifact={"items": [{"row": {"title": "A"}}]},
    )

    memory = session.working_memory
    assert memory["last_tool_result"]["answer_preview"] == "hello"
    assert memory["last_tool_result"]["targets"] == ["alignment"]
    assert memory["last_displayed_list"]["tool"] == "query_library_metadata"
    assert memory["last_library_metadata_result"] == {"items": [{"row": {"title": "A"}}]}
