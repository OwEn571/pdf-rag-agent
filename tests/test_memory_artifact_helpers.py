from __future__ import annotations

from app.domain.models import SessionContext
from app.services.memory_artifact_helpers import (
    answer_from_recent_tool_artifact_reference,
    chinese_ordinal_value,
    latest_list_tool_artifact,
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
