from __future__ import annotations

from app.domain.models import SessionContext, SessionTurn
from app.services.intent_marker_matching import query_matches_any
from app.services.memory_intents import (
    MEMORY_INTENT_MARKERS,
    contains_ordinal_reference,
    is_memory_comparison_query,
    is_pdf_agent_topology_design_query,
    is_short_followup,
    looks_like_memory_reference,
    looks_like_recent_tool_result_reference,
)


def test_memory_intent_markers_use_centralized_profiles() -> None:
    assert query_matches_any("上一轮", "上一轮", MEMORY_INTENT_MARKERS["memory_reference"])
    assert query_matches_any(
        "pdf agent",
        "pdfagent",
        MEMORY_INTENT_MARKERS["pdf_agent_explicit"],
    )
    assert not query_matches_any("随便聊聊", "随便聊聊", MEMORY_INTENT_MARKERS["short_followup"])


def test_memory_intents_detect_references_and_short_followups() -> None:
    assert contains_ordinal_reference("第一篇具体讲什么？")
    assert contains_ordinal_reference("what about the 2nd?")
    assert looks_like_memory_reference("上一轮列表里的第一篇")
    assert is_short_followup("这个公式呢")
    assert not is_short_followup("请详细重新检索并总结这个方向的所有后续论文和关键证据")


def test_memory_intents_detect_recent_tool_result_references() -> None:
    session = SessionContext(
        session_id="s1",
        turns=[SessionTurn(query="推荐几篇论文", answer="...")],
        working_memory={"last_displayed_list": {"items": [{"title": "A"}]}},
    )
    artifact_session = SessionContext(
        session_id="s2",
        turns=[SessionTurn(query="列出结果", answer="...")],
        working_memory={"tool_results": [{"artifact": {"items": [{"title": "B"}]}}]},
    )

    assert looks_like_recent_tool_result_reference("第一篇为什么？", session=session)
    assert looks_like_recent_tool_result_reference("上面列表", session=artifact_session)
    assert not looks_like_recent_tool_result_reference("第一篇为什么？", session=None)


def test_memory_intents_detect_comparison_and_pdf_agent_design_query() -> None:
    lowered = "pdf agent 多智能体系统应该用什么拓扑设计"
    compact = "pdfagent多智能体系统应该用什么拓扑设计"

    assert is_memory_comparison_query("这两者有什么区别")
    assert is_memory_comparison_query("compare the two")
    assert is_pdf_agent_topology_design_query(lowered=lowered, compact=compact)
    assert not is_pdf_agent_topology_design_query(lowered="ordinary rag question", compact="ordinaryragquestion")
