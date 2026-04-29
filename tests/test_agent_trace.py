from __future__ import annotations

import json

from app.services.agent_trace_diff import diff_agent_traces
from app.services.agent_trace import write_agent_trace


def test_write_agent_trace_records_events_and_compact_final(tmp_path) -> None:
    path = write_agent_trace(
        data_dir=tmp_path,
        session_id="demo/session",
        events=[
            {"event": "tool_call", "data": {"type": "tool_use", "name": "search_corpus"}},
            {"event": "observation", "data": {"type": "tool_result", "name": "search_corpus"}},
        ],
        final_payload={"answer": "hello" * 400, "session_id": "demo/session"},
        execution_steps=[{"node": "agent_loop", "summary": "search_corpus"}],
    )

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert path.parent.name == "demo_session"
    assert lines[0]["event"] == "tool_call"
    assert lines[1]["data"]["type"] == "tool_result"
    assert lines[2]["event"] == "final"
    assert lines[2]["data"]["answer_chars"] == 2000
    assert len(lines[2]["data"]["answer_preview"]) == 1200
    assert lines[2]["data"]["execution_steps"][0]["node"] == "agent_loop"


def test_diff_agent_traces_ignores_volatile_fields() -> None:
    expected = [
        {"index": 1, "event": "tool_call", "data": {"id": "a", "type": "tool_use", "name": "search_corpus"}},
        {
            "index": 2,
            "event": "observation",
            "data": {"id": "a", "type": "tool_result", "name": "search_corpus", "ok": True, "took_ms": 10},
        },
        {
            "index": 3,
            "event": "final",
            "data": {
                "answer_chars": 380,
                "answer_preview": "old",
                "execution_steps": [{"node": "agent_loop", "summary": "old"}],
            },
        },
    ]
    actual = [
        {"index": 1, "event": "tool_call", "data": {"id": "b", "type": "tool_use", "name": "search_corpus"}},
        {
            "index": 2,
            "event": "observation",
            "data": {"id": "b", "type": "tool_result", "name": "search_corpus", "ok": True, "took_ms": 200},
        },
        {
            "index": 3,
            "event": "final",
            "data": {
                "answer_chars": 399,
                "answer_preview": "new",
                "execution_steps": [{"node": "agent_loop", "summary": "new"}],
            },
        },
    ]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is True
    assert diff.differences == []


def test_diff_agent_traces_reports_stable_signal_changes() -> None:
    expected = [
        {"event": "tool_call", "data": {"type": "tool_use", "name": "search_corpus"}},
        {"event": "observation", "data": {"type": "tool_result", "name": "search_corpus", "ok": True}},
    ]
    actual = [
        {"event": "tool_call", "data": {"type": "tool_use", "name": "web_search"}},
        {"event": "observation", "data": {"type": "tool_result", "name": "web_search", "ok": False}},
    ]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is False
    assert "search_corpus" in diff.differences[0]
    assert "web_search" in diff.differences[0]


def test_diff_agent_traces_reports_ask_human_question_changes() -> None:
    expected = [
        {
            "event": "ask_human",
            "data": {"type": "ask_human", "question": "选哪篇？", "options": [{"label": "A"}]},
        }
    ]
    actual = [
        {
            "event": "ask_human",
            "data": {"type": "ask_human", "question": "确认哪个公式？", "options": []},
        }
    ]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is False
    assert "选哪篇" in diff.differences[0]
    assert "确认哪个公式" in diff.differences[0]


def test_diff_agent_traces_reports_todo_update_changes() -> None:
    expected = [
        {
            "event": "todo_update",
            "data": {"type": "todo_update", "items": [{"id": "1", "text": "查找表格证据", "status": "doing"}]},
        }
    ]
    actual = [
        {
            "event": "todo_update",
            "data": {"type": "todo_update", "items": [{"id": "1", "text": "解释指标定义", "status": "pending"}]},
        }
    ]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is False
    assert "查找表格证据" in diff.differences[0]
    assert "解释指标定义" in diff.differences[0]
