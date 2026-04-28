from __future__ import annotations

import json

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
