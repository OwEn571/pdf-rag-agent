from __future__ import annotations

import json

from app.api.routes import _format_sse, _stream_error_events


def test_stream_error_events_use_normalized_protocol_payloads() -> None:
    events = _stream_error_events(RuntimeError("boom"))

    assert events[0] == ("error", {"message": "boom", "type": "error"})
    assert events[1][0] == "final"
    assert events[1][1]["type"] == "final"
    assert events[1][1]["answer"] == ""
    assert events[1][1]["error"] == "boom"
    assert events[1][1]["citations"] == []
    assert events[1][1]["usage"] == {}


def test_format_sse_preserves_event_name_and_json_payload() -> None:
    chunk = _format_sse("final", {"type": "final", "answer": "你好"})

    assert chunk.startswith("event: final\n")
    data_line = next(line for line in chunk.splitlines() if line.startswith("data: "))
    assert json.loads(data_line.removeprefix("data: ")) == {"type": "final", "answer": "你好"}
