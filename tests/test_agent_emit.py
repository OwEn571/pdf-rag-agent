from __future__ import annotations

from app.services import agent_emit
from app.services.agent_emit import AgentEventRecorder, write_turn_trace_safe


def test_agent_event_recorder_normalizes_and_forwards_events() -> None:
    forwarded: list[dict[str, object]] = []
    recorder = AgentEventRecorder(callback=forwarded.append)

    recorder.emit("tool_call", {"tool": "search_corpus", "arguments": {"query": "DPO"}})

    assert recorder.events == forwarded
    assert recorder.events[0]["event"] == "tool_call"
    assert recorder.events[0]["data"]["type"] == "tool_use"
    assert recorder.events[0]["data"]["name"] == "search_corpus"
    assert recorder.events[0]["data"]["input"] == {"query": "DPO"}


def test_write_turn_trace_safe_respects_disabled_flag(tmp_path) -> None:
    path = write_turn_trace_safe(
        enabled=False,
        data_dir=tmp_path,
        session_id="demo",
        events=[],
        final_payload={},
        execution_steps=[],
    )

    assert path is None
    assert not (tmp_path / "traces").exists()


def test_write_turn_trace_safe_writes_trace(tmp_path) -> None:
    path = write_turn_trace_safe(
        enabled=True,
        data_dir=tmp_path,
        session_id="demo",
        events=[{"event": "session", "data": {"session_id": "demo"}}],
        final_payload={"answer": "hello"},
        execution_steps=[{"node": "agent_loop", "summary": "compose"}],
    )

    assert path is not None
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert '"event": "session"' in content
    assert '"answer_preview": "hello"' in content


def test_write_turn_trace_safe_catches_writer_errors(monkeypatch, tmp_path) -> None:
    def fail_writer(**_: object) -> None:
        raise OSError("nope")

    monkeypatch.setattr(agent_emit, "write_agent_trace", fail_writer)

    path = write_turn_trace_safe(
        enabled=True,
        data_dir=tmp_path,
        session_id="demo",
        events=[],
        final_payload={},
        execution_steps=[],
    )

    assert path is None
