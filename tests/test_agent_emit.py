from __future__ import annotations

from app.domain.models import QueryContract
from app.services import agent_emit
from app.services.agent_emit import (
    AgentEventRecorder,
    emit_agent_step,
    emit_agent_tool_call,
    record_agent_observation,
    write_turn_trace_safe,
)


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


def test_record_agent_observation_canonicalizes_tool_and_step() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    steps: list[dict[str, object]] = []

    record_agent_observation(
        emit=lambda event, payload: events.append((event, payload)),
        execution_steps=steps,
        tool="answer_conversation",
        summary="done",
        payload={"intent": "library_status"},
    )

    assert events == [
        (
            "observation",
            {
                "tool": "compose",
                "summary": "done",
                "payload": {"intent": "library_status", "raw_tool": "answer_conversation"},
            },
        )
    ]
    assert steps == [{"node": "agent_tool:compose", "summary": "done"}]


def test_emit_agent_tool_call_and_step_use_protocol_payloads() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    emit_agent_tool_call(
        emit=lambda event, payload: events.append((event, payload)),
        tool="get_library_status",
        arguments={"query": "多少论文"},
    )
    emit_agent_step(
        emit=lambda event, payload: events.append((event, payload)),
        index=1,
        action="search_corpus",
        contract=QueryContract(clean_query="DPO是什么", targets=["DPO"]),
        arguments={"query": "DPO"},
    )

    assert events[0] == (
        "tool_call",
        {"tool": "compose", "arguments": {"query": "多少论文", "raw_tool": "get_library_status"}},
    )
    assert events[1][0] == "agent_step"
    assert events[1][1]["index"] == 1
    assert events[1][1]["arguments"] == {"query": "DPO"}
    assert "DPO" in str(events[1][1]["message"])
