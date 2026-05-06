from __future__ import annotations

from app.domain.models import SessionContext
from app.services.agent.context import AgentRunContext


def test_agent_run_context_collects_events_and_steps() -> None:
    forwarded: list[dict[str, object]] = []
    context = AgentRunContext.create(
        session_id="demo",
        session=SessionContext(session_id="demo"),
        event_callback=forwarded.append,
    )

    context.emit("observation", {"tool": "compose", "summary": "done", "payload": {"answer": "ok"}})
    context.execution_steps.append({"node": "agent_tool:compose", "summary": "done"})

    assert context.session_id == "demo"
    assert context.events == forwarded
    assert context.events[0]["data"]["type"] == "tool_result"
    assert context.events[0]["data"]["name"] == "compose"
    assert context.execution_steps == [{"node": "agent_tool:compose", "summary": "done"}]
