from __future__ import annotations

from typing import Any, Callable

from app.services.agent.tools import all_agent_tool_names
from app.services.contracts.context import canonical_agent_tool


def canonical_agent_event_tool(tool: str, *, canonical_names: set[str] | None = None) -> str:
    return canonical_agent_tool(
        tool=tool,
        aliases={},
        canonical_names=canonical_names or all_agent_tool_names(),
    )


def record_agent_observation(
    *,
    emit: Callable[[str, dict[str, Any]], None],
    execution_steps: list[dict[str, Any]],
    tool: str,
    summary: str,
    payload: dict[str, Any],
    canonical_names: set[str] | None = None,
) -> None:
    canonical_tool = canonical_agent_event_tool(tool, canonical_names=canonical_names)
    event_payload = dict(payload)
    if canonical_tool != tool:
        event_payload.setdefault("raw_tool", tool)
    emit("observation", {"tool": canonical_tool, "summary": summary, "payload": event_payload})
    execution_steps.append({"node": f"agent_tool:{canonical_tool}", "summary": summary})


def emit_agent_tool_call(
    *,
    emit: Callable[[str, dict[str, Any]], None],
    tool: str,
    arguments: dict[str, Any],
    canonical_names: set[str] | None = None,
) -> None:
    canonical_tool = canonical_agent_event_tool(tool, canonical_names=canonical_names)
    event_arguments = dict(arguments)
    if canonical_tool != tool:
        event_arguments.setdefault("raw_tool", tool)
    emit("tool_call", {"tool": canonical_tool, "arguments": event_arguments})
