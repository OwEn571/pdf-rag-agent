from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.domain.models import QueryContract
from app.services.agent_step_messages import agent_step_message
from app.services.agent_tools import all_agent_tool_names
from app.services.contract_context import LEGACY_TOOL_NAME_ALIASES, canonical_agent_tool
from app.services.agent_events import normalize_agent_event
from app.services.agent_trace import write_agent_trace

AgentEventCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class AgentEventRecorder:
    callback: AgentEventCallback | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, event: str, data: dict[str, Any]) -> None:
        item = {"event": event, "data": normalize_agent_event(event, data)}
        self.events.append(item)
        if self.callback is not None:
            self.callback(item)


def write_turn_trace_safe(
    *,
    enabled: bool,
    data_dir: Path,
    session_id: str,
    events: list[dict[str, Any]],
    final_payload: dict[str, Any],
    execution_steps: list[dict[str, Any]],
    logger: logging.Logger | None = None,
) -> Path | None:
    if not enabled:
        return None
    try:
        return write_agent_trace(
            data_dir=data_dir,
            session_id=session_id,
            events=events,
            final_payload=final_payload,
            execution_steps=execution_steps,
        )
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.warning("failed to write agent trace: %s", exc)
        return None


def canonical_agent_event_tool(tool: str) -> str:
    return canonical_agent_tool(
        tool=tool,
        aliases=LEGACY_TOOL_NAME_ALIASES,
        canonical_names=all_agent_tool_names(),
    )


def record_agent_observation(
    *,
    emit: Callable[[str, dict[str, Any]], None],
    execution_steps: list[dict[str, Any]],
    tool: str,
    summary: str,
    payload: dict[str, Any],
) -> None:
    canonical_tool = canonical_agent_event_tool(tool)
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
) -> None:
    canonical_tool = canonical_agent_event_tool(tool)
    event_arguments = dict(arguments)
    if canonical_tool != tool:
        event_arguments.setdefault("raw_tool", tool)
    emit("tool_call", {"tool": canonical_tool, "arguments": event_arguments})


def emit_agent_step(
    *,
    emit: Callable[[str, dict[str, Any]], None],
    index: int,
    action: str,
    contract: QueryContract,
    arguments: dict[str, Any] | None = None,
) -> None:
    emit(
        "agent_step",
        {
            "index": index,
            "action": action,
            "arguments": dict(arguments or {}),
            "message": agent_step_message(action=action, contract=contract),
        },
    )
