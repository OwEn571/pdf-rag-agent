from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
