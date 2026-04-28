from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from app.domain.models import SessionContext
from app.services.agent_emit import AgentEventCallback, AgentEventRecorder


@dataclass(slots=True)
class AgentRunContext:
    session_id: str
    session: SessionContext
    event_recorder: AgentEventRecorder
    execution_steps: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        session: SessionContext,
        event_callback: AgentEventCallback | None = None,
    ) -> "AgentRunContext":
        return cls(
            session_id=session_id,
            session=session,
            event_recorder=AgentEventRecorder(callback=event_callback),
        )

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.event_recorder.events

    @property
    def emit(self) -> Callable[[str, dict[str, Any]], None]:
        return self.event_recorder.emit
