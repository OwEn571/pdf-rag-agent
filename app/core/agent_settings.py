from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentSettings:
    max_agent_steps: int = 8
    max_parallel_tools: int = 4
    confidence_floor: float = 0.6
    max_clarification_attempts: int = 2
    disambiguation_auto_resolve_threshold: float = 0.85
    disambiguation_recommend_threshold: float = 0.65

    @classmethod
    def from_settings(cls, settings: Any) -> "AgentSettings":
        defaults = cls()
        return cls(
            max_agent_steps=_bounded_int(
                getattr(settings, "agent_max_steps", defaults.max_agent_steps),
                default=defaults.max_agent_steps,
                minimum=1,
                maximum=64,
            ),
            max_parallel_tools=_bounded_int(
                getattr(settings, "agent_max_parallel_tools", defaults.max_parallel_tools),
                default=defaults.max_parallel_tools,
                minimum=1,
                maximum=16,
            ),
            confidence_floor=_bounded_float(
                getattr(settings, "agent_confidence_floor", defaults.confidence_floor),
                default=defaults.confidence_floor,
                minimum=0.0,
                maximum=1.0,
            ),
            max_clarification_attempts=_bounded_int(
                getattr(settings, "agent_max_clarification_attempts", defaults.max_clarification_attempts),
                default=defaults.max_clarification_attempts,
                minimum=0,
                maximum=10,
            ),
            disambiguation_auto_resolve_threshold=_bounded_float(
                getattr(
                    settings,
                    "agent_disambiguation_auto_resolve_threshold",
                    defaults.disambiguation_auto_resolve_threshold,
                ),
                default=defaults.disambiguation_auto_resolve_threshold,
                minimum=0.0,
                maximum=1.0,
            ),
            disambiguation_recommend_threshold=_bounded_float(
                getattr(
                    settings,
                    "agent_disambiguation_recommend_threshold",
                    defaults.disambiguation_recommend_threshold,
                ),
                default=defaults.disambiguation_recommend_threshold,
                minimum=0.0,
                maximum=1.0,
            ),
        )


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _bounded_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))
