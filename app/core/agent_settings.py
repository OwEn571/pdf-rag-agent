from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentSettings:
    max_agent_steps: int = 8
    max_parallel_tools: int = 4
    max_calls_per_tool: int = 3
    confidence_floor: float = 0.6
    answer_logprobs_enabled: bool = False
    answer_logprobs_min_tokens: int = 3
    answer_self_consistency_enabled: bool = False
    answer_self_consistency_samples: int = 3
    generic_claim_solver_enabled: bool = False
    generic_claim_solver_shadow_enabled: bool = False
    # Comma-separated list of goal names to keep on the blocked list.
    # Empty string means "use the default blocked set" (see schema_claims.py).
    generic_claim_solver_blocked_goals: str = ""
    llm_driven_loop_enabled: bool = False
    dynamic_tools_enabled: bool = False
    dynamic_tool_deployment_id: str = "local"
    dynamic_tool_timeout_seconds: float = 2.0
    dynamic_tool_memory_mb: int = 256
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
            max_calls_per_tool=_bounded_int(
                getattr(settings, "agent_max_calls_per_tool", defaults.max_calls_per_tool),
                default=defaults.max_calls_per_tool,
                minimum=1,
                maximum=12,
            ),
            confidence_floor=_bounded_float(
                getattr(settings, "agent_confidence_floor", defaults.confidence_floor),
                default=defaults.confidence_floor,
                minimum=0.0,
                maximum=1.0,
            ),
            answer_logprobs_enabled=_bounded_bool(
                getattr(settings, "agent_answer_logprobs_enabled", defaults.answer_logprobs_enabled),
                default=defaults.answer_logprobs_enabled,
            ),
            answer_logprobs_min_tokens=_bounded_int(
                getattr(settings, "agent_answer_logprobs_min_tokens", defaults.answer_logprobs_min_tokens),
                default=defaults.answer_logprobs_min_tokens,
                minimum=1,
                maximum=512,
            ),
            answer_self_consistency_enabled=_bounded_bool(
                getattr(
                    settings,
                    "agent_answer_self_consistency_enabled",
                    defaults.answer_self_consistency_enabled,
                ),
                default=defaults.answer_self_consistency_enabled,
            ),
            answer_self_consistency_samples=_bounded_int(
                getattr(
                    settings,
                    "agent_answer_self_consistency_samples",
                    defaults.answer_self_consistency_samples,
                ),
                default=defaults.answer_self_consistency_samples,
                minimum=2,
                maximum=5,
            ),
            generic_claim_solver_enabled=_bounded_bool(
                getattr(settings, "agent_generic_claim_solver_enabled", defaults.generic_claim_solver_enabled),
                default=defaults.generic_claim_solver_enabled,
            ),
            generic_claim_solver_shadow_enabled=_bounded_bool(
                getattr(
                    settings,
                    "agent_generic_claim_solver_shadow_enabled",
                    defaults.generic_claim_solver_shadow_enabled,
                ),
                default=defaults.generic_claim_solver_shadow_enabled,
            ),
            generic_claim_solver_blocked_goals=_bounded_string(
                getattr(
                    settings,
                    "agent_generic_claim_solver_blocked_goals",
                    defaults.generic_claim_solver_blocked_goals,
                ),
                default=defaults.generic_claim_solver_blocked_goals,
                maximum=512,
            ),
            llm_driven_loop_enabled=_bounded_bool(
                getattr(settings, "agent_llm_driven_loop_enabled", defaults.llm_driven_loop_enabled),
                default=defaults.llm_driven_loop_enabled,
            ),
            dynamic_tools_enabled=_bounded_bool(
                getattr(settings, "agent_dynamic_tools_enabled", defaults.dynamic_tools_enabled),
                default=defaults.dynamic_tools_enabled,
            ),
            dynamic_tool_deployment_id=_bounded_string(
                getattr(settings, "agent_dynamic_tool_deployment_id", defaults.dynamic_tool_deployment_id),
                default=defaults.dynamic_tool_deployment_id,
                maximum=128,
            ),
            dynamic_tool_timeout_seconds=_bounded_float(
                getattr(settings, "agent_dynamic_tool_timeout_seconds", defaults.dynamic_tool_timeout_seconds),
                default=defaults.dynamic_tool_timeout_seconds,
                minimum=0.05,
                maximum=30.0,
            ),
            dynamic_tool_memory_mb=_bounded_int(
                getattr(settings, "agent_dynamic_tool_memory_mb", defaults.dynamic_tool_memory_mb),
                default=defaults.dynamic_tool_memory_mb,
                minimum=64,
                maximum=2048,
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


def _bounded_string(value: Any, *, default: str, maximum: int) -> str:
    text = " ".join(str(value or "").split())[:maximum]
    return text or default


def _bounded_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _bounded_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default
