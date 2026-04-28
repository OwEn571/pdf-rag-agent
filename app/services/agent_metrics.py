from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

try:
    from prometheus_client import Counter, Histogram, REGISTRY
except Exception:  # noqa: BLE001
    Counter = None
    Histogram = None
    REGISTRY = None


def _existing_collector(name: str) -> Any | None:
    if REGISTRY is None:
        return None
    collectors = getattr(REGISTRY, "_names_to_collectors", {})
    if not isinstance(collectors, dict):
        return None
    return collectors.get(name)


def _counter(name: str, documentation: str, labelnames: list[str]) -> Any | None:
    if Counter is None:
        return None
    existing = _existing_collector(name)
    if existing is not None:
        return existing
    try:
        return Counter(name, documentation, labelnames)
    except ValueError:
        return _existing_collector(name)


def _histogram(name: str, documentation: str, labelnames: list[str]) -> Any | None:
    if Histogram is None:
        return None
    existing = _existing_collector(name)
    if existing is not None:
        return existing
    try:
        return Histogram(name, documentation, labelnames)
    except ValueError:
        return _existing_collector(name)


TOOL_CALLS_TOTAL = _counter(
    "tool_calls_total",
    "Agent tool calls by tool name and success status.",
    ["name", "ok"],
)
TOOL_LATENCY_SECONDS = _histogram(
    "tool_latency_seconds",
    "Agent tool execution latency in seconds by tool name.",
    ["name"],
)
_ACTIVE_TOOL_EXECUTION: ContextVar[int] = ContextVar("active_agent_tool_execution", default=0)


def metrics_available() -> bool:
    return TOOL_CALLS_TOTAL is not None and TOOL_LATENCY_SECONDS is not None


def record_agent_event(event: str, payload: dict[str, Any]) -> None:
    if event != "observation":
        return
    if _inside_tool_execution():
        return
    if str(payload.get("summary", "") or "") == "tool_loop_ready":
        return
    name = _tool_name(payload)
    if not name:
        return
    record_tool_call(name=name, ok=_ok_label(payload.get("ok", True)))
    took_ms = payload.get("took_ms")
    if isinstance(took_ms, (int, float)) and took_ms >= 0:
        record_tool_latency(name=name, seconds=float(took_ms) / 1000.0)


def record_tool_execution(*, name: str, ok: bool, elapsed_seconds: float) -> None:
    tool_name = str(name or "unknown").strip() or "unknown"
    record_tool_call(name=tool_name, ok="true" if ok else "false")
    if elapsed_seconds >= 0:
        record_tool_latency(name=tool_name, seconds=elapsed_seconds)


def begin_tool_execution() -> Token[int]:
    return _ACTIVE_TOOL_EXECUTION.set(_ACTIVE_TOOL_EXECUTION.get() + 1)


def end_tool_execution(token: Token[int]) -> None:
    _ACTIVE_TOOL_EXECUTION.reset(token)


def record_tool_call(*, name: str, ok: str) -> None:
    if TOOL_CALLS_TOTAL is None:
        return
    try:
        TOOL_CALLS_TOTAL.labels(name=name, ok=ok).inc()
    except Exception:  # noqa: BLE001
        return


def record_tool_latency(*, name: str, seconds: float) -> None:
    if TOOL_LATENCY_SECONDS is None:
        return
    try:
        TOOL_LATENCY_SECONDS.labels(name=name).observe(seconds)
    except Exception:  # noqa: BLE001
        return


def _tool_name(payload: dict[str, Any]) -> str:
    return str(payload.get("name", "") or payload.get("tool", "") or "unknown").strip()


def _ok_label(value: Any) -> str:
    if isinstance(value, str):
        return "false" if value.strip().lower() in {"0", "false", "no", "n"} else "true"
    return "true" if bool(value) else "false"


def _inside_tool_execution() -> bool:
    return _ACTIVE_TOOL_EXECUTION.get() > 0
