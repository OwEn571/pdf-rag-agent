from __future__ import annotations

from typing import Any

from app.services.agent_metrics import record_agent_event


def normalize_agent_event(event: str, data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(data)
    payload.setdefault("type", _event_type(event))
    if event == "tool_call":
        tool = str(payload.get("tool", "") or payload.get("name", "") or "").strip()
        arguments = payload.get("arguments", payload.get("input", {}))
        payload.setdefault("id", _event_id(event=event, name=tool))
        payload.setdefault("name", tool)
        payload.setdefault("input", arguments if isinstance(arguments, dict) else {})
    elif event == "observation":
        tool = str(payload.get("tool", "") or payload.get("name", "") or "").strip()
        output = payload.get("payload", payload.get("output", {}))
        payload.setdefault("id", _event_id(event=event, name=tool))
        payload.setdefault("name", tool)
        payload.setdefault("ok", True)
        payload.setdefault("output", output if isinstance(output, dict) else {"value": output})
        payload.setdefault("took_ms", None)
    elif event == "answer_delta":
        text = str(payload.get("text", payload.get("delta", payload.get("content", ""))) or "")
        payload.setdefault("text", text)
    elif event == "agent_step":
        payload.setdefault("status", "doing")
    elif event == "error":
        payload.setdefault("message", str(payload.get("message", "") or "agent error"))
    record_agent_event(event, payload)
    return payload


def _event_type(event: str) -> str:
    return {
        "tool_call": "tool_use",
        "observation": "tool_result",
        "answer_delta": "answer_delta",
        "final": "final",
        "error": "error",
    }.get(event, event)


def _event_id(*, event: str, name: str) -> str:
    return f"{event}:{name or 'unknown'}"
