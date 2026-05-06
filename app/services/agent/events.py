from __future__ import annotations

from typing import Any

from app.services.agent.metrics import record_agent_event
from app.services.tools.registry_helpers import normalize_todo_items


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
    elif event in {"thinking_delta", "answer_delta"}:
        text = str(payload.get("text", payload.get("delta", payload.get("content", ""))) or "")
        payload.setdefault("text", text)
    elif event == "ask_human":
        payload.setdefault("question", str(payload.get("question", "") or ""))
        options = payload.get("options", [])
        payload["options"] = options if isinstance(options, list) else []
        payload.setdefault("reason", str(payload.get("reason", "") or ""))
    elif event == "todo_update":
        payload["items"] = normalize_todo_items(payload.get("items", []))
    elif event == "plan":
        payload.setdefault("payload", dict(data))
        payload.setdefault("items", _plan_items(payload))
    elif event == "verification":
        payload.setdefault("payload", dict(data))
        payload.setdefault("status", str(payload.get("status", "") or "unknown"))
    elif event == "confidence":
        payload.setdefault("value", _confidence_value(payload))
        payload.setdefault("basis", str(payload.get("basis", "") or "unknown"))
    elif event == "final":
        payload.setdefault("answer", str(payload.get("answer", "") or ""))
        citations = payload.get("citations", [])
        payload["citations"] = citations if isinstance(citations, list) else []
        usage = payload.get("usage", {})
        payload["usage"] = usage if isinstance(usage, dict) else {}
    elif event == "agent_step":
        payload.setdefault("status", "doing")
    elif event == "error":
        payload.setdefault("message", str(payload.get("message", "") or "agent error"))
    elif event == "solver_selection":
        payload.setdefault("selected", str(payload.get("selected", "") or "unknown"))
    record_agent_event(event, payload)
    return payload


def _event_type(event: str) -> str:
    return {
        "tool_call": "tool_use",
        "observation": "tool_result",
        "thinking_delta": "thinking_delta",
        "answer_delta": "answer_delta",
        "ask_human": "ask_human",
        "todo_update": "todo_update",
        "plan": "plan",
        "verification": "verification",
        "confidence": "confidence",
        "solver_selection": "solver_selection",
        "solver_shadow": "solver_shadow",
        "final": "final",
        "error": "error",
    }.get(event, event)


def _event_id(*, event: str, name: str) -> str:
    return f"{event}:{name or 'unknown'}"


def _plan_items(payload: dict[str, Any]) -> list[dict[str, str]]:
    sequence = payload.get("solver_sequence", payload.get("actions", []))
    if not isinstance(sequence, list):
        return []
    items: list[dict[str, str]] = []
    for index, raw_item in enumerate(sequence, start=1):
        text = str(raw_item).strip()
        if not text:
            continue
        items.append({"id": f"plan-{index}", "text": text, "status": "pending"})
    return items


def _confidence_value(payload: dict[str, Any]) -> float | None:
    value = payload.get("value", payload.get("score"))
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, parsed))
