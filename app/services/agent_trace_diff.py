from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TraceDiff:
    ok: bool
    differences: list[str]
    expected_signature: list[dict[str, Any]]
    actual_signature: list[dict[str, Any]]


def load_agent_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid trace json at {path}:{line_number}") from exc
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def diff_agent_traces(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    *,
    max_differences: int = 20,
) -> TraceDiff:
    expected_signature = trace_signature(expected)
    actual_signature = trace_signature(actual)
    differences = _signature_differences(
        expected_signature=expected_signature,
        actual_signature=actual_signature,
        max_differences=max_differences,
    )
    return TraceDiff(
        ok=not differences,
        differences=differences,
        expected_signature=expected_signature,
        actual_signature=actual_signature,
    )


def diff_agent_trace_files(expected_path: Path, actual_path: Path, *, max_differences: int = 20) -> TraceDiff:
    return diff_agent_traces(
        load_agent_trace(expected_path),
        load_agent_trace(actual_path),
        max_differences=max_differences,
    )


def trace_signature(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signature: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        event = str(row.get("event", "") or "message")
        data = row.get("data", {})
        if not isinstance(data, dict):
            data = {}
        signature.append(_event_signature(event=event, data=data))
    return signature


def _event_signature(*, event: str, data: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {"event": event}
    event_type = str(data.get("type", "") or "")
    if event_type:
        item["type"] = event_type
    name = str(data.get("name", "") or data.get("tool", "") or "")
    if name:
        item["name"] = name
    if "ok" in data:
        item["ok"] = bool(data.get("ok"))
    if "count" in data:
        item["count"] = _safe_int(data.get("count"))
    status = str(data.get("status", "") or "")
    if status:
        item["status"] = status
    if event == "verification" and isinstance(data.get("recommended_action"), str):
        item["recommended_action"] = data["recommended_action"]
    if event == "contract":
        item["interaction_mode"] = str(data.get("interaction_mode", "") or "")
        item["relation"] = str(data.get("relation", "") or "")
        notes = _notes(data.get("notes"))
        router_action = _note_value(notes=notes, prefix="router_action=")
        router_tags = _note_values(notes=notes, prefix="router_tag=")
        if router_action:
            item["router_action"] = router_action
        if router_tags:
            item["router_tags"] = router_tags
        intent_kind = _note_value(notes=notes, prefix="intent_kind=")
        if intent_kind:
            item["intent_kind"] = intent_kind
    if event == "confidence":
        item["basis"] = str(data.get("basis", "") or "")
        item["score_bucket"] = _confidence_score_bucket(data.get("value", data.get("score")))
    if event == "ask_human":
        item["question"] = str(data.get("question", "") or "")[:200]
        item["options_count"] = len(data.get("options", [])) if isinstance(data.get("options"), list) else 0
    if event == "todo_update":
        item["todo_items"] = _todo_items_signature(data.get("items"))
    if event == "final":
        item["execution_nodes"] = _execution_nodes(data.get("execution_steps"))
        if "answer_chars" in data:
            item["answer_chars_bucket"] = _answer_chars_bucket(data.get("answer_chars"))
    return item


def _signature_differences(
    *,
    expected_signature: list[dict[str, Any]],
    actual_signature: list[dict[str, Any]],
    max_differences: int,
) -> list[str]:
    differences: list[str] = []
    if len(expected_signature) != len(actual_signature):
        differences.append(f"event_count expected={len(expected_signature)} actual={len(actual_signature)}")
    for index, (expected, actual) in enumerate(zip(expected_signature, actual_signature), start=1):
        if expected == actual:
            continue
        differences.append(f"event[{index}] expected={expected} actual={actual}")
        if len(differences) >= max_differences:
            break
    return differences


def _execution_nodes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    nodes: list[str] = []
    for item in value:
        if isinstance(item, dict):
            node = str(item.get("node", "") or "").strip()
            if node:
                nodes.append(node)
    return nodes


def _todo_items_signature(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, str]] = []
    for raw_item in value[:12]:
        if not isinstance(raw_item, dict):
            continue
        item_id = str(raw_item.get("id", "") or "").strip()
        text = " ".join(str(raw_item.get("text", "") or "").split())[:160]
        status = str(raw_item.get("status", "") or "").strip()
        if item_id or text or status:
            items.append({"id": item_id, "text": text, "status": status})
    return items


def _answer_chars_bucket(value: Any) -> str:
    count = _safe_int(value)
    if count == 0:
        return "0"
    if count < 400:
        return "<400"
    if count < 1200:
        return "<1200"
    return ">=1200"


def _confidence_score_bucket(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if score < 0.4:
        return "<0.4"
    if score < 0.6:
        return "<0.6"
    if score < 0.8:
        return "<0.8"
    return ">=0.8"


def _notes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _note_value(*, notes: list[str], prefix: str) -> str:
    for note in notes:
        if note.startswith(prefix):
            return note.split("=", 1)[1]
    return ""


def _note_values(*, notes: list[str], prefix: str) -> list[str]:
    return [note.split("=", 1)[1] for note in notes if note.startswith(prefix)]


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
