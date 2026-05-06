from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.contracts.context import note_value, note_values


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
    name = str(data.get("name", "") or data.get("tool", "") or data.get("action", "") or "")
    if name:
        item["name"] = name
    if "ok" in data:
        item["ok"] = bool(data.get("ok"))
    if event == "tool_call" or event_type == "tool_use":
        tool_input = _tool_input_signature(data.get("input", data.get("arguments")))
        if tool_input:
            item["input"] = tool_input
    if event == "agent_step":
        tool_input = _tool_input_signature(data.get("arguments"))
        if tool_input:
            item["arguments"] = tool_input
    if "count" in data:
        item["count"] = _safe_int(data.get("count"))
    status = str(data.get("status", "") or "")
    if status:
        item["status"] = status
    if event == "verification" and isinstance(data.get("recommended_action"), str):
        item["recommended_action"] = data["recommended_action"]
    if event == "plan":
        item["solver_sequence"] = _string_list(data.get("solver_sequence"))
        item["required_claims"] = _string_list(data.get("required_claims"))
    if event == "claims":
        item["claim_signature"] = _claims_signature(data.get("items"))
    if event == "solver_shadow":
        item["solver_shadow_signature"] = _solver_shadow_signature(data)
    if event in {"evidence", "web_search"}:
        item["evidence_signature"] = _evidence_signature(data.get("items"))
    if event == "contract":
        item["interaction_mode"] = str(data.get("interaction_mode", "") or "")
        item["relation"] = str(data.get("relation", "") or "")
        notes = _notes(data.get("notes"))
        router_action = note_value(notes=notes, prefix="router_action=")
        router_tags = note_values(notes=notes, prefix="router_tag=")
        if router_action:
            item["router_action"] = router_action
        if router_tags:
            item["router_tags"] = router_tags
        intent_kind = note_value(notes=notes, prefix="intent_kind=")
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
        item["citation_signature"] = _citations_signature(data.get("citations"))
        item["runtime_summary_signature"] = _runtime_summary_signature(data.get("runtime_summary"))
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


def _tool_input_signature(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    signature: dict[str, Any] = {}
    for key in sorted(value)[:16]:
        key_text = str(key).strip()
        if not key_text:
            continue
        signature[key_text] = _compact_json_value(value[key], depth=0)
    return signature


def _compact_json_value(value: Any, *, depth: int) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return _compact_text(value, limit=180)
    if depth >= 2:
        return _compact_text(value, limit=120)
    if isinstance(value, list):
        return [_compact_json_value(item, depth=depth + 1) for item in value[:8]]
    if isinstance(value, dict):
        return {
            str(key).strip(): _compact_json_value(value[key], depth=depth + 1)
            for key in sorted(value)[:12]
            if str(key).strip()
        }
    return _compact_text(value, limit=120)


def _claims_signature(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    signature: list[dict[str, Any]] = []
    for raw_item in value[:16]:
        if not isinstance(raw_item, dict):
            continue
        structured_data = raw_item.get("structured_data")
        if not isinstance(structured_data, dict):
            structured_data = {}
        item: dict[str, Any] = {
            "type": str(raw_item.get("claim_type", "") or ""),
            "entity": _compact_text(raw_item.get("entity"), limit=120),
            "value": _compact_text(raw_item.get("value"), limit=160),
            "paper_ids": _string_list(raw_item.get("paper_ids"))[:8],
            "evidence_ids": _string_list(raw_item.get("evidence_ids"))[:12],
        }
        source = str(structured_data.get("source", "") or "")
        if source:
            item["source"] = source
        signature.append(item)
    return signature


def _solver_shadow_signature(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "selected": str(value.get("selected", "") or ""),
        "schema": _solver_shadow_claim_summary(value.get("schema")),
        "deterministic": _solver_shadow_claim_summary(value.get("deterministic")),
    }


def _solver_shadow_claim_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"count": 0, "types": [], "paper_ids": [], "evidence_ids": [], "sources": {}}
    sources = value.get("sources")
    if not isinstance(sources, dict):
        sources = {}
    return {
        "count": _safe_int(value.get("count")),
        "types": _string_list(value.get("types"))[:12],
        "paper_ids": _string_list(value.get("paper_ids"))[:12],
        "evidence_ids": _string_list(value.get("evidence_ids"))[:16],
        "sources": {str(key): _safe_int(sources[key]) for key in sorted(sources)},
    }


def _evidence_signature(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    signature: list[dict[str, Any]] = []
    for raw_item in value[:16]:
        if not isinstance(raw_item, dict):
            continue
        metadata = raw_item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        item: dict[str, Any] = {
            "doc_id": str(raw_item.get("doc_id", "") or ""),
            "paper_id": str(raw_item.get("paper_id", "") or ""),
            "title": _compact_text(raw_item.get("title"), limit=140),
            "page": _safe_int(raw_item.get("page")),
            "block_type": str(raw_item.get("block_type", "") or ""),
            "snippet": _compact_text(raw_item.get("snippet"), limit=180),
        }
        caption = _compact_text(raw_item.get("caption"), limit=120)
        if caption:
            item["caption"] = caption
        source = str(metadata.get("source", "") or metadata.get("provider", "") or "")
        if source:
            item["source"] = source
        path = _compact_text(raw_item.get("file_path"), limit=160)
        if path:
            item["path"] = path
        signature.append(item)
    return signature


def _citations_signature(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    signature: list[dict[str, Any]] = []
    for raw_item in value[:12]:
        if not isinstance(raw_item, dict):
            continue
        item: dict[str, Any] = {
            "doc_id": str(raw_item.get("doc_id", "") or ""),
            "paper_id": str(raw_item.get("paper_id", "") or ""),
            "title": _compact_text(raw_item.get("title"), limit=140),
            "page": _safe_int(raw_item.get("page")),
            "block_type": str(raw_item.get("block_type", "") or ""),
            "snippet": _compact_text(raw_item.get("snippet"), limit=160),
        }
        path = _compact_text(raw_item.get("file_path"), limit=160)
        if path:
            item["path"] = path
        signature.append(item)
    return signature


def _runtime_summary_signature(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    grounding = value.get("grounding")
    if not isinstance(grounding, dict):
        grounding = {}
    answer_generation = value.get("answer_generation")
    if not isinstance(answer_generation, dict):
        answer_generation = {}
    answer_confidence = answer_generation.get("confidence")
    if not isinstance(answer_confidence, dict):
        answer_confidence = {}
    claim_sources = grounding.get("claim_sources")
    if not isinstance(claim_sources, dict):
        claim_sources = {}
    return {
        "verification_status": str(grounding.get("verification_status", "") or ""),
        "claim_count": _safe_int(grounding.get("claim_count")),
        "citation_count": _safe_int(grounding.get("citation_count")),
        "claim_sources": {str(key): _safe_int(claim_sources[key]) for key in sorted(claim_sources)},
        "answer_confidence_basis": str(answer_confidence.get("basis", "") or ""),
        "answer_confidence_bucket": _confidence_score_bucket(answer_confidence.get("score")),
    }


def _compact_text(value: Any, *, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
