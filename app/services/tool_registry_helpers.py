from __future__ import annotations

from typing import Any

from app.domain.models import EvidenceBlock, SessionContext
from app.services.evidence_tools import evidence_from_payload


def normalize_todo_items(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, str]] = []
    allowed_statuses = {"pending", "doing", "done", "cancelled"}
    for index, raw in enumerate(value, start=1):
        if not isinstance(raw, dict):
            continue
        text = " ".join(str(raw.get("text", "") or "").split())
        if not text:
            continue
        item_id = " ".join(str(raw.get("id", "") or "").split()) or f"todo-{index}"
        status = str(raw.get("status", "") or "pending").strip()
        if status not in allowed_statuses:
            status = "pending"
        items.append({"id": item_id, "text": text, "status": status})
    return items


def store_session_todos(session: SessionContext, items: list[dict[str, str]]) -> None:
    memory = dict(session.working_memory or {})
    memory["todos"] = items
    session.working_memory = memory


def format_task_results_answer(task_results: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for index, result in enumerate(task_results, start=1):
        prompt = str(result.get("prompt", "") or f"子任务 {index}").strip()
        answer = str(result.get("answer", "") or "").strip()
        if not answer:
            continue
        sections.append(f"## {index}. {prompt}\n\n{answer}")
    return "\n\n".join(sections).strip()


def format_fetched_urls_answer(fetched_urls: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for item in fetched_urls:
        url = str(item.get("url", "") or "").strip()
        if not url:
            continue
        if not bool(item.get("ok")):
            sections.append(f"- `{url}`：读取失败（{item.get('error', 'unknown_error')}）")
            continue
        title = str(item.get("title", "") or url).strip()
        text = str(item.get("text", "") or "").strip()
        sections.append(f"### {title}\n\n来源：{url}\n\n{text}")
    return "\n\n".join(sections).strip()


def format_summaries_answer(summaries: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        str(item.get("summary", "") or "").strip()
        for item in summaries
        if str(item.get("summary", "") or "").strip()
    )


def focus_values(raw: Any, fallback: list[str]) -> list[str]:
    values = raw if isinstance(raw, list) else fallback
    return [str(item).strip() for item in list(values or []) if str(item).strip()]


def evidence_blocks_from_state(state: dict[str, Any]) -> list[EvidenceBlock]:
    evidence: list[EvidenceBlock] = []
    for key in ("evidence", "web_evidence"):
        for item in list(state.get(key, []) or []):
            if isinstance(item, EvidenceBlock):
                evidence.append(item)
    evidence.extend(evidence_from_payload(state.get("fetched_urls", [])))
    return list({item.doc_id or item.snippet: item for item in evidence}.values())


def summary_source_from_state(state: dict[str, Any]) -> str:
    fetched_text = "\n".join(str(item.get("text", "") or "") for item in list(state.get("fetched_urls", []) or []))
    task_text = "\n".join(str(item.get("answer", "") or "") for item in list(state.get("task_results", []) or []))
    return "\n".join(part for part in [fetched_text, task_text] if part.strip())


def coerce_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))
