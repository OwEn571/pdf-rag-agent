from __future__ import annotations

import hashlib
from typing import Any

from app.domain.models import EvidenceBlock, QueryContract, SessionContext
from app.services.contract_context import contract_answer_slots, note_value
from app.services.evidence_tools import evidence_from_payload, verify_claim_against_evidence
from app.services.url_fetcher import FetchUrlResult


def tool_input_from_state(state: dict[str, Any], name: str) -> dict[str, Any]:
    tool_inputs = state.get("tool_inputs", {})
    if not isinstance(tool_inputs, dict):
        return {}
    payload = tool_inputs.get(name, {})
    return dict(payload) if isinstance(payload, dict) else {}


def conversation_intent_summary(contract: QueryContract) -> dict[str, Any]:
    notes = [str(item) for item in contract.notes]
    intent_kind = note_value(notes=notes, prefix="intent_kind=") or contract.interaction_mode
    return {
        "kind": intent_kind,
        "answer_slots": contract_answer_slots(contract),
        "requested_fields": contract.requested_fields,
        "targets": contract.targets,
    }


def research_intent_summary(contract: QueryContract) -> tuple[str, dict[str, Any]]:
    answer_slots = contract_answer_slots(contract)
    summary = "/".join(answer_slots or contract.requested_fields or [contract.interaction_mode])
    payload = {
        "answer_slots": answer_slots,
        "requested_fields": contract.requested_fields,
        "required_modalities": contract.required_modalities,
        "targets": contract.targets,
        "continuation_mode": contract.continuation_mode,
    }
    return summary, payload


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


def fetch_url_payload(result: FetchUrlResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "url": result.url,
        "title": result.title,
        "text": result.text,
        "error": result.error,
        "status_code": result.status_code,
    }


def fetch_url_evidence(result: FetchUrlResult) -> EvidenceBlock | None:
    if not result.ok:
        return None
    doc_id = "web::fetch::" + hashlib.sha1(result.url.encode("utf-8")).hexdigest()[:16]
    return EvidenceBlock(
        doc_id=doc_id,
        paper_id=doc_id,
        title=result.title or result.url,
        file_path=result.url,
        page=0,
        block_type="web",
        caption=result.url,
        snippet=result.text[:1600],
        score=0.75,
        metadata={"source": "fetch_url", "url": result.url, "status_code": result.status_code},
    )


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


def verify_claim_tool_payload(*, planned_input: dict[str, Any], state: dict[str, Any]) -> tuple[dict[str, Any], str]:
    claim = str(planned_input.get("claim", "") or "").strip()
    evidence = evidence_from_payload(planned_input.get("evidence", [])) or evidence_blocks_from_state(state)
    min_overlap = coerce_int(planned_input.get("min_overlap", 2), default=2, minimum=1, maximum=20)
    check = verify_claim_against_evidence(claim=claim, evidence=evidence, min_overlap=min_overlap)
    payload = {
        "claim": claim,
        "status": check.status,
        "confidence": check.confidence,
        "supporting_evidence_ids": check.supporting_evidence_ids,
        "matched_terms": check.matched_terms,
        "missing_terms": check.missing_terms,
        "min_overlap": min_overlap,
        "reason": check.reason,
    }
    return payload, f"{check.status}:{check.confidence:.2f}"
