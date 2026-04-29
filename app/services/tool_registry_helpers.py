from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from app.domain.models import EvidenceBlock, QueryContract, SessionContext, VerificationReport
from app.services.contract_context import contract_answer_slots, note_value
from app.services.evidence_tools import (
    evidence_from_payload,
    summarize_evidence,
    summarize_text,
    verify_claim_against_evidence,
)
from app.services.learnings import remember_learning
from app.services.proposed_tools import propose_tool as record_tool_proposal
from app.services.url_fetcher import FetchUrlResult
from app.services.web_evidence import merge_evidence


def tool_input_from_state(state: dict[str, Any], name: str) -> dict[str, Any]:
    tool_inputs = state.get("tool_inputs", {})
    if not isinstance(tool_inputs, dict):
        return {}
    payload = tool_inputs.get(name, {})
    return dict(payload) if isinstance(payload, dict) else {}


def tool_inputs_by_name(agent_plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_items = agent_plan.get("tool_call_args", []) if isinstance(agent_plan, dict) else []
    if not isinstance(raw_items, list):
        return {}
    tool_inputs: dict[str, dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        args = item.get("args", {})
        if not name or not isinstance(args, dict):
            continue
        tool_inputs.setdefault(name, dict(args))
    return tool_inputs


def planned_tool_input_from_state(state: dict[str, Any], name: str) -> dict[str, Any]:
    planned = tool_input_from_state(state, name)
    if planned:
        return planned
    current = state.get("current_tool_input", {})
    return dict(current) if isinstance(current, dict) else {}


def tool_loop_ready_observation(
    *,
    tool: str,
    actions: list[str],
    tool_inputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "tool": tool,
        "summary": "tool_loop_ready",
        "payload": {"actions": actions, "tool_inputs": tool_inputs},
    }


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


def record_tool_observation(
    *,
    agent: Any,
    emit: Any,
    execution_steps: list[dict[str, Any]],
    tool: str,
    summary: str,
    payload: Any,
) -> None:
    agent._record_agent_observation(
        emit=emit,
        execution_steps=execution_steps,
        tool=tool,
        summary=summary,
        payload=payload,
    )


def library_metadata_tool_request(*, planned_input: dict[str, Any], fallback_query: str) -> dict[str, Any]:
    metadata_query = str(planned_input.get("query", "") or fallback_query).strip()
    return {**planned_input, "query": metadata_query}


def library_metadata_observation_payload(*, result: dict[str, Any], answer: str) -> tuple[str, dict[str, Any]]:
    return (
        f"rows={result.get('row_count', 0)}",
        {
            "sql": result.get("sql", ""),
            "columns": result.get("columns", []),
            "row_count": result.get("row_count", 0),
            "truncated": result.get("truncated", False),
            "error": result.get("error", ""),
            "has_answer": bool(answer),
        },
    )


def store_conversation_answer_result(
    *,
    agent: Any,
    state: dict[str, Any],
    session: SessionContext,
    contract: QueryContract,
    emit: Any,
    tool: str,
    query: str,
    answer: str,
    artifact: dict[str, Any] | None = None,
) -> dict[str, int]:
    remember_kwargs: dict[str, Any] = {
        "session": session,
        "contract": contract,
        "tool": tool,
        "query": query,
        "answer": answer,
    }
    if artifact is not None:
        remember_kwargs["artifact"] = artifact
    agent._remember_conversation_tool_result(**remember_kwargs)
    agent._set_conversation_answer(state=state, answer=answer, emit=emit)
    return {"chars": len(answer)}


def read_memory_tool_payload(
    *,
    agent: Any,
    session: SessionContext,
    active_title_limit: int | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    context = agent._session_conversation_context(session)
    active_context = session.active_research_context_payload()
    payload_active_context = dict(active_context)
    if active_title_limit is not None:
        payload_active_context["titles"] = list(active_context.get("titles", []) or [])[:active_title_limit]
        payload_active_context["active_titles"] = list(active_context.get("active_titles", []) or [])[:active_title_limit]
    call_arguments = {"turn_count": len(session.turns), "active_targets": active_context["targets"]}
    payload = {
        "active_research_context": payload_active_context,
        "has_working_memory": bool(context.get("working_memory")),
    }
    return call_arguments, f"turns={len(session.turns)}", payload


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


def todo_write_tool_payload(
    *,
    planned_input: dict[str, Any],
    session: SessionContext,
) -> tuple[list[dict[str, str]], dict[str, Any], str]:
    items = normalize_todo_items(planned_input.get("items", []))
    store_session_todos(session, items)
    return items, {"items": items}, f"todos={len(items)}"


def format_task_results_answer(task_results: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for index, result in enumerate(task_results, start=1):
        prompt = str(result.get("prompt", "") or f"子任务 {index}").strip()
        answer = str(result.get("answer", "") or "").strip()
        if not answer:
            continue
        sections.append(f"## {index}. {prompt}\n\n{answer}")
    return "\n\n".join(sections).strip()


def task_tool_request(*, planned_input: dict[str, Any], fallback_prompt: str) -> dict[str, Any]:
    prompt = " ".join(str(planned_input.get("prompt", "") or planned_input.get("description", "") or fallback_prompt).split())
    raw_tools_allowed = planned_input.get("tools_allowed", [])
    tools_allowed = string_list_values(raw_tools_allowed)
    return {
        "prompt": prompt,
        "description": str(planned_input.get("description", "") or ""),
        "tools_allowed": tools_allowed,
        "max_steps": planned_input.get("max_steps", None),
    }


def task_result_observation_payload(*, request: dict[str, Any], result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    answer = str(result.get("answer", "") or "")
    return (
        f"task_answer_chars={len(answer)}",
        {
            "prompt": str(request.get("prompt", "") or ""),
            "verification": result.get("verification", {}),
            "answer_chars": len(answer),
        },
    )


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


def fetch_url_tool_request(planned_input: dict[str, Any]) -> dict[str, Any]:
    return {
        "url": str(planned_input.get("url", "") or "").strip(),
        "max_chars": planned_input.get("max_chars", 4000),
    }


def fetch_url_payload(result: FetchUrlResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "url": result.url,
        "title": result.title,
        "text": result.text,
        "error": result.error,
        "status_code": result.status_code,
    }


def fetch_url_tool_payload(result: FetchUrlResult) -> tuple[dict[str, Any], str, dict[str, Any]]:
    payload = fetch_url_payload(result)
    return payload, "ok" if result.ok else result.error, {**payload, "text": result.text[:600]}


def store_fetched_url_payload(*, state: dict[str, Any], payload: dict[str, Any]) -> None:
    state.setdefault("fetched_urls", []).append(payload)


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


def conversation_artifact_answer_from_state(state: dict[str, Any]) -> tuple[bool, str, list[Any]]:
    if state.get("task_results"):
        task_results = list(state.get("task_results", []) or [])
        answer = format_task_results_answer(task_results)
        citations = [
            citation
            for result in task_results
            if isinstance(result, dict)
            for citation in list(result.get("citations", []) or [])
        ]
        return True, answer, citations
    if state.get("fetched_urls"):
        return True, format_fetched_urls_answer(list(state.get("fetched_urls", []) or [])), []
    if state.get("summaries"):
        return True, format_summaries_answer(list(state.get("summaries", []) or [])), []
    return False, "", []


def conversation_clarification_report(contract: QueryContract) -> dict[str, Any]:
    missing_fields = [
        str(note).split("=", 1)[1]
        for note in contract.notes
        if str(note).startswith("ambiguous_slot=") and "=" in str(note)
    ] or ["user_choice"]
    return {
        "status": "clarify",
        "missing_fields": missing_fields,
        "recommended_action": "ask_human",
    }


def compose_done_payload(state: dict[str, Any]) -> dict[str, bool]:
    return {"has_answer": bool(state.get("answer"))}


def store_citation_candidates_payload(*, state: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    state["citation_candidates"] = candidates
    return f"candidates={len(candidates)}", {"titles": [item["title"] for item in candidates[:6]]}


def store_citation_lookup_payload(*, state: dict[str, Any], lookup: dict[str, Any]) -> tuple[str, dict[str, int]]:
    state["citation_lookup"] = lookup
    return (
        f"web_enabled={lookup.get('web_enabled')}, evidence={len(lookup.get('evidence', []) or [])}",
        {"result_count": len(lookup.get("results", []) or [])},
    )


def citation_ranking_result_payload(lookup: dict[str, Any]) -> tuple[list[EvidenceBlock], list[str], dict[str, str], str]:
    evidence = [item for item in list(lookup.get("evidence", []) or []) if isinstance(item, EvidenceBlock)]
    citation_doc_ids = [
        str(item.get("doc_id", ""))
        for item in list(lookup.get("results", []) or [])
        if item.get("citation_count") is not None and item.get("doc_id")
    ]
    counted = [item for item in list(lookup.get("results", []) or []) if item.get("citation_count") is not None]
    report = {
        "status": "pass" if counted else "retry",
        "recommended_action": "ranked_by_external_citation_count" if counted else "citation_count_not_found_in_web_snippets",
    }
    return evidence, citation_doc_ids, report, f"counted={len(counted)}"


def reflect_previous_answer_payload(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    excluded_titles = sorted(state["excluded_titles"])
    return f"excluded_titles={len(excluded_titles)}", {"excluded_titles": excluded_titles}


def ensure_research_clarification_report(state: dict[str, Any]) -> VerificationReport:
    verification = state.get("verification")
    if not isinstance(verification, VerificationReport) or verification.status != "clarify":
        verification = VerificationReport(
            status="clarify",
            missing_fields=["user_choice"],
            recommended_action="ask_human",
        )
        state["verification"] = verification
    return verification


def research_compose_observation_payload(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    verification = state.get("verification")
    return (
        str(getattr(verification, "status", "pending")),
        {
            "claim_count": len(state.get("claims", []) or []),
            "verification": verification.model_dump() if isinstance(verification, VerificationReport) else None,
        },
    )


def focus_values(raw: Any, fallback: list[str]) -> list[str]:
    values = raw if isinstance(raw, list) else fallback
    return [str(item).strip() for item in list(values or []) if str(item).strip()]


def string_list_values(raw: Any) -> list[str]:
    values = raw if isinstance(raw, list) else []
    return [str(item).strip() for item in list(values or []) if str(item).strip()]


def read_pdf_page_tool_request(*, planned_input: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    paper_id = str(planned_input.get("paper_id", "") or "").strip()
    if not paper_id:
        screened = list(state.get("screened_papers", []) or [])
        paper_id = str(getattr(screened[0], "paper_id", "") or "") if screened else ""
    page_from = coerce_int(planned_input.get("page_from", 1), default=1, minimum=1, maximum=10000)
    page_to = coerce_int(planned_input.get("page_to", page_from), default=page_from, minimum=page_from, maximum=10000)
    max_chars = coerce_int(planned_input.get("max_chars", 4000), default=4000, minimum=200, maximum=20000)
    return {"paper_id": paper_id, "page_from": page_from, "page_to": page_to, "max_chars": max_chars}


def grep_corpus_tool_request(*, planned_input: dict[str, Any], state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rewritten_queries = list(state.get("rewritten_queries", []) or [])
    pattern = str(
        planned_input.get("regex", "")
        or planned_input.get("pattern", "")
        or (rewritten_queries[0] if rewritten_queries else "")
    ).strip()
    scope = str(planned_input.get("scope", "") or "auto").strip()
    paper_ids = string_list_values(planned_input.get("paper_ids", []))
    max_hits = planned_input.get("max_hits", 20)
    request = {"pattern": pattern, "scope": scope, "paper_ids": paper_ids, "max_hits": max_hits}
    return request, {"regex": pattern, "scope": scope, "paper_ids": paper_ids, "max_hits": max_hits}


def query_rewrite_tool_request(*, planned_input: dict[str, Any], contract: QueryContract) -> dict[str, Any]:
    query = str(planned_input.get("query", "") or contract.clean_query).strip()
    targets = string_list_values(planned_input.get("targets", contract.targets))
    max_queries = coerce_int(planned_input.get("max_queries", 3), default=3, minimum=1, maximum=8)
    return {
        "query": query,
        "targets": targets,
        "mode": str(planned_input.get("mode", "") or "multi_query"),
        "max_queries": max_queries,
    }


def query_rewrite_tool_payload(*, result: Any, state: dict[str, Any]) -> tuple[dict[str, Any], str]:
    raw_payload = result.payload() if hasattr(result, "payload") else {}
    payload = dict(raw_payload) if isinstance(raw_payload, dict) else {}
    state.setdefault("query_rewrites", []).append(payload)
    state["rewritten_queries"] = list(payload.get("queries", []) or [])
    return payload, f"queries={len(state['rewritten_queries'])}"


def search_corpus_strategy(planned_input: dict[str, Any]) -> str:
    return str(planned_input.get("strategy", "") or "auto").strip()


def search_corpus_observation_payload(state: dict[str, Any]) -> tuple[str, dict[str, int]]:
    paper_count = len(state.get("screened_papers", []) or [])
    evidence_count = len(state.get("evidence", []) or [])
    return (
        f"papers={paper_count}, evidence={evidence_count}",
        {"paper_count": paper_count, "evidence_count": evidence_count},
    )


def atomic_search_tool_request(
    *,
    name: str,
    planned_input: dict[str, Any],
    state: dict[str, Any],
    default_limit: int,
) -> dict[str, Any]:
    rewritten_queries = list(state.get("rewritten_queries", []) or [])
    contract: QueryContract = state["contract"]
    query = str(planned_input.get("query", "") or (rewritten_queries[0] if rewritten_queries else contract.clean_query)).strip()
    request: dict[str, Any] = {
        "query": query,
        "contract": contract,
        "scope": str(planned_input.get("scope", "") or "auto").strip(),
        "paper_ids": string_list_values(planned_input.get("paper_ids", [])),
        "limit": planned_input.get("top_k", planned_input.get("limit", default_limit)),
    }
    if name == "hybrid_search":
        request["alpha"] = planned_input.get("alpha", 0.5)
    return request


def atomic_search_observation_payload(
    *,
    request: dict[str, Any],
    evidence: list[EvidenceBlock],
    paper_count: int,
) -> tuple[str, dict[str, Any]]:
    return (
        f"evidence={len(evidence)}",
        {
            "query": request.get("query", ""),
            "scope": request.get("scope", ""),
            "evidence_count": len(evidence),
            "paper_count": paper_count,
            "sources": list(dict.fromkeys(str(item.metadata.get("search_source", "")) for item in evidence if item.metadata)),
        },
    )


def rerank_tool_request(
    *,
    planned_input: dict[str, Any],
    state: dict[str, Any],
    default_top_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract: QueryContract = state["contract"]
    query = str(planned_input.get("query", "") or contract.clean_query).strip()
    focus = string_list_values(planned_input.get("focus", contract.targets))
    top_k = coerce_int(planned_input.get("top_k", default_top_k), default=default_top_k, minimum=1, maximum=50)
    candidate_evidence = evidence_from_payload(planned_input.get("candidates", []))
    source_evidence = candidate_evidence or [
        item for item in list(state.get("evidence", []) or []) if isinstance(item, EvidenceBlock)
    ]
    request = {"query": query, "evidence": source_evidence, "top_k": top_k, "focus": focus}
    payload_context = {
        "used_explicit_candidates": bool(candidate_evidence),
        "input_candidate_count": len(source_evidence),
    }
    return request, payload_context


def rerank_observation_payload(
    *,
    request: dict[str, Any],
    payload_context: dict[str, Any],
    evidence: list[EvidenceBlock],
) -> tuple[str, dict[str, Any]]:
    return (
        f"evidence={len(evidence)}",
        {
            "query": request.get("query", ""),
            "focus": list(request.get("focus", []) or []),
            **payload_context,
            "evidence_count": len(evidence),
            "top_doc_ids": [item.doc_id for item in evidence[:5]],
        },
    )


def store_research_evidence_result(
    *,
    agent: Any,
    state: dict[str, Any],
    evidence: list[EvidenceBlock],
) -> list[Any]:
    state["evidence"] = merge_evidence(list(state.get("evidence", []) or []), evidence)
    papers = [
        paper
        for paper_id in list(dict.fromkeys(item.paper_id for item in evidence if item.paper_id))
        if (paper := agent._candidate_from_paper_id(paper_id)) is not None
    ]
    if papers:
        existing_screened = {paper.paper_id for paper in list(state.get("screened_papers", []) or [])}
        state["screened_papers"] = [
            *list(state.get("screened_papers", []) or []),
            *[paper for paper in papers if paper.paper_id not in existing_screened],
        ]
        existing_candidates = {paper.paper_id for paper in list(state.get("candidate_papers", []) or [])}
        state["candidate_papers"] = [
            *list(state.get("candidate_papers", []) or []),
            *[paper for paper in papers if paper.paper_id not in existing_candidates],
        ]
    return papers


def evidence_event_payload(evidence: list[EvidenceBlock]) -> dict[str, Any]:
    return {"count": len(evidence), "items": [item.model_dump() for item in evidence]}


def store_fetch_url_evidence_result(
    *,
    agent: Any,
    state: dict[str, Any],
    evidence: EvidenceBlock | None,
) -> dict[str, Any] | None:
    if evidence is None:
        return None
    state["web_evidence"] = [*list(state.get("web_evidence", []) or []), evidence]
    state["evidence"] = merge_evidence(list(state.get("evidence", []) or []), [evidence])
    return evidence_event_payload([evidence])


def evidence_result_observation_payload(
    *,
    payload: dict[str, Any],
    evidence: list[EvidenceBlock],
    paper_count: int,
) -> tuple[str, dict[str, Any]]:
    return f"evidence={len(evidence)}", {**payload, "evidence_count": len(evidence), "paper_count": paper_count}


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


def remember_tool_payload(
    *,
    data_dir: Path,
    planned_input: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    key = str(planned_input.get("key", "") or "general").strip()
    content = str(planned_input.get("content", "") or "").strip()
    path = remember_learning(data_dir=data_dir, key=key, content=content)
    state.setdefault("learnings", []).append({"key": key, "path": str(path), "content": content})
    payload = {"key": key, "path": str(path), "content_chars": len(content)}
    return payload, f"key={key}"


def propose_tool_payload(agent: Any, planned_input: dict[str, Any]) -> dict[str, Any]:
    settings = getattr(agent, "settings", None)
    data_dir = Path(getattr(settings, "data_dir", "data"))
    try:
        proposal = record_tool_proposal(
            data_dir=data_dir,
            name=str(planned_input.get("name", "") or ""),
            description=str(planned_input.get("description", "") or ""),
            input_schema=dict(planned_input.get("input_schema", {}) or {}),
            python_code=str(planned_input.get("python_code", "") or ""),
            rationale=str(planned_input.get("rationale", "") or ""),
        )
    except (TypeError, ValueError) as exc:
        return {
            "status": "rejected",
            "error": str(exc),
            "admin_approval_required": True,
        }
    return {
        **proposal.payload(),
        "admin_approval_required": True,
    }


def summarize_tool_payload(
    *,
    planned_input: dict[str, Any],
    state: dict[str, Any],
    targets: list[str],
    fallback_to_summary_source: bool,
) -> dict[str, Any]:
    focus = focus_values(planned_input.get("focus", targets), targets)
    target_words = coerce_int(planned_input.get("target_words", 120), default=120, minimum=20, maximum=1000)
    text = str(planned_input.get("text", "") or "").strip()
    if text:
        summary = summarize_text(text=text, target_words=target_words, focus=focus)
        source_chars = len(text)
    else:
        evidence = evidence_from_payload(planned_input.get("evidence", [])) or evidence_blocks_from_state(state)
        if evidence or not fallback_to_summary_source:
            summary = summarize_evidence(evidence=evidence, target_words=target_words, focus=focus)
            source_chars = sum(len(item.snippet or "") for item in evidence)
        else:
            source_text = summary_source_from_state(state)
            summary = summarize_text(text=source_text, target_words=target_words, focus=focus)
            source_chars = len(source_text)
    return {"summary": summary, "source_chars": source_chars, "focus": focus, "target_words": target_words}


def store_summary_payload(*, state: dict[str, Any], payload: dict[str, Any]) -> str:
    state.setdefault("summaries", []).append(payload)
    return f"chars={len(str(payload.get('summary', '') or ''))}"


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


def store_claim_check_payload(*, state: dict[str, Any], payload: dict[str, Any]) -> None:
    state.setdefault("claim_checks", []).append(payload)
    state.setdefault("tool_verifications", []).append(payload)


def store_tool_proposal_payload(*, state: dict[str, Any], payload: dict[str, Any]) -> str:
    state.setdefault("tool_proposals", []).append(payload)
    return str(payload.get("status", ""))
