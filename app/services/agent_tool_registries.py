from __future__ import annotations

import hashlib
from typing import Any, Callable

from app.domain.models import EvidenceBlock, QueryContract, SessionContext, VerificationReport
from app.services.agent_tools import RegisteredAgentTool
from app.services.learnings import remember_learning
from app.services.url_fetcher import fetch_url as fetch_url_text

EmitFn = Callable[[str, dict[str, Any]], None]


def _normalize_todo_items(value: Any) -> list[dict[str, str]]:
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


def _store_session_todos(session: SessionContext, items: list[dict[str, str]]) -> None:
    memory = dict(session.working_memory or {})
    memory["todos"] = items
    session.working_memory = memory


def _format_task_results_answer(task_results: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for index, result in enumerate(task_results, start=1):
        prompt = str(result.get("prompt", "") or f"子任务 {index}").strip()
        answer = str(result.get("answer", "") or "").strip()
        if not answer:
            continue
        sections.append(f"## {index}. {prompt}\n\n{answer}")
    return "\n\n".join(sections).strip()


def _format_fetched_urls_answer(fetched_urls: list[dict[str, Any]]) -> str:
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


def _task_plan_with_allow_list(plan: dict[str, Any], tools_allowed: list[str]) -> dict[str, Any]:
    if not tools_allowed:
        return plan
    allowed = {str(item) for item in tools_allowed if str(item).strip()}
    actions = [str(item) for item in list(plan.get("actions", []) or []) if str(item) in allowed]
    tool_call_args = [
        item
        for item in list(plan.get("tool_call_args", []) or [])
        if isinstance(item, dict) and str(item.get("name", "") or "") in allowed
    ]
    return {**plan, "actions": actions, "tool_call_args": tool_call_args}


def _coerce_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def build_conversation_tool_registry(
    *,
    agent: Any,
    state: dict[str, Any],
    contract: QueryContract,
    query: str,
    session: SessionContext,
    max_web_results: int,
    emit: EmitFn,
    execution_steps: list[dict[str, Any]],
) -> dict[str, RegisteredAgentTool]:
    def tool_input(name: str) -> dict[str, Any]:
        tool_inputs = state.get("tool_inputs", {})
        if not isinstance(tool_inputs, dict):
            return {}
        payload = tool_inputs.get(name, {})
        return dict(payload) if isinstance(payload, dict) else {}

    def intent_summary() -> dict[str, Any]:
        notes = [str(item) for item in contract.notes]
        answer_slots = [str(item).strip() for item in list(getattr(contract, "answer_slots", []) or []) if str(item).strip()]
        if not answer_slots:
            answer_slots = [
                note.split("=", 1)[1]
                for note in notes
                if note.startswith("answer_slot=") and "=" in note
            ]
        intent_kind = next(
            (
                note.split("=", 1)[1]
                for note in notes
                if note.startswith("intent_kind=") and "=" in note
            ),
            contract.interaction_mode,
        )
        return {
            "kind": intent_kind,
            "answer_slots": answer_slots,
            "requested_fields": contract.requested_fields,
            "targets": contract.targets,
        }

    def understand_user_intent() -> None:
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="understand_user_intent",
            summary=f"intent={intent_summary()['kind']}",
            payload=intent_summary(),
        )

    def answer_conversation() -> None:
        agent._emit_agent_tool_call(emit=emit, tool="answer_conversation", arguments={"intent": intent_summary()})
        answer = agent._compose_conversation_response(contract=contract, query=query, session=session)
        agent._remember_conversation_tool_result(session=session, contract=contract, tool="answer_conversation", query=query, answer=answer)
        agent._set_conversation_answer(state=state, answer=answer, emit=emit)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="answer_conversation",
            summary="conversation_answer_ready",
            payload={"chars": len(answer)},
        )

    def get_library_status() -> None:
        agent._emit_agent_tool_call(emit=emit, tool="get_library_status", arguments={"query": query})
        answer = agent._compose_library_status_response(query=query)
        agent._remember_conversation_tool_result(session=session, contract=contract, tool="get_library_status", query=query, answer=answer)
        agent._set_conversation_answer(state=state, answer=answer, emit=emit)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="get_library_status",
            summary="library_status_ready",
            payload={"chars": len(answer)},
        )

    def query_library_metadata() -> None:
        state["library_metadata_attempted"] = True
        planned_input = tool_input("query_library_metadata")
        metadata_query = str(planned_input.get("query", "") or query).strip()
        agent._emit_agent_tool_call(
            emit=emit,
            tool="query_library_metadata",
            arguments={**planned_input, "query": metadata_query},
        )
        result = agent._compose_library_metadata_query_response(query=metadata_query)
        state["library_metadata_result"] = result
        answer = str(result.get("answer", "") or "").strip()
        if answer:
            artifact = agent._conversation_tool_result_artifact(tool="query_library_metadata", result=result)
            agent._remember_conversation_tool_result(
                session=session,
                contract=contract,
                tool="query_library_metadata",
                query=query,
                answer=answer,
                artifact=artifact,
            )
            agent._set_conversation_answer(state=state, answer=answer, emit=emit)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="query_library_metadata",
            summary=f"rows={result.get('row_count', 0)}",
            payload={
                "sql": result.get("sql", ""),
                "columns": result.get("columns", []),
                "row_count": result.get("row_count", 0),
                "truncated": result.get("truncated", False),
                "error": result.get("error", ""),
                "has_answer": bool(answer),
            },
        )

    def get_library_recommendation() -> None:
        agent._emit_agent_tool_call(emit=emit, tool="get_library_recommendation", arguments={"query": query})
        answer = agent._compose_library_recommendation_response(query=query, session=session)
        agent._remember_conversation_tool_result(
            session=session,
            contract=contract,
            tool="get_library_recommendation",
            query=query,
            answer=answer,
        )
        agent._set_conversation_answer(state=state, answer=answer, emit=emit)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="get_library_recommendation",
            summary="library_recommendation_ready",
            payload={"chars": len(answer)},
        )

    def read_conversation_memory() -> None:
        context = agent._session_conversation_context(session)
        active_context = session.active_research_context_payload()
        agent._emit_agent_tool_call(
            emit=emit,
            tool="read_conversation_memory",
            arguments={"turn_count": len(session.turns), "active_targets": active_context["targets"]},
        )
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="read_conversation_memory",
            summary=f"turns={len(session.turns)}",
            payload={"active_research_context": active_context, "has_working_memory": bool(context.get("working_memory"))},
        )

    def read_memory() -> None:
        read_conversation_memory()

    def todo_write() -> None:
        planned_input = tool_input("todo_write") or dict(state.get("current_tool_input", {}) or {})
        items = _normalize_todo_items(planned_input.get("items", []))
        _store_session_todos(session, items)
        emit("todo_update", {"items": items})
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="todo_write",
            summary=f"todos={len(items)}",
            payload={"items": items},
        )

    def remember() -> None:
        planned_input = tool_input("remember") or dict(state.get("current_tool_input", {}) or {})
        key = str(planned_input.get("key", "") or "general").strip()
        content = str(planned_input.get("content", "") or "").strip()
        path = remember_learning(data_dir=agent.settings.data_dir, key=key, content=content)
        state.setdefault("learnings", []).append({"key": key, "path": str(path), "content": content})
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="remember",
            summary=f"key={key}",
            payload={"key": key, "path": str(path), "content_chars": len(content)},
        )

    def run_task() -> None:
        planned_input = tool_input("Task") or dict(state.get("current_tool_input", {}) or {})
        prompt = " ".join(str(planned_input.get("prompt", "") or planned_input.get("description", "") or query).split())
        if not prompt:
            return
        raw_tools_allowed = planned_input.get("tools_allowed", [])
        tools_allowed = [
            str(item).strip()
            for item in (raw_tools_allowed if isinstance(raw_tools_allowed, list) else [])
            if str(item).strip()
        ]
        max_steps = planned_input.get("max_steps", None)
        sub_contract = agent._extract_query_contract(
            query=prompt,
            session=session,
            mode="auto",
            clarification_choice=None,
        )
        sub_plan = agent._plan_agent_actions(contract=sub_contract, session=session, use_web_search=False)
        sub_plan = _task_plan_with_allow_list(sub_plan, tools_allowed)
        emit("agent_plan", {"subtask": prompt, "task_tool": True, **sub_plan})
        agent._emit_agent_tool_call(
            emit=emit,
            tool="Task",
            arguments={
                "description": planned_input.get("description", ""),
                "prompt": prompt,
                "tools_allowed": tools_allowed,
                "max_steps": max_steps,
            },
        )
        if sub_contract.interaction_mode == "conversation":
            sub_state = agent.runtime.execute_conversation_tools(
                contract=sub_contract,
                query=prompt,
                session=session,
                agent_plan=sub_plan,
                max_web_results=max_web_results,
                emit=emit,
                execution_steps=execution_steps,
            )
            answer = str(sub_state.get("answer", "") or "")
            citations = list(sub_state.get("citations", []) or [])
            verification_payload = dict(sub_state.get("verification_report", {}) or {"status": "pass"})
            result = {
                "prompt": prompt,
                "answer": answer,
                "citations": citations,
                "verification": verification_payload,
                "contract": sub_contract.model_dump(),
            }
        else:
            sub_state = agent.runtime.run_research_agent_loop(
                contract=sub_contract,
                session=session,
                agent_plan=sub_plan,
                web_enabled=False,
                explicit_web_search=False,
                max_web_results=max_web_results,
                emit=emit,
                execution_steps=execution_steps,
            )
            verification = sub_state.get("verification")
            answer, citations = agent._compose_answer(
                contract=sub_state["contract"],
                claims=sub_state["claims"],
                evidence=sub_state["evidence"],
                papers=sub_state["screened_papers"],
                verification=verification,
                session=session,
            )
            result = {
                "prompt": prompt,
                "answer": answer,
                "citations": citations,
                "verification": verification.model_dump() if hasattr(verification, "model_dump") else {},
                "contract": sub_state["contract"].model_dump(),
            }
        state.setdefault("task_results", []).append(result)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="Task",
            summary=f"task_answer_chars={len(str(result.get('answer', '') or ''))}",
            payload={
                "prompt": prompt,
                "verification": result.get("verification", {}),
                "answer_chars": len(str(result.get("answer", "") or "")),
            },
        )

    def answer_from_memory() -> None:
        agent._emit_agent_tool_call(
            emit=emit,
            tool="answer_from_memory",
            arguments={"query": query, "targets": contract.targets},
        )
        answer = agent._compose_memory_followup_answer(query=query, session=session, contract=contract)
        agent._remember_conversation_tool_result(session=session, contract=contract, tool="answer_from_memory", query=query, answer=answer)
        agent._set_conversation_answer(state=state, answer=answer, emit=emit)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="answer_from_memory",
            summary="memory_answer_ready",
            payload={"chars": len(answer)},
        )

    def synthesize_previous_results() -> None:
        agent._emit_agent_tool_call(emit=emit, tool="synthesize_previous_results", arguments={"targets": contract.targets})
        answer = agent._compose_memory_synthesis_answer(query=query, session=session, contract=contract)
        bindings = agent._active_memory_bindings(session)
        state["citations"] = agent._citations_from_memory_bindings(bindings)
        agent._remember_conversation_tool_result(session=session, contract=contract, tool="synthesize_previous_results", query=query, answer=answer)
        agent._set_conversation_answer(state=state, answer=answer, emit=emit)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="synthesize_previous_results",
            summary="memory_synthesis_ready",
            payload={"binding_count": len(bindings), "chars": len(answer)},
        )

    def recover_previous_recommendation_candidates() -> None:
        candidates = agent._select_citation_ranking_candidates(session=session, query=query, limit=6)
        state["citation_candidates"] = candidates
        agent._emit_agent_tool_call(emit=emit, tool="recover_previous_recommendation_candidates", arguments={"query": query, "limit": 6})
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="recover_previous_recommendation_candidates",
            summary=f"candidates={len(candidates)}",
            payload={"titles": [item["title"] for item in candidates[:6]]},
        )

    def web_citation_lookup() -> None:
        candidates = list(state.get("citation_candidates", []) or [])
        lookup = agent._lookup_candidate_citation_counts(
            candidates=candidates,
            max_web_results=max_web_results,
            emit=emit,
            execution_steps=execution_steps,
        )
        state["citation_lookup"] = lookup
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="web_citation_lookup",
            summary=f"web_enabled={lookup.get('web_enabled')}, evidence={len(lookup.get('evidence', []) or [])}",
            payload={"result_count": len(lookup.get("results", []) or [])},
        )

    def rank_by_verified_citation_count() -> None:
        candidates = list(state.get("citation_candidates", []) or [])
        lookup = dict(state.get("citation_lookup", {}) or {})
        answer = agent._format_citation_ranking_answer(
            candidates=candidates,
            citation_results=list(lookup.get("results", []) or []),
            web_enabled=bool(lookup.get("web_enabled")),
        )
        evidence = [item for item in list(lookup.get("evidence", []) or []) if isinstance(item, EvidenceBlock)]
        citation_doc_ids = [
            str(item.get("doc_id", ""))
            for item in list(lookup.get("results", []) or [])
            if item.get("citation_count") is not None and item.get("doc_id")
        ]
        state["citations"] = agent._dedupe_citations(agent._citations_from_doc_ids(citation_doc_ids, evidence))
        counted = [item for item in list(lookup.get("results", []) or []) if item.get("citation_count") is not None]
        state["verification_report"] = {
            "status": "pass" if counted else "retry",
            "recommended_action": "ranked_by_external_citation_count" if counted else "citation_count_not_found_in_web_snippets",
        }
        agent._remember_conversation_tool_result(
            session=session,
            contract=contract,
            tool="rank_by_verified_citation_count",
            query=query,
            answer=answer,
        )
        agent._set_conversation_answer(state=state, answer=answer, emit=emit)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="rank_by_verified_citation_count",
            summary=f"counted={len(counted)}",
            payload=state["verification_report"],
        )

    def web_search() -> None:
        is_citation_turn = "citation_count_ranking" in contract.requested_fields or "citation_count_requires_web" in contract.notes
        if is_citation_turn:
            if not state.get("citation_candidates"):
                recover_previous_recommendation_candidates()
            web_citation_lookup()
            return
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="web_search",
            summary="conversation_web_search_skipped",
            payload={"intent": intent_summary()},
        )

    def fetch_url() -> None:
        planned_input = tool_input("fetch_url") or dict(state.get("current_tool_input", {}) or {})
        url = str(planned_input.get("url", "") or "").strip()
        max_chars = planned_input.get("max_chars", 4000)
        agent._emit_agent_tool_call(emit=emit, tool="fetch_url", arguments={"url": url, "max_chars": max_chars})
        result = fetch_url_text(client=agent.clients.http_client, url=url, max_chars=max_chars)
        payload = {
            "ok": result.ok,
            "url": result.url,
            "title": result.title,
            "text": result.text,
            "error": result.error,
            "status_code": result.status_code,
        }
        state.setdefault("fetched_urls", []).append(payload)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="fetch_url",
            summary="ok" if result.ok else result.error,
            payload={**payload, "text": result.text[:600]},
        )

    def compose() -> None:
        if state.get("task_results") and not state.get("answer"):
            answer = _format_task_results_answer(list(state.get("task_results", []) or []))
            if answer:
                citations = [
                    citation
                    for result in list(state.get("task_results", []) or [])
                    for citation in list(result.get("citations", []) or [])
                ]
                state["citations"] = citations
                agent._set_conversation_answer(state=state, answer=answer, emit=emit)
        elif state.get("fetched_urls") and not state.get("answer"):
            answer = _format_fetched_urls_answer(list(state.get("fetched_urls", []) or []))
            if answer:
                agent._set_conversation_answer(state=state, answer=answer, emit=emit)
        elif contract.relation == "library_status":
            if not state.get("library_metadata_attempted"):
                query_library_metadata()
            if not state.get("answer"):
                get_library_status()
        elif contract.relation == "library_recommendation":
            get_library_recommendation()
        elif contract.relation == "memory_followup":
            if not state.get("answer"):
                answer_from_memory()
        elif contract.relation == "memory_synthesis":
            if not state.get("answer"):
                synthesize_previous_results()
        elif contract.relation == "library_citation_ranking":
            if not state.get("citation_candidates"):
                recover_previous_recommendation_candidates()
            if not state.get("citation_lookup"):
                web_citation_lookup()
            rank_by_verified_citation_count()
        elif not state.get("answer"):
            answer_conversation()
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="compose",
            summary="done",
            payload={"has_answer": bool(state.get("answer"))},
        )

    def ask_human() -> None:
        missing_fields = [
            str(note).split("=", 1)[1]
            for note in contract.notes
            if str(note).startswith("ambiguous_slot=") and "=" in str(note)
        ] or ["user_choice"]
        state["verification_report"] = {
            "status": "clarify",
            "missing_fields": missing_fields,
            "recommended_action": "ask_human",
        }
        answer = agent._clarification_question(contract, session)
        agent._set_conversation_answer(state=state, answer=answer, emit=emit)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="ask_human",
            summary="conversation_clarification",
            payload=state["verification_report"],
        )

    def compose_or_ask_human() -> None:
        if not state.get("answer"):
            compose()
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="compose_or_ask_human",
            summary="done",
            payload={"has_answer": bool(state.get("answer"))},
        )

    return {
        "read_memory": RegisteredAgentTool("read_memory", read_memory),
        "todo_write": RegisteredAgentTool("todo_write", todo_write),
        "remember": RegisteredAgentTool("remember", remember),
        "Task": RegisteredAgentTool("Task", run_task),
        "web_search": RegisteredAgentTool("web_search", web_search),
        "fetch_url": RegisteredAgentTool("fetch_url", fetch_url),
        "query_library_metadata": RegisteredAgentTool("query_library_metadata", query_library_metadata),
        "compose": RegisteredAgentTool("compose", compose, terminal=True),
        "ask_human": RegisteredAgentTool("ask_human", ask_human, terminal=True),
    }


def build_research_tool_registry(
    *,
    agent: Any,
    state: dict[str, Any],
    session: SessionContext,
    web_enabled: bool,
    explicit_web_search: bool,
    max_web_results: int,
    emit: EmitFn,
    execution_steps: list[dict[str, Any]],
) -> dict[str, RegisteredAgentTool]:
    def tool_input(name: str) -> dict[str, Any]:
        tool_inputs = state.get("tool_inputs", {})
        if not isinstance(tool_inputs, dict):
            return {}
        payload = tool_inputs.get(name, {})
        return dict(payload) if isinstance(payload, dict) else {}

    def understand_user_intent() -> None:
        contract: QueryContract = state["contract"]
        notes = [str(item) for item in contract.notes]
        answer_slots = [str(item).strip() for item in list(getattr(contract, "answer_slots", []) or []) if str(item).strip()]
        if not answer_slots:
            answer_slots = [
                note.split("=", 1)[1]
                for note in notes
                if note.startswith("answer_slot=") and "=" in note
            ]
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="understand_user_intent",
            summary="/".join(answer_slots or contract.requested_fields or [contract.interaction_mode]),
            payload={
                "answer_slots": answer_slots,
                "requested_fields": contract.requested_fields,
                "required_modalities": contract.required_modalities,
                "targets": contract.targets,
                "continuation_mode": contract.continuation_mode,
            },
        )

    def reflect_previous_answer() -> None:
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="reflect_previous_answer",
            summary=f"excluded_titles={len(state['excluded_titles'])}",
            payload={"excluded_titles": sorted(state["excluded_titles"])},
        )

    def read_memory() -> None:
        context = agent._session_conversation_context(session)
        active_context = session.active_research_context_payload()
        agent._emit_agent_tool_call(
            emit=emit,
            tool="read_memory",
            arguments={"turn_count": len(session.turns), "active_targets": active_context["targets"]},
        )
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="read_memory",
            summary=f"turns={len(session.turns)}",
            payload={
                "active_research_context": {
                    **active_context,
                    "titles": list(active_context.get("titles", []) or [])[:4],
                    "active_titles": list(active_context.get("active_titles", []) or [])[:4],
                },
                "has_working_memory": bool(context.get("working_memory")),
            },
        )

    def todo_write() -> None:
        planned_input = tool_input("todo_write") or dict(state.get("current_tool_input", {}) or {})
        items = _normalize_todo_items(planned_input.get("items", []))
        _store_session_todos(session, items)
        emit("todo_update", {"items": items})
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="todo_write",
            summary=f"todos={len(items)}",
            payload={"items": items},
        )

    def remember() -> None:
        planned_input = tool_input("remember") or dict(state.get("current_tool_input", {}) or {})
        key = str(planned_input.get("key", "") or "general").strip()
        content = str(planned_input.get("content", "") or "").strip()
        path = remember_learning(data_dir=agent.settings.data_dir, key=key, content=content)
        state.setdefault("learnings", []).append({"key": key, "path": str(path), "content": content})
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="remember",
            summary=f"key={key}",
            payload={"key": key, "path": str(path), "content_chars": len(content)},
        )

    def search_papers() -> None:
        agent._agent_search_papers(
            state=state,
            session=session,
            emit=emit,
            execution_steps=execution_steps,
        )

    def search_corpus() -> None:
        if not state.get("screened_papers"):
            search_papers()
        if not state.get("evidence"):
            search_evidence()
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="search_corpus",
            summary=f"papers={len(state.get('screened_papers', []) or [])}, evidence={len(state.get('evidence', []) or [])}",
            payload={
                "paper_count": len(state.get("screened_papers", []) or []),
                "evidence_count": len(state.get("evidence", []) or []),
            },
        )

    def search_evidence() -> None:
        agent._agent_search_evidence(
            state=state,
            emit=emit,
            execution_steps=execution_steps,
        )

    def run_atomic_search(name: str) -> None:
        planned_input = tool_input(name) or dict(state.get("current_tool_input", {}) or {})
        query = str(planned_input.get("query", "") or state["contract"].clean_query).strip()
        scope = str(planned_input.get("scope", "") or "auto").strip()
        top_k = planned_input.get("top_k", planned_input.get("limit", agent.settings.evidence_limit_default))
        raw_paper_ids = planned_input.get("paper_ids", [])
        paper_ids = [
            str(item).strip()
            for item in (raw_paper_ids if isinstance(raw_paper_ids, list) else [])
            if str(item).strip()
        ]
        kwargs: dict[str, Any] = {
            "query": query,
            "contract": state["contract"],
            "scope": scope,
            "paper_ids": paper_ids,
            "limit": top_k,
        }
        if name == "hybrid_search":
            kwargs["alpha"] = planned_input.get("alpha", 0.5)
        evidence = getattr(agent.retriever, name)(**kwargs)
        state["evidence"] = agent._merge_evidence(list(state.get("evidence", []) or []), evidence)
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
        emit("evidence", {"count": len(state["evidence"]), "items": [item.model_dump() for item in state["evidence"]]})
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool=name,
            summary=f"evidence={len(evidence)}",
            payload={
                "query": query,
                "scope": scope,
                "evidence_count": len(evidence),
                "paper_count": len(papers),
                "sources": list(dict.fromkeys(str(item.metadata.get("search_source", "")) for item in evidence if item.metadata)),
            },
        )

    def bm25_search() -> None:
        run_atomic_search("bm25_search")

    def vector_search() -> None:
        run_atomic_search("vector_search")

    def hybrid_search() -> None:
        run_atomic_search("hybrid_search")

    def rerank() -> None:
        planned_input = tool_input("rerank") or dict(state.get("current_tool_input", {}) or {})
        query = str(planned_input.get("query", "") or state["contract"].clean_query).strip()
        raw_focus = planned_input.get("focus", state["contract"].targets)
        focus = [
            str(item).strip()
            for item in (raw_focus if isinstance(raw_focus, list) else [])
            if str(item).strip()
        ]
        top_k = planned_input.get("top_k", agent.settings.evidence_limit_default)
        evidence = agent.retriever.rerank_evidence(
            query=query,
            evidence=list(state.get("evidence", []) or []),
            top_k=top_k,
            focus=focus,
        )
        state["evidence"] = evidence
        emit("evidence", {"count": len(evidence), "items": [item.model_dump() for item in evidence]})
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="rerank",
            summary=f"evidence={len(evidence)}",
            payload={
                "query": query,
                "focus": focus,
                "evidence_count": len(evidence),
                "top_doc_ids": [item.doc_id for item in evidence[:5]],
            },
        )

    def add_evidence_result(name: str, evidence: list[EvidenceBlock], payload: dict[str, Any]) -> None:
        state["evidence"] = agent._merge_evidence(list(state.get("evidence", []) or []), evidence)
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
        emit("evidence", {"count": len(state["evidence"]), "items": [item.model_dump() for item in state["evidence"]]})
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool=name,
            summary=f"evidence={len(evidence)}",
            payload={**payload, "evidence_count": len(evidence), "paper_count": len(papers)},
        )

    def read_pdf_page() -> None:
        planned_input = tool_input("read_pdf_page") or dict(state.get("current_tool_input", {}) or {})
        paper_id = str(planned_input.get("paper_id", "") or "").strip()
        if not paper_id:
            screened = list(state.get("screened_papers", []) or [])
            paper_id = str(getattr(screened[0], "paper_id", "") or "") if screened else ""
        page_from = _coerce_int(planned_input.get("page_from", 1), default=1, minimum=1, maximum=10000)
        page_to = _coerce_int(planned_input.get("page_to", page_from), default=page_from, minimum=page_from, maximum=10000)
        max_chars = _coerce_int(planned_input.get("max_chars", 4000), default=4000, minimum=200, maximum=20000)
        evidence = agent.retriever.read_pdf_pages(
            paper_id=paper_id,
            page_from=page_from,
            page_to=page_to,
            max_chars=max_chars,
        )
        add_evidence_result(
            "read_pdf_page",
            evidence,
            {"paper_id": paper_id, "page_from": page_from, "page_to": page_to, "max_chars": max_chars},
        )

    def grep_corpus() -> None:
        planned_input = tool_input("grep_corpus") or dict(state.get("current_tool_input", {}) or {})
        pattern = str(planned_input.get("regex", "") or planned_input.get("pattern", "") or "").strip()
        scope = str(planned_input.get("scope", "") or "auto").strip()
        raw_paper_ids = planned_input.get("paper_ids", [])
        paper_ids = [
            str(item).strip()
            for item in (raw_paper_ids if isinstance(raw_paper_ids, list) else [])
            if str(item).strip()
        ]
        max_hits = planned_input.get("max_hits", 20)
        evidence = agent.retriever.grep_corpus(
            pattern=pattern,
            scope=scope,
            paper_ids=paper_ids,
            max_hits=max_hits,
        )
        add_evidence_result(
            "grep_corpus",
            evidence,
            {"regex": pattern, "scope": scope, "paper_ids": paper_ids, "max_hits": max_hits},
        )

    def web_search() -> None:
        agent._agent_web_search(
            state=state,
            web_enabled=web_enabled,
            max_web_results=max_web_results,
            emit=emit,
            execution_steps=execution_steps,
        )

    def fetch_url() -> None:
        planned_input = tool_input("fetch_url") or dict(state.get("current_tool_input", {}) or {})
        url = str(planned_input.get("url", "") or "").strip()
        max_chars = planned_input.get("max_chars", 4000)
        agent._emit_agent_tool_call(emit=emit, tool="fetch_url", arguments={"url": url, "max_chars": max_chars})
        result = fetch_url_text(client=agent.clients.http_client, url=url, max_chars=max_chars)
        payload = {
            "ok": result.ok,
            "url": result.url,
            "title": result.title,
            "text": result.text,
            "error": result.error,
            "status_code": result.status_code,
        }
        state.setdefault("fetched_urls", []).append(payload)
        if result.ok:
            doc_id = "web::fetch::" + hashlib.sha1(result.url.encode("utf-8")).hexdigest()[:16]
            evidence = EvidenceBlock(
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
            state["web_evidence"] = [*list(state.get("web_evidence", []) or []), evidence]
            state["evidence"] = agent._merge_evidence(list(state.get("evidence", []) or []), [evidence])
            emit("web_search", {"count": 1, "items": [evidence.model_dump()]})
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="fetch_url",
            summary="ok" if result.ok else result.error,
            payload={**payload, "text": result.text[:600]},
        )

    def solve_claims() -> None:
        agent._agent_solve_claims(
            state=state,
            session=session,
            explicit_web_search=explicit_web_search,
            max_web_results=max_web_results,
            emit=emit,
            execution_steps=execution_steps,
        )

    def verify_grounding() -> None:
        if state["verification"] is None:
            agent._agent_verify_grounding(
                state=state,
                session=session,
                explicit_web_search=explicit_web_search,
                max_web_results=max_web_results,
                emit=emit,
                execution_steps=execution_steps,
            )

    def ask_human() -> None:
        verification = state.get("verification")
        if not isinstance(verification, VerificationReport) or verification.status != "clarify":
            state["verification"] = VerificationReport(
                status="clarify",
                missing_fields=["user_choice"],
                recommended_action="ask_human",
            )
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="ask_human",
            summary=str(state["verification"].recommended_action),
            payload=state["verification"].model_dump(),
        )

    def compose() -> None:
        verification = state.get("verification")
        if not isinstance(verification, VerificationReport) or verification.status != "clarify":
            if not state.get("claims"):
                solve_claims()
            if state.get("verification") is None:
                verify_grounding()
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="compose",
            summary=str(getattr(state.get("verification"), "status", "pending")),
            payload={
                "claim_count": len(state.get("claims", []) or []),
                "verification": state["verification"].model_dump()
                if isinstance(state.get("verification"), VerificationReport)
                else None,
            },
        )

    def compose_or_ask_human() -> None:
        compose()

    return {
        "read_memory": RegisteredAgentTool("read_memory", read_memory),
        "todo_write": RegisteredAgentTool("todo_write", todo_write),
        "remember": RegisteredAgentTool("remember", remember),
        "bm25_search": RegisteredAgentTool("bm25_search", bm25_search),
        "vector_search": RegisteredAgentTool("vector_search", vector_search),
        "hybrid_search": RegisteredAgentTool("hybrid_search", hybrid_search),
        "rerank": RegisteredAgentTool("rerank", rerank),
        "read_pdf_page": RegisteredAgentTool("read_pdf_page", read_pdf_page),
        "grep_corpus": RegisteredAgentTool("grep_corpus", grep_corpus),
        "search_corpus": RegisteredAgentTool("search_corpus", search_corpus),
        "compose": RegisteredAgentTool("compose", compose, terminal=True),
        "web_search": RegisteredAgentTool("web_search", web_search),
        "fetch_url": RegisteredAgentTool("fetch_url", fetch_url),
        "ask_human": RegisteredAgentTool("ask_human", ask_human, terminal=True),
    }
