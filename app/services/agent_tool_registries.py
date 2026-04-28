from __future__ import annotations

from typing import Any, Callable

from app.domain.models import EvidenceBlock, QueryContract, SessionContext, VerificationReport
from app.services.agent_task import run_task_subagent
from app.services.agent_tools import RegisteredAgentTool
from app.services.query_rewrite import rewrite_query
from app.services.tool_registry_helpers import (
    atomic_search_observation_payload,
    atomic_search_tool_request,
    conversation_artifact_answer_from_state,
    conversation_intent_summary,
    evidence_event_payload,
    evidence_result_observation_payload,
    fetch_url_evidence,
    fetch_url_tool_payload,
    fetch_url_tool_request,
    grep_corpus_tool_request,
    library_metadata_observation_payload,
    library_metadata_tool_request,
    planned_tool_input_from_state,
    propose_tool_payload,
    query_rewrite_tool_payload,
    query_rewrite_tool_request,
    read_memory_tool_payload,
    read_pdf_page_tool_request,
    remember_tool_payload,
    rerank_observation_payload,
    rerank_tool_request,
    research_intent_summary,
    search_corpus_observation_payload,
    search_corpus_strategy,
    store_claim_check_payload,
    store_fetch_url_evidence_result,
    store_research_evidence_result,
    summarize_tool_payload,
    task_result_observation_payload,
    task_tool_request,
    todo_write_tool_payload,
    verify_claim_tool_payload,
)
from app.services.url_fetcher import fetch_url as fetch_url_text

EmitFn = Callable[[str, dict[str, Any]], None]


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
    def intent_summary() -> dict[str, Any]:
        return conversation_intent_summary(contract)

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
        planned_input = planned_tool_input_from_state(state, "query_library_metadata")
        request = library_metadata_tool_request(planned_input=planned_input, fallback_query=query)
        agent._emit_agent_tool_call(
            emit=emit,
            tool="query_library_metadata",
            arguments=request,
        )
        result = agent._compose_library_metadata_query_response(query=str(request.get("query", "") or ""))
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
        summary, payload = library_metadata_observation_payload(result=result, answer=answer)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="query_library_metadata",
            summary=summary,
            payload=payload,
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
        call_arguments, summary, payload = read_memory_tool_payload(agent=agent, session=session)
        agent._emit_agent_tool_call(
            emit=emit,
            tool="read_conversation_memory",
            arguments=call_arguments,
        )
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="read_conversation_memory",
            summary=summary,
            payload=payload,
        )

    def read_memory() -> None:
        read_conversation_memory()

    def todo_write() -> None:
        planned_input = planned_tool_input_from_state(state, "todo_write")
        items, payload, summary = todo_write_tool_payload(planned_input=planned_input, session=session)
        emit("todo_update", {"items": items})
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="todo_write",
            summary=summary,
            payload=payload,
        )

    def remember() -> None:
        planned_input = planned_tool_input_from_state(state, "remember")
        payload, summary = remember_tool_payload(data_dir=agent.settings.data_dir, planned_input=planned_input, state=state)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="remember",
            summary=summary,
            payload=payload,
        )

    def propose_tool() -> None:
        planned_input = planned_tool_input_from_state(state, "propose_tool")
        payload = propose_tool_payload(agent, planned_input)
        state.setdefault("tool_proposals", []).append(payload)
        emit("tool_proposal", payload)
        if payload.get("status") == "pending_review" and not state.get("answer"):
            agent._set_conversation_answer(state=state, answer="已记录工具提案，等待人工审核后才能启用。", emit=emit)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="propose_tool",
            summary=str(payload.get("status", "")),
            payload=payload,
        )

    def summarize() -> None:
        planned_input = planned_tool_input_from_state(state, "summarize")
        payload = summarize_tool_payload(
            planned_input=planned_input,
            state=state,
            targets=contract.targets,
            fallback_to_summary_source=True,
        )
        state.setdefault("summaries", []).append(payload)
        if payload["summary"] and not state.get("answer"):
            agent._set_conversation_answer(state=state, answer=payload["summary"], emit=emit)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="summarize",
            summary=f"chars={len(payload['summary'])}",
            payload=payload,
        )

    def verify_claim() -> None:
        planned_input = planned_tool_input_from_state(state, "verify_claim")
        payload, summary = verify_claim_tool_payload(planned_input=planned_input, state=state)
        store_claim_check_payload(state=state, payload=payload)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="verify_claim",
            summary=summary,
            payload=payload,
        )

    def run_task() -> None:
        planned_input = planned_tool_input_from_state(state, "Task")
        request = task_tool_request(planned_input=planned_input, fallback_prompt=query)
        if not request["prompt"]:
            return
        result = run_task_subagent(
            agent=agent,
            prompt=str(request["prompt"]),
            description=str(request["description"]),
            tools_allowed=list(request["tools_allowed"]),
            max_steps=request["max_steps"],
            session=session,
            max_web_results=max_web_results,
            emit=emit,
            execution_steps=execution_steps,
        )
        state.setdefault("task_results", []).append(result)
        summary, payload = task_result_observation_payload(request=request, result=result)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="Task",
            summary=summary,
            payload=payload,
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
        planned_input = planned_tool_input_from_state(state, "fetch_url")
        request = fetch_url_tool_request(planned_input)
        agent._emit_agent_tool_call(emit=emit, tool="fetch_url", arguments=request)
        result = fetch_url_text(client=agent.clients.http_client, **request)
        payload, summary, observation_payload = fetch_url_tool_payload(result)
        state.setdefault("fetched_urls", []).append(payload)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="fetch_url",
            summary=summary,
            payload=observation_payload,
        )

    def compose() -> None:
        artifact_answer_matched, answer, citations = conversation_artifact_answer_from_state(state)
        if artifact_answer_matched and not state.get("answer"):
            if answer:
                if citations:
                    state["citations"] = citations
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
        "propose_tool": RegisteredAgentTool("propose_tool", propose_tool),
        "summarize": RegisteredAgentTool("summarize", summarize),
        "verify_claim": RegisteredAgentTool("verify_claim", verify_claim),
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
    def understand_user_intent() -> None:
        contract: QueryContract = state["contract"]
        summary, payload = research_intent_summary(contract)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="understand_user_intent",
            summary=summary,
            payload=payload,
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
        call_arguments, summary, payload = read_memory_tool_payload(agent=agent, session=session, active_title_limit=4)
        agent._emit_agent_tool_call(
            emit=emit,
            tool="read_memory",
            arguments=call_arguments,
        )
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="read_memory",
            summary=summary,
            payload=payload,
        )

    def todo_write() -> None:
        planned_input = planned_tool_input_from_state(state, "todo_write")
        items, payload, summary = todo_write_tool_payload(planned_input=planned_input, session=session)
        emit("todo_update", {"items": items})
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="todo_write",
            summary=summary,
            payload=payload,
        )

    def remember() -> None:
        planned_input = planned_tool_input_from_state(state, "remember")
        payload, summary = remember_tool_payload(data_dir=agent.settings.data_dir, planned_input=planned_input, state=state)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="remember",
            summary=summary,
            payload=payload,
        )

    def propose_tool() -> None:
        planned_input = planned_tool_input_from_state(state, "propose_tool")
        payload = propose_tool_payload(agent, planned_input)
        state.setdefault("tool_proposals", []).append(payload)
        emit("tool_proposal", payload)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="propose_tool",
            summary=str(payload.get("status", "")),
            payload=payload,
        )

    def search_papers() -> None:
        agent._agent_search_papers(
            state=state,
            session=session,
            emit=emit,
            execution_steps=execution_steps,
        )

    def search_corpus() -> None:
        planned_input = planned_tool_input_from_state(state, "search_corpus")
        strategy = search_corpus_strategy(planned_input)
        if strategy in {"bm25", "vector", "hybrid"}:
            run_atomic_search(f"{strategy}_search")
            return
        if not state.get("screened_papers"):
            search_papers()
        if not state.get("evidence"):
            search_evidence()
        summary, payload = search_corpus_observation_payload(state)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="search_corpus",
            summary=summary,
            payload=payload,
        )

    def search_evidence() -> None:
        agent._agent_search_evidence(
            state=state,
            emit=emit,
            execution_steps=execution_steps,
        )

    def run_atomic_search(name: str) -> None:
        planned_input = planned_tool_input_from_state(state, name)
        request = atomic_search_tool_request(
            name=name,
            planned_input=planned_input,
            state=state,
            default_limit=agent.settings.evidence_limit_default,
        )
        evidence = getattr(agent.retriever, name)(**request)
        papers = store_research_evidence_result(agent=agent, state=state, evidence=evidence)
        emit("evidence", evidence_event_payload(list(state.get("evidence", []) or [])))
        summary, payload = atomic_search_observation_payload(request=request, evidence=evidence, paper_count=len(papers))
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool=name,
            summary=summary,
            payload=payload,
        )

    def bm25_search() -> None:
        run_atomic_search("bm25_search")

    def vector_search() -> None:
        run_atomic_search("vector_search")

    def hybrid_search() -> None:
        run_atomic_search("hybrid_search")

    def rerank() -> None:
        planned_input = planned_tool_input_from_state(state, "rerank")
        default_top_k = getattr(agent.settings, "evidence_limit_default", 12)
        request, payload_context = rerank_tool_request(
            planned_input=planned_input,
            state=state,
            default_top_k=default_top_k,
        )
        evidence = agent.retriever.rerank_evidence(**request)
        state["evidence"] = evidence
        emit("evidence", evidence_event_payload(evidence))
        summary, payload = rerank_observation_payload(request=request, payload_context=payload_context, evidence=evidence)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="rerank",
            summary=summary,
            payload=payload,
        )

    def add_evidence_result(name: str, evidence: list[EvidenceBlock], payload: dict[str, Any]) -> None:
        papers = store_research_evidence_result(agent=agent, state=state, evidence=evidence)
        emit("evidence", evidence_event_payload(list(state.get("evidence", []) or [])))
        summary, observation_payload = evidence_result_observation_payload(
            payload=payload,
            evidence=evidence,
            paper_count=len(papers),
        )
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool=name,
            summary=summary,
            payload=observation_payload,
        )

    def read_pdf_page() -> None:
        planned_input = planned_tool_input_from_state(state, "read_pdf_page")
        request = read_pdf_page_tool_request(planned_input=planned_input, state=state)
        evidence = agent.retriever.read_pdf_pages(**request)
        add_evidence_result("read_pdf_page", evidence, request)

    def grep_corpus() -> None:
        planned_input = planned_tool_input_from_state(state, "grep_corpus")
        request, payload = grep_corpus_tool_request(planned_input=planned_input, state=state)
        evidence = agent.retriever.grep_corpus(**request)
        add_evidence_result("grep_corpus", evidence, payload)

    def query_rewrite() -> None:
        planned_input = planned_tool_input_from_state(state, "query_rewrite")
        contract: QueryContract = state["contract"]
        result = rewrite_query(**query_rewrite_tool_request(planned_input=planned_input, contract=contract))
        payload, summary = query_rewrite_tool_payload(result=result, state=state)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="query_rewrite",
            summary=summary,
            payload=payload,
        )

    def summarize() -> None:
        planned_input = planned_tool_input_from_state(state, "summarize")
        contract: QueryContract = state["contract"]
        payload = summarize_tool_payload(
            planned_input=planned_input,
            state=state,
            targets=contract.targets,
            fallback_to_summary_source=False,
        )
        state.setdefault("summaries", []).append(payload)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="summarize",
            summary=f"chars={len(payload['summary'])}",
            payload=payload,
        )

    def verify_claim() -> None:
        planned_input = planned_tool_input_from_state(state, "verify_claim")
        payload, summary = verify_claim_tool_payload(planned_input=planned_input, state=state)
        store_claim_check_payload(state=state, payload=payload)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="verify_claim",
            summary=summary,
            payload=payload,
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
        planned_input = planned_tool_input_from_state(state, "fetch_url")
        request = fetch_url_tool_request(planned_input)
        agent._emit_agent_tool_call(emit=emit, tool="fetch_url", arguments=request)
        result = fetch_url_text(client=agent.clients.http_client, **request)
        payload, summary, observation_payload = fetch_url_tool_payload(result)
        state.setdefault("fetched_urls", []).append(payload)
        event_payload = store_fetch_url_evidence_result(agent=agent, state=state, evidence=fetch_url_evidence(result))
        if event_payload is not None:
            emit("web_search", event_payload)
        agent._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="fetch_url",
            summary=summary,
            payload=observation_payload,
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
        "propose_tool": RegisteredAgentTool("propose_tool", propose_tool),
        "bm25_search": RegisteredAgentTool("bm25_search", bm25_search),
        "vector_search": RegisteredAgentTool("vector_search", vector_search),
        "hybrid_search": RegisteredAgentTool("hybrid_search", hybrid_search),
        "rerank": RegisteredAgentTool("rerank", rerank),
        "read_pdf_page": RegisteredAgentTool("read_pdf_page", read_pdf_page),
        "grep_corpus": RegisteredAgentTool("grep_corpus", grep_corpus),
        "query_rewrite": RegisteredAgentTool("query_rewrite", query_rewrite),
        "summarize": RegisteredAgentTool("summarize", summarize),
        "verify_claim": RegisteredAgentTool("verify_claim", verify_claim),
        "search_corpus": RegisteredAgentTool("search_corpus", search_corpus),
        "compose": RegisteredAgentTool("compose", compose, terminal=True),
        "web_search": RegisteredAgentTool("web_search", web_search),
        "fetch_url": RegisteredAgentTool("fetch_url", fetch_url),
        "ask_human": RegisteredAgentTool("ask_human", ask_human, terminal=True),
    }
