from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from app.domain.models import EvidenceBlock, QueryContract, SessionContext, VerificationReport
from app.services.agent.tool_events import emit_agent_tool_call as emit_agent_tool_call_event
from app.services.agent.task import run_task_subagent
from app.services.agent.tools import RegisteredAgentTool, all_agent_tool_names
from app.services.library.citation_ranking import (
    CITATION_LOOKUP_STAGE,
    format_citation_ranking_answer,
    lookup_candidate_citation_counts,
    select_citation_ranking_candidates,
    semantic_scholar_citation_evidence,
)
from app.services.clarification.questions import build_agent_clarification_question
from app.services.contracts.conversation_memory import active_memory_bindings, memory_binding_doc_ids
from app.services.answers.conversation_state import set_conversation_answer
from app.services.contracts.context import contract_has_note
from app.services.answers.evidence_presentation import citations_from_doc_ids, dedupe_citations
from app.services.memory.artifacts import conversation_tool_result_artifact
from app.services.answers.memory_followup import (
    compose_memory_followup_answer,
    compose_memory_synthesis_answer,
)
from app.services.tools.proposals import run_tool_proposal_sandbox
from app.services.planning.query_rewrite import rewrite_query
from app.services.agent.research_search_handlers import (
    agent_search_evidence,
    agent_search_papers,
    agent_web_search,
)
from app.services.agent.research_compose_handlers import agent_solve_claims
from app.services.agent.research_verification_handlers import agent_verify_grounding
from app.services.contracts.session_context import agent_session_conversation_context
from app.services.tools.registry_helpers import (
    atomic_search_observation_payload,
    atomic_search_tool_request,
    citation_ranking_result_payload,
    conversation_artifact_answer_from_state,
    conversation_clarification_report,
    conversation_intent_summary,
    compose_done_payload,
    evidence_event_payload,
    evidence_result_observation_payload,
    ensure_research_clarification_report,
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
    record_tool_observation,
    read_memory_tool_payload,
    read_pdf_page_tool_request,
    reflect_previous_answer_payload,
    remember_tool_payload,
    rerank_observation_payload,
    rerank_tool_request,
    research_compose_observation_payload,
    research_intent_summary,
    search_corpus_observation_payload,
    search_corpus_strategy,
    store_claim_check_payload,
    store_citation_candidates_payload,
    store_citation_lookup_payload,
    store_conversation_answer_result,
    store_fetched_url_payload,
    store_fetch_url_evidence_result,
    store_research_evidence_result,
    store_summary_payload,
    store_tool_proposal_payload,
    summarize_tool_payload,
    task_result_observation_payload,
    task_tool_request,
    todo_write_tool_payload,
    verify_claim_tool_payload,
)
from app.services.retrieval.url_fetcher import fetch_url as fetch_url_text

EmitFn = Callable[[str, dict[str, Any]], None]

CONVERSATION_ANSWER_STAGE = "conversation_answer"
LIBRARY_STATUS_STAGE = "library_status"
LIBRARY_RECOMMENDATION_STAGE = "library_recommendation"
INTENT_UNDERSTANDING_STAGE = "intent_understanding"
CONVERSATION_MEMORY_READ_STAGE = "conversation_memory_read"
MEMORY_ANSWER_STAGE = "memory_answer"
MEMORY_SYNTHESIS_STAGE = "memory_synthesis"
RECOMMENDATION_RECOVERY_STAGE = "recommendation_recovery"
PREVIOUS_ANSWER_REFLECTION_STAGE = "previous_answer_reflection"
CITATION_RANKING_STAGE = "citation_count_ranking"


def _dynamic_tool_manifests(agent: Any) -> list[dict[str, Any]]:
    enabled = bool(getattr(getattr(agent, "agent_settings", None), "dynamic_tools_enabled", False))
    if not enabled:
        return []
    manifests = getattr(agent, "dynamic_tool_manifests", [])
    return [dict(item) for item in list(manifests or []) if isinstance(item, dict) and str(item.get("name") or "").strip()]


def _add_dynamic_tools(
    *,
    tools: dict[str, RegisteredAgentTool],
    agent: Any,
    state: dict[str, Any],
    contract: QueryContract,
    emit: EmitFn,
    record_observation: Callable[..., None],
) -> dict[str, RegisteredAgentTool]:
    for manifest in _dynamic_tool_manifests(agent):
        name = str(manifest.get("name") or "").strip()
        proposal_path = str(manifest.get("proposal_path") or "").strip()
        if not name or not proposal_path or name in tools:
            continue

        def run_dynamic_tool(arguments: dict[str, Any] | None = None, *, tool_name: str = name, path: str = proposal_path) -> None:
            planned_input = dict(arguments or {}) or planned_tool_input_from_state(state, tool_name)
            dynamic_canonical_names = {*all_agent_tool_names(), tool_name}
            emit_agent_tool_call_event(
                emit=emit,
                tool=tool_name,
                arguments=planned_input,
                canonical_names=dynamic_canonical_names,
            )
            settings = getattr(agent, "agent_settings", None)
            report = run_tool_proposal_sandbox(
                proposal_path=Path(path),
                args=planned_input,
                timeout_seconds=float(getattr(settings, "dynamic_tool_timeout_seconds", 2.0)),
                memory_limit_mb=int(getattr(settings, "dynamic_tool_memory_mb", 256)),
            )
            state.setdefault("dynamic_tool_results", []).append({"tool": tool_name, "report": report})
            result = report.get("result")
            if (
                report.get("status") == "pass"
                and contract.interaction_mode == "conversation"
                and isinstance(result, dict)
                and str(result.get("answer", "") or "").strip()
                and not state.get("answer")
            ):
                set_conversation_answer(
                    state=state,
                    answer=str(result.get("answer", "") or "").strip(),
                    emit=emit,
                )
            record_observation(
                tool=tool_name,
                summary=f"dynamic_tool:{report.get('status', 'unknown')}",
                payload={
                    "status": report.get("status"),
                    "proposal_id": report.get("proposal_id"),
                    "tool_name": report.get("tool_name"),
                    "code_sha256": report.get("code_sha256"),
                    "duration_ms": report.get("duration_ms"),
                    "result": result if report.get("status") == "pass" else None,
                    "error": report.get("error") if report.get("status") != "pass" else None,
                },
                canonical_names=dynamic_canonical_names,
            )

        tools[name] = RegisteredAgentTool(name, run_dynamic_tool, accepts_arguments=True)
    return tools


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
    def record_observation(
        *,
        tool: str,
        summary: str,
        payload: Any,
        canonical_names: set[str] | None = None,
    ) -> None:
        record_tool_observation(
            agent=agent,
            emit=emit,
            execution_steps=execution_steps,
            tool=tool,
            summary=summary,
            payload=payload,
            canonical_names=canonical_names,
        )

    def planned_input(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return dict(arguments or {}) or planned_tool_input_from_state(state, name)

    def intent_summary() -> dict[str, Any]:
        return conversation_intent_summary(contract)

    def understand_user_intent() -> None:
        record_observation(
            tool="read_memory",
            summary=f"intent={intent_summary()['kind']}",
            payload={"stage": INTENT_UNDERSTANDING_STAGE, **intent_summary()},
        )

    def answer_conversation() -> None:
        if state.get("answer"):
            return
        emit_agent_tool_call_event(
            emit=emit,
            tool="compose",
            arguments={"stage": CONVERSATION_ANSWER_STAGE, "intent": intent_summary()},
        )
        answer = agent._compose_conversation_response(contract=contract, query=query, session=session)
        payload = store_conversation_answer_result(
            agent=agent, state=state, session=session, contract=contract, emit=emit,
            tool="answer_conversation", query=query, answer=answer,
        )
        record_observation(
            tool="compose",
            summary="conversation_answer_ready",
            payload={"stage": CONVERSATION_ANSWER_STAGE, **payload},
        )

    def get_library_status() -> None:
        if state.get("answer"):
            return
        emit_agent_tool_call_event(
            emit=emit,
            tool="compose",
            arguments={"stage": LIBRARY_STATUS_STAGE, "query": query},
        )
        answer = agent._compose_library_status_response(query=query)
        payload = store_conversation_answer_result(
            agent=agent, state=state, session=session, contract=contract, emit=emit,
            tool="get_library_status", query=query, answer=answer,
        )
        record_observation(
            tool="compose",
            summary="library_status_ready",
            payload={"stage": LIBRARY_STATUS_STAGE, **payload},
        )

    def query_library_metadata(arguments: dict[str, Any] | None = None) -> None:
        state["library_metadata_attempted"] = True
        request = library_metadata_tool_request(planned_input=planned_input("query_library_metadata", arguments), fallback_query=query)
        emit_agent_tool_call_event(
            emit=emit,
            tool="query_library_metadata",
            arguments=request,
        )
        result = agent._compose_library_metadata_query_response(query=str(request.get("query", "") or ""))
        state["library_metadata_result"] = result
        answer = str(result.get("answer", "") or "").strip()
        if answer:
            artifact = conversation_tool_result_artifact(tool="query_library_metadata", result=result)
            store_conversation_answer_result(
                agent=agent, state=state, session=session, contract=contract, emit=emit,
                tool="query_library_metadata", query=query, answer=answer, artifact=artifact,
            )
        summary, payload = library_metadata_observation_payload(result=result, answer=answer)
        record_observation(
            tool="query_library_metadata",
            summary=summary,
            payload=payload,
        )

    def get_library_recommendation() -> None:
        if state.get("answer"):
            return
        emit_agent_tool_call_event(
            emit=emit,
            tool="compose",
            arguments={"stage": LIBRARY_RECOMMENDATION_STAGE, "query": query},
        )
        answer = agent._compose_library_recommendation_response(query=query, session=session)
        payload = store_conversation_answer_result(
            agent=agent, state=state, session=session, contract=contract, emit=emit,
            tool="get_library_recommendation", query=query, answer=answer,
        )
        record_observation(
            tool="compose",
            summary="library_recommendation_ready",
            payload={"stage": LIBRARY_RECOMMENDATION_STAGE, **payload},
        )

    def read_conversation_memory() -> None:
        call_arguments, summary, payload = read_memory_tool_payload(agent=agent, session=session)
        emit_agent_tool_call_event(
            emit=emit,
            tool="read_memory",
            arguments={"stage": CONVERSATION_MEMORY_READ_STAGE, **call_arguments},
        )
        record_observation(
            tool="read_memory",
            summary=summary,
            payload={"stage": CONVERSATION_MEMORY_READ_STAGE, **payload},
        )

    def todo_write(arguments: dict[str, Any] | None = None) -> None:
        items, payload, summary = todo_write_tool_payload(planned_input=planned_input("todo_write", arguments), session=session)
        emit("todo_update", {"items": items})
        record_observation(
            tool="todo_write",
            summary=summary,
            payload=payload,
        )

    def remember(arguments: dict[str, Any] | None = None) -> None:
        payload, summary = remember_tool_payload(data_dir=agent.settings.data_dir, planned_input=planned_input("remember", arguments), state=state)
        record_observation(
            tool="remember",
            summary=summary,
            payload=payload,
        )

    def propose_tool(arguments: dict[str, Any] | None = None) -> None:
        payload = propose_tool_payload(agent, planned_input("propose_tool", arguments))
        summary = store_tool_proposal_payload(state=state, payload=payload)
        emit("tool_proposal", payload)
        if payload.get("status") == "pending_review" and not state.get("answer"):
            set_conversation_answer(state=state, answer="已记录工具提案，等待人工审核后才能启用。", emit=emit)
        record_observation(
            tool="propose_tool",
            summary=summary,
            payload=payload,
        )

    def summarize(arguments: dict[str, Any] | None = None) -> None:
        payload = summarize_tool_payload(
            planned_input=planned_input("summarize", arguments),
            state=state,
            targets=contract.targets,
            fallback_to_summary_source=True,
        )
        summary = store_summary_payload(state=state, payload=payload)
        if payload["summary"] and not state.get("answer"):
            set_conversation_answer(state=state, answer=payload["summary"], emit=emit)
        record_observation(
            tool="summarize",
            summary=summary,
            payload=payload,
        )

    def verify_claim(arguments: dict[str, Any] | None = None) -> None:
        payload, summary = verify_claim_tool_payload(planned_input=planned_input("verify_claim", arguments), state=state)
        store_claim_check_payload(state=state, payload=payload)
        record_observation(
            tool="verify_claim",
            summary=summary,
            payload=payload,
        )

    def run_task(arguments: dict[str, Any] | None = None) -> None:
        request = task_tool_request(planned_input=planned_input("Task", arguments), fallback_prompt=query)
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
        record_observation(
            tool="Task",
            summary=summary,
            payload=payload,
        )

    def answer_from_memory() -> None:
        if state.get("answer"):
            return
        emit_agent_tool_call_event(
            emit=emit,
            tool="read_memory",
            arguments={"stage": MEMORY_ANSWER_STAGE, "query": query, "targets": contract.targets},
        )
        answer = compose_memory_followup_answer(
            query=query,
            session=session,
            contract=contract,
            clients=agent.clients,
            conversation_context=lambda current_session, *, max_chars=24000: agent_session_conversation_context(
                current_session,
                settings=agent.settings,
                max_chars=max_chars,
            ),
            clean_text=agent._clean_common_ocr_artifacts,
        )
        payload = store_conversation_answer_result(
            agent=agent, state=state, session=session, contract=contract, emit=emit,
            tool="answer_from_memory", query=query, answer=answer,
        )
        record_observation(
            tool="read_memory",
            summary="memory_answer_ready",
            payload={"stage": MEMORY_ANSWER_STAGE, **payload},
        )

    def synthesize_previous_results() -> None:
        if state.get("answer"):
            return
        emit_agent_tool_call_event(
            emit=emit,
            tool="read_memory",
            arguments={"stage": MEMORY_SYNTHESIS_STAGE, "targets": contract.targets},
        )
        answer = compose_memory_synthesis_answer(
            query=query,
            session=session,
            contract=contract,
            clients=agent.clients,
            conversation_context=lambda current_session, *, max_chars=24000: agent_session_conversation_context(
                current_session,
                settings=agent.settings,
                max_chars=max_chars,
            ),
            clean_text=agent._clean_common_ocr_artifacts,
        )
        bindings = active_memory_bindings(session)
        state["citations"] = dedupe_citations(
            citations_from_doc_ids(
                memory_binding_doc_ids(bindings),
                [],
                block_doc_lookup=agent.retriever.block_doc_by_id,
                paper_doc_lookup=agent.retriever.paper_doc_by_id,
            )
        )
        payload = store_conversation_answer_result(
            agent=agent, state=state, session=session, contract=contract, emit=emit,
            tool="synthesize_previous_results", query=query, answer=answer,
        )
        record_observation(
            tool="read_memory",
            summary="memory_synthesis_ready",
            payload={"stage": MEMORY_SYNTHESIS_STAGE, "binding_count": len(bindings), **payload},
        )

    def recover_previous_recommendation_candidates() -> None:
        if state.get("citation_candidates"):
            return
        candidates = select_citation_ranking_candidates(
            paper_documents=list(agent.retriever.paper_documents()),
            session=session,
            query=query,
            limit=6,
            rank_library_papers_for_recommendation=agent._rank_library_papers_for_recommendation,
        )
        summary, payload = store_citation_candidates_payload(state=state, candidates=candidates)
        emit_agent_tool_call_event(
            emit=emit,
            tool="read_memory",
            arguments={"stage": RECOMMENDATION_RECOVERY_STAGE, "query": query, "limit": 6},
        )
        record_observation(
            tool="read_memory",
            summary=summary,
            payload={"stage": RECOMMENDATION_RECOVERY_STAGE, **payload},
        )

    def web_citation_lookup() -> None:
        if state.get("citation_lookup"):
            return
        candidates = list(state.get("citation_candidates", []) or [])
        lookup = lookup_candidate_citation_counts(
            candidates=candidates,
            max_web_results=max_web_results,
            web_search=agent.web_search,
            emit=emit,
            emit_tool_call=lambda tool, arguments: emit_agent_tool_call_event(emit=emit, tool=tool, arguments=arguments),
            record_observation=lambda tool, summary, payload: record_observation(
                tool=tool,
                summary=summary,
                payload=payload,
            ),
            semantic_scholar_lookup=lambda title: semantic_scholar_citation_evidence(
                title=title,
                web_search=agent.web_search,
                timeout_seconds=float(agent.settings.tavily_timeout_seconds),
            ),
        )
        summary, payload = store_citation_lookup_payload(state=state, lookup=lookup)
        record_observation(
            tool="web_search",
            summary=summary,
            payload={"stage": CITATION_LOOKUP_STAGE, **payload},
        )

    def rank_by_verified_citation_count() -> None:
        candidates = list(state.get("citation_candidates", []) or [])
        lookup = dict(state.get("citation_lookup", {}) or {})
        answer = format_citation_ranking_answer(
            candidates=candidates,
            citation_results=list(lookup.get("results", []) or []),
            web_enabled=bool(lookup.get("web_enabled")),
        )
        evidence, citation_doc_ids, report, summary = citation_ranking_result_payload(lookup)
        state["citations"] = dedupe_citations(
            citations_from_doc_ids(
                citation_doc_ids,
                evidence,
                block_doc_lookup=agent.retriever.block_doc_by_id,
                paper_doc_lookup=agent.retriever.paper_doc_by_id,
            )
        )
        state["verification_report"] = report
        store_conversation_answer_result(
            agent=agent, state=state, session=session, contract=contract, emit=emit,
            tool="rank_by_verified_citation_count", query=query, answer=answer,
        )
        record_observation(
            tool="compose",
            summary=summary,
            payload={"stage": CITATION_RANKING_STAGE, **state["verification_report"]},
        )

    def web_search() -> None:
        is_citation_turn = "citation_count_ranking" in contract.requested_fields or contract_has_note(
            contract,
            "citation_count_requires_web",
        )
        if is_citation_turn:
            if not state.get("citation_candidates"):
                recover_previous_recommendation_candidates()
            web_citation_lookup()
            return
        record_observation(
            tool="web_search",
            summary="conversation_web_search_skipped",
            payload={"intent": intent_summary()},
        )

    def fetch_url(arguments: dict[str, Any] | None = None) -> None:
        request = fetch_url_tool_request(planned_input("fetch_url", arguments))
        emit_agent_tool_call_event(emit=emit, tool="fetch_url", arguments=request)
        result = fetch_url_text(client=agent.clients.http_client, **request)
        payload, summary, observation_payload = fetch_url_tool_payload(result)
        store_fetched_url_payload(state=state, payload=payload)
        record_observation(
            tool="fetch_url",
            summary=summary,
            payload=observation_payload,
        )

    # Dispatch table: relation -> ordered list of step functions.
    # Each step is self-guarded (returns early if preconditions already met).
    # Add new conversation relation types here instead of writing if/elif chains.
    _COMPOSE_RELATION_STEPS: dict[str, list[Callable[[], None]]] = {
        "library_status": [query_library_metadata, get_library_status],
        "library_recommendation": [get_library_recommendation],
        "memory_followup": [answer_from_memory],
        "memory_synthesis": [synthesize_previous_results],
        "library_citation_ranking": [
            recover_previous_recommendation_candidates,
            web_citation_lookup,
            rank_by_verified_citation_count,
        ],
    }

    def compose() -> None:
        artifact_answer_matched, answer, citations = conversation_artifact_answer_from_state(state)
        if artifact_answer_matched and not state.get("answer"):
            if answer:
                if citations:
                    state["citations"] = citations
                set_conversation_answer(state=state, answer=answer, emit=emit)
        else:
            steps = _COMPOSE_RELATION_STEPS.get(contract.relation)
            if steps is not None:
                for step_fn in steps:
                    step_fn()
            elif not state.get("answer"):
                answer_conversation()
        record_observation(
            tool="compose",
            summary="done",
            payload=compose_done_payload(state),
        )

    def ask_human() -> None:
        state["verification_report"] = conversation_clarification_report(contract)
        answer = build_agent_clarification_question(
            contract=contract,
            session=session,
            clients=agent.clients,
            settings=agent.settings,
        )
        set_conversation_answer(state=state, answer=answer, emit=emit)
        record_observation(
            tool="ask_human",
            summary="conversation_clarification",
            payload=state["verification_report"],
        )

    tools = {
        "read_memory": RegisteredAgentTool("read_memory", read_conversation_memory),
        "todo_write": RegisteredAgentTool("todo_write", todo_write, accepts_arguments=True),
        "remember": RegisteredAgentTool("remember", remember, accepts_arguments=True),
        "propose_tool": RegisteredAgentTool("propose_tool", propose_tool, accepts_arguments=True),
        "summarize": RegisteredAgentTool("summarize", summarize, accepts_arguments=True),
        "verify_claim": RegisteredAgentTool("verify_claim", verify_claim, accepts_arguments=True),
        "Task": RegisteredAgentTool("Task", run_task, accepts_arguments=True),
        "web_search": RegisteredAgentTool("web_search", web_search),
        "fetch_url": RegisteredAgentTool("fetch_url", fetch_url, accepts_arguments=True),
        "query_library_metadata": RegisteredAgentTool("query_library_metadata", query_library_metadata, accepts_arguments=True),
        "compose": RegisteredAgentTool("compose", compose, terminal=True),
        "ask_human": RegisteredAgentTool("ask_human", ask_human, terminal=True),
    }
    return _add_dynamic_tools(
        tools=tools,
        agent=agent,
        state=state,
        contract=contract,
        emit=emit,
        record_observation=record_observation,
    )


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
    def record_observation(
        *,
        tool: str,
        summary: str,
        payload: Any,
        canonical_names: set[str] | None = None,
    ) -> None:
        record_tool_observation(
            agent=agent,
            emit=emit,
            execution_steps=execution_steps,
            tool=tool,
            summary=summary,
            payload=payload,
            canonical_names=canonical_names,
        )

    def planned_input(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return dict(arguments or {}) or planned_tool_input_from_state(state, name)

    def understand_user_intent() -> None:
        contract: QueryContract = state["contract"]
        summary, payload = research_intent_summary(contract)
        record_observation(
            tool="read_memory",
            summary=summary,
            payload={"stage": INTENT_UNDERSTANDING_STAGE, **payload},
        )

    def reflect_previous_answer() -> None:
        summary, payload = reflect_previous_answer_payload(state)
        record_observation(
            tool="read_memory",
            summary=summary,
            payload={"stage": PREVIOUS_ANSWER_REFLECTION_STAGE, **payload},
        )

    def read_memory() -> None:
        call_arguments, summary, payload = read_memory_tool_payload(agent=agent, session=session, active_title_limit=4)
        emit_agent_tool_call_event(
            emit=emit,
            tool="read_memory",
            arguments=call_arguments,
        )
        record_observation(
            tool="read_memory",
            summary=summary,
            payload=payload,
        )

    def todo_write(arguments: dict[str, Any] | None = None) -> None:
        items, payload, summary = todo_write_tool_payload(planned_input=planned_input("todo_write", arguments), session=session)
        emit("todo_update", {"items": items})
        record_observation(
            tool="todo_write",
            summary=summary,
            payload=payload,
        )

    def remember(arguments: dict[str, Any] | None = None) -> None:
        payload, summary = remember_tool_payload(data_dir=agent.settings.data_dir, planned_input=planned_input("remember", arguments), state=state)
        record_observation(
            tool="remember",
            summary=summary,
            payload=payload,
        )

    def propose_tool(arguments: dict[str, Any] | None = None) -> None:
        payload = propose_tool_payload(agent, planned_input("propose_tool", arguments))
        summary = store_tool_proposal_payload(state=state, payload=payload)
        emit("tool_proposal", payload)
        record_observation(
            tool="propose_tool",
            summary=summary,
            payload=payload,
        )

    def search_papers(arguments: dict[str, Any] | None = None) -> None:
        agent_search_papers(
            agent=agent,
            state=state,
            session=session,
            emit=emit,
            execution_steps=execution_steps,
            tool_input=planned_input("search_corpus", arguments),
        )

    def search_corpus(arguments: dict[str, Any] | None = None) -> None:
        request_input = planned_input("search_corpus", arguments)
        strategy = search_corpus_strategy(request_input)
        if strategy in {"bm25", "vector", "hybrid"}:
            run_atomic_search(f"{strategy}_search", request_input)
            return
        if not state.get("screened_papers"):
            search_papers(request_input)
        if not state.get("evidence"):
            search_evidence(request_input)
        summary, payload = search_corpus_observation_payload(state)
        record_observation(
            tool="search_corpus",
            summary=summary,
            payload=payload,
        )

    def search_evidence(arguments: dict[str, Any] | None = None) -> None:
        agent_search_evidence(
            agent=agent,
            state=state,
            emit=emit,
            execution_steps=execution_steps,
            tool_input=planned_input("search_corpus", arguments),
        )

    def run_atomic_search(name: str, arguments: dict[str, Any] | None = None) -> None:
        request = atomic_search_tool_request(
            name=name,
            planned_input=planned_input(name, arguments),
            state=state,
            default_limit=agent.settings.evidence_limit_default,
        )
        evidence = getattr(agent.retriever, name)(**request)
        papers = store_research_evidence_result(agent=agent, state=state, evidence=evidence)
        emit("evidence", evidence_event_payload(list(state.get("evidence", []) or [])))
        summary, payload = atomic_search_observation_payload(request=request, evidence=evidence, paper_count=len(papers))
        record_observation(
            tool=name,
            summary=summary,
            payload=payload,
        )

    def bm25_search(arguments: dict[str, Any] | None = None) -> None:
        run_atomic_search("bm25_search", arguments)

    def vector_search(arguments: dict[str, Any] | None = None) -> None:
        run_atomic_search("vector_search", arguments)

    def hybrid_search(arguments: dict[str, Any] | None = None) -> None:
        run_atomic_search("hybrid_search", arguments)

    def rerank(arguments: dict[str, Any] | None = None) -> None:
        default_top_k = getattr(agent.settings, "evidence_limit_default", 12)
        request, payload_context = rerank_tool_request(
            planned_input=planned_input("rerank", arguments),
            state=state,
            default_top_k=default_top_k,
        )
        evidence = agent.retriever.rerank_evidence(**request)
        state["evidence"] = evidence
        emit("evidence", evidence_event_payload(evidence))
        summary, payload = rerank_observation_payload(request=request, payload_context=payload_context, evidence=evidence)
        record_observation(
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
        record_observation(
            tool=name,
            summary=summary,
            payload=observation_payload,
        )

    def read_pdf_page(arguments: dict[str, Any] | None = None) -> None:
        request = read_pdf_page_tool_request(planned_input=planned_input("read_pdf_page", arguments), state=state)
        evidence = agent.retriever.read_pdf_pages(**request)
        add_evidence_result("read_pdf_page", evidence, request)

    def grep_corpus(arguments: dict[str, Any] | None = None) -> None:
        request, payload = grep_corpus_tool_request(planned_input=planned_input("grep_corpus", arguments), state=state)
        evidence = agent.retriever.grep_corpus(**request)
        add_evidence_result("grep_corpus", evidence, payload)

    def query_rewrite(arguments: dict[str, Any] | None = None) -> None:
        contract: QueryContract = state["contract"]
        result = rewrite_query(**query_rewrite_tool_request(planned_input=planned_input("query_rewrite", arguments), contract=contract))
        payload, summary = query_rewrite_tool_payload(result=result, state=state)
        record_observation(
            tool="query_rewrite",
            summary=summary,
            payload=payload,
        )

    def summarize(arguments: dict[str, Any] | None = None) -> None:
        contract: QueryContract = state["contract"]
        payload = summarize_tool_payload(
            planned_input=planned_input("summarize", arguments),
            state=state,
            targets=contract.targets,
            fallback_to_summary_source=False,
        )
        summary = store_summary_payload(state=state, payload=payload)
        record_observation(
            tool="summarize",
            summary=summary,
            payload=payload,
        )

    def verify_claim(arguments: dict[str, Any] | None = None) -> None:
        payload, summary = verify_claim_tool_payload(planned_input=planned_input("verify_claim", arguments), state=state)
        store_claim_check_payload(state=state, payload=payload)
        record_observation(
            tool="verify_claim",
            summary=summary,
            payload=payload,
        )

    def run_task(arguments: dict[str, Any] | None = None) -> None:
        contract: QueryContract = state["contract"]
        request = task_tool_request(planned_input=planned_input("Task", arguments), fallback_prompt=contract.clean_query)
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
        record_observation(
            tool="Task",
            summary=summary,
            payload=payload,
        )

    def web_search(arguments: dict[str, Any] | None = None) -> None:
        agent_web_search(
            agent=agent,
            state=state,
            web_enabled=web_enabled,
            max_web_results=max_web_results,
            emit=emit,
            execution_steps=execution_steps,
            tool_input=planned_input("web_search", arguments),
        )

    def fetch_url(arguments: dict[str, Any] | None = None) -> None:
        request = fetch_url_tool_request(planned_input("fetch_url", arguments))
        emit_agent_tool_call_event(emit=emit, tool="fetch_url", arguments=request)
        result = fetch_url_text(client=agent.clients.http_client, **request)
        payload, summary, observation_payload = fetch_url_tool_payload(result)
        store_fetched_url_payload(state=state, payload=payload)
        event_payload = store_fetch_url_evidence_result(agent=agent, state=state, evidence=fetch_url_evidence(result))
        if event_payload is not None:
            emit("web_search", event_payload)
        record_observation(
            tool="fetch_url",
            summary=summary,
            payload=observation_payload,
        )

    def solve_claims() -> None:
        agent_solve_claims(
            agent=agent,
            state=state,
            session=session,
            explicit_web_search=explicit_web_search,
            emit=emit,
            execution_steps=execution_steps,
        )

    def verify_grounding() -> None:
        if state["verification"] is None:
            agent_verify_grounding(
                agent=agent,
                state=state,
                session=session,
                emit=emit,
                execution_steps=execution_steps,
            )

    def ask_human() -> None:
        verification = ensure_research_clarification_report(state)
        record_observation(
            tool="ask_human",
            summary=str(verification.recommended_action),
            payload=verification.model_dump(),
        )

    def compose() -> None:
        verification = state.get("verification")
        if not isinstance(verification, VerificationReport) or verification.status != "clarify":
            if not state.get("claims"):
                solve_claims()
            if state.get("verification") is None:
                verify_grounding()
        summary, payload = research_compose_observation_payload(state)
        record_observation(
            tool="compose",
            summary=summary,
            payload=payload,
        )

    tools = {
        "read_memory": RegisteredAgentTool("read_memory", read_memory),
        "todo_write": RegisteredAgentTool("todo_write", todo_write, accepts_arguments=True),
        "remember": RegisteredAgentTool("remember", remember, accepts_arguments=True),
        "propose_tool": RegisteredAgentTool("propose_tool", propose_tool, accepts_arguments=True),
        "bm25_search": RegisteredAgentTool("bm25_search", bm25_search, accepts_arguments=True),
        "vector_search": RegisteredAgentTool("vector_search", vector_search, accepts_arguments=True),
        "hybrid_search": RegisteredAgentTool("hybrid_search", hybrid_search, accepts_arguments=True),
        "rerank": RegisteredAgentTool("rerank", rerank, accepts_arguments=True),
        "read_pdf_page": RegisteredAgentTool("read_pdf_page", read_pdf_page, accepts_arguments=True),
        "grep_corpus": RegisteredAgentTool("grep_corpus", grep_corpus, accepts_arguments=True),
        "query_rewrite": RegisteredAgentTool("query_rewrite", query_rewrite, accepts_arguments=True),
        "summarize": RegisteredAgentTool("summarize", summarize, accepts_arguments=True),
        "verify_claim": RegisteredAgentTool("verify_claim", verify_claim, accepts_arguments=True),
        "Task": RegisteredAgentTool("Task", run_task, accepts_arguments=True),
        "search_corpus": RegisteredAgentTool("search_corpus", search_corpus, accepts_arguments=True),
        "compose": RegisteredAgentTool("compose", compose, terminal=True),
        "web_search": RegisteredAgentTool("web_search", web_search, accepts_arguments=True),
        "fetch_url": RegisteredAgentTool("fetch_url", fetch_url, accepts_arguments=True),
        "ask_human": RegisteredAgentTool("ask_human", ask_human, terminal=True),
    }
    return _add_dynamic_tools(
        tools=tools,
        agent=agent,
        state=state,
        contract=state["contract"],
        emit=emit,
        record_observation=record_observation,
    )
