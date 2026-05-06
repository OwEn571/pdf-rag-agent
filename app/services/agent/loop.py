from __future__ import annotations

import logging
from typing import Any

from app.domain.models import (
    AssistantCitation,
    AssistantResponse,
    QueryContract,
    SessionTurn,
    VerificationReport,
)
from app.services.agent.compound import run_compound_query_if_needed
from app.services.agent.contract_extraction import extract_agent_query_contract
from app.services.agent.context import AgentRunContext
from app.services.agent.emit import write_turn_trace_safe
from app.services.agent.runtime_summary import build_runtime_summary
from app.services.agent.runtime_helpers import claim_focus_titles
from app.services.clarification.intents import (
    clarification_options_from_contract_notes,
    clarification_tracking_key,
    clear_pending_clarification,
    remember_clarification_attempt,
    reset_clarification_tracking,
    store_pending_clarification,
)
from app.services.clarification.questions import build_agent_clarification_question
from app.services.clarification.limit_runtime import force_best_effort_after_clarification_limit
from app.services.infra.confidence import confidence_from_logprobs, confidence_from_self_consistency, confidence_payload
from app.services.planning.query_shaping import should_use_web_search
from app.services.memory.research import remember_research_outcome
from app.services.contracts.context import contract_notes
from app.services.contracts.session_context import active_research_from_contract, conversation_active_research_from_contract
from app.services.answers.citation_whitelist import audit_answer_citations, build_answer_whitelist


def finish_agent_turn(
    *,
    settings: Any,
    run_context: AgentRunContext,
    final_payload: dict[str, Any],
    logger: logging.Logger | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    write_turn_trace_safe(
        enabled=bool(getattr(settings, "agent_trace_enabled", True)),
        data_dir=settings.data_dir,
        session_id=run_context.session_id,
        events=run_context.events,
        final_payload=final_payload,
        execution_steps=run_context.execution_steps,
        logger=logger,
    )
    return final_payload, run_context.events


def run_compound_turn_if_needed(
    *,
    agent: Any,
    run_context: AgentRunContext,
    query: str,
    clarification_choice: dict[str, Any] | None,
) -> dict[str, Any] | None:
    return run_compound_query_if_needed(
        agent=agent,
        query=query,
        session_id=run_context.session_id,
        session=run_context.session,
        clarification_choice=clarification_choice,
        emit=run_context.emit,
        execution_steps=run_context.execution_steps,
    )


def run_standard_turn(
    *,
    agent: Any,
    run_context: AgentRunContext,
    query: str,
    use_web_search: bool,
    max_web_results: int,
    clarification_choice: dict[str, Any] | None,
    stream_answer: bool,
) -> dict[str, Any]:
    session = run_context.session
    run_context.emit("thinking_delta", {"text": "正在分析问题意图..."})
    contract = extract_agent_query_contract(
        agent=agent,
        query=query,
        session=session,
        clarification_choice=clarification_choice,
    )
    web_enabled = should_use_web_search(use_web_search=use_web_search, contract=contract)
    if web_enabled:
        contract = contract.model_copy(update={"allow_web_search": True})
    run_context.emit("contract", contract.model_dump())
    run_context.execution_steps.append({"node": "query_contract_extractor", "summary": contract.relation})
    agent_plan = agent.planner.plan_actions(contract=contract, session=session, use_web_search=web_enabled)
    run_context.emit("agent_plan", agent_plan)
    run_context.execution_steps.append({"node": "agent_planner", "summary": " -> ".join(agent_plan.get("actions", []))})
    if contract.interaction_mode == "conversation":
        return run_conversation_turn(
            agent=agent,
            run_context=run_context,
            query=query,
            contract=contract,
            agent_plan=agent_plan,
            max_web_results=max_web_results,
        )
    return run_research_turn(
        agent=agent,
        run_context=run_context,
        query=query,
        contract=contract,
        agent_plan=agent_plan,
        web_enabled=web_enabled,
        explicit_web_search=use_web_search,
        max_web_results=max_web_results,
        stream_answer=stream_answer,
    )


def run_conversation_turn(
    *,
    agent: Any,
    run_context: AgentRunContext,
    query: str,
    contract: QueryContract,
    agent_plan: dict[str, Any],
    max_web_results: int,
) -> dict[str, Any]:
    session = run_context.session
    execution_steps = run_context.execution_steps
    conversation_state = agent.runtime.execute_conversation_tools(
        contract=contract,
        query=query,
        session=session,
        agent_plan=agent_plan,
        max_web_results=max_web_results,
        emit=run_context.emit,
        execution_steps=execution_steps,
    )
    answer = str(conversation_state.get("answer", ""))
    citations = [
        item
        for item in list(conversation_state.get("citations", []) or [])
        if isinstance(item, AssistantCitation)
    ]
    verification_payload = dict(conversation_state.get("verification_report", {}) or {"status": "pass"})
    conversation_needs_human = verification_payload.get("status") == "clarify"
    session.last_relation = contract.relation
    citation_titles = [item.title for item in citations if item.title]
    active_research = conversation_active_research_from_contract(contract, titles=citation_titles)
    if conversation_needs_human:
        verification = VerificationReport(
            status="clarify",
            missing_fields=_string_list(verification_payload.get("missing_fields")),
            unsupported_claims=_string_list(verification_payload.get("unsupported_claims")),
            contradictory_claims=_string_list(verification_payload.get("contradictory_claims")),
            recommended_action=str(verification_payload.get("recommended_action", "") or "ask_human"),
        )
        clarification_options = clarification_options_from_contract_notes(contract)
        store_pending_clarification(session=session, contract=contract, options=clarification_options)
        remember_clarification_attempt(
            session=session,
            key=clarification_tracking_key(
                contract=contract,
                verification=verification,
                options=clarification_options,
            ),
        )
    else:
        clear_pending_clarification(session)
        reset_clarification_tracking(session)
    agent.sessions.commit_turn(
        session,
        SessionTurn.from_contract(
            query=query,
            answer=answer,
            contract=contract,
            interaction_mode="conversation",
            titles=citation_titles,
        ),
        active=active_research,
    )
    response = AssistantResponse(
        session_id=run_context.session_id,
        interaction_mode="conversation",
        answer=answer,
        citations=citations,
        query_contract=contract.model_dump(),
        research_plan_summary=agent_plan,
        runtime_summary=build_runtime_summary(
            contract=contract,
            active_research_context=session.active_research_context_payload(),
            tool_plan=agent_plan,
            execution_steps=execution_steps,
            verification_report=verification_payload,
            citations=citations,
        ),
        execution_steps=execution_steps,
        verification_report=verification_payload,
        needs_human=conversation_needs_human,
        clarification_question=build_agent_clarification_question(
            contract=contract,
            session=session,
            clients=agent.clients,
            settings=agent.settings,
        )
        if conversation_needs_human
        else "",
        clarification_options=clarification_options_from_contract_notes(contract) if conversation_needs_human else [],
    )
    return response.model_dump()


def run_research_turn(
    *,
    agent: Any,
    run_context: AgentRunContext,
    query: str,
    contract: QueryContract,
    agent_plan: dict[str, Any],
    web_enabled: bool,
    explicit_web_search: bool,
    max_web_results: int,
    stream_answer: bool,
) -> dict[str, Any]:
    session = run_context.session
    execution_steps = run_context.execution_steps
    agent_state = agent.runtime.run_research_agent_loop(
        contract=contract,
        session=session,
        agent_plan=agent_plan,
        web_enabled=web_enabled,
        explicit_web_search=explicit_web_search,
        max_web_results=max_web_results,
        emit=run_context.emit,
        execution_steps=execution_steps,
    )
    contract = agent_state["contract"]
    plan = agent_state["plan"]
    screened_papers = agent_state["screened_papers"]
    evidence = agent_state["evidence"]
    claims = agent_state["claims"]
    verification = agent_state["verification"]

    if isinstance(verification, VerificationReport) and verification.status == "clarify":
        forced_state = force_best_effort_after_clarification_limit(
            state=agent_state,
            session=session,
            web_enabled=web_enabled,
            explicit_web_search=explicit_web_search,
            max_web_results=max_web_results,
            emit=run_context.emit,
            execution_steps=execution_steps,
            runtime=agent.runtime,
            max_clarification_attempts=agent.agent_settings.max_clarification_attempts,
        )
        if forced_state is not None:
            agent_state = forced_state
            contract = agent_state["contract"]
            plan = agent_state["plan"]
            screened_papers = agent_state["screened_papers"]
            evidence = agent_state["evidence"]
            claims = agent_state["claims"]
            verification = agent_state["verification"]

    answer_logprobs: list[float] = []
    request_answer_logprobs = bool(
        stream_answer and getattr(agent.agent_settings, "answer_logprobs_enabled", False)
    )
    answer, citations = agent._compose_answer(
        contract=contract,
        claims=claims,
        evidence=evidence,
        papers=screened_papers,
        verification=verification,
        session=session,
        stream_callback=(lambda text: run_context.emit("answer_delta", {"text": text})) if stream_answer else None,
        logprob_callback=answer_logprobs.extend if request_answer_logprobs else None,
        request_logprobs=request_answer_logprobs,
    )
    answer_confidence = None
    if request_answer_logprobs:
        answer_confidence = confidence_payload(
            confidence_from_logprobs(
                answer_logprobs,
                min_tokens=int(getattr(agent.agent_settings, "answer_logprobs_min_tokens", 3)),
            )
        )
        agent_state["answer_logprob_confidence"] = answer_confidence
        run_context.emit("confidence", answer_confidence)
    if getattr(agent.agent_settings, "answer_self_consistency_enabled", False):
        try:
            requested_samples = int(getattr(agent.agent_settings, "answer_self_consistency_samples", 3))
        except (TypeError, ValueError):
            requested_samples = 3
        requested_samples = max(2, min(5, requested_samples))
        answer_samples = [answer]
        for _ in range(max(0, requested_samples - 1)):
            sample_answer, _ = agent._compose_answer(
                contract=contract,
                claims=claims,
                evidence=evidence,
                papers=screened_papers,
                verification=verification,
                session=session,
                stream_callback=None,
                logprob_callback=None,
                request_logprobs=False,
            )
            answer_samples.append(sample_answer)
        self_consistency_confidence = confidence_payload(
            confidence_from_self_consistency(answer_samples, min_samples=requested_samples)
        )
        agent_state["answer_self_consistency_confidence"] = self_consistency_confidence
        run_context.emit("confidence", self_consistency_confidence)
        if answer_confidence is None or self_consistency_confidence["score"] < answer_confidence["score"]:
            answer_confidence = self_consistency_confidence
    # ── P0-1: Citation whitelist audit ──
    _allowed_titles = build_answer_whitelist(
        evidence=evidence,
        citations=citations,
        screened_papers=screened_papers,
    )
    _violations = audit_answer_citations(
        answer=answer,
        allowed_titles=_allowed_titles,
        max_citation_index=len(citations),
    )
    if _violations:
        # Retry once with explicit whitelist constraint
        _retry_contract = contract.model_copy(update={
            "notes": list(dict.fromkeys([
                *contract_notes(contract),
                "citation_whitelist_violation",
                "上一轮引用了不在证据集合中的论文，请只引用 evidence/citations 中存在的标题。",
            ]))
        })
        try:
            answer, citations = agent._compose_answer(
                contract=_retry_contract,
                claims=claims,
                evidence=evidence,
                papers=screened_papers,
                verification=verification,
                session=session,
                stream_callback=(lambda text: run_context.emit("answer_delta", {"text": text})) if stream_answer else None,
                logprob_callback=None,
                request_logprobs=False,
            )
        except Exception:
            pass  # keep original answer on retry failure
    def paper_title_lookup(paper_id: str) -> str | None:
        doc = agent.retriever.paper_doc_by_id(paper_id)
        if doc is None:
            return None
        return str((doc.metadata or {}).get("title", ""))

    focus_titles = claim_focus_titles(claims=claims, papers=screened_papers, paper_title_lookup=paper_title_lookup)
    active_titles = focus_titles if verification.status == "pass" else []
    if verification.status == "pass":
        remember_research_outcome(
            session=session,
            contract=contract,
            answer=answer,
            claims=claims,
            papers=screened_papers,
            evidence=evidence,
            citations=citations,
            candidate_lookup=agent._candidate_from_paper_id,
            verification=verification,  # P0-8: prevents best_effort from polluting target_bindings
        )
    session.last_relation = contract.relation
    active_research = active_research_from_contract(contract, titles=active_titles)
    session.answered_titles = list(dict.fromkeys([*session.answered_titles, *active_research.titles]))
    if verification.status == "clarify":
        clarification_options = clarification_options_from_contract_notes(contract)
        store_pending_clarification(session=session, contract=contract, options=clarification_options)
        remember_clarification_attempt(
            session=session,
            key=clarification_tracking_key(
                contract=contract,
                verification=verification,
                options=clarification_options,
            ),
        )
    else:
        clear_pending_clarification(session)
        reset_clarification_tracking(session)
    agent.sessions.commit_turn(
        session,
        SessionTurn.from_contract(
            query=query,
            answer=answer,
            contract=contract,
            titles=focus_titles,
        ),
        active=active_research,
    )

    response = AssistantResponse(
        session_id=run_context.session_id,
        interaction_mode=contract.interaction_mode,
        answer=answer,
        citations=citations,
        query_contract=contract.model_dump(),
        research_plan_summary=plan.model_dump(),
        runtime_summary=build_runtime_summary(
            contract=contract,
            active_research_context=session.active_research_context_payload(),
            tool_plan=agent_plan,
            research_plan=plan.model_dump(),
            execution_steps=execution_steps,
            verification_report=verification.model_dump(),
            answer_confidence=answer_confidence,
            claims=claims,
            citations=citations,
        ),
        execution_steps=execution_steps,
        verification_report=verification.model_dump(),
        needs_human=verification.status == "clarify",
        clarification_question=build_agent_clarification_question(
            contract=contract,
            session=session,
            clients=agent.clients,
            settings=agent.settings,
        )
        if verification.status == "clarify"
        else "",
        clarification_options=clarification_options_from_contract_notes(contract) if verification.status == "clarify" else [],
    )
    return response.model_dump()


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []
