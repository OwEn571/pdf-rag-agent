from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable
from uuid import uuid4

from app.core.agent_settings import AgentSettings
from app.core.config import Settings
from app.domain.models import (
    AssistantCitation,
    AssistantResponse,
    CandidatePaper,
    Claim,
    DisambiguationJudgeDecision,
    EvidenceBlock,
    QueryContract,
    ResearchPlan,
    SessionContext,
    VerificationReport,
)
from app.services.agent_context import AgentRunContext
from app.services.agent_loop import finish_agent_turn, run_compound_turn_if_needed, run_standard_turn
from app.services.agent_task import run_task_subagent
from app.services.model_clients import ModelClients
from app.services.learnings import load_learnings
from app.services.agent_emit import (
    emit_agent_tool_call as emit_agent_tool_call_event,
    record_agent_observation as record_agent_observation_event,
)
from app.services.agent_events import normalize_agent_event
from app.services.agent_planner import AgentPlanner
from app.services.agent_runtime import AgentRuntime
from app.services.agent_runtime_summary import build_runtime_summary
from app.services.agent_runtime_helpers import (
    claim_focus_titles,
    clarify_retry_verification_if_needed,
    excluded_focus_titles,
    clarification_limit_decision,
    prepare_retry_research_materials,
    promote_best_effort_state_after_clarification_limit,
    reflect_agent_state_decision,
    retry_research_limits,
    refresh_selected_ambiguity_materials,
    run_agent_paper_search,
    run_retry_verification_from_materials,
    screen_agent_papers,
    search_agent_evidence,
    solve_agent_state_claims,
    verification_observation_payload,
    verify_grounding_tool_call_arguments,
)
from app.services.clarification_intents import (
    acronym_evidence_from_corpus as build_acronym_evidence_from_corpus,
    acronym_options_from_evidence as build_acronym_options_from_evidence,
    clarification_tracking_key,
    clarification_options_from_contract_notes,
    contract_from_pending_clarification,
    disambiguation_goal_markers,
    evidence_disambiguation_options,
    disambiguation_judge_human_prompt,
    disambiguation_judge_option_payload,
    disambiguation_judge_system_prompt,
    next_clarification_attempt,
    remember_clarification_attempt,
    resolve_disambiguation_judge_decision,
    selected_clarification_paper_id,
    store_pending_clarification,
)
from app.services.clarification_question_helpers import build_clarification_question
from app.services.compound_task_helpers import (
    compound_task_label,
    compound_task_result_from_task_payload,
)
from app.services.contract_normalization import (
    normalize_contract_targets,
)
from app.services.contextual_contract_helpers import (
    contextual_active_paper_contract,
    formula_answer_correction_contract,
    formula_contextual_paper_contract,
    formula_followup_target,
    formula_location_followup_contract,
    formula_query_allows_paper_context,
    paper_context_supports_formula_target,
    paper_from_query_hint,
    paper_scope_correction_contract,
)
from app.services.conversation_memory_contract import (
    active_memory_bindings,
    apply_conversation_memory_to_contract,
    target_binding_from_memory,
)
from app.services.conversation_contract_helpers import normalize_conversation_tool_contract
from app.services.followup_intents import (
    is_negative_correction_query,
    looks_like_active_paper_reference,
    looks_like_formula_answer_correction,
    looks_like_formula_location_correction,
    looks_like_paper_scope_correction,
)
from app.services.followup_relationship_contracts import (
    inherit_followup_relationship_contract,
    normalize_followup_direction_contract,
)
from app.services.evidence_presentation import (
    chunk_text,
    citations_from_doc_ids,
)
from app.services.followup_candidate_helpers import (
    expand_followup_candidate_pool,
    rank_followup_candidates,
    resolve_followup_seed_papers,
    selected_followup_candidate_assessment,
)
from app.services.intent import IntentRecognizer
from app.services.intent_router import LLMIntentRouter, query_contract_from_router_decision
from app.services.pdf_rendering import render_pdf_page_image_data_url
from app.services.query_shaping import (
    extract_targets,
    is_short_acronym,
)
from app.services.research_planning import (
    research_plan_goals,
)
from app.services.research_memory import remember_compound_outcome, remember_research_outcome
from app.services.tool_registry_helpers import tool_input_from_state
from app.services.web_evidence import (
    build_web_research_claim,
    collect_web_evidence,
    search_agent_web_evidence,
)
from app.services.web_search import TavilyWebSearchClient
from app.services.agent_mixins import (
    AnswerComposerMixin,
    ClaimVerifierMixin,
    ConceptReasoningMixin,
    EntityDefinitionMixin,
    FollowupRoutingMixin,
    SolverPipelineMixin,
)
from app.services.retrieval import DualIndexRetriever
from app.services.session_store import SessionStore
from app.services.session_context_helpers import (
    apply_session_history_compression,
    session_conversation_context,
    session_history_compression_payload,
    session_history_compression_system_prompt,
    session_history_compression_window,
    session_llm_history_messages,
    truncate_context_text,
)

logger = logging.getLogger(__name__)


class ResearchAssistantAgentV4(
    FollowupRoutingMixin,
    AnswerComposerMixin,
    ConceptReasoningMixin,
    EntityDefinitionMixin,
    SolverPipelineMixin,
    ClaimVerifierMixin,
):
    def __init__(
        self,
        *,
        settings: Settings,
        retriever: DualIndexRetriever,
        clients: ModelClients,
        sessions: SessionStore,
        web_search: TavilyWebSearchClient | None = None,
    ) -> None:
        self.settings = settings
        self.agent_settings = AgentSettings.from_settings(settings)
        self.retriever = retriever
        self.clients = clients
        self.sessions = sessions
        self.web_search = web_search or TavilyWebSearchClient(settings)
        self.planner = AgentPlanner(
            clients=self.clients,
            conversation_context=self._session_conversation_context,
            conversation_messages=lambda session: session_llm_history_messages(
                session,
                max_turns=6,
                answer_limit=900,
            ),
            is_negative_correction_query=is_negative_correction_query,
            confidence_floor=self.agent_settings.confidence_floor,
        )
        self.intent_router = IntentRecognizer(
            clients=self.clients,
            conversation_context=lambda session: self._session_conversation_context(session, max_chars=12000),
            conversation_messages=lambda session: session_llm_history_messages(
                session,
                max_turns=6,
                answer_limit=900,
            ),
            normalize_targets=lambda targets, requested_fields: normalize_contract_targets(
                targets=targets,
                requested_fields=requested_fields,
                canonicalize_targets=self.retriever.canonicalize_targets,
            ),
        )
        self.llm_intent_router = LLMIntentRouter(
            clients=self.clients,
            conversation_context=lambda session: self._session_conversation_context(session, max_chars=12000),
            conversation_messages=lambda session: session_llm_history_messages(
                session,
                max_turns=6,
                answer_limit=900,
            ),
        )
        self.runtime = AgentRuntime(agent=self)
        self._rendered_page_data_url_cache: dict[tuple[str, int], str] = {}

    def chat(
        self,
        *,
        query: str,
        session_id: str | None = None,
        mode: str = "auto",
        use_web_search: bool = False,
        max_web_results: int = 3,
        clarification_choice: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result, _ = self._run(
            query=query,
            session_id=session_id,
            mode=mode,
            use_web_search=use_web_search,
            max_web_results=max_web_results,
            clarification_choice=clarification_choice,
        )
        return result

    async def achat(
        self,
        *,
        query: str,
        session_id: str | None = None,
        mode: str = "auto",
        use_web_search: bool = False,
        max_web_results: int = 3,
        clarification_choice: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self.chat,
            query=query,
            session_id=session_id,
            mode=mode,
            use_web_search=use_web_search,
            max_web_results=max_web_results,
            clarification_choice=clarification_choice,
        )

    async def astream_chat_events(
        self,
        *,
        query: str,
        session_id: str | None = None,
        mode: str = "auto",
        use_web_search: bool = False,
        max_web_results: int = 3,
        clarification_choice: dict[str, Any] | None = None,
    ) -> Any:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        def emit_event(item: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        worker = asyncio.create_task(
            asyncio.to_thread(
                self._run,
                query=query,
                session_id=session_id,
                mode=mode,
                use_web_search=use_web_search,
                max_web_results=max_web_results,
                clarification_choice=clarification_choice,
                event_callback=emit_event,
            )
        )
        worker.add_done_callback(lambda _task: loop.call_soon_threadsafe(queue.put_nowait, sentinel))
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item
        result, emitted_events = await worker
        final_payload = {k: v for k, v in result.items() if k != "answer"}
        answer_was_streamed = any(item.get("event") == "answer_delta" for item in emitted_events)
        if not answer_was_streamed:
            for chunk in chunk_text(str(result.get("answer", "")), size=28):
                yield {"event": "answer_delta", "data": {"text": chunk}}
        yield {
            "event": "final",
            "data": normalize_agent_event("final", final_payload | {"answer": result.get("answer", "")}),
        }

    def _run(
        self,
        *,
        query: str,
        session_id: str | None,
        mode: str,
        use_web_search: bool,
        max_web_results: int,
        clarification_choice: dict[str, Any] | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        resolved_session_id = session_id or uuid4().hex[:12]
        session = self.sessions.get(resolved_session_id)
        self._compress_session_history_if_needed(session)
        run_context = AgentRunContext.create(
            session_id=resolved_session_id,
            session=session,
            event_callback=event_callback,
        )
        emit = run_context.emit

        emit("session", {"session_id": resolved_session_id})
        compound_result = run_compound_turn_if_needed(
            agent=self,
            run_context=run_context,
            query=query,
            clarification_choice=clarification_choice,
        )
        if compound_result is not None:
            return finish_agent_turn(
                settings=self.settings,
                run_context=run_context,
                final_payload=compound_result,
                logger=logger,
            )

        payload = run_standard_turn(
            agent=self,
            run_context=run_context,
            query=query,
            mode=mode,
            use_web_search=use_web_search,
            max_web_results=max_web_results,
            clarification_choice=clarification_choice,
            stream_answer=event_callback is not None,
        )
        return finish_agent_turn(
            settings=self.settings,
            run_context=run_context,
            final_payload=payload,
            logger=logger,
        )

    def _set_conversation_answer(
        self,
        *,
        state: dict[str, Any],
        answer: str,
        emit: Callable[[str, dict[str, Any]], None],
    ) -> None:
        state["answer"] = answer
        for chunk in chunk_text(str(answer or ""), size=96):
            emit("answer_delta", {"text": chunk})

    def _runtime_summary(
        self,
        *,
        contract: QueryContract,
        session: SessionContext | None = None,
        tool_plan: dict[str, Any] | None = None,
        research_plan: dict[str, Any] | None = None,
        execution_steps: list[dict[str, Any]] | None = None,
        verification_report: dict[str, Any] | None = None,
        answer_confidence: dict[str, Any] | None = None,
        claims: list[Claim] | None = None,
        citations: list[AssistantCitation] | None = None,
    ) -> dict[str, Any]:
        return build_runtime_summary(
            contract=contract,
            active_research_context=session.active_research_context_payload() if session is not None else None,
            tool_plan=tool_plan,
            research_plan=research_plan,
            execution_steps=execution_steps,
            verification_report=verification_report,
            answer_confidence=answer_confidence,
            claims=claims,
            citations=citations,
        )

    def _execute_compound_task_subagent(
        self,
        *,
        contract: QueryContract,
        session: SessionContext,
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        task_result = run_task_subagent(
            agent=self,
            prompt=contract.clean_query,
            description=compound_task_label(contract),
            tools_allowed=[],
            max_steps=8,
            session=session,
            max_web_results=3,
            emit=emit,
            execution_steps=execution_steps,
            contract=contract,
        )
        result = compound_task_result_from_task_payload(task_result, fallback_contract=contract)
        result_contract = result.get("contract")
        relation = result_contract.relation if isinstance(result_contract, QueryContract) else contract.relation
        self._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="Task",
            summary=f"compound_subtask:{relation}",
            payload={
                "prompt": contract.clean_query,
                "relation": relation,
                "verification": task_result.get("verification", {}),
                "answer_chars": len(str(result.get("answer", "") or "")),
            },
        )
        return result

    def _force_best_effort_after_clarification_limit(
        self,
        *,
        state: dict[str, Any],
        session: SessionContext,
        web_enabled: bool,
        explicit_web_search: bool,
        max_web_results: int,
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        contract: QueryContract = state["contract"]
        verification = state.get("verification")
        clarification_options = clarification_options_from_contract_notes(contract)
        clarification_key = clarification_tracking_key(
            contract=contract,
            verification=verification,
            options=clarification_options,
        )
        next_attempt = next_clarification_attempt(session=session, key=clarification_key)
        decision = clarification_limit_decision(
            contract=contract,
            verification=verification,
            next_attempt=next_attempt,
            max_attempts=self.agent_settings.max_clarification_attempts,
            options=clarification_options,
        )
        if decision is None:
            return None

        self._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="clarification_limit",
            summary=decision.summary,
            payload=decision.observation_payload,
        )

        forced_state = self.runtime.run_research_agent_loop(
            contract=decision.forced_contract,
            session=session,
            agent_plan=decision.forced_plan,
            web_enabled=web_enabled,
            explicit_web_search=explicit_web_search,
            max_web_results=max_web_results,
            emit=emit,
            execution_steps=execution_steps,
        )
        return promote_best_effort_state_after_clarification_limit(forced_state)

    def _record_agent_observation(
        self,
        *,
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
        tool: str,
        summary: str,
        payload: dict[str, Any],
    ) -> None:
        record_agent_observation_event(
            emit=emit,
            execution_steps=execution_steps,
            tool=tool,
            summary=summary,
            payload=payload,
        )

    def _emit_agent_tool_call(
        self,
        *,
        emit: Callable[[str, dict[str, Any]], None],
        tool: str,
        arguments: dict[str, Any],
    ) -> None:
        emit_agent_tool_call_event(emit=emit, tool=tool, arguments=arguments)

    def _agent_search_papers(
        self,
        *,
        state: dict[str, Any],
        session: SessionContext,
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
    ) -> None:
        contract: QueryContract = state["contract"]
        plan: ResearchPlan = state["plan"]
        tool_input = tool_input_from_state(state, "search_corpus")
        excluded_titles: set[str] = state["excluded_titles"]
        active = session.effective_active_research()
        result = run_agent_paper_search(
            contract=contract,
            plan=plan,
            tool_input=tool_input,
            active_targets=list(active.targets),
            excluded_titles=excluded_titles,
            search_papers=lambda query, search_contract, limit: self.retriever.search_papers(
                query=query,
                contract=search_contract,
                limit=limit,
            ),
            paper_lookup=self._candidate_from_paper_id,
            screen_papers=lambda search_contract, search_plan, candidates, search_excluded_titles: self._screen_agent_papers(
                contract=search_contract,
                plan=search_plan,
                candidate_papers=candidates,
                excluded_titles=search_excluded_titles,
            ),
        )
        self._emit_agent_tool_call(emit=emit, tool="search_corpus", arguments=result.tool_call_arguments)
        state["contract"] = result.contract
        candidate_papers = result.candidate_papers
        screened_papers = result.screened_papers
        state["candidate_papers"] = candidate_papers
        state["screened_papers"] = screened_papers
        state["precomputed_evidence"] = result.precomputed_evidence
        emit("candidate_papers", {"count": len(candidate_papers), "items": [item.model_dump() for item in candidate_papers]})
        emit("screened_papers", {"count": len(screened_papers), "items": [item.model_dump() for item in screened_papers]})
        self._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="search_corpus",
            summary=result.observation_summary,
            payload=result.observation_payload,
        )

    def _screen_agent_papers(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        candidate_papers: list[CandidatePaper],
        excluded_titles: set[str],
    ) -> tuple[list[CandidatePaper], list[EvidenceBlock] | None]:
        return screen_agent_papers(
            contract=contract,
            plan=plan,
            candidate_papers=candidate_papers,
            excluded_titles=excluded_titles,
            paper_lookup=self._candidate_from_paper_id,
            paper_summary_text=lambda paper_id: self._paper_summary_text(paper_id),
            prefer_identity_matching_papers=lambda candidates, targets: self._prefer_identity_matching_papers(
                candidates=candidates,
                targets=targets,
            ),
            search_entity_evidence=lambda query, search_contract, limit: self.retriever.search_entity_evidence(
                query=query,
                contract=search_contract,
                limit=limit,
            ),
            ground_entity_papers=lambda candidates, evidence, limit: self._ground_entity_papers(
                candidates=candidates,
                evidence=evidence,
                limit=limit,
            ),
        )

    def _agent_search_evidence(
        self,
        *,
        state: dict[str, Any],
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
    ) -> None:
        contract: QueryContract = state["contract"]
        plan: ResearchPlan = state["plan"]
        tool_input = tool_input_from_state(state, "search_corpus")
        screened_papers: list[CandidatePaper] = state["screened_papers"]
        excluded_titles: set[str] = state["excluded_titles"]
        result = search_agent_evidence(
            contract=contract,
            plan=plan,
            tool_input=tool_input,
            screened_papers=screened_papers,
            precomputed_evidence=state.get("precomputed_evidence"),
            excluded_titles=excluded_titles,
            search_concept_evidence=lambda query, search_contract, paper_ids, limit: self.retriever.search_concept_evidence(
                query=query,
                contract=search_contract,
                paper_ids=paper_ids,
                limit=limit,
            ),
            expand_evidence=lambda paper_ids, query, search_contract, limit: self.retriever.expand_evidence(
                paper_ids=paper_ids,
                query=query,
                contract=search_contract,
                limit=limit,
            ),
        )
        self._emit_agent_tool_call(emit=emit, tool="search_corpus", arguments=result.tool_call_arguments)
        evidence = result.evidence
        state["evidence"] = evidence
        emit("evidence", {"count": len(evidence), "items": [item.model_dump() for item in evidence]})
        self._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="search_corpus",
            summary=result.observation_summary,
            payload=result.observation_payload,
        )

    def _agent_web_search(
        self,
        *,
        state: dict[str, Any],
        web_enabled: bool,
        max_web_results: int,
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
    ) -> None:
        contract: QueryContract = state["contract"]
        tool_input = tool_input_from_state(state, "web_search")
        result = search_agent_web_evidence(
            contract=contract,
            existing_evidence=state["evidence"],
            tool_input=tool_input,
            web_enabled=web_enabled,
            max_web_results=max_web_results,
            collect=lambda search_contract, enabled, limit, query: collect_web_evidence(
                web_search=self.web_search,
                contract=search_contract,
                use_web_search=enabled,
                max_web_results=limit,
                query_override=query,
            ),
        )
        self._emit_agent_tool_call(emit=emit, tool="web_search", arguments=result.tool_call_arguments)
        web_evidence = result.web_evidence
        state["web_evidence"] = web_evidence
        if web_evidence:
            state["evidence"] = result.merged_evidence
            emit("web_search", {"count": len(web_evidence), "items": [item.model_dump() for item in web_evidence]})
            emit("evidence", {"count": len(state["evidence"]), "items": [item.model_dump() for item in state["evidence"]]})
        self._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="web_search",
            summary=result.observation_summary,
            payload=result.observation_payload,
        )

    def _agent_solve_claims(
        self,
        *,
        state: dict[str, Any],
        session: SessionContext,
        explicit_web_search: bool,
        max_web_results: int,
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
    ) -> None:
        contract: QueryContract = state["contract"]
        plan: ResearchPlan = state["plan"]
        screened_papers: list[CandidatePaper] = state["screened_papers"]
        evidence: list[EvidenceBlock] = state["evidence"]
        self._emit_agent_tool_call(
            emit=emit,
            tool="compose",
            arguments={"stage": "solve_claims", "solver_sequence": plan.solver_sequence, "evidence_count": len(evidence)},
        )
        ambiguity_options = self._disambiguation_options_from_evidence(
            contract=contract,
            session=session,
            papers=screened_papers,
            evidence=evidence,
        )
        if ambiguity_options:
            judge_decision = self._judge_disambiguation_options(contract=contract, options=ambiguity_options)
            resolution = resolve_disambiguation_judge_decision(
                contract=contract,
                options=ambiguity_options,
                judge_decision=judge_decision,
                auto_resolve_threshold=self.agent_settings.disambiguation_auto_resolve_threshold,
                recommend_threshold=self.agent_settings.disambiguation_recommend_threshold,
            )
            self._record_agent_observation(
                emit=emit,
                execution_steps=execution_steps,
                tool=resolution.observation_tool,
                summary=resolution.observation_summary,
                payload=resolution.observation_payload,
            )
            if resolution.auto_resolve and resolution.selected_option is not None:
                contract = resolution.contract
                state["contract"] = contract
                self._refresh_state_for_selected_ambiguity(
                    state=state,
                    selected=resolution.selected_option,
                    emit=emit,
                    execution_steps=execution_steps,
                )
                claims = solve_agent_state_claims(
                    state=state,
                    explicit_web=explicit_web_search,
                    solve_claims=lambda item_contract, item_plan, item_papers, item_evidence: self._run_solvers(
                        contract=item_contract,
                        plan=item_plan,
                        papers=item_papers,
                        evidence=item_evidence,
                        session=session,
                        use_web_search=explicit_web_search,
                        max_web_results=max_web_results,
                    ),
                    build_claim=lambda item_contract, item_evidence: build_web_research_claim(
                        contract=item_contract,
                        web_evidence=item_evidence,
                    ),
                )
                state["claims"] = claims
            else:
                state["contract"] = resolution.contract
                state["claims"] = []
                state["verification"] = resolution.verification
        else:
            claims = solve_agent_state_claims(
                state=state,
                explicit_web=explicit_web_search,
                solve_claims=lambda item_contract, item_plan, item_papers, item_evidence: self._run_solvers(
                    contract=item_contract,
                    plan=item_plan,
                    papers=item_papers,
                    evidence=item_evidence,
                    session=session,
                    use_web_search=explicit_web_search,
                    max_web_results=max_web_results,
                ),
                build_claim=lambda item_contract, item_evidence: build_web_research_claim(
                    contract=item_contract,
                    web_evidence=item_evidence,
                ),
            )
            state["claims"] = claims
        claims = state["claims"]
        emit("claims", {"count": len(claims), "items": [item.model_dump() for item in claims]})
        self._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="compose",
            summary=f"claims={len(claims)}",
            payload={"stage": "solve_claims", "claim_count": len(claims), "claim_types": [item.claim_type for item in claims]},
        )

    def _agent_verify_grounding(
        self,
        *,
        state: dict[str, Any],
        session: SessionContext,
        explicit_web_search: bool,
        max_web_results: int,
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
    ) -> None:
        verification = state.get("verification")
        if isinstance(verification, VerificationReport) and verification.status == "clarify":
            self._record_agent_observation(
                emit=emit,
                execution_steps=execution_steps,
                tool="verify_claim",
                summary=verification.status,
                payload=verification_observation_payload(verification),
            )
            return
        contract: QueryContract = state["contract"]
        plan: ResearchPlan = state["plan"]
        claims: list[Claim] = state["claims"]
        screened_papers: list[CandidatePaper] = state["screened_papers"]
        evidence: list[EvidenceBlock] = state["evidence"]
        self._emit_agent_tool_call(
            emit=emit,
            tool="verify_claim",
            arguments=verify_grounding_tool_call_arguments(plan=plan, claims=claims),
        )
        verification = self._verify_claims(
            contract=contract,
            plan=plan,
            claims=claims,
            papers=screened_papers,
            evidence=evidence,
        )
        state["verification"] = verification
        if verification.status == "retry" and plan.retry_budget > 0:
            self._agent_retry_after_verification(
                state=state,
                session=session,
                explicit_web_search=explicit_web_search,
                max_web_results=max_web_results,
                emit=emit,
                execution_steps=execution_steps,
            )
            verification = state["verification"]
        verification = clarify_retry_verification_if_needed(contract=contract, verification=verification)
        state["verification"] = verification
        self._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="verify_claim",
            summary=verification.status,
            payload=verification_observation_payload(verification),
        )

    def _agent_retry_after_verification(
        self,
        *,
        state: dict[str, Any],
        session: SessionContext,
        explicit_web_search: bool,
        max_web_results: int,
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
    ) -> None:
        contract: QueryContract = state["contract"]
        plan: ResearchPlan = state["plan"]
        excluded_titles: set[str] = state["excluded_titles"]
        retry_limits = retry_research_limits(plan)
        self._emit_agent_tool_call(
            emit=emit,
            tool="retry_research",
            arguments={
                "reason": state["verification"].recommended_action if state.get("verification") else "",
                "paper_limit": retry_limits.paper_limit,
                "evidence_limit": retry_limits.evidence_limit,
            },
        )
        retry_materials = prepare_retry_research_materials(
            contract=contract,
            plan=plan,
            excluded_titles=excluded_titles,
            search_papers=lambda query, search_contract, limit: self.retriever.search_papers(
                query=query,
                contract=search_contract,
                limit=limit,
            ),
            paper_lookup=self._candidate_from_paper_id,
            search_concept_evidence=lambda query, search_contract, paper_ids, limit: self.retriever.search_concept_evidence(
                query=query,
                contract=search_contract,
                paper_ids=paper_ids,
                limit=limit,
            ),
            search_entity_evidence=lambda query, search_contract, limit: self.retriever.search_entity_evidence(
                query=query,
                contract=search_contract,
                limit=limit,
            ),
            expand_evidence=lambda paper_ids, query, search_contract, limit: self.retriever.expand_evidence(
                paper_ids=paper_ids,
                query=query,
                contract=search_contract,
                limit=limit,
            ),
            ground_entity_papers=lambda candidates, evidence, limit: self._ground_entity_papers(
                candidates=candidates,
                evidence=evidence,
                limit=limit,
            ),
        )
        retry_result = run_retry_verification_from_materials(
            contract=contract,
            plan=plan,
            materials=retry_materials,
            solve_claims=lambda retry_plan, retry_papers, retry_evidence: self._run_solvers(
                contract=contract,
                plan=retry_plan,
                papers=retry_papers,
                evidence=retry_evidence,
                session=session,
                use_web_search=explicit_web_search,
                max_web_results=max_web_results,
            ),
            verify_claims=lambda retry_plan, retry_claims, retry_papers, retry_evidence: self._verify_claims(
                contract=contract,
                plan=retry_plan,
                claims=retry_claims,
                papers=retry_papers,
                evidence=retry_evidence,
            ),
            prefer_identity_matching_papers=lambda candidates, targets: self._prefer_identity_matching_papers(
                candidates=candidates,
                targets=targets,
            ),
        )
        if retry_result.should_replace_materials:
            state["screened_papers"] = retry_result.candidate_papers
            state["evidence"] = retry_result.evidence
            state["claims"] = retry_result.claims
        state["verification"] = retry_result.verification
        self._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="retry_research",
            summary=retry_result.observation_summary,
            payload=retry_result.observation_payload,
        )

    def _agent_reflect(
        self,
        *,
        state: dict[str, Any],
        session: SessionContext,
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
    ) -> None:
        verification = state.get("verification")
        if not isinstance(verification, VerificationReport):
            verification = VerificationReport(
                status="clarify",
                missing_fields=["verified_claims"],
                recommended_action="clarify_after_reflection",
            )
            state["verification"] = verification
        reflection = self._reflect_agent_state(
            contract=state["contract"],
            session=session,
            claims=state["claims"],
            papers=state["screened_papers"],
            evidence=state["evidence"],
            verification=verification,
            excluded_titles=state["excluded_titles"],
        )
        if reflection.get("decision") == "clarify":
            state["verification"] = VerificationReport(
                status="clarify",
                missing_fields=[str(item) for item in reflection.get("missing_fields", ["agent_reflection"])],
                recommended_action=str(reflection.get("recommended_action", "clarify_after_reflection")),
            )
        state["reflection"] = reflection
        emit("reflection", reflection)
        execution_steps.append({"node": "agent_reflection", "summary": str(reflection.get("decision", state["verification"].status))})

    def _extract_query_contract(
        self,
        *,
        query: str,
        session: SessionContext,
        mode: str,
        clarification_choice: dict[str, Any] | None = None,
    ) -> QueryContract:
        clean_query = " ".join(query.strip().split())
        clarified_contract = contract_from_pending_clarification(
            clean_query=clean_query,
            session=session,
            clarification_choice=clarification_choice,
        )
        if clarified_contract is not None:
            return clarified_contract
        targets = extract_targets(clean_query)
        contract = self._contract_from_llm_tool_router(
            clean_query=clean_query,
            session=session,
            extracted_targets=targets,
        )
        if contract is None:
            if self.agent_settings.legacy_intent_fallback_enabled:
                contract = self.intent_router.contract_for_query(
                    clean_query=clean_query,
                    session=session,
                    extracted_targets=targets,
                )
            else:
                contract = self._router_miss_clarification_contract(clean_query=clean_query)
        contract = self._normalize_conversation_tool_contract(
            contract=contract,
            clean_query=clean_query,
            session=session,
        )
        if contract.interaction_mode == "conversation":
            return contract
        refined_contract = self._refine_followup_contract(contract=contract, session=session)
        refined_contract = self._resolve_formula_answer_correction_contract(contract=refined_contract, session=session)
        refined_contract = self._resolve_formula_contextual_paper_contract(contract=refined_contract, session=session)
        refined_contract = self._resolve_formula_location_followup_contract(contract=refined_contract, session=session)
        refined_contract = self._resolve_paper_scope_correction_contract(contract=refined_contract, session=session)
        refined_contract = self._resolve_contextual_active_paper_contract(contract=refined_contract, session=session)
        refined_contract = inherit_followup_relationship_contract(
            contract=refined_contract,
            session=session,
            normalize_targets=lambda targets, requested_fields: normalize_contract_targets(
                targets=targets,
                requested_fields=requested_fields,
                canonicalize_targets=self.retriever.canonicalize_targets,
            ),
        )
        refined_contract = normalize_followup_direction_contract(
            contract=refined_contract,
            normalize_targets=lambda targets, requested_fields: normalize_contract_targets(
                targets=targets,
                requested_fields=requested_fields,
                canonicalize_targets=self.retriever.canonicalize_targets,
            ),
        )
        return apply_conversation_memory_to_contract(
            contract=refined_contract,
            session=session,
            selected_clarification_paper_id=selected_clarification_paper_id(refined_contract),
        )

    @staticmethod
    def _router_miss_clarification_contract(*, clean_query: str) -> QueryContract:
        return QueryContract(
            clean_query=clean_query,
            interaction_mode="conversation",
            relation="clarify_user_intent",
            targets=[],
            answer_slots=["clarify"],
            requested_fields=[],
            required_modalities=[],
            answer_shape="narrative",
            precision_requirement="normal",
            continuation_mode="fresh",
            allow_web_search=False,
            notes=[
                "structured_intent",
                "llm_tool_router",
                "router_unavailable",
                "legacy_intent_fallback_disabled",
                "intent_needs_clarification",
                "low_intent_confidence",
            ],
        )

    def _contract_from_llm_tool_router(
        self,
        *,
        clean_query: str,
        session: SessionContext,
        extracted_targets: list[str],
    ) -> QueryContract | None:
        decision = self.llm_intent_router.route(query=clean_query, session=session)
        return query_contract_from_router_decision(
            decision=decision,
            clean_query=clean_query,
            session=session,
            extracted_targets=extracted_targets,
            normalize_targets=lambda targets, requested_fields: normalize_contract_targets(
                targets=targets,
                requested_fields=requested_fields,
                canonicalize_targets=self.retriever.canonicalize_targets,
            ),
        )

    def _normalize_conversation_tool_contract(
        self,
        *,
        contract: QueryContract,
        clean_query: str,
        session: SessionContext,
    ) -> QueryContract:
        return normalize_conversation_tool_contract(
            contract=contract,
            clean_query=clean_query,
            session=session,
            paper_from_query_hint=self._paper_from_query_hint,
        )

    def _session_conversation_context(self, session: SessionContext, *, max_chars: int = 24000) -> dict[str, Any]:
        """Return the retained conversation as the LLM-facing working memory."""

        try:
            persistent_learnings = load_learnings(data_dir=self.settings.data_dir, max_chars=4000)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load persistent learnings: %s", exc)
            persistent_learnings = ""
        return session_conversation_context(
            session,
            persistent_learnings=persistent_learnings,
            max_chars=max_chars,
        )

    def _remember_research_outcome(
        self,
        *,
        session: SessionContext,
        contract: QueryContract,
        answer: str,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        citations: list[AssistantCitation],
    ) -> None:
        remember_research_outcome(
            session=session,
            contract=contract,
            answer=answer,
            claims=claims,
            papers=papers,
            evidence=evidence,
            citations=citations,
            candidate_lookup=self._candidate_from_paper_id,
        )

    def _remember_compound_outcome(
        self,
        *,
        session: SessionContext,
        clean_query: str,
        subtask_results: list[dict[str, Any]],
    ) -> None:
        remember_compound_outcome(
            session=session,
            clean_query=clean_query,
            subtask_results=subtask_results,
            candidate_lookup=self._candidate_from_paper_id,
        )

    def _resolve_formula_answer_correction_contract(self, *, contract: QueryContract, session: SessionContext) -> QueryContract:
        active = session.effective_active_research()
        active_formula = active.relation == "formula_lookup" or "formula" in {str(field) for field in active.requested_fields}
        if not active_formula or not active.targets:
            return contract
        if not looks_like_formula_answer_correction(contract.clean_query):
            return contract
        title = active.titles[0] if active.titles else ""
        paper = self._paper_from_query_hint(title) if title else None
        return formula_answer_correction_contract(contract=contract, active=active, paper=paper)

    def _resolve_formula_location_followup_contract(self, *, contract: QueryContract, session: SessionContext) -> QueryContract:
        active = session.effective_active_research()
        active_formula = (
            active.relation == "formula_lookup"
            or "formula" in {str(field) for field in active.requested_fields}
        )
        if not active_formula or not active.targets:
            return contract
        if contract.interaction_mode == "conversation":
            return contract
        if not looks_like_formula_location_correction(contract.clean_query):
            return contract
        paper = self._paper_from_query_hint(contract.clean_query)
        if paper is None:
            return contract
        target = formula_followup_target(
            contract=contract,
            active=session.effective_active_research(),
            paper=paper,
        )
        if not target:
            return contract
        return formula_location_followup_contract(contract=contract, paper=paper, target=target)

    def _resolve_formula_contextual_paper_contract(self, *, contract: QueryContract, session: SessionContext) -> QueryContract:
        goals = research_plan_goals(contract)
        if contract.interaction_mode != "research" or "formula" not in goals or not contract.targets:
            return contract
        if selected_clarification_paper_id(contract) or "exclude_previous_focus" in contract.notes:
            return contract
        active = session.effective_active_research()
        context_text = " ".join([*active.titles, *active.targets]).strip()
        if not context_text:
            return contract
        paper = self._paper_from_query_hint(context_text)
        if paper is None:
            return contract
        if not formula_query_allows_paper_context(
            contract=contract,
            active=session.effective_active_research(),
            paper=paper,
        ):
            return contract
        target = str(contract.targets[0] or "").strip()
        if not target or not paper_context_supports_formula_target(
            block_documents=self.retriever.block_documents_for_paper(paper.paper_id, limit=256),
            target=target,
        ):
            return contract
        return formula_contextual_paper_contract(contract=contract, paper=paper, target=target)

    def _resolve_paper_scope_correction_contract(self, *, contract: QueryContract, session: SessionContext) -> QueryContract:
        if contract.interaction_mode != "research" or selected_clarification_paper_id(contract):
            return contract
        active = session.effective_active_research()
        if not active.has_content() or not active.targets:
            return contract
        if not looks_like_paper_scope_correction(contract.clean_query):
            return contract
        paper = self._paper_from_query_hint(contract.clean_query)
        if paper is None:
            return contract
        return paper_scope_correction_contract(contract=contract, active=active, paper=paper)

    def _resolve_contextual_active_paper_contract(self, *, contract: QueryContract, session: SessionContext) -> QueryContract:
        if contract.interaction_mode != "research" or selected_clarification_paper_id(contract):
            return contract
        if "exclude_previous_focus" in contract.notes or is_negative_correction_query(contract.clean_query):
            return contract
        if not looks_like_active_paper_reference(contract.clean_query):
            return contract
        active = session.effective_active_research()
        if not active.titles:
            return contract
        paper = self._paper_from_query_hint(" ".join(active.titles))
        if paper is None:
            return contract
        return contextual_active_paper_contract(contract=contract, paper=paper)

    def _paper_from_query_hint(self, query: str) -> CandidatePaper | None:
        return paper_from_query_hint(
            query,
            paper_documents=self.retriever.paper_documents(),
            candidate_lookup=self._candidate_from_paper_id,
        )

    def _remember_clarification_attempt(
        self,
        *,
        session: SessionContext,
        contract: QueryContract,
        verification: VerificationReport,
    ) -> None:
        key = clarification_tracking_key(
            contract=contract,
            verification=verification,
            options=clarification_options_from_contract_notes(contract),
        )
        remember_clarification_attempt(session=session, key=key)

    def _disambiguation_options_from_evidence(
        self,
        *,
        contract: QueryContract,
        session: SessionContext,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> list[dict[str, Any]]:
        target = str(contract.targets[0] or "").strip() if contract.targets else ""
        target_binding_exists = bool(target and target_binding_from_memory(session=session, target=target))
        return evidence_disambiguation_options(
            contract=contract,
            target_binding_exists=target_binding_exists,
            is_negative_correction=is_negative_correction_query(contract.clean_query),
            initial_options=lambda: self._acronym_options_from_evidence(target=target, papers=papers, evidence=evidence),
            broad_options=lambda: self._acronym_options_from_evidence(
                target=target,
                papers=papers,
                evidence=self.retriever.search_concept_evidence(
                    query=target,
                    contract=contract,
                    limit=max(self.settings.evidence_limit_default, 96),
                ),
            ),
            corpus_options=lambda: self._acronym_options_from_evidence(
                target=target,
                papers=papers,
                evidence=self._acronym_evidence_from_corpus(target=target, limit=160),
            ),
            excluded_titles=excluded_focus_titles(
                session=session,
                contract=contract,
                is_negative_correction_query=is_negative_correction_query,
            ),
        )

    def _judge_disambiguation_options(
        self,
        *,
        contract: QueryContract,
        options: list[dict[str, Any]],
    ) -> DisambiguationJudgeDecision | None:
        if self.clients.chat is None or len(options) < 2:
            return None
        payload = self.clients.invoke_json(
            system_prompt=disambiguation_judge_system_prompt(),
            human_prompt=disambiguation_judge_human_prompt(
                contract=contract,
                candidate_options=[self._disambiguation_judge_payload(option) for option in options[:8]],
            ),
            fallback={},
        )
        if not isinstance(payload, dict):
            return None
        try:
            return DisambiguationJudgeDecision.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("disambiguation judge returned invalid payload: %s", exc)
            return None

    def _disambiguation_judge_payload(self, option: dict[str, Any]) -> dict[str, Any]:
        paper_id = str(option.get("paper_id", "") or "").strip()
        paper = self._candidate_from_paper_id(paper_id) if paper_id else None
        return disambiguation_judge_option_payload(option=option, paper=paper)

    def _refresh_state_for_selected_ambiguity(
        self,
        *,
        state: dict[str, Any],
        selected: dict[str, Any],
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
    ) -> None:
        candidate_pool: list[CandidatePaper] = [
            *list(state.get("screened_papers") or []),
            *list(state.get("candidate_papers") or []),
        ]
        contract: QueryContract = state["contract"]
        plan: ResearchPlan = state["plan"]
        excluded_titles: set[str] = state["excluded_titles"]
        refresh = refresh_selected_ambiguity_materials(
            selected=selected,
            contract=contract,
            plan=plan,
            candidate_papers=candidate_pool,
            existing_evidence=list(state.get("evidence") or []),
            excluded_titles=excluded_titles,
            paper_lookup=self._candidate_from_paper_id,
            search_concept_evidence=lambda query, search_contract, paper_ids, limit: self.retriever.search_concept_evidence(
                query=query,
                contract=search_contract,
                paper_ids=paper_ids,
                limit=limit,
            ),
            expand_evidence=lambda paper_ids, query, search_contract, limit: self.retriever.expand_evidence(
                paper_ids=paper_ids,
                query=query,
                contract=search_contract,
                limit=limit,
            ),
        )
        if refresh is None:
            return
        if refresh.selected_papers:
            state["screened_papers"] = refresh.selected_papers
            emit("screened_papers", {"count": len(state["screened_papers"]), "items": [item.model_dump() for item in state["screened_papers"]]})
        evidence = refresh.evidence
        if refresh.evidence_refreshed:
            self._record_agent_observation(
                emit=emit,
                execution_steps=execution_steps,
                tool="search_corpus",
                summary=f"auto_resolved_evidence={len(evidence)}",
                payload={"stage": "search_evidence", "selected_paper_id": refresh.paper_id, "evidence_count": len(evidence)},
            )
        state["evidence"] = evidence
        emit("evidence", {"count": len(evidence), "items": [item.model_dump() for item in evidence]})

    def _acronym_options_from_evidence(
        self,
        *,
        target: str,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> list[dict[str, Any]]:
        return build_acronym_options_from_evidence(
            target=target,
            papers=papers,
            evidence=evidence,
            paper_lookup=self._candidate_from_paper_id,
        )

    def _acronym_evidence_from_corpus(self, *, target: str, limit: int) -> list[EvidenceBlock]:
        return build_acronym_evidence_from_corpus(
            target=target,
            limit=limit,
            paper_documents=lambda: self.retriever.paper_documents(),
            block_documents_for_paper=lambda paper_id, block_limit: self.retriever.block_documents_for_paper(
                paper_id,
                limit=block_limit,
            ),
        )

    def _store_pending_clarification(self, *, session: SessionContext, contract: QueryContract) -> None:
        options = clarification_options_from_contract_notes(contract)
        store_pending_clarification(session=session, contract=contract, options=options)

    def _plan_agent_actions(self, *, contract: QueryContract, session: SessionContext, use_web_search: bool) -> dict[str, Any]:
        return self.planner.plan_actions(
            contract=contract,
            session=session,
            use_web_search=use_web_search,
        )

    def _reflect_agent_state(
        self,
        *,
        contract: QueryContract,
        session: SessionContext,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        verification: VerificationReport,
        excluded_titles: set[str],
    ) -> dict[str, Any]:
        focus_titles = self._claim_focus_titles(claims=claims, papers=papers)
        target = str(contract.targets[0] or "").strip() if contract.targets else ""
        return reflect_agent_state_decision(
            contract=contract,
            claims=claims,
            focus_titles=focus_titles,
            verification=verification,
            excluded_titles=excluded_titles,
            target_binding_exists=bool(target and target_binding_from_memory(session=session, target=target)),
            ambiguity_option_count=lambda: len(
                self._acronym_options_from_evidence(target=target, papers=papers, evidence=evidence)
            ),
        )

    def _compress_session_history_if_needed(self, session: SessionContext) -> None:
        if self.clients.chat is None:
            return
        retained_turns, older_turns = session_history_compression_window(
            session,
            max_turns=self.settings.agent_history_max_turns,
        )
        if not older_turns:
            return
        compressed = self.clients.invoke_text(
            system_prompt=session_history_compression_system_prompt(),
            human_prompt=json.dumps(
                session_history_compression_payload(session, older_turns=older_turns),
                ensure_ascii=False,
            ),
            fallback=session.summary,
        ).strip()
        apply_session_history_compression(session, compressed=compressed, retained_turns=retained_turns)
        self.sessions.upsert(session)

    def _claim_focus_titles(self, *, claims: list[Claim], papers: list[CandidatePaper]) -> list[str]:
        def paper_title_lookup(paper_id: str) -> str | None:
            doc = self.retriever.paper_doc_by_id(paper_id)
            if doc is None:
                return None
            return str((doc.metadata or {}).get("title", ""))

        return claim_focus_titles(claims=claims, papers=papers, paper_title_lookup=paper_title_lookup)

    def _resolve_followup_seed_papers(
        self,
        *,
        contract: QueryContract,
        candidates: list[CandidatePaper],
        session: SessionContext,
    ) -> list[CandidatePaper]:
        return resolve_followup_seed_papers(
            contract=contract,
            candidates=candidates,
            active_titles=session.effective_active_research().titles,
            clients=self.clients,
            paper_summary_text=lambda paper_id: self._paper_summary_text(paper_id),
        )

    def _expand_followup_candidate_pool(
        self,
        *,
        contract: QueryContract,
        seed_papers: list[CandidatePaper],
        initial_candidates: list[CandidatePaper],
    ) -> list[CandidatePaper]:
        return expand_followup_candidate_pool(
            contract=contract,
            seed_papers=seed_papers,
            initial_candidates=initial_candidates,
            paper_limit_default=int(self.settings.paper_limit_default),
            paper_summary_text=lambda paper_id: self._paper_summary_text(paper_id),
            search_papers=lambda query, search_contract, limit: self.retriever.search_papers(
                query=query,
                contract=search_contract,
                limit=limit,
            ),
        )

    def _rank_followup_candidates(
        self,
        *,
        contract: QueryContract,
        seed_papers: list[CandidatePaper],
        candidates: list[CandidatePaper],
        evidence: list[EvidenceBlock] | None = None,
    ) -> list[dict[str, Any]]:
        return rank_followup_candidates(
            contract=contract,
            seed_papers=seed_papers,
            candidates=candidates,
            evidence=evidence or [],
            clients=self.clients,
            paper_summary_text=lambda paper_id: self._paper_summary_text(paper_id),
            selected_candidate_assessment=lambda paper: selected_followup_candidate_assessment(
                contract=contract,
                seed_papers=seed_papers,
                paper=paper,
                evidence=evidence or [],
                clients=self.clients,
                expand_evidence=lambda paper_ids, query, evidence_contract, limit: self.retriever.expand_evidence(
                    paper_ids=paper_ids,
                    query=query,
                    contract=evidence_contract,
                    limit=limit,
                ),
                paper_summary_text=lambda paper_id: self._paper_summary_text(paper_id),
                coerce_confidence=lambda value: self._coerce_confidence(value),
            ),
            coerce_confidence=lambda value: self._coerce_confidence(value),
        )

    def _paper_summary_text(self, paper_id: str) -> str:
        doc = self.retriever.paper_doc_by_id(paper_id)
        if doc is None:
            return ""
        meta = dict(doc.metadata or {})
        return str(meta.get("generated_summary") or meta.get("abstract_note") or doc.page_content[:400]).strip()

    def _citations_from_doc_ids(self, doc_ids: list[str], evidence: list[EvidenceBlock]) -> list[AssistantCitation]:
        return citations_from_doc_ids(
            doc_ids,
            evidence,
            block_doc_lookup=self.retriever.block_doc_by_id,
            paper_doc_lookup=self.retriever.paper_doc_by_id,
        )

    def _prefer_identity_matching_papers(self, *, candidates: list[CandidatePaper], targets: list[str]) -> list[CandidatePaper]:
        matched = [item for item in candidates if self._paper_identity_matches_targets(paper=item, targets=targets)]
        return matched or candidates

    def _render_page_image_data_url(self, *, file_path: str, page: int) -> str:
        return render_pdf_page_image_data_url(
            file_path=file_path,
            page=page,
            pdf_render_dpi=int(self.settings.pdf_render_dpi),
            timeout_seconds=float(self.settings.figure_vlm_timeout_seconds),
            cache=self._rendered_page_data_url_cache,
            logger=logger,
        )

    def _clarification_question(self, contract: QueryContract, session: SessionContext) -> str:
        return build_clarification_question(
            contract=contract,
            session=session,
            clients=self.clients,
            conversation_context=lambda current_session: self._session_conversation_context(current_session),
        )
