from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable
from uuid import uuid4

from app.core.agent_settings import AgentSettings
from app.core.config import Settings
from app.domain.models import (
    ActiveResearch,
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
    emit_agent_step as emit_agent_step_event,
    emit_agent_tool_call as emit_agent_tool_call_event,
    record_agent_observation as record_agent_observation_event,
)
from app.services.agent_planner import AgentPlanner
from app.services.agent_runtime import AgentRuntime
from app.services.agent_runtime_summary import build_runtime_summary
from app.services.agent_runtime_helpers import (
    claim_focus_titles,
    clarify_retry_verification_if_needed,
    entity_evidence_limit,
    excluded_focus_titles,
    filter_candidate_papers_by_excluded_titles,
    filter_evidence_by_excluded_titles,
    clarification_limit_decision,
    prepare_retry_research_materials,
    promote_best_effort_state_after_clarification_limit,
    retry_research_limits,
    refresh_selected_ambiguity_materials,
    screen_agent_papers,
    search_agent_candidate_papers,
    search_agent_evidence,
)
from app.services.agent_tools import agent_tool_manifest, all_agent_tool_names
from app.services.clarification_intents import (
    acronym_evidence_from_corpus as build_acronym_evidence_from_corpus,
    acronym_options_from_evidence as build_acronym_options_from_evidence,
    ambiguity_options_from_notes,
    clarification_tracking_key,
    clarification_options_from_contract_notes,
    clear_pending_clarification,
    apply_disambiguation_judge_recommendation,
    contract_needs_evidence_disambiguation,
    contract_with_auto_resolved_ambiguity,
    contract_from_pending_clarification,
    contract_from_selected_clarification_option,
    contract_with_ambiguity_options,
    disambiguation_goal_markers,
    evidence_disambiguation_options,
    disambiguation_judge_human_prompt,
    disambiguation_judge_option_payload,
    disambiguation_judge_summary,
    disambiguation_judge_system_prompt,
    disambiguation_missing_fields,
    judge_allows_auto_resolve,
    next_clarification_attempt,
    remember_clarification_attempt,
    reset_clarification_tracking,
    selected_option_from_judge_decision,
    selected_clarification_paper_id,
    store_pending_clarification,
)
from app.services.clarification_question_helpers import build_clarification_question
from app.services.citation_ranking import (
    format_citation_ranking_answer,
    lookup_candidate_citation_counts,
    select_citation_ranking_candidates,
    semantic_scholar_citation_evidence,
)
from app.services.compound_intents import should_try_compound_decomposition as should_try_compound_decomposition_query
from app.services.compound_task_helpers import (
    compose_compound_comparison_answer,
    comparison_results_with_memory,
    compound_subtask_contract_from_payload,
    compound_subtask_relation_from_slots,
    compound_task_label,
    compound_task_result_from_task_payload,
    llm_decompose_compound_query,
    merge_redundant_field_subtasks,
)
from app.services.contract_context import (
    LEGACY_TOOL_NAME_ALIASES,
    canonical_tools,
)
from app.services.contract_normalization import (
    normalize_contract_targets,
    normalize_lookup_text,
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
    llm_memory_followup_contract,
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
    build_figure_contexts,
    chunk_text,
    citations_from_doc_ids,
    paper_recommendation_reason,
)
from app.services.followup_candidate_helpers import (
    expand_followup_candidate_pool,
    llm_validate_followup_candidate,
    rank_followup_candidates,
    resolve_followup_seed_papers,
    selected_followup_candidate_assessment,
)
from app.services.figure_intents import figure_signal_score
from app.services.intent import IntentRecognizer
from app.services.memory_followup_answers import (
    compose_formula_interpretation_followup_answer,
    compose_language_preference_followup_answer,
    compose_memory_followup_answer,
    compose_memory_synthesis_answer,
)
from app.services.pdf_rendering import render_pdf_page_image_data_url
from app.services.query_shaping import (
    extract_targets,
    is_short_acronym,
    paper_query_text,
    should_use_web_search,
)
from app.services.research_planning import (
    build_research_plan,
    research_plan_goals,
)
from app.services.research_memory import remember_compound_outcome, remember_research_outcome
from app.services.tool_registry_helpers import coerce_int, tool_input_from_state
from app.services.web_evidence import (
    build_web_research_claim,
    collect_web_evidence,
    search_agent_web_evidence,
    solve_claims_with_web_research,
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
    make_active_research,
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
            conversation_messages=lambda session: self._session_llm_history_messages(
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
            conversation_messages=lambda session: self._session_llm_history_messages(
                session,
                max_turns=6,
                answer_limit=900,
            ),
            normalize_targets=lambda targets, requested_fields: self._normalize_contract_targets(
                targets=targets,
                requested_fields=requested_fields,
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
        yield {"event": "final", "data": final_payload | {"answer": result.get("answer", "")}}

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
            claims=claims,
            citations=citations,
        )

    @staticmethod
    def _canonical_tools(raw_tools: list[Any]) -> list[str]:
        return canonical_tools(
            raw_tools=raw_tools,
            aliases=LEGACY_TOOL_NAME_ALIASES,
            canonical_names=all_agent_tool_names(),
        )

    def _compose_memory_synthesis_answer(self, *, query: str, session: SessionContext, contract: QueryContract) -> str:
        return compose_memory_synthesis_answer(
            query=query,
            session=session,
            contract=contract,
            clients=self.clients,
            conversation_context=self._session_conversation_context,
            clean_text=self._clean_common_ocr_artifacts,
        )

    def _compose_memory_followup_answer(self, *, query: str, session: SessionContext, contract: QueryContract) -> str:
        return compose_memory_followup_answer(
            query=query,
            session=session,
            contract=contract,
            clients=self.clients,
            conversation_context=self._session_conversation_context,
            clean_text=self._clean_common_ocr_artifacts,
        )

    def _compose_formula_interpretation_followup_answer(
        self,
        *,
        query: str,
        session: SessionContext,
        contract: QueryContract,
    ) -> str:
        return compose_formula_interpretation_followup_answer(
            query=query,
            session=session,
            contract=contract,
            clients=self.clients,
            conversation_context=self._session_conversation_context,
            clean_text=self._clean_common_ocr_artifacts,
        )

    def _compose_language_preference_followup_answer(
        self,
        *,
        query: str,
        session: SessionContext,
        contract: QueryContract,
    ) -> str:
        return compose_language_preference_followup_answer(
            query=query,
            session=session,
            contract=contract,
            clients=self.clients,
            conversation_context=self._session_conversation_context,
            clean_text=self._clean_common_ocr_artifacts,
        )

    def _llm_memory_followup_contract(
        self,
        *,
        clean_query: str,
        session: SessionContext,
        current_contract: QueryContract,
    ) -> QueryContract | None:
        return llm_memory_followup_contract(
            clean_query=clean_query,
            session=session,
            current_contract=current_contract,
            clients=self.clients,
            conversation_context=self._session_conversation_context,
        )

    def _should_try_compound_decomposition(self, clean_query: str, *, session: SessionContext | None = None) -> bool:
        return should_try_compound_decomposition_query(clean_query, session=session)

    def _llm_decompose_compound_query(self, *, clean_query: str, session: SessionContext) -> list[QueryContract]:
        return llm_decompose_compound_query(
            clean_query=clean_query,
            session=session,
            clients=self.clients,
            available_tools=list(agent_tool_manifest()),
            conversation_context=self._session_conversation_context,
            history_messages=self._session_llm_history_messages,
            target_normalizer=lambda targets, fields: self._normalize_contract_targets(
                targets=targets,
                requested_fields=fields,
            ),
        )

    def _subtask_contract_from_payload(
        self,
        payload: object,
        *,
        fallback_query: str,
        index: int,
    ) -> QueryContract | None:
        return compound_subtask_contract_from_payload(
            payload,
            fallback_query=fallback_query,
            index=index,
            target_normalizer=lambda targets, fields: self._normalize_contract_targets(
                targets=targets,
                requested_fields=fields,
            ),
        )

    @staticmethod
    def _subtask_relation_from_slots(
        *,
        answer_slots: list[str],
        requested_fields: list[str],
        targets: list[str],
    ) -> str:
        return compound_subtask_relation_from_slots(
            answer_slots=answer_slots,
            requested_fields=requested_fields,
            targets=targets,
        )

    def _merge_redundant_field_subtasks(self, subcontracts: list[QueryContract]) -> list[QueryContract]:
        return merge_redundant_field_subtasks(subcontracts)

    def _execute_compound_conversation_subtask(
        self,
        *,
        contract: QueryContract,
        session: SessionContext,
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._execute_compound_task_subagent(
            contract=contract,
            session=session,
            emit=emit,
            execution_steps=execution_steps,
        )

    def _execute_compound_research_subtask(
        self,
        *,
        contract: QueryContract,
        session: SessionContext,
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._execute_compound_task_subagent(
            contract=contract,
            session=session,
            emit=emit,
            execution_steps=execution_steps,
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

    def _compose_compound_comparison_answer(
        self,
        *,
        query: str,
        subtask_results: list[dict[str, Any]],
        session: SessionContext,
        comparison_contract: QueryContract | None = None,
    ) -> str:
        return compose_compound_comparison_answer(
            query=query,
            subtask_results=subtask_results,
            session=session,
            comparison_contract=comparison_contract,
            clients=self.clients,
            clean_text=self._clean_common_ocr_artifacts,
        )

    def _comparison_results_with_memory(
        self,
        *,
        subtask_results: list[dict[str, Any]],
        session: SessionContext,
        comparison_contract: QueryContract | None,
    ) -> list[dict[str, Any]]:
        return comparison_results_with_memory(
            subtask_results=subtask_results,
            session=session,
            comparison_contract=comparison_contract,
        )

    def _select_citation_ranking_candidates(
        self,
        *,
        session: SessionContext,
        query: str,
        limit: int,
    ) -> list[dict[str, str]]:
        return select_citation_ranking_candidates(
            paper_documents=list(self.retriever.paper_documents()),
            session=session,
            query=query,
            limit=limit,
            rank_library_papers_for_recommendation=self._rank_library_papers_for_recommendation,
        )

    def _lookup_candidate_citation_counts(
        self,
        *,
        candidates: list[dict[str, str]],
        max_web_results: int,
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return lookup_candidate_citation_counts(
            candidates=candidates,
            max_web_results=max_web_results,
            web_search=self.web_search,
            emit=emit,
            emit_tool_call=lambda tool, arguments: self._emit_agent_tool_call(
                emit=emit,
                tool=tool,
                arguments=arguments,
            ),
            record_observation=lambda tool, summary, payload: self._record_agent_observation(
                emit=emit,
                execution_steps=execution_steps,
                tool=tool,
                summary=summary,
                payload=payload,
            ),
            semantic_scholar_lookup=lambda title: self._semantic_scholar_citation_evidence(title=title),
        )

    def _semantic_scholar_citation_evidence(self, *, title: str) -> EvidenceBlock | None:
        return semantic_scholar_citation_evidence(
            title=title,
            web_search=self.web_search,
            timeout_seconds=float(self.settings.tavily_timeout_seconds),
        )

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
        next_attempt = self._next_clarification_attempt(session=session, contract=contract, verification=verification)
        decision = clarification_limit_decision(
            contract=contract,
            verification=verification,
            next_attempt=next_attempt,
            max_attempts=self.agent_settings.max_clarification_attempts,
            options=self._clarification_options(contract),
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

    def _emit_agent_step(
        self,
        *,
        emit: Callable[[str, dict[str, Any]], None],
        index: int,
        action: str,
        contract: QueryContract,
        state: dict[str, Any],
        arguments: dict[str, Any] | None = None,
    ) -> None:
        emit_agent_step_event(
            emit=emit,
            index=index,
            action=action,
            contract=contract,
            arguments=arguments,
        )

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
        paper_limit = coerce_int(
            tool_input.get("top_k", plan.paper_limit),
            default=plan.paper_limit,
            minimum=1,
            maximum=50,
        )
        if paper_limit != plan.paper_limit:
            plan = plan.model_copy(update={"paper_limit": paper_limit})
        excluded_titles: set[str] = state["excluded_titles"]
        paper_query = str(tool_input.get("query", "") or "").strip() or paper_query_text(contract)
        self._emit_agent_tool_call(
            emit=emit,
            tool="search_corpus",
            arguments={
                "stage": "search_papers",
                "query": paper_query,
                "limit": plan.paper_limit,
                "requested_fields": contract.requested_fields,
                "modalities": contract.required_modalities,
            },
        )
        active = session.effective_active_research()
        paper_result = search_agent_candidate_papers(
            contract=contract,
            paper_query=paper_query,
            paper_limit=plan.paper_limit,
            active_targets=list(active.targets),
            excluded_titles=excluded_titles,
            search_papers=lambda query, search_contract, limit: self.retriever.search_papers(
                query=query,
                contract=search_contract,
                limit=limit,
            ),
            paper_lookup=self._candidate_from_paper_id,
        )
        state["contract"] = paper_result.contract
        contract = paper_result.contract
        candidate_papers = paper_result.candidate_papers
        screened_papers, precomputed_evidence = self._screen_agent_papers(
            contract=contract,
            plan=plan,
            candidate_papers=candidate_papers,
            excluded_titles=excluded_titles,
        )
        state["candidate_papers"] = candidate_papers
        state["screened_papers"] = screened_papers
        state["precomputed_evidence"] = precomputed_evidence
        emit("candidate_papers", {"count": len(candidate_papers), "items": [item.model_dump() for item in candidate_papers]})
        emit("screened_papers", {"count": len(screened_papers), "items": [item.model_dump() for item in screened_papers]})
        self._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="search_corpus",
            summary=f"candidates={len(candidate_papers)}, selected={len(screened_papers)}",
            payload={
                "stage": "search_papers",
                "candidate_count": len(candidate_papers),
                "selected_count": len(screened_papers),
                "selected_titles": [item.title for item in screened_papers[:5]],
            },
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
        self._emit_agent_tool_call(
            emit=emit,
            tool="search_corpus",
            arguments={
                "stage": "search_evidence",
                "query": result.query,
                "paper_ids": [item.paper_id for item in screened_papers],
                "limit": result.limit,
                "modalities": contract.required_modalities,
            },
        )
        evidence = result.evidence
        state["evidence"] = evidence
        emit("evidence", {"count": len(evidence), "items": [item.model_dump() for item in evidence]})
        self._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="search_corpus",
            summary=f"evidence={len(evidence)}",
            payload={
                "stage": "search_evidence",
                "evidence_count": len(evidence),
                "block_types": list(dict.fromkeys(item.block_type for item in evidence[:12])),
            },
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
            collect=lambda search_contract, enabled, limit, query: self._collect_web_evidence(
                contract=search_contract,
                use_web_search=enabled,
                max_web_results=limit,
                query_override=query,
            ),
        )
        self._emit_agent_tool_call(
            emit=emit,
            tool="web_search",
            arguments={
                "query": result.query,
                "max_results": result.max_results,
                "enabled": web_enabled,
            },
        )
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
            summary=f"web_evidence={len(web_evidence)}",
            payload={"web_evidence_count": len(web_evidence)},
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
            selected_option = selected_option_from_judge_decision(
                decision=judge_decision,
                options=ambiguity_options,
            )
            auto_resolve = selected_option is not None and judge_allows_auto_resolve(
                judge_decision,
                threshold=self.agent_settings.disambiguation_auto_resolve_threshold,
            )
            self._record_agent_observation(
                emit=emit,
                execution_steps=execution_steps,
                tool="resolve_ambiguity" if auto_resolve else "detect_ambiguity",
                summary=disambiguation_judge_summary(
                    options=ambiguity_options,
                    judge_decision=judge_decision,
                ),
                payload={
                    "options": ambiguity_options[:4],
                    "judge_decision": judge_decision.model_dump() if judge_decision is not None else {},
                },
            )
            if auto_resolve and selected_option is not None:
                contract = contract_with_auto_resolved_ambiguity(
                    contract=contract,
                    selected=selected_option,
                    decision=judge_decision,
                )
                state["contract"] = contract
                self._refresh_state_for_selected_ambiguity(
                    state=state,
                    selected=selected_option,
                    emit=emit,
                    execution_steps=execution_steps,
                )
                contract = state["contract"]
                screened_papers = state["screened_papers"]
                evidence = state["evidence"]
                claims = solve_claims_with_web_research(
                    contract=contract,
                    web_evidence=state["web_evidence"],
                    explicit_web=explicit_web_search,
                    solve_claims=lambda: self._run_solvers(
                        contract=contract,
                        plan=plan,
                        papers=screened_papers,
                        evidence=evidence,
                        session=session,
                        use_web_search=explicit_web_search,
                        max_web_results=max_web_results,
                    ),
                    build_claim=lambda item_contract, item_evidence: self._build_web_research_claim(
                        contract=item_contract,
                        web_evidence=item_evidence,
                    ),
                )
                state["claims"] = claims
            else:
                ambiguity_options = apply_disambiguation_judge_recommendation(
                    options=ambiguity_options,
                    decision=judge_decision,
                    recommend_threshold=self.agent_settings.disambiguation_recommend_threshold,
                )
                contract = contract_with_ambiguity_options(contract=contract, options=ambiguity_options)
                state["contract"] = contract
                state["claims"] = []
                state["verification"] = VerificationReport(
                    status="clarify",
                    missing_fields=disambiguation_missing_fields(contract),
                    recommended_action="clarify_ambiguous_entity",
                )
        else:
            claims = solve_claims_with_web_research(
                contract=contract,
                web_evidence=state["web_evidence"],
                explicit_web=explicit_web_search,
                solve_claims=lambda: self._run_solvers(
                    contract=contract,
                    plan=plan,
                    papers=screened_papers,
                    evidence=evidence,
                    session=session,
                    use_web_search=explicit_web_search,
                    max_web_results=max_web_results,
                ),
                build_claim=lambda item_contract, item_evidence: self._build_web_research_claim(
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
                payload={"stage": "verify_grounding", **verification.model_dump()},
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
            arguments={"stage": "verify_grounding", "claim_count": len(claims), "required_claims": plan.required_claims},
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
            payload={"stage": "verify_grounding", **verification.model_dump()},
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
        broader_candidates = retry_materials.candidate_papers
        broader_evidence = retry_materials.evidence
        goals = retry_materials.goals
        retry_plan = plan.model_copy(update={"retry_budget": 0})
        retry_claims = self._run_solvers(
            contract=contract,
            plan=retry_plan,
            papers=broader_candidates,
            evidence=broader_evidence,
            session=session,
            use_web_search=explicit_web_search,
            max_web_results=max_web_results,
        )
        retry_report = self._verify_claims(
            contract=contract,
            plan=retry_plan,
            claims=retry_claims,
            papers=broader_candidates,
            evidence=broader_evidence,
        )
        if retry_report.status == "pass":
            state["screened_papers"] = (
                self._prefer_identity_matching_papers(candidates=broader_candidates, targets=contract.targets)
                if "figure_conclusion" in goals and contract.targets
                else broader_candidates
            )
            state["evidence"] = broader_evidence
            state["claims"] = retry_claims
            state["verification"] = retry_report
        else:
            state["verification"] = retry_report
        self._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="retry_research",
            summary=f"retry_status={state['verification'].status}",
            payload={
                "candidate_count": len(broader_candidates),
                "evidence_count": len(broader_evidence),
                "claim_count": len(retry_claims),
                "status": state["verification"].status,
            },
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
        contract = self.intent_router.contract_for_query(
            clean_query=clean_query,
            session=session,
            extracted_targets=targets,
        )
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
            normalize_targets=lambda targets, requested_fields: self._normalize_contract_targets(
                targets=targets,
                requested_fields=requested_fields,
            ),
        )
        refined_contract = normalize_followup_direction_contract(
            contract=refined_contract,
            normalize_targets=lambda targets, requested_fields: self._normalize_contract_targets(
                targets=targets,
                requested_fields=requested_fields,
            ),
        )
        return apply_conversation_memory_to_contract(
            contract=refined_contract,
            session=session,
            selected_clarification_paper_id=selected_clarification_paper_id(refined_contract),
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

        return session_conversation_context(
            session,
            persistent_learnings=self._persistent_learnings_context(),
            max_chars=max_chars,
        )

    def _persistent_learnings_context(self) -> str:
        try:
            return load_learnings(data_dir=self.settings.data_dir, max_chars=4000)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load persistent learnings: %s", exc)
            return ""

    def _session_llm_history_messages(
        self,
        session: SessionContext,
        *,
        max_turns: int = 4,
        answer_limit: int = 700,
    ) -> list[dict[str, str]]:
        """Render recent turns as real chat messages for context-aware LLM calls."""

        return session_llm_history_messages(session, max_turns=max_turns, answer_limit=answer_limit)

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

    @staticmethod
    def _contract_from_selected_clarification_option(
        *,
        clean_query: str,
        target: str,
        selected: dict[str, Any],
        notes_extra: list[str] | None = None,
        resolution_note: str = "resolved_human_choice",
        resolution_subject: str = "用户选择的含义是",
    ) -> QueryContract:
        return contract_from_selected_clarification_option(
            clean_query=clean_query,
            target=target,
            selected=selected,
            notes_extra=notes_extra,
            resolution_note=resolution_note,
            resolution_subject=resolution_subject,
        )

    def _next_clarification_attempt(
        self,
        *,
        session: SessionContext,
        contract: QueryContract,
        verification: VerificationReport,
    ) -> int:
        key = self._clarification_key(contract=contract, verification=verification)
        return next_clarification_attempt(session=session, key=key)

    def _remember_clarification_attempt(
        self,
        *,
        session: SessionContext,
        contract: QueryContract,
        verification: VerificationReport,
    ) -> None:
        key = self._clarification_key(contract=contract, verification=verification)
        remember_clarification_attempt(session=session, key=key)

    @staticmethod
    def _reset_clarification_tracking(session: SessionContext) -> None:
        reset_clarification_tracking(session)

    def _clarification_key(self, *, contract: QueryContract, verification: VerificationReport) -> str:
        return clarification_tracking_key(
            contract=contract,
            verification=verification,
            options=self._clarification_options(contract),
        )

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
            excluded_titles=self._excluded_focus_titles(session=session, contract=contract),
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

    @staticmethod
    def _contract_with_ambiguity_options(*, contract: QueryContract, options: list[dict[str, Any]]) -> QueryContract:
        return contract_with_ambiguity_options(contract=contract, options=options)

    def _clarification_options(self, contract: QueryContract) -> list[dict[str, Any]]:
        return clarification_options_from_contract_notes(contract)

    def _store_pending_clarification(self, *, session: SessionContext, contract: QueryContract) -> None:
        options = self._clarification_options(contract)
        store_pending_clarification(session=session, contract=contract, options=options)

    @staticmethod
    def _clear_pending_clarification(session: SessionContext) -> None:
        clear_pending_clarification(session)

    @staticmethod
    def _make_active_research(
        *,
        relation: str,
        targets: list[str],
        titles: list[str],
        requested_fields: list[str],
        required_modalities: list[str],
        answer_shape: str,
        precision_requirement: str,
        clean_query: str,
    ) -> ActiveResearch:
        return make_active_research(
            relation=relation,
            targets=targets,
            titles=titles,
            requested_fields=requested_fields,
            required_modalities=required_modalities,
            answer_shape=answer_shape,
            precision_requirement=precision_requirement,
            clean_query=clean_query,
        )

    def _excluded_focus_titles(self, *, session: SessionContext, contract: QueryContract) -> set[str]:
        return excluded_focus_titles(
            session=session,
            contract=contract,
            is_negative_correction_query=is_negative_correction_query,
        )

    @staticmethod
    def _filter_candidate_papers_by_excluded_titles(
        candidates: list[CandidatePaper],
        *,
        excluded_titles: set[str],
    ) -> list[CandidatePaper]:
        return filter_candidate_papers_by_excluded_titles(candidates, excluded_titles=excluded_titles)

    @staticmethod
    def _filter_evidence_by_excluded_titles(
        evidence: list[EvidenceBlock],
        *,
        excluded_titles: set[str],
    ) -> list[EvidenceBlock]:
        return filter_evidence_by_excluded_titles(evidence, excluded_titles=excluded_titles)

    def _entity_evidence_limit(self, *, contract: QueryContract, plan: ResearchPlan, excluded_titles: set[str]) -> int:
        return entity_evidence_limit(contract=contract, plan=plan, excluded_titles=excluded_titles)

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
        repeated_excluded = bool(excluded_titles & {normalize_lookup_text(title) for title in focus_titles})
        if repeated_excluded:
            return {
                "decision": "clarify",
                "reason": "The candidate answer still points to a paper the user just rejected.",
                "missing_fields": ["different_interpretation"],
                "recommended_action": "clarify_or_search_alternative",
                "focus_titles": focus_titles,
            }
        if verification.status == "clarify":
            return {
                "decision": "clarify",
                "reason": verification.recommended_action or "human clarification required",
                "missing_fields": verification.missing_fields,
                "recommended_action": verification.recommended_action,
                "focus_titles": focus_titles,
            }
        goals = research_plan_goals(contract)
        if contract_needs_evidence_disambiguation(contract):
            if target_binding_from_memory(session=session, target=contract.targets[0]) and "exclude_previous_focus" not in contract.notes:
                option_count = 1
            else:
                option_count = len(self._acronym_options_from_evidence(target=contract.targets[0], papers=papers, evidence=evidence))
            if option_count > 1 and not claims and not ambiguity_options_from_notes(contract.notes):
                return {
                    "decision": "clarify",
                    "reason": "Multiple acronym meanings remain unresolved.",
                    "missing_fields": disambiguation_missing_fields(contract),
                    "recommended_action": "clarify_ambiguous_entity",
                    "focus_titles": focus_titles,
                }
        return {
            "decision": verification.status,
            "reason": "grounding verified" if verification.status == "pass" else verification.recommended_action,
            "focus_titles": focus_titles,
        }

    def _normalize_contract_targets(self, *, targets: list[str], requested_fields: list[str]) -> list[str]:
        return normalize_contract_targets(
            targets=targets,
            requested_fields=requested_fields,
            canonicalize_targets=self.retriever.canonicalize_targets,
        )

    def _build_research_plan(self, contract: QueryContract) -> ResearchPlan:
        return build_research_plan(contract=contract, settings=self.settings)

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

    def _should_use_web_search(self, *, use_web_search: bool, contract: QueryContract) -> bool:
        return should_use_web_search(use_web_search=use_web_search, contract=contract)

    def _collect_web_evidence(
        self,
        *,
        contract: QueryContract,
        use_web_search: bool,
        max_web_results: int,
        query_override: str = "",
    ) -> list[EvidenceBlock]:
        return collect_web_evidence(
            web_search=self.web_search,
            contract=contract,
            use_web_search=use_web_search,
            max_web_results=max_web_results,
            query_override=query_override,
        )

    @staticmethod
    def _build_web_research_claim(*, contract: QueryContract, web_evidence: list[EvidenceBlock]) -> Claim:
        return build_web_research_claim(contract=contract, web_evidence=web_evidence)

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
            selected_candidate_assessment=lambda paper: self._selected_followup_candidate_assessment(
                contract=contract,
                seed_papers=seed_papers,
                paper=paper,
                evidence=evidence or [],
            ),
            coerce_confidence=lambda value: self._coerce_confidence(value),
        )

    def _selected_followup_candidate_assessment(
        self,
        *,
        contract: QueryContract,
        seed_papers: list[CandidatePaper],
        paper: CandidatePaper,
        evidence: list[EvidenceBlock],
    ) -> dict[str, Any]:
        return selected_followup_candidate_assessment(
            contract=contract,
            seed_papers=seed_papers,
            paper=paper,
            evidence=evidence,
            clients=self.clients,
            expand_evidence=lambda paper_ids, query, evidence_contract, limit: self.retriever.expand_evidence(
                paper_ids=paper_ids,
                query=query,
                contract=evidence_contract,
                limit=limit,
            ),
            paper_summary_text=lambda paper_id: self._paper_summary_text(paper_id),
            coerce_confidence=lambda value: self._coerce_confidence(value),
        )

    def _llm_validate_followup_candidate(
        self,
        *,
        contract: QueryContract,
        seed_papers: list[CandidatePaper],
        paper: CandidatePaper,
        relationship_evidence: list[EvidenceBlock],
    ) -> dict[str, Any]:
        return llm_validate_followup_candidate(
            contract=contract,
            seed_papers=seed_papers,
            paper=paper,
            relationship_evidence=relationship_evidence,
            clients=self.clients,
            paper_summary_text=lambda paper_id: self._paper_summary_text(paper_id),
            coerce_confidence=lambda value: self._coerce_confidence(value),
        )

    def _paper_summary_text(self, paper_id: str) -> str:
        doc = self.retriever.paper_doc_by_id(paper_id)
        if doc is None:
            return ""
        meta = dict(doc.metadata or {})
        return str(meta.get("generated_summary") or meta.get("abstract_note") or doc.page_content[:400]).strip()

    def _paper_recommendation_reason(self, paper: CandidatePaper) -> str:
        summary = self._paper_summary_text(paper.paper_id)
        return paper_recommendation_reason(summary)

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

    def _build_figure_contexts(self, evidence: list[EvidenceBlock], limit: int = 2) -> list[dict[str, Any]]:
        return build_figure_contexts(evidence, limit=limit)

    @staticmethod
    def _figure_signal_score(text: str) -> int:
        return figure_signal_score(text)

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
