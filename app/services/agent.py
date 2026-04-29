from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import httpx

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
    SessionTurn,
    VerificationReport,
)
from app.services.agent_context import AgentRunContext
from app.services.agent_loop import finish_agent_turn, run_compound_turn_if_needed, run_standard_turn
from app.services.agent_task import run_task_subagent
from app.services.model_clients import ModelClients
from app.services.learnings import load_learnings
from app.services.agent_planner import AgentPlanner
from app.services.agent_runtime import AgentRuntime
from app.services.agent_tools import agent_tool_manifest, all_agent_tool_names
from app.services.clarification_intents import (
    CLARIFICATION_OPTION_SCHEMA_VERSION,
    ambiguity_options_from_notes,
    clarification_option_public_payload,
    looks_like_clarification_choice_text,
    pending_clarification_selection_index,
)
from app.services.citation_ranking import (
    extract_citation_count_from_evidence,
    format_citation_ranking_answer,
    parse_citation_count,
    title_token_overlap,
)
from app.services.compound_intents import should_try_compound_decomposition_heuristic
from app.services.confidence import confidence_from_verification_report, confidence_payload
from app.services.contract_context import (
    LEGACY_TOOL_NAME_ALIASES,
    canonical_agent_tool,
    canonical_tools,
    contract_allows_active_context_override,
    contract_answer_slots,
    contract_topic_state,
    intent_kind_from_contract,
    note_float,
    note_value,
    note_values,
    observed_tool_names,
)
from app.services.contract_normalization import (
    clean_contract_target_text,
    is_structural_target_reference,
    normalize_contract_targets,
    normalize_lookup_text,
    normalize_modalities,
)
from app.services.followup_intents import (
    formula_query_allows_active_paper_context,
    is_formula_interpretation_followup_query,
    is_language_preference_followup,
    is_memory_synthesis_query,
    is_negative_correction_query,
    looks_like_active_paper_reference,
    looks_like_contextual_metric_query,
    looks_like_formula_answer_correction,
    looks_like_formula_location_correction,
    looks_like_paper_scope_correction,
)
from app.services.followup_relationship_intents import (
    followup_relationship_recheck_requested,
    followup_relevance_score,
    has_followup_domain_signal,
    has_followup_seed_intro_signal,
    has_followup_soft_relation_signal,
    has_followup_support_relation_signal,
    target_relation_cue_near_text,
)
from app.services.evidence_presentation import (
    build_figure_contexts,
    chunk_text,
    citations_from_doc_ids,
    paper_recommendation_reason,
    safe_year,
)
from app.services.figure_intents import figure_signal_score
from app.services.intent import IntentRecognizer
from app.services.library_intents import (
    citation_ranking_has_library_context,
    is_citation_ranking_query,
    is_library_count_query,
    is_library_status_query,
    is_scoped_library_recommendation_query,
    library_query_prefers_previous_candidates,
)
from app.services.query_shaping import (
    evidence_query_text,
    extract_targets,
    is_short_acronym,
    matches_target,
    paper_query_text,
    should_use_concept_evidence,
    should_use_web_search,
)
from app.services.research_planning import (
    build_research_plan,
    research_plan_goals,
)
from app.services.web_evidence import (
    build_web_research_claim,
    merge_evidence,
    should_add_web_claim,
    web_include_domains,
    web_query_text,
    web_search_topic,
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

logger = logging.getLogger(__name__)
ALLOWED_SUBPROCESS_COMMANDS = {"pdftoppm"}

def _subprocess_command_allowed(command: list[str]) -> bool:
    if not command:
        return False
    executable = str(command[0] or "").strip()
    if not executable:
        return False
    return executable == Path(executable).name and executable in ALLOWED_SUBPROCESS_COMMANDS


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
        notes = [str(item) for item in list(contract.notes or [])]
        answer_slots = contract_answer_slots(contract)
        ambiguous_slots = note_values(notes=notes, prefix="ambiguous_slot=")
        topic_state = contract_topic_state(contract)
        planned_raw = list((tool_plan or {}).get("actions", []) or [])
        observed_raw = observed_tool_names(execution_steps or [])
        planned = self._canonical_tools(planned_raw)
        observed = self._canonical_tools(observed_raw)
        verification_status = str((verification_report or {}).get("status") or "")
        verifier_confidence = confidence_payload(confidence_from_verification_report(verification_report or {}))
        selected_paper_id = note_value(notes=notes, prefix="selected_paper_id=")
        selected_title = note_value(notes=notes, prefix="memory_title=")
        binding_sources = [
            note
            for note in notes
            if note
            in {
                "resolved_from_conversation_memory",
                "resolved_from_user_paper_hint",
                "formula_contextual_paper_binding",
                "formula_location_followup",
                "exclude_previous_focus",
            }
        ]
        contract_context = {
            "topic_state": topic_state,
            "target_aliases": note_values(notes=notes, prefix="target_alias="),
            "selected_paper_id": selected_paper_id,
            "selected_title": selected_title,
            "binding_sources": binding_sources,
            "needs_clarification": "intent_needs_clarification" in notes,
            "clarification_reasons": list(
                dict.fromkeys([*ambiguous_slots, *[note for note in notes if note in {"low_intent_confidence"}]])
            ),
        }
        claim_source_counts: dict[str, int] = {}
        for claim in list(claims or []):
            if not isinstance(claim, Claim):
                continue
            source = str(dict(claim.structured_data or {}).get("source") or "legacy_solver")
            claim_source_counts[source] = claim_source_counts.get(source, 0) + 1
        canonical_tool_names = all_agent_tool_names()
        summary = {
            "intent": {
                "kind": note_value(notes=notes, prefix="intent_kind=") or intent_kind_from_contract(contract),
                "confidence": note_float(notes=notes, prefix="intent_confidence="),
                "goal": contract.clean_query,
                "mode": contract.interaction_mode,
                "relation": contract.relation,
                "targets": list(contract.targets),
                "answer_slots": answer_slots,
                "ambiguous_slots": ambiguous_slots,
                "needs_local_corpus": contract.interaction_mode == "research",
                "needs_web": bool(contract.allow_web_search),
                "refers_previous_turn": contract.continuation_mode == "followup",
                "topic_state": topic_state,
                "active_topic": note_value(notes=notes, prefix="active_topic="),
            },
            "tool_loop": {
                "mode": "react_loop",
                "planned_tools": planned,
                "observed_tools": observed,
                "raw_planned_tools": [str(item) for item in planned_raw],
                "raw_observed_tools": observed_raw,
                "legacy_tools": [
                    tool for tool in observed_raw if tool not in canonical_tool_names and tool not in {"agent_loop", "conversation_agent_loop"}
                ],
            },
            "grounding": {
                "verification_status": verification_status or "pending",
                "confidence": verifier_confidence,
                "claim_count": len(claims or []),
                "citation_count": len(citations or []),
                "claim_sources": claim_source_counts,
                "research_solver_sequence": list((research_plan or {}).get("solver_sequence", []) or []),
            },
            "contract_context": contract_context,
        }
        if session is not None:
            summary["active_research_context"] = session.active_research_context_payload()
        return summary

    @staticmethod
    def _canonical_tools(raw_tools: list[Any]) -> list[str]:
        return canonical_tools(
            raw_tools=raw_tools,
            aliases=LEGACY_TOOL_NAME_ALIASES,
            canonical_names=all_agent_tool_names(),
        )

    def _compose_memory_synthesis_answer(self, *, query: str, session: SessionContext, contract: QueryContract) -> str:
        if self.clients.chat is not None:
            text = self.clients.invoke_text(
                system_prompt=(
                    "你是论文研究 Agent 的会话记忆综合器。"
                    "只基于 conversation_context 中已经发生的工具结果、回答、claims/引用线索来回答当前追问；"
                    "不要重新检索，不要编造未出现过的新事实。"
                    "如果用户问比较/区别，先给一句总览，再用简洁表格或要点比较。"
                    "如果记忆里的某一项证据不足，要明确说证据不足。"
                    "输出中文 Markdown。"
                ),
                human_prompt=json.dumps(
                    {
                        "current_query": query,
                        "current_contract": contract.model_dump(),
                        "conversation_context": self._session_conversation_context(session),
                    },
                    ensure_ascii=False,
                ),
                fallback="",
            ).strip()
            if text:
                return self._clean_common_ocr_artifacts(text)
        last_compound = dict((session.working_memory or {}).get("last_compound_query", {}) or {})
        rows = []
        for item in list(last_compound.get("subtasks", []) or [])[:4]:
            targets = item.get("targets") or []
            target = str(targets[0]) if targets else str(item.get("clean_query", "对象"))
            preview = " ".join(str(item.get("answer_preview", "")).split())
            rows.append(f"- **{target}**：{preview[:260] if preview else '上一轮没有留下足够细节。'}")
        if rows:
            return "基于上一轮已经完成的检索结果，先做保守比较：\n\n" + "\n".join(rows)
        return "这轮追问看起来是在比较上一轮对象，但当前会话记忆里没有足够可综合的工具结果。"

    def _compose_memory_followup_answer(self, *, query: str, session: SessionContext, contract: QueryContract) -> str:
        requested = {str(item) for item in list(contract.requested_fields or [])}
        if "formula_interpretation" in requested:
            return self._compose_formula_interpretation_followup_answer(query=query, session=session, contract=contract)
        if "answer_language_preference" in requested:
            return self._compose_language_preference_followup_answer(query=query, session=session, contract=contract)
        artifact_answer = self._answer_from_recent_tool_artifact_reference(query=query, session=session)
        if artifact_answer:
            return artifact_answer
        if self.clients.chat is not None:
            text = self.clients.invoke_text(
                system_prompt=(
                    "你是论文研究 Agent 的通用会话记忆问答工具。"
                "你的输入是完整 conversation_context，其中包含历史用户问题、助手回答、工具结果摘要、working_memory。"
                "请只基于这些记忆回答当前追问；不要重新推荐、不要重新检索、不要编造未出现过的新事实。"
                "只适用于回答上一轮工具输出本身的依据、选择理由、排序理由、措辞解释。"
                "如果用户是在问某篇论文/模型/方法本身的正文内容、核心结论、实验结果、方法细节，"
                "不要在这里编答案，应说明需要进入论文检索/阅读工具。"
                "如果记忆不足以回答，就说清楚缺什么，并建议下一步调用哪个工具，而不是假装知道。"
                "输出简洁中文 Markdown，像在继续聊天，不要复读整份上一轮答案。"
            ),
                human_prompt=json.dumps(
                    {
                        "current_query": query,
                        "current_contract": contract.model_dump(),
                        "conversation_context": self._session_conversation_context(session),
                    },
                    ensure_ascii=False,
                ),
                fallback="",
            ).strip()
            if text:
                return self._clean_common_ocr_artifacts(text)
        previous = session.turns[-1].answer if session.turns else ""
        if previous:
            compact = " ".join(previous.split())
            return f"我根据上一轮结果回答：{compact[:420]}"
        return "当前会话记忆里没有足够的上一轮工具结果来回答这个追问。"

    def _answer_from_recent_tool_artifact_reference(self, *, query: str, session: SessionContext) -> str:
        item_index = self._referenced_list_item_index(query)
        if item_index is None:
            return ""
        artifact = self._latest_list_tool_artifact(session)
        if not artifact:
            return ""
        items = [item for item in list(artifact.get("items", []) or []) if isinstance(item, dict)]
        if item_index < 0:
            return ""
        if item_index >= len(items):
            source_query = str(artifact.get("query", "") or "上一轮工具结果").strip()
            return f"上一轮“{source_query}”只保留了 {len(items)} 条结构化结果，找不到第 {item_index + 1} 条。"
        item = items[item_index]
        row = dict(item.get("row", {}) or {})
        ordinal = int(item.get("ordinal", item_index + 1) or item_index + 1)
        source_query = str(artifact.get("query", "") or "").strip()
        source_tool = str(artifact.get("tool", "") or "tool").strip()
        lines = []
        if source_query:
            lines.append(f"按上一轮“{source_query}”的 `{source_tool}` 结果，第 {ordinal} 条是：")
        else:
            lines.append(f"按上一轮 `{source_tool}` 结果，第 {ordinal} 条是：")
        lines.append("")
        title = str(row.get("title", "") or row.get("paper_title", "") or "").strip()
        if title:
            lines.append(f"- 标题：{title}")
        for key, label in [
            ("year", "年份"),
            ("year_int", "年份"),
            ("authors", "作者"),
            ("author", "作者"),
            ("paper_id", "paper_id"),
            ("categories", "分类"),
            ("tags", "标签"),
        ]:
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if not text or (key == "year_int" and str(row.get("year", "") or "").strip()):
                continue
            lines.append(f"- {label}：{text}")
        if len(lines) <= 2:
            for key, value in row.items():
                text = "" if value is None else str(value).strip()
                if text:
                    lines.append(f"- {key}：{text}")
        return "\n".join(lines).strip()

    @staticmethod
    def _latest_list_tool_artifact(session: SessionContext) -> dict[str, Any]:
        memory = dict(session.working_memory or {})
        direct = memory.get("last_displayed_list")
        if isinstance(direct, dict) and isinstance(direct.get("items"), list):
            return dict(direct)
        for result in reversed([item for item in list(memory.get("tool_results", []) or []) if isinstance(item, dict)]):
            artifact = result.get("artifact")
            if isinstance(artifact, dict) and isinstance(artifact.get("items"), list):
                merged = dict(artifact)
                merged.setdefault("query", result.get("query", ""))
                merged.setdefault("tool", result.get("tool", ""))
                return merged
        return {}

    @classmethod
    def _referenced_list_item_index(cls, query: str) -> int | None:
        compact = re.sub(r"\s+", "", str(query or "").strip().lower())
        if not compact:
            return None
        digit_match = re.search(r"第(\d+)(篇|个|项|条|篇论文|篇文章)?", compact)
        if digit_match:
            return max(0, int(digit_match.group(1)) - 1)
        chinese_match = re.search(r"第([一二三四五六七八九十两]+)(篇|个|项|条|篇论文|篇文章)?", compact)
        if chinese_match:
            value = cls._chinese_ordinal_value(chinese_match.group(1))
            if value is not None:
                return max(0, value - 1)
        english_ordinals = {
            "first": 1,
            "1st": 1,
            "second": 2,
            "2nd": 2,
            "third": 3,
            "3rd": 3,
            "fourth": 4,
            "4th": 4,
            "fifth": 5,
            "5th": 5,
            "sixth": 6,
            "6th": 6,
            "seventh": 7,
            "7th": 7,
            "eighth": 8,
            "8th": 8,
            "ninth": 9,
            "9th": 9,
            "tenth": 10,
            "10th": 10,
        }
        for token, value in english_ordinals.items():
            if token in compact:
                return value - 1
        return None

    @staticmethod
    def _chinese_ordinal_value(text: str) -> int | None:
        digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        raw = str(text or "").strip()
        if not raw:
            return None
        if raw == "十":
            return 10
        if "十" in raw:
            left, _, right = raw.partition("十")
            tens = digits.get(left, 1 if left == "" else 0)
            ones = digits.get(right, 0) if right else 0
            value = tens * 10 + ones
            return value if value > 0 else None
        return digits.get(raw)

    def _compose_formula_interpretation_followup_answer(
        self,
        *,
        query: str,
        session: SessionContext,
        contract: QueryContract,
    ) -> str:
        previous_turns = [
            turn
            for turn in reversed(session.turns)
            if turn.relation == "formula_lookup"
            or "formula" in {str(item) for item in list(turn.requested_fields or [])}
            or "formula" in {str(item) for item in list(turn.answer_slots or [])}
        ]
        previous = previous_turns[0] if previous_turns else (session.turns[-1] if session.turns else None)
        previous_answer = previous.answer if previous is not None else ""
        if self.clients.chat is not None and previous_answer:
            text = self.clients.invoke_text(
                system_prompt=(
                    "你是论文公式讲解器。用户当前是在追问上一轮已经给出的公式应该如何理解。"
                    "只能基于上一轮回答、变量解释和会话记忆来解释，不要重新检索、不要引入新论文事实。"
                    "不要完整重抄公式，不要重新列一遍变量表；最多引用 1-2 个关键符号。"
                    "用简洁中文 Markdown 输出，重点讲：这个式子在优化什么、正负样本如何影响方向、"
                    "参考策略/温度系数/sigmoid 或 log-ratio 的直觉，以及最容易误解的边界。"
                    "所有数学符号必须用 KaTeX 可渲染的标准 LaTeX 并包在 $...$ 中，例如 $\\pi_{\\theta}$、"
                    "$\\pi_{\\mathrm{ref}}$、$y_w$、$y_l$、$\\log \\sigma$。"
                    "不要输出 $pi_{theta}$、$pi_mathrmref$、$frac...$ 这类缺少反斜杠或大括号的裸符号；"
                    "如果不确定 LaTeX 写法，就改用中文描述，不要写半截公式。"
                ),
                human_prompt=json.dumps(
                    {
                        "current_query": query,
                        "current_contract": contract.model_dump(),
                        "previous_formula_query": previous.query if previous is not None else "",
                        "previous_formula_answer": previous_answer,
                        "conversation_context": self._session_conversation_context(session, max_chars=12000),
                    },
                    ensure_ascii=False,
                ),
                fallback="",
            ).strip()
            if text:
                return self._clean_common_ocr_artifacts(text)
        if previous_answer:
            compact = " ".join(previous_answer.split())
            return (
                "## 怎么读\n\n"
                "这条公式先看优化方向：它想让偏好回答相对参考策略更可能，让劣选回答相对参考策略更不可能。"
                "再看缩放项：温度系数控制偏好信号强度，sigmoid/log-ratio 把“偏好回答是否已经明显强于劣选回答”变成训练权重。\n\n"
                f"上一轮公式摘要：{compact[:360]}"
            )
        return "我需要上一轮已经定位到的公式，才能继续解释它的直觉。"

    def _compose_language_preference_followup_answer(
        self,
        *,
        query: str,
        session: SessionContext,
        contract: QueryContract,
    ) -> str:
        previous = session.turns[-1].answer if session.turns else ""
        if self.clients.chat is not None and previous:
            text = self.clients.invoke_text(
                system_prompt=(
                    "你是论文研究 Agent 的回答语言修正器。用户指出上一条回答中英文混杂，要求中文。"
                    "请只基于上一条回答做中文化改写或简短确认，不要检索论文，不要新增引用，不要编造新事实。"
                    "公式、变量符号、论文标题和不可翻译专名可以保留英文；变量解释、句子说明必须使用中文。"
                    "输出简洁中文 Markdown。"
                ),
                human_prompt=json.dumps(
                    {
                        "current_query": query,
                        "current_contract": contract.model_dump(),
                        "previous_answer": previous,
                        "conversation_context": self._session_conversation_context(session, max_chars=8000),
                    },
                    ensure_ascii=False,
                ),
                fallback="",
            ).strip()
            if text:
                return self._clean_common_ocr_artifacts(text)
        return "好的，后续我会用中文说明；公式符号和论文标题会保留原样，变量解释和推理过程用中文。"

    def _llm_memory_followup_contract(
        self,
        *,
        clean_query: str,
        session: SessionContext,
        current_contract: QueryContract,
    ) -> QueryContract | None:
        if self.clients.chat is None or not session.turns:
            return None
        payload = self.clients.invoke_json(
            system_prompt=(
                "你是论文研究 Agent 的会话记忆追问判别器。"
                "判断当前用户问题是否只是在追问上一轮工具输出本身，而不需要读取新的论文证据。"
                "只有当问题可以主要基于 conversation_context 回答时，返回 should_use_memory=true。"
                "典型 true：为什么这么推荐、推荐理由、上一轮排序依据、上一轮回答里的某个结论依据。"
                "典型 false：用户问某篇论文具体说了什么、核心结论、方法、实验结果、图表、公式、更多细节；"
                "这类问题虽然要用记忆解析指代，但必须交给后续研究工具检索正文证据。"
                "如果当前问题需要新的论文检索、外部动态信息、全新主题，返回 false。"
                "只输出 JSON：should_use_memory, reason, targets, requested_fields, answer_shape。"
            ),
            human_prompt=json.dumps(
                {
                    "current_query": clean_query,
                    "current_contract": current_contract.model_dump(),
                    "conversation_context": self._session_conversation_context(session),
                },
                ensure_ascii=False,
            ),
            fallback={},
        )
        if not isinstance(payload, dict) or not bool(payload.get("should_use_memory")):
            return None
        raw_targets = payload.get("targets", [])
        targets = [str(item).strip() for item in raw_targets if str(item).strip()] if isinstance(raw_targets, list) else []
        raw_fields = payload.get("requested_fields", [])
        requested_fields = [str(item).strip() for item in raw_fields if str(item).strip()] if isinstance(raw_fields, list) else ["answer"]
        answer_shape = str(payload.get("answer_shape", current_contract.answer_shape) or "").strip().lower()
        if answer_shape not in {"bullets", "narrative", "table"}:
            answer_shape = "narrative"
        return QueryContract(
            clean_query=clean_query,
            interaction_mode="conversation",
            relation="memory_followup",
            targets=targets,
            requested_fields=requested_fields or ["answer"],
            required_modalities=[],
            answer_shape=answer_shape,
            precision_requirement="normal",
            continuation_mode="followup",
            notes=["agent_tool", "llm_memory_followup", str(payload.get("reason", ""))[:180]],
        )

    @staticmethod
    def _is_formula_interpretation_followup(*, clean_query: str, session: SessionContext) -> bool:
        active = session.effective_active_research()
        had_formula_context = active.relation == "formula_lookup" or any(
            turn.relation == "formula_lookup"
            or "formula" in {str(item) for item in list(turn.requested_fields or [])}
            or "formula" in {str(item) for item in list(turn.answer_slots or [])}
            for turn in session.turns[-3:]
        )
        return is_formula_interpretation_followup_query(clean_query, had_formula_context=had_formula_context)

    @staticmethod
    def _is_language_preference_followup(*, clean_query: str, session: SessionContext) -> bool:
        return is_language_preference_followup(clean_query, has_turns=bool(session.turns))

    @staticmethod
    def _is_comparison_query(query: str) -> bool:
        normalized = " ".join(str(query or "").lower().split())
        return any(token in normalized for token in ["区别", "比较", "对比", "difference", "compare", "vs"])

    def _active_memory_bindings(self, session: SessionContext) -> list[dict[str, Any]]:
        bindings = dict((session.working_memory or {}).get("target_bindings", {}) or {})
        selected: list[dict[str, Any]] = []
        for target in session.effective_active_research().targets:
            binding = bindings.get(normalize_lookup_text(target))
            if isinstance(binding, dict):
                selected.append(dict(binding))
        if len(selected) >= 2:
            return selected
        for binding in bindings.values():
            if isinstance(binding, dict) and binding not in selected:
                selected.append(dict(binding))
            if len(selected) >= 4:
                break
        return selected

    def _citations_from_memory_bindings(self, bindings: list[dict[str, Any]]) -> list[AssistantCitation]:
        doc_ids: list[str] = []
        for binding in bindings:
            for doc_id in list(binding.get("evidence_ids", []) or [])[:2]:
                if str(doc_id).strip():
                    doc_ids.append(str(doc_id).strip())
            paper_id = str(binding.get("paper_id", "") or "").strip()
            if paper_id:
                doc_ids.append(f"paper::{paper_id}")
        return self._dedupe_citations(self._citations_from_doc_ids(list(dict.fromkeys(doc_ids)), []))

    def _should_try_compound_decomposition(self, clean_query: str, *, session: SessionContext | None = None) -> bool:
        normalized = normalize_lookup_text(clean_query)
        memory = dict((session.working_memory if session is not None else {}) or {})
        bindings = dict(memory.get("target_bindings", {}) or {})
        has_memory_context = bool(bindings or (session is not None and session.effective_active_research().targets))
        target_count = len({target.lower() for target in extract_targets(clean_query)})
        return should_try_compound_decomposition_heuristic(
            clean_query,
            normalized_query=normalized,
            target_count=target_count,
            has_memory_context=has_memory_context,
        )

    def _llm_decompose_compound_query(self, *, clean_query: str, session: SessionContext) -> list[QueryContract]:
        if self.clients.chat is None:
            return []
        system_prompt = (
                "你是论文研究 Agent 的任务分解器，不是最终回答器。"
                "你的任务是判断当前用户消息是否包含多个可执行子任务，并把它们拆成有序 QueryContract。"
                "检索论文只是工具，子任务应围绕用户真实意图组织，而不是套模板。"
                "你可以参考 available_tools，但你不调用工具；planner/executor 会基于你的子任务调用工具。"
                "如果问题只有一个任务，输出 is_compound=false 和空 subtasks。"
                "如果一句话中有多个需求，例如多个公式查询、总结+实验结果、查询+比较、数量+推荐，输出 is_compound=true。"
                "但同一篇论文/同一实体的多个字段（例如“核心结论是什么，实验结果如何”）不是 compound；"
                "必须合并为一个 QueryContract，并把 requested_fields 写成多个字段。"
                "每个 subtask 字段为 clean_query, interaction_mode, intent_kind, continuation_mode, targets, answer_slots, "
                "requested_fields, required_modalities, answer_shape, precision_requirement, notes。"
                "不要输出 relation；用 answer_slots/requested_fields 表达子任务目标。"
                "可用 answer_slots 包括 library_status, library_recommendation, origin, formula, followup_research, "
                "entity_definition, topology_discovery, topology_recommendation, figure, paper_summary, metric_value, "
                "concept_definition, paper_recommendation, comparison, general_answer。"
                "interaction_mode 只能是 conversation 或 research。"
                "required_modalities 只能使用 page_text, paper_card, table, caption, figure。"
                "answer_shape 只能是 bullets, narrative, table。precision_requirement 只能是 exact, high, normal。"
                "公式查询使用 answer_slots=[formula] + requested_fields=[formula, variable_explanation] + required_modalities=[page_text, table]。"
                "库状态/库列表/库元信息问题使用 library_status，必须 interaction_mode=conversation，targets=[]，不要走 research 检索；"
                "例如按年份、作者、标签、分类、PDF 有无统计或筛选当前库内论文。"
                "库内默认推荐问题使用 library_recommendation，必须 interaction_mode=conversation，targets=[]。"
                "比较/综合使用 answer_slots=[comparison]，并且应放在其依赖的检索子任务之后。"
                "targets 只能放实体本身，不要把“公式、结果、summary”等任务词拼进 target。"
                "只输出 JSON：is_compound, reason, subtasks。"
        )
        human_payload = {
            "current_query": clean_query,
            "available_tools": agent_tool_manifest(),
            "conversation_context": self._session_conversation_context(session, max_chars=10000),
        }
        invoke_json_messages = getattr(self.clients, "invoke_json_messages", None)
        if callable(invoke_json_messages):
            payload = invoke_json_messages(
                system_prompt=system_prompt,
                messages=[
                    *self._session_llm_history_messages(session),
                    {"role": "user", "content": json.dumps(human_payload, ensure_ascii=False)},
                ],
                fallback={},
            )
        else:
            payload = self.clients.invoke_json(
                system_prompt=system_prompt,
                human_prompt=json.dumps(human_payload, ensure_ascii=False),
                fallback={},
            )
        if not isinstance(payload, dict) or not bool(payload.get("is_compound")):
            return []
        raw_subtasks = payload.get("subtasks", [])
        if not isinstance(raw_subtasks, list):
            return []
        contracts: list[QueryContract] = []
        for index, item in enumerate(raw_subtasks[:5]):
            contract = self._subtask_contract_from_payload(item, fallback_query=clean_query, index=index)
            if contract is not None:
                contracts.append(contract)
        return contracts if len(contracts) >= 2 else []

    def _subtask_contract_from_payload(
        self,
        payload: object,
        *,
        fallback_query: str,
        index: int,
    ) -> QueryContract | None:
        if not isinstance(payload, dict):
            return None
        allowed_relations = {
            "library_status",
            "library_recommendation",
            "origin_lookup",
            "formula_lookup",
            "followup_research",
            "entity_definition",
            "topology_discovery",
            "topology_recommendation",
            "figure_question",
            "paper_summary_results",
            "metric_value_lookup",
            "concept_definition",
            "paper_recommendation",
            "general_question",
            "comparison_synthesis",
        }
        continuation_mode = str(payload.get("continuation_mode", "") or "").strip().lower()
        if continuation_mode not in {"fresh", "followup", "context_switch"}:
            continuation_mode = "fresh" if index == 0 else "followup"
        clean_query = " ".join(str(payload.get("clean_query", "") or fallback_query).strip().split())
        raw_targets = payload.get("targets", [])
        targets = [str(item).strip() for item in raw_targets if str(item).strip()] if isinstance(raw_targets, list) else []
        raw_answer_slots = payload.get("answer_slots", [])
        if isinstance(raw_answer_slots, str):
            raw_answer_slots = [raw_answer_slots]
        answer_slots = [str(item).strip() for item in raw_answer_slots if str(item).strip()] if isinstance(raw_answer_slots, list) else []
        raw_requested_fields = payload.get("requested_fields", [])
        requested_fields = [str(item).strip() for item in raw_requested_fields if str(item).strip()] if isinstance(raw_requested_fields, list) else []
        targets = self._normalize_contract_targets(targets=targets, requested_fields=requested_fields)
        relation = str(payload.get("relation", "") or "").strip()
        if relation not in allowed_relations:
            relation = self._subtask_relation_from_slots(
                answer_slots=answer_slots,
                requested_fields=requested_fields,
                targets=targets,
            )
        if relation not in allowed_relations:
            return None
        interaction_mode = str(payload.get("interaction_mode", "") or "").strip().lower()
        if interaction_mode not in {"conversation", "research"}:
            interaction_mode = "conversation" if relation in {"library_status", "library_recommendation", "comparison_synthesis"} else "research"
        if relation in {"library_status", "library_recommendation", "comparison_synthesis"}:
            interaction_mode = "conversation"
        if relation in {"library_status", "library_recommendation"}:
            targets = []
            requested_fields = []
        raw_required_modalities = payload.get("required_modalities", [])
        required_modalities = self._normalize_modalities(
            [str(item).strip() for item in raw_required_modalities if str(item).strip()] if isinstance(raw_required_modalities, list) else [],
            relation=relation,
        )
        if relation == "formula_lookup":
            requested_fields = [*requested_fields, *[field for field in ["formula", "variable_explanation"] if field not in requested_fields]]
            required_modalities = [*required_modalities, *[modality for modality in ["page_text", "table"] if modality not in required_modalities]]
            interaction_mode = "research"
        if interaction_mode == "conversation":
            required_modalities = []
        elif not required_modalities:
            required_modalities = ["page_text", "paper_card"]
        if interaction_mode == "research" and not requested_fields:
            requested_fields = ["answer"]
        answer_shape = str(payload.get("answer_shape", "") or "").strip().lower()
        if answer_shape not in {"bullets", "narrative", "table"}:
            answer_shape = "table" if relation == "comparison_synthesis" else "narrative"
        precision_requirement = str(payload.get("precision_requirement", "") or "").strip().lower()
        if precision_requirement not in {"exact", "high", "normal"}:
            precision_requirement = "exact" if relation in {"formula_lookup", "metric_value_lookup"} else "high"
        raw_notes = payload.get("notes", [])
        notes = [str(item).strip() for item in raw_notes if str(item).strip()] if isinstance(raw_notes, list) else []
        notes = list(dict.fromkeys([*notes, "compound_subtask", *[f"answer_slot={slot}" for slot in answer_slots], f"subtask_{relation}"]))
        return QueryContract(
            clean_query=clean_query,
            interaction_mode=interaction_mode,
            relation=relation,
            targets=targets,
            answer_slots=answer_slots,
            requested_fields=requested_fields,
            required_modalities=required_modalities,
            answer_shape=answer_shape,
            precision_requirement=precision_requirement,  # type: ignore[arg-type]
            continuation_mode=continuation_mode,  # type: ignore[arg-type]
            notes=notes,
        )

    @staticmethod
    def _subtask_relation_from_slots(
        *,
        answer_slots: list[str],
        requested_fields: list[str],
        targets: list[str],
    ) -> str:
        slots = {"_".join(str(item or "").strip().lower().replace("-", "_").split()) for item in answer_slots}
        fields = {"_".join(str(item or "").strip().lower().replace("-", "_").split()) for item in requested_fields}
        tokens = slots | fields
        if "library_status" in tokens:
            return "library_status"
        if "library_recommendation" in tokens:
            return "library_recommendation"
        if "comparison" in tokens or "synthesis" in tokens:
            return "comparison_synthesis"
        if "origin" in tokens or {"paper_title", "year"} <= tokens:
            return "origin_lookup"
        if "formula" in tokens:
            return "formula_lookup"
        if "followup_research" in tokens or "followup_papers" in tokens:
            return "followup_research"
        if "figure" in tokens or "figure_conclusion" in tokens:
            return "figure_question"
        if "metric_value" in tokens:
            return "metric_value_lookup"
        if "paper_summary" in tokens or "summary" in tokens or "results" in tokens:
            return "paper_summary_results"
        if "paper_recommendation" in tokens or "recommended_papers" in tokens:
            return "paper_recommendation"
        if "topology_recommendation" in tokens or "best_topology" in tokens:
            return "topology_recommendation"
        if "topology_discovery" in tokens or "relevant_papers" in tokens:
            return "topology_discovery"
        if "entity_definition" in tokens or "entity_type" in tokens or ("definition" in tokens and targets):
            return "entity_definition"
        if "concept_definition" in tokens or "definition" in tokens:
            return "concept_definition"
        return "general_question"

    def _merge_redundant_field_subtasks(self, subcontracts: list[QueryContract]) -> list[QueryContract]:
        mergeable_relations = {
            "paper_summary_results",
            "metric_value_lookup",
            "entity_definition",
            "concept_definition",
            "formula_lookup",
            "figure_question",
            "general_question",
            "followup_research",
        }
        merged: list[QueryContract] = []
        by_key: dict[tuple[str, str, tuple[str, ...]], int] = {}
        precision_rank = {"normal": 0, "high": 1, "exact": 2}
        for contract in subcontracts:
            normalized_targets = tuple(normalize_lookup_text(target) for target in contract.targets if target)
            key = (contract.interaction_mode, contract.relation, normalized_targets)
            if contract.relation not in mergeable_relations or key not in by_key:
                by_key[key] = len(merged)
                merged.append(contract)
                continue
            existing_index = by_key[key]
            existing = merged[existing_index]
            requested_fields = list(dict.fromkeys([*existing.requested_fields, *contract.requested_fields]))
            required_modalities = list(dict.fromkeys([*existing.required_modalities, *contract.required_modalities]))
            notes = list(dict.fromkeys([*existing.notes, *contract.notes, "merged_same_target_fields"]))
            clean_query = existing.clean_query
            if contract.clean_query and contract.clean_query not in clean_query:
                clean_query = f"{clean_query}；{contract.clean_query}"
            precision = (
                contract.precision_requirement
                if precision_rank.get(contract.precision_requirement, 0) > precision_rank.get(existing.precision_requirement, 0)
                else existing.precision_requirement
            )
            merged[existing_index] = existing.model_copy(
                update={
                    "clean_query": clean_query,
                    "requested_fields": requested_fields or existing.requested_fields,
                    "required_modalities": required_modalities or existing.required_modalities,
                    "precision_requirement": precision,
                    "notes": notes,
                }
            )
        return merged

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
            description=self._compound_task_label(contract),
            tools_allowed=[],
            max_steps=8,
            session=session,
            max_web_results=3,
            emit=emit,
            execution_steps=execution_steps,
            contract=contract,
        )
        result = self._compound_task_result_from_task_payload(task_result, fallback_contract=contract)
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

    @staticmethod
    def _compound_task_result_from_task_payload(
        task_result: dict[str, Any],
        *,
        fallback_contract: QueryContract,
    ) -> dict[str, Any]:
        contract = task_result.get("contract_obj")
        if not isinstance(contract, QueryContract):
            raw_contract = task_result.get("contract")
            if isinstance(raw_contract, dict):
                try:
                    contract = QueryContract.model_validate(raw_contract)
                except Exception:  # noqa: BLE001
                    contract = fallback_contract
            else:
                contract = fallback_contract
        verification = task_result.get("verification_obj")
        if not isinstance(verification, VerificationReport):
            raw_verification = task_result.get("verification")
            if isinstance(raw_verification, dict):
                try:
                    verification = VerificationReport.model_validate(raw_verification)
                except Exception:  # noqa: BLE001
                    verification = VerificationReport(status="pass", recommended_action="task_subagent")
            else:
                verification = VerificationReport(status="pass", recommended_action="task_subagent")
        return {
            "contract": contract,
            "answer": str(task_result.get("answer", "") or ""),
            "citations": list(task_result.get("citations", []) or []),
            "claims": list(task_result.get("claims", []) or []),
            "evidence": list(task_result.get("evidence", []) or []),
            "verification": verification,
        }

    def _compose_compound_comparison_answer(
        self,
        *,
        query: str,
        subtask_results: list[dict[str, Any]],
        session: SessionContext,
        comparison_contract: QueryContract | None = None,
    ) -> str:
        comparable_results = self._comparison_results_with_memory(
            subtask_results=subtask_results,
            session=session,
            comparison_contract=comparison_contract,
        )
        comparable = [
            {
                "relation": result["contract"].relation if isinstance(result.get("contract"), QueryContract) else "",
                "targets": result["contract"].targets if isinstance(result.get("contract"), QueryContract) else [],
                "answer": str(result.get("answer", "")),
                "claims": [claim.model_dump() for claim in list(result.get("claims", []) or []) if isinstance(claim, Claim)],
            }
            for result in comparable_results
        ]
        if self.clients.chat is not None:
            text = self.clients.invoke_text(
                system_prompt=(
                    "你是论文研究助手的多子任务综合器。"
                    "只基于输入的子任务答案和 claims 做比较，不要引入外部记忆。"
                    "请用简洁中文 Markdown 输出：先给 1 句总览，再用表格比较目标函数/优化信号/是否需要 reward model/使用场景，最后给读法建议。"
                    "如果某个子任务证据不足，要明确说证据不足，不要补公式。"
                ),
                human_prompt=json.dumps(
                    {
                        "query": query,
                        "subtasks": comparable,
                    },
                    ensure_ascii=False,
                ),
                fallback="",
            ).strip()
            if text:
                return self._clean_common_ocr_artifacts(text)
        rows: list[str] = []
        for result in comparable:
            targets = result.get("targets") or []
            target = str(targets[0]) if targets else "对象"
            answer = " ".join(str(result.get("answer", "")).split())
            rows.append(f"- **{target}**：{answer[:260] if answer else '当前证据不足。'}")
        return "基于前两个子任务的证据，可以先做保守比较：\n\n" + "\n".join(rows)

    def _comparison_results_with_memory(
        self,
        *,
        subtask_results: list[dict[str, Any]],
        session: SessionContext,
        comparison_contract: QueryContract | None,
    ) -> list[dict[str, Any]]:
        augmented = list(subtask_results)
        present_targets = {
            normalize_lookup_text(target)
            for result in augmented
            if isinstance(result.get("contract"), QueryContract)
            for target in result["contract"].targets
            if str(target).strip()
        }
        requested_targets = list(comparison_contract.targets if comparison_contract is not None else [])
        if not requested_targets:
            requested_targets = list(
                dict.fromkeys(
                    [*session.effective_active_research().targets, *[item.get("target", "") for item in self._active_memory_bindings(session)]]
                )
            )
        bindings = dict((session.working_memory or {}).get("target_bindings", {}) or {})
        for target in requested_targets:
            clean_target = str(target or "").strip()
            key = normalize_lookup_text(clean_target)
            if not key or key in present_targets:
                continue
            binding = bindings.get(key)
            if not isinstance(binding, dict):
                continue
            relation = str(binding.get("relation", "") or "followup_research")
            requested_fields = [str(item) for item in list(binding.get("requested_fields", []) or []) if str(item)]
            contract = QueryContract(
                clean_query=str(binding.get("clean_query", "") or clean_target),
                relation=relation,
                targets=[str(binding.get("target", "") or clean_target)],
                requested_fields=requested_fields or ["answer"],
                required_modalities=[str(item) for item in list(binding.get("required_modalities", []) or []) if str(item)] or ["page_text"],
                continuation_mode="followup",
                notes=["restored_from_session_memory_for_comparison"],
            )
            augmented.append(
                {
                    "contract": contract,
                    "answer": str(binding.get("answer_preview", "") or ""),
                    "citations": [],
                    "claims": [],
                    "evidence": [],
                    "verification": VerificationReport(status="pass", recommended_action="memory_comparison_context"),
                }
            )
            present_targets.add(key)
        return augmented

    @staticmethod
    def _dedupe_citations(citations: list[AssistantCitation]) -> list[AssistantCitation]:
        seen: set[tuple[str, str, int, str]] = set()
        deduped: list[AssistantCitation] = []
        for citation in citations:
            key = (citation.title, citation.file_path, citation.page, citation.block_type)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(citation)
        return deduped

    @staticmethod
    def _compound_section_heading(*, contract: QueryContract, index: int) -> str:
        return f"## {index}. {ResearchAssistantAgentV4._compound_task_label(contract)}"

    @staticmethod
    def _compound_research_progress_markdown(*, contract: QueryContract, index: int) -> str:
        heading = ResearchAssistantAgentV4._compound_section_heading(contract=contract, index=index)
        if contract.relation == "formula_lookup":
            target = contract.targets[0] if contract.targets else "目标对象"
            return f"{heading}\n\n好的，我现在去查询 **{target}** 的公式。"
        return heading

    @staticmethod
    def _compound_task_label(contract: QueryContract) -> str:
        if contract.relation == "library_status":
            return "查看论文库概览和文章预览"
        if contract.relation == "library_recommendation":
            return "从库内给出默认推荐"
        if contract.relation == "formula_lookup":
            target = contract.targets[0] if contract.targets else "目标对象"
            return f"查询 {target} 公式"
        if contract.relation == "comparison_synthesis":
            target_text = " 和 ".join(contract.targets) if contract.targets else "前面结果"
            return f"比较 {target_text}"
        return contract.clean_query

    @staticmethod
    def _demote_markdown_headings(answer: str) -> str:
        return re.sub(r"^(#{1,5})\\s+", lambda match: "#" + match.group(1) + " ", str(answer or "").strip(), flags=re.M)

    @staticmethod
    def _format_compound_section(*, contract: QueryContract, answer: str, index: int) -> str:
        normalized = ResearchAssistantAgentV4._demote_markdown_headings(str(answer or "").strip())
        return f"{ResearchAssistantAgentV4._compound_section_heading(contract=contract, index=index)}\n\n{normalized}".strip()

    def _select_citation_ranking_candidates(
        self,
        *,
        session: SessionContext,
        query: str,
        limit: int,
    ) -> list[dict[str, str]]:
        docs: list[dict[str, object]] = []
        seen_paper_ids: set[str] = set()
        by_title: dict[str, dict[str, object]] = {}
        for doc in self.retriever.paper_documents():
            meta = dict(doc.metadata or {})
            paper_id = str(meta.get("paper_id", "")).strip()
            title = str(meta.get("title", "") or "").strip()
            if not paper_id or paper_id in seen_paper_ids or not title:
                continue
            seen_paper_ids.add(paper_id)
            docs.append(meta)
            by_title[self._normalize_title_key(title)] = meta

        selected: list[dict[str, str]] = []
        selected_keys: set[str] = set()

        def add_candidate(*, title: str, year: str = "", reason: str = "") -> None:
            clean_title = " ".join(str(title or "").split()).strip()
            if not clean_title:
                return
            key = self._normalize_title_key(clean_title)
            if not key or key in selected_keys:
                return
            meta = by_title.get(key)
            selected_keys.add(key)
            selected.append(
                {
                    "title": str(meta.get("title", clean_title) if meta else clean_title),
                    "year": str(meta.get("year", year) if meta else year),
                    "paper_id": str(meta.get("paper_id", "") if meta else ""),
                    "reason": reason or str(meta.get("generated_summary", "") if meta else ""),
                }
            )

        if library_query_prefers_previous_candidates(query):
            for turn in reversed(session.turns[-4:]):
                if turn.relation not in {"library_recommendation", "compound_query", "library_citation_ranking"}:
                    continue
                for title, year in re.findall(r"《([^》]{2,220})》(?:（(\d{4})）)?", turn.answer):
                    add_candidate(title=title, year=year)
                    if len(selected) >= limit:
                        break
                if selected:
                    break

        if not selected:
            for item in self._rank_library_papers_for_recommendation(docs=docs, query=query, limit=limit):
                add_candidate(title=item["title"], year=item.get("year", ""), reason=item.get("reason", ""))
                if len(selected) >= limit:
                    break
        return selected[:limit]

    def _lookup_candidate_citation_counts(
        self,
        *,
        candidates: list[dict[str, str]],
        max_web_results: int,
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        web_enabled = bool(self.web_search.is_configured)
        all_evidence: list[EvidenceBlock] = []
        results: list[dict[str, Any]] = []
        per_title_limit = max(2, min(int(max_web_results or 3), 4))
        include_domains = [
            "semanticscholar.org",
            "openalex.org",
            "paperswithcode.com",
            "dblp.org",
            "scholar.google.com",
        ]
        for candidate in candidates:
            title = candidate["title"]
            web_query = f"\"{title}\" citation count citations"
            self._emit_agent_tool_call(
                emit=emit,
                tool="web_citation_lookup",
                arguments={
                    "title": title,
                    "query": web_query,
                    "max_results": per_title_limit,
                    "enabled": web_enabled,
                },
            )
            evidence: list[EvidenceBlock] = []
            if web_enabled:
                direct_evidence = self._semantic_scholar_citation_evidence(title=title)
                if direct_evidence is not None:
                    evidence.append(direct_evidence)
                if not evidence:
                    evidence = self.web_search.search(
                        query=web_query,
                        max_results=per_title_limit,
                        topic="general",
                        include_domains=include_domains,
                    )
                if evidence:
                    emit("web_search", {"count": len(evidence), "items": [item.model_dump() for item in evidence]})
                    all_evidence.extend(evidence)
            extracted = extract_citation_count_from_evidence(title=title, evidence=evidence)
            result = {
                **candidate,
                "citation_count": extracted.get("citation_count"),
                "source_title": extracted.get("source_title", ""),
                "source_url": extracted.get("source_url", ""),
                "doc_id": extracted.get("doc_id", ""),
                "source_snippet": extracted.get("source_snippet", ""),
            }
            results.append(result)
            summary = (
                f"{title}: citations={result['citation_count']}"
                if result["citation_count"] is not None
                else f"{title}: citation count unavailable"
            )
            self._record_agent_observation(
                emit=emit,
                execution_steps=execution_steps,
                tool="web_citation_lookup",
                summary=summary,
                payload={
                    "title": title,
                    "web_evidence_count": len(evidence),
                    "citation_count": result["citation_count"],
                    "source_url": result["source_url"],
                },
            )
        return {"web_enabled": web_enabled, "results": results, "evidence": all_evidence}

    def _semantic_scholar_citation_evidence(self, *, title: str) -> EvidenceBlock | None:
        if type(self.web_search).__name__ != "TavilyWebSearchClient":
            return None
        try:
            response = httpx.get(
                "https://api.semanticscholar.org/graph/v1/paper/search/match",
                params={
                    "query": title,
                    "fields": "title,year,citationCount,url",
                },
                timeout=min(max(float(self.settings.tavily_timeout_seconds), 2.0), 5.0),
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.info("semantic scholar citation lookup failed for %s: %s", title, exc)
            return None
        records = payload.get("data", [])
        if not isinstance(records, list):
            return None
        best_record: dict[str, Any] | None = None
        best_overlap = 0.0
        for record in records:
            if not isinstance(record, dict):
                continue
            record_title = str(record.get("title", "") or "").strip()
            overlap = title_token_overlap(title, record_title)
            if overlap > best_overlap:
                best_overlap = overlap
                best_record = record
        if best_record is None or best_overlap < 0.55:
            return None
        count = parse_citation_count(str(best_record.get("citationCount", "")))
        if count is None:
            return None
        record_title = str(best_record.get("title", "") or title).strip()
        url = str(best_record.get("url", "") or "").strip() or "https://www.semanticscholar.org/search"
        year = str(best_record.get("year", "") or "").strip()
        doc_id = "web::semantic-scholar::" + hashlib.sha1(f"{record_title}\n{url}".encode("utf-8")).hexdigest()[:16]
        snippet = (
            f"Semantic Scholar citationCount: {count:,}. "
            f"Matched paper title: {record_title}."
        )
        return EvidenceBlock(
            doc_id=doc_id,
            paper_id=doc_id,
            title=f"{record_title} | Semantic Scholar",
            file_path=url,
            page=0,
            block_type="web",
            caption=url,
            snippet=snippet,
            score=best_overlap,
            metadata={
                "source": "semantic_scholar",
                "query": title,
                "year": year,
                "citation_count": count,
                "title_overlap": best_overlap,
            },
        )

    @staticmethod
    def _normalize_title_key(title: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(title or "").lower())

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
        if not isinstance(verification, VerificationReport) or verification.status != "clarify":
            return None
        next_attempt = self._next_clarification_attempt(session=session, contract=contract, verification=verification)
        if next_attempt < self.agent_settings.max_clarification_attempts:
            return None

        options = self._clarification_options(contract)
        forced_contract = contract
        if options:
            selected = options[0]
            forced_contract = self._contract_from_selected_clarification_option(
                clean_query=contract.clean_query,
                target=contract.targets[0] if contract.targets else str(selected.get("target", "") or ""),
                selected=selected,
                notes_extra=["clarification_limit_reached", "assumed_most_likely_intent"],
            )
            summary = f"selected={selected.get('meaning') or selected.get('title') or 'first_option'}"
        else:
            notes = list(dict.fromkeys([*contract.notes, "clarification_limit_reached", "best_effort_answer"]))
            forced_contract = contract.model_copy(update={"notes": notes})
            summary = verification.recommended_action or "best_effort_answer"

        self._record_agent_observation(
            emit=emit,
            execution_steps=execution_steps,
            tool="clarification_limit",
            summary=summary,
            payload={
                "max_attempts": self.agent_settings.max_clarification_attempts,
                "attempt": next_attempt,
                "assumption": summary,
            },
        )

        forced_plan = {
            "thought": "Clarification limit reached; proceed with the most likely intent and provide a grounded best-effort answer.",
            "actions": ["search_corpus", "compose"],
            "stop_conditions": ["best_effort_answer"],
        }
        forced_state = self.runtime.run_research_agent_loop(
            contract=forced_contract,
            session=session,
            agent_plan=forced_plan,
            web_enabled=web_enabled,
            explicit_web_search=explicit_web_search,
            max_web_results=max_web_results,
            emit=emit,
            execution_steps=execution_steps,
        )
        forced_verification = forced_state.get("verification")
        if (
            isinstance(forced_verification, VerificationReport)
            and forced_verification.status == "clarify"
            and forced_state.get("claims")
        ):
            forced_state["verification"] = VerificationReport(
                status="pass",
                recommended_action="best_effort_after_clarification_limit",
            )
            forced_state["contract"] = forced_state["contract"].model_copy(
                update={
                    "notes": list(
                        dict.fromkeys(
                            [
                                *forced_state["contract"].notes,
                                "clarification_limit_reached",
                                "best_effort_after_clarification_limit",
                            ]
                        )
                    )
                }
            )
        return forced_state

    def _record_agent_observation(
        self,
        *,
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
        tool: str,
        summary: str,
        payload: dict[str, Any],
    ) -> None:
        canonical_tool = self._canonical_agent_tool(tool)
        event_payload = dict(payload)
        if canonical_tool != tool:
            event_payload.setdefault("raw_tool", tool)
        emit("observation", {"tool": canonical_tool, "summary": summary, "payload": event_payload})
        execution_steps.append({"node": f"agent_tool:{canonical_tool}", "summary": summary})

    def _emit_agent_tool_call(
        self,
        *,
        emit: Callable[[str, dict[str, Any]], None],
        tool: str,
        arguments: dict[str, Any],
    ) -> None:
        canonical_tool = self._canonical_agent_tool(tool)
        event_arguments = dict(arguments)
        if canonical_tool != tool:
            event_arguments.setdefault("raw_tool", tool)
        emit("tool_call", {"tool": canonical_tool, "arguments": event_arguments})

    @staticmethod
    def _canonical_agent_tool(tool: str) -> str:
        return canonical_agent_tool(
            tool=tool,
            aliases=LEGACY_TOOL_NAME_ALIASES,
            canonical_names=all_agent_tool_names(),
        )

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
        emit(
            "agent_step",
            {
                "index": index,
                "action": action,
                "arguments": dict(arguments or {}),
                "message": self._agent_step_message(action=action, contract=contract, state=state),
            },
        )

    @staticmethod
    def _agent_step_message(*, action: str, contract: QueryContract, state: dict[str, Any]) -> str:
        target_text = " / ".join(contract.targets) if contract.targets else "当前问题"
        messages = {
            "read_memory": "读取会话工作记忆，确认上一轮目标、选择和工具结果。",
            "search_corpus": f"从本地论文库检索与 {target_text} 相关的论文和证据块。",
            "bm25_search": f"对 {target_text} 相关内容做关键词检索，优先召回精确术语、公式和标题。",
            "vector_search": f"对 {target_text} 相关内容做语义向量检索，补足改写表达。",
            "hybrid_search": f"对 {target_text} 相关内容做混合检索，融合关键词和语义召回。",
            "rerank": "按当前问题重新排序已收集证据，优先保留最相关片段。",
            "read_pdf_page": "读取本地论文 PDF 索引中的指定页文本、表格或图注块。",
            "grep_corpus": "用精确字符串或正则在本地论文库中查找公式、术语和片段。",
            "query_rewrite": "改写当前问题，生成多路本地检索查询。",
            "summarize": "压缩文本或当前证据，生成面向后续推理的短摘要。",
            "verify_claim": "检查具体 claim 是否被当前或传入证据支持。",
            "compose": "基于当前记忆或证据进入最终整理；研究问题会先完成内部求解和校验。",
            "todo_write": "更新可见任务列表，让多步检索/验证过程可以被前端追踪。",
            "remember": "把可复用的学习或用户偏好持久化，供后续轮次读取。",
            "propose_tool": "记录一个待人工审核的新工具提案，不执行其中的代码。",
            "Task": "派发一个独立子任务，通过同一套工具循环收集结果。",
            "understand_user_intent": f"先确认任务类型：{contract.relation}，目标是 {target_text}。",
            "reflect_previous_answer": "先反思上一轮回答，排除已经被用户否定的解释。",
            "answer_conversation": "调用对话工具处理普通交流，不从主流程直接回答。",
            "get_library_status": "调用论文库状态工具读取当前索引、分类和文章预览。",
            "query_library_metadata": "调用只读库元信息 SQL 工具，按当前问题查询论文标题、作者、年份、分类、标签等索引字段。",
            "get_library_recommendation": "调用库内推荐工具，基于当前论文库挑出值得优先读的论文。",
            "answer_from_memory": "调用通用记忆问答工具，回答用户对上一轮工具结果的追问。",
            "read_conversation_memory": "读取会话工作记忆，继承上一轮工具结果和目标绑定。",
            "synthesize_previous_results": "基于已保留的工具结果做综合，不重新猜测。",
            "recover_previous_recommendation_candidates": "先恢复上一轮推荐候选，避免凭空换一批论文。",
            "web_citation_lookup": "引用数是外部动态指标，逐篇调用 Web citation 检索工具。",
            "rank_by_verified_citation_count": "只用抽取得到的 citation count 做排序，并说明证据边界。",
            "web_search": "本地证据不够或问题需要外部动态信息，补充 Web 检索。",
            "fetch_url": "读取一个已知 HTTPS URL 的正文，并执行 SSRF 安全校验。",
            "ask_human": "当前存在实质歧义，需要用户选择后再继续。",
            "compose_or_ask_human": "证据链检查完成，整理最终回答或交互选项。",
        }
        return messages.get(action, action)

    @staticmethod
    def _state_tool_input(state: dict[str, Any], tool: str) -> dict[str, Any]:
        tool_inputs = state.get("tool_inputs", {})
        if not isinstance(tool_inputs, dict):
            return {}
        payload = tool_inputs.get(tool, {})
        return dict(payload) if isinstance(payload, dict) else {}

    @staticmethod
    def _tool_int_arg(
        payload: dict[str, Any],
        key: str,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        try:
            parsed = int(payload.get(key, default))
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

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
        tool_input = self._state_tool_input(state, "search_corpus")
        paper_limit = self._tool_int_arg(tool_input, "top_k", default=plan.paper_limit, minimum=1, maximum=50)
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
        candidate_papers = self.retriever.search_papers(query=paper_query, contract=contract, limit=plan.paper_limit)
        if excluded_titles:
            candidate_papers = self._filter_candidate_papers_by_excluded_titles(
                candidate_papers,
                excluded_titles=excluded_titles,
            )
        active = session.effective_active_research()
        if not candidate_papers and contract.continuation_mode == "followup" and active.targets:
            fallback_contract = contract.model_copy(update={"targets": list(active.targets)})
            candidate_papers = self.retriever.search_papers(
                query=paper_query_text(fallback_contract),
                contract=fallback_contract,
                limit=plan.paper_limit,
            )
            if excluded_titles:
                candidate_papers = self._filter_candidate_papers_by_excluded_titles(
                    candidate_papers,
                    excluded_titles=excluded_titles,
                )
            state["contract"] = fallback_contract
            contract = fallback_contract
        selected_paper_id = self._selected_clarification_paper_id(contract)
        if selected_paper_id:
            selected = [item for item in candidate_papers if item.paper_id == selected_paper_id]
            if not selected:
                paper = self._candidate_from_paper_id(selected_paper_id)
                selected = [paper] if paper is not None else []
            if selected:
                candidate_papers = selected
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
        selected_paper_id = self._selected_clarification_paper_id(contract)
        if selected_paper_id:
            selected = [item for item in candidate_papers if item.paper_id == selected_paper_id]
            if not selected:
                paper = self._candidate_from_paper_id(selected_paper_id)
                selected = [paper] if paper is not None else []
            if selected:
                candidate_papers = selected
        screened_papers = candidate_papers
        precomputed_evidence: list[EvidenceBlock] | None = None
        goals = research_plan_goals(contract)
        if goals & {"followup_papers", "candidate_relationship", "strict_followup"}:
            screened_papers = self._filter_followup_candidates(contract=contract, candidates=candidate_papers)
        elif "formula" in goals and contract.targets:
            screened_papers = self._prefer_identity_matching_papers(candidates=candidate_papers, targets=contract.targets)
        elif "figure_conclusion" in goals and contract.targets:
            screened_papers = self._prefer_identity_matching_papers(candidates=candidate_papers, targets=contract.targets)
        elif goals & {"entity_type", "role_in_context"}:
            entity_evidence_limit = self._entity_evidence_limit(
                contract=contract,
                plan=plan,
                excluded_titles=excluded_titles,
            )
            precomputed_evidence = self.retriever.search_entity_evidence(
                query=evidence_query_text(contract),
                contract=contract,
                limit=entity_evidence_limit,
            )
            if selected_paper_id:
                precomputed_evidence = [item for item in precomputed_evidence if item.paper_id == selected_paper_id]
            if excluded_titles:
                precomputed_evidence = self._filter_evidence_by_excluded_titles(
                    precomputed_evidence,
                    excluded_titles=excluded_titles,
                )
            grounded_papers = self._ground_entity_papers(
                candidates=candidate_papers,
                evidence=precomputed_evidence,
                limit=plan.paper_limit,
            )
            if grounded_papers:
                screened_papers = grounded_papers
        return screened_papers, precomputed_evidence

    def _agent_search_evidence(
        self,
        *,
        state: dict[str, Any],
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
    ) -> None:
        contract: QueryContract = state["contract"]
        plan: ResearchPlan = state["plan"]
        tool_input = self._state_tool_input(state, "search_corpus")
        evidence_limit = self._tool_int_arg(tool_input, "top_k", default=plan.evidence_limit, minimum=1, maximum=50)
        if evidence_limit != plan.evidence_limit:
            plan = plan.model_copy(update={"evidence_limit": evidence_limit})
        screened_papers: list[CandidatePaper] = state["screened_papers"]
        excluded_titles: set[str] = state["excluded_titles"]
        evidence_query = str(tool_input.get("query", "") or "").strip() or evidence_query_text(contract)
        self._emit_agent_tool_call(
            emit=emit,
            tool="search_corpus",
            arguments={
                "stage": "search_evidence",
                "query": evidence_query,
                "paper_ids": [item.paper_id for item in screened_papers],
                "limit": plan.evidence_limit,
                "modalities": contract.required_modalities,
            },
        )
        precomputed_evidence = state.get("precomputed_evidence")
        if should_use_concept_evidence(contract):
            evidence = self.retriever.search_concept_evidence(
                query=evidence_query,
                contract=contract,
                paper_ids=[item.paper_id for item in screened_papers],
                limit=plan.evidence_limit,
            )
            if not evidence:
                evidence = self.retriever.expand_evidence(
                    paper_ids=[item.paper_id for item in screened_papers],
                    query=evidence_query,
                    contract=contract,
                    limit=plan.evidence_limit,
                )
        else:
            evidence = precomputed_evidence or self.retriever.expand_evidence(
                paper_ids=[item.paper_id for item in screened_papers],
                query=evidence_query,
                contract=contract,
                limit=plan.evidence_limit,
            )
        if excluded_titles:
            evidence = self._filter_evidence_by_excluded_titles(evidence, excluded_titles=excluded_titles)
        selected_paper_id = self._selected_clarification_paper_id(contract)
        if selected_paper_id:
            evidence = [item for item in evidence if item.paper_id == selected_paper_id]
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
        tool_input = self._state_tool_input(state, "web_search")
        web_query = str(tool_input.get("query", "") or "").strip() or web_query_text(contract)
        result_limit = self._tool_int_arg(tool_input, "max_results", default=max_web_results, minimum=1, maximum=20)
        self._emit_agent_tool_call(
            emit=emit,
            tool="web_search",
            arguments={
                "query": web_query,
                "max_results": result_limit,
                "enabled": web_enabled,
            },
        )
        web_evidence = self._collect_web_evidence(
            contract=contract,
            use_web_search=web_enabled,
            max_web_results=result_limit,
            query_override=web_query,
        )
        state["web_evidence"] = web_evidence
        if web_evidence:
            state["evidence"] = merge_evidence(state["evidence"], web_evidence)
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
            selected_option = self._selected_option_from_judge_decision(
                decision=judge_decision,
                options=ambiguity_options,
            )
            auto_resolve = selected_option is not None and self._judge_allows_auto_resolve(judge_decision)
            self._record_agent_observation(
                emit=emit,
                execution_steps=execution_steps,
                tool="resolve_ambiguity" if auto_resolve else "detect_ambiguity",
                summary=self._disambiguation_judge_summary(
                    options=ambiguity_options,
                    judge_decision=judge_decision,
                ),
                payload={
                    "options": ambiguity_options[:4],
                    "judge_decision": judge_decision.model_dump() if judge_decision is not None else {},
                },
            )
            if auto_resolve and selected_option is not None:
                contract = self._contract_with_auto_resolved_ambiguity(
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
                claims = self._run_solvers(
                    contract=contract,
                    plan=plan,
                    papers=screened_papers,
                    evidence=evidence,
                    session=session,
                    use_web_search=explicit_web_search,
                    max_web_results=max_web_results,
                )
                web_evidence: list[EvidenceBlock] = state["web_evidence"]
                if web_evidence and should_add_web_claim(
                    contract=contract,
                    claims=claims,
                    explicit_web=explicit_web_search,
                ):
                    claims.append(self._build_web_research_claim(contract=contract, web_evidence=web_evidence))
                state["claims"] = claims
            else:
                ambiguity_options = self._apply_disambiguation_judge_recommendation(
                    options=ambiguity_options,
                    decision=judge_decision,
                )
                contract = self._contract_with_ambiguity_options(contract=contract, options=ambiguity_options)
                state["contract"] = contract
                state["claims"] = []
                state["verification"] = VerificationReport(
                    status="clarify",
                    missing_fields=self._disambiguation_missing_fields(contract),
                    recommended_action="clarify_ambiguous_entity",
                )
        else:
            claims = self._run_solvers(
                contract=contract,
                plan=plan,
                papers=screened_papers,
                evidence=evidence,
                session=session,
                use_web_search=explicit_web_search,
                max_web_results=max_web_results,
            )
            web_evidence: list[EvidenceBlock] = state["web_evidence"]
            if web_evidence and should_add_web_claim(
                contract=contract,
                claims=claims,
                explicit_web=explicit_web_search,
            ):
                claims.append(self._build_web_research_claim(contract=contract, web_evidence=web_evidence))
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
        goals = research_plan_goals(contract)
        if verification.status == "retry" and contract.targets and (
            goals & {"definition", "mechanism", "examples", "figure_conclusion", "answer", "general_answer"}
        ):
            verification = VerificationReport(
                status="clarify",
                missing_fields=["relevant_evidence"],
                recommended_action="clarify_target",
            )
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
        self._emit_agent_tool_call(
            emit=emit,
            tool="retry_research",
            arguments={
                "reason": state["verification"].recommended_action if state.get("verification") else "",
                "paper_limit": max(plan.paper_limit + 4, 10),
                "evidence_limit": max(plan.evidence_limit + 12, int(plan.evidence_limit * 1.5)),
            },
        )
        broader_candidates = self.retriever.search_papers(
            query=paper_query_text(contract),
            contract=contract,
            limit=max(plan.paper_limit + 4, 10),
        )
        if excluded_titles:
            broader_candidates = self._filter_candidate_papers_by_excluded_titles(
                broader_candidates,
                excluded_titles=excluded_titles,
            )
        selected_paper_id = self._selected_clarification_paper_id(contract)
        if selected_paper_id:
            selected = [item for item in broader_candidates if item.paper_id == selected_paper_id]
            if not selected:
                paper = self._candidate_from_paper_id(selected_paper_id)
                selected = [paper] if paper is not None else []
            if selected:
                broader_candidates = selected
        goals = research_plan_goals(contract)
        if should_use_concept_evidence(contract):
            broader_evidence = self.retriever.search_concept_evidence(
                query=evidence_query_text(contract),
                contract=contract,
                paper_ids=[item.paper_id for item in broader_candidates],
                limit=max(plan.evidence_limit + 12, int(plan.evidence_limit * 1.5)),
            )
        elif goals & {"entity_type", "role_in_context"}:
            broader_evidence = self.retriever.search_entity_evidence(
                query=evidence_query_text(contract),
                contract=contract,
                limit=max(
                    self._entity_evidence_limit(contract=contract, plan=plan, excluded_titles=excluded_titles),
                    plan.evidence_limit + 12,
                    int(plan.evidence_limit * 1.5),
                ),
            )
            if excluded_titles:
                broader_evidence = self._filter_evidence_by_excluded_titles(
                    broader_evidence,
                    excluded_titles=excluded_titles,
                )
            broader_candidates = self._ground_entity_papers(
                candidates=broader_candidates,
                evidence=broader_evidence,
                limit=max(plan.paper_limit + 4, 10),
            )
        else:
            broader_evidence = self.retriever.expand_evidence(
                paper_ids=[item.paper_id for item in broader_candidates],
                query=evidence_query_text(contract),
                contract=contract,
                limit=max(plan.evidence_limit + 12, int(plan.evidence_limit * 1.5)),
            )
        if excluded_titles:
            broader_evidence = self._filter_evidence_by_excluded_titles(
                broader_evidence,
                excluded_titles=excluded_titles,
            )
        if selected_paper_id:
            broader_evidence = [item for item in broader_evidence if item.paper_id == selected_paper_id]
        retry_claims = self._run_solvers(
            contract=contract,
            plan=plan.model_copy(update={"retry_budget": 0}),
            papers=broader_candidates,
            evidence=broader_evidence,
            session=session,
            use_web_search=explicit_web_search,
            max_web_results=max_web_results,
        )
        retry_report = self._verify_claims(
            contract=contract,
            plan=plan.model_copy(update={"retry_budget": 0}),
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
        clarified_contract = self._contract_from_pending_clarification(
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
        refined_contract = self._inherit_followup_relationship_contract(contract=refined_contract, session=session)
        refined_contract = self._normalize_followup_direction_contract(contract=refined_contract)
        return self._apply_conversation_memory_to_contract(contract=refined_contract, session=session)

    @staticmethod
    def _library_status_contract(clean_query: str) -> QueryContract:
        return QueryContract(
            clean_query=clean_query,
            interaction_mode="conversation",
            relation="library_status",
            targets=[],
            requested_fields=[],
            required_modalities=[],
            answer_shape="bullets",
            precision_requirement="exact",
            continuation_mode="fresh",
            notes=["self_knowledge", "dynamic_library_stats"],
        )

    @staticmethod
    def _library_recommendation_contract(clean_query: str) -> QueryContract:
        return QueryContract(
            clean_query=clean_query,
            interaction_mode="conversation",
            relation="library_recommendation",
            targets=[],
            requested_fields=[],
            required_modalities=[],
            answer_shape="bullets",
            precision_requirement="normal",
            continuation_mode="fresh",
            notes=["self_knowledge", "dynamic_library_recommendation"],
        )

    def _normalize_conversation_tool_contract(
        self,
        *,
        contract: QueryContract,
        clean_query: str,
        session: SessionContext,
    ) -> QueryContract:
        if is_citation_ranking_query(clean_query) and citation_ranking_has_library_context(
            clean_query=clean_query,
            session=session,
        ):
            return QueryContract(
                clean_query=clean_query,
                interaction_mode="conversation",
                relation="library_citation_ranking",
                targets=[],
                requested_fields=["citation_count_ranking"],
                required_modalities=[],
                answer_shape="table",
                precision_requirement="normal",
                continuation_mode="followup" if session.turns else "fresh",
                allow_web_search=True,
                notes=["agent_tool", "external_metric", "citation_count_requires_web"],
            )
        active = session.effective_active_research()
        active_formula = active.relation == "formula_lookup" or "formula" in {str(field) for field in active.requested_fields}
        if active_formula and active.targets and looks_like_formula_answer_correction(clean_query):
            return self._formula_answer_correction_contract(contract=contract, session=session)
        if self._is_formula_interpretation_followup(clean_query=clean_query, session=session):
            return QueryContract(
                clean_query=clean_query,
                interaction_mode="conversation",
                relation="memory_followup",
                targets=list(active.targets),
                requested_fields=["formula_interpretation"],
                required_modalities=[],
                answer_shape="narrative",
                precision_requirement="normal",
                continuation_mode="followup",
                notes=["agent_tool", "formula_interpretation_followup"],
            )
        if self._is_language_preference_followup(clean_query=clean_query, session=session):
            active = session.effective_active_research()
            return QueryContract(
                clean_query=clean_query,
                interaction_mode="conversation",
                relation="memory_followup",
                targets=list(active.targets),
                requested_fields=["answer_language_preference"],
                required_modalities=[],
                answer_shape="narrative",
                precision_requirement="normal",
                continuation_mode="followup",
                notes=["agent_tool", "answer_language_preference"],
            )
        if is_memory_synthesis_query(clean_query) and (
            len(self._active_memory_bindings(session)) >= 2
            or len(list(dict((session.working_memory or {}).get("last_compound_query", {}) or {}).get("subtasks", []) or [])) >= 2
        ):
            targets = list(dict.fromkeys(session.effective_active_research().targets))
            return QueryContract(
                clean_query=clean_query,
                interaction_mode="conversation",
                relation="memory_synthesis",
                targets=targets,
                requested_fields=["comparison", "synthesis"],
                required_modalities=[],
                answer_shape="table" if self._is_comparison_query(clean_query) else "narrative",
                precision_requirement="high",
                continuation_mode="followup",
                notes=["agent_tool", "conversation_memory_synthesis"],
            )
        if is_scoped_library_recommendation_query(clean_query) and not is_library_count_query(clean_query):
            return self._library_recommendation_contract(clean_query).model_copy(
                update={"notes": list(dict.fromkeys([*contract.notes, "agent_tool", "dynamic_library_recommendation"]))}
            )
        if is_library_status_query(clean_query):
            return self._library_status_contract(clean_query).model_copy(
                update={"notes": list(dict.fromkeys([*contract.notes, "agent_tool", "dynamic_library_stats"]))}
            )
        if contract.relation in {
            "greeting",
            "self_identity",
            "capability",
            "library_status",
            "library_recommendation",
            "memory_followup",
            "clarify_user_intent",
            "correction_without_context",
            "memory_synthesis",
            "library_citation_ranking",
        }:
            return contract.model_copy(
                update={
                    "interaction_mode": "conversation",
                    "required_modalities": [],
                    "notes": list(dict.fromkeys([*contract.notes, "agent_tool"])),
                }
            )
        return contract

    def _normalize_followup_direction_contract(self, *, contract: QueryContract) -> QueryContract:
        clean_query = " ".join(str(contract.clean_query or "").split())
        lowered = clean_query.lower()
        if not clean_query or ("后续" not in clean_query and "follow" not in lowered and "successor" not in lowered):
            return contract
        direction_match = re.search(
            r"^(?P<candidate>.+?)\s*(?:真是|是否是|是不是|是|算是|属于|is|are)\s*(?P<seed>.+?)\s*的?\s*(?:严格)?\s*(?:后续|扩展|继承|follow[- ]?up|successor)",
            clean_query,
            flags=re.IGNORECASE,
        )
        if direction_match is None:
            return contract
        candidate_title = " ".join(str(direction_match.group("candidate") or "").strip(" ，,。？?").split())
        candidate_title = re.sub(r"(?:真|真的|是否|是不是)\s*$", "", candidate_title).strip()
        seed_text = " ".join(str(direction_match.group("seed") or "").strip(" ，,。？?").split())
        seed_targets = self._normalize_contract_targets(targets=[seed_text], requested_fields=contract.requested_fields)
        if not seed_targets:
            return contract
        notes = [note for note in contract.notes if not str(note).startswith("candidate_title=")]
        if candidate_title:
            notes.append(f"candidate_title={candidate_title}")
        notes.append("followup_direction_resolved")
        requested_fields = list(dict.fromkeys([*contract.requested_fields, "candidate_relationship", "evidence"]))
        required_modalities = list(dict.fromkeys([*contract.required_modalities, "paper_card", "page_text"]))
        answer_shape = contract.answer_shape if contract.answer_shape in {"bullets", "table"} else "bullets"
        return contract.model_copy(
            update={
                "interaction_mode": "research",
                "relation": "followup_research",
                "targets": seed_targets,
                "requested_fields": requested_fields,
                "required_modalities": required_modalities,
                "answer_shape": answer_shape,
                "precision_requirement": "high",
                "notes": list(dict.fromkeys(notes)),
            }
        )

    def _inherit_followup_relationship_contract(self, *, contract: QueryContract, session: SessionContext) -> QueryContract:
        memory = dict(session.working_memory or {})
        relationship = dict(memory.get("last_followup_relationship", {}) or {})
        if not relationship:
            return contract
        clean_query = str(contract.clean_query or "")
        normalized_query = normalize_lookup_text(clean_query)
        if not followup_relationship_recheck_requested(clean_query, normalized_query):
            return contract
        if not contract_allows_active_context_override(contract) and contract.relation != "followup_research":
            return contract
        goals = research_plan_goals(contract)
        relation_like = bool(goals & {"followup_papers", "candidate_relationship", "summary", "results", "answer", "general_answer"})
        if not relation_like and contract.continuation_mode != "followup":
            return contract
        seed_target = str(relationship.get("seed_target", "") or "").strip()
        candidate_title = str(relationship.get("candidate_title", "") or "").strip()
        if not seed_target or not candidate_title:
            return contract
        targets = self._normalize_contract_targets(targets=[seed_target], requested_fields=contract.requested_fields) or [seed_target]
        notes = [note for note in contract.notes if not str(note).startswith("candidate_title=")]
        notes.extend(
            [
                f"candidate_title={candidate_title}",
                "inherited_followup_relationship",
                "strict_followup_validation",
            ]
        )
        return contract.model_copy(
            update={
                "clean_query": f"{candidate_title} 是否是 {seed_target} 的严格后续工作？",
                "interaction_mode": "research",
                "relation": "followup_research",
                "targets": targets,
                "requested_fields": ["candidate_relationship", "strict_followup", "evidence"],
                "required_modalities": ["paper_card", "page_text"],
                "answer_shape": "bullets",
                "precision_requirement": "high",
                "continuation_mode": "followup",
                "notes": list(dict.fromkeys(notes)),
            }
        )

    def _session_conversation_context(self, session: SessionContext, *, max_chars: int = 24000) -> dict[str, Any]:
        """Return the retained conversation as the LLM-facing working memory."""

        def turn_payload(turn: SessionTurn, *, answer_limit: int) -> dict[str, Any]:
            return {
                "user_query": turn.query,
                "assistant_answer": self._truncate_context_text(turn.answer, limit=answer_limit),
                "query_contract": {
                    "relation": turn.relation,
                    "interaction_mode": turn.interaction_mode,
                    "clean_query": turn.clean_query,
                    "targets": turn.targets,
                    "answer_slots": turn.answer_slots,
                    "requested_fields": turn.requested_fields,
                    "required_modalities": turn.required_modalities,
                    "answer_shape": turn.answer_shape,
                    "precision_requirement": turn.precision_requirement,
                },
                "citation_titles": turn.titles,
            }

        answer_limit = 1800
        payload = {
            "summary_of_compressed_older_turns": session.summary,
            "active_research_context": session.active_research_context_payload(),
            "pending_clarification": {
                "type": session.pending_clarification_type,
                "target": session.pending_clarification_target,
                "options": session.pending_clarification_options,
            },
            "working_memory": session.working_memory,
            "persistent_learnings": self._persistent_learnings_context(),
            "turns": [turn_payload(turn, answer_limit=answer_limit) for turn in session.turns],
        }
        serialized = json.dumps(payload, ensure_ascii=False)
        if len(serialized) <= max_chars:
            return payload

        compact_turns: list[dict[str, Any]] = []
        for index, turn in enumerate(session.turns):
            limit = 900 if index >= max(0, len(session.turns) - 4) else 280
            compact_turns.append(turn_payload(turn, answer_limit=limit))
        payload["turns"] = compact_turns
        payload["context_compression_note"] = "Older answers were shortened because the raw conversation context was near the prompt budget."
        return payload

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

        messages: list[dict[str, str]] = []
        if session.summary:
            messages.append(
                {
                    "role": "human",
                    "content": "以下是更早对话的压缩摘要，请用于解析后续指代，不要把摘要当成新问题：\n" + session.summary,
                }
            )
        for turn in session.turns[-max_turns:]:
            user_query = str(turn.query or "").strip()
            if user_query:
                messages.append({"role": "user", "content": user_query})
            answer = self._truncate_context_text(turn.answer, limit=answer_limit)
            metadata = {
                "relation": turn.relation,
                "interaction_mode": turn.interaction_mode,
                "targets": turn.targets,
                "answer_slots": turn.answer_slots,
                "requested_fields": turn.requested_fields,
                "citation_titles": turn.titles,
            }
            messages.append(
                {
                    "role": "assistant",
                    "content": answer + "\n\n[上一轮工具上下文]\n" + json.dumps(metadata, ensure_ascii=False),
                }
            )
        return messages

    @staticmethod
    def _truncate_context_text(text: str, *, limit: int) -> str:
        compact = str(text or "").strip()
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 3)].rstrip() + "..."

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
        memory = dict(session.working_memory or {})
        bindings = dict(memory.get("target_bindings", {}) or {})
        paper_by_id = {paper.paper_id: paper for paper in papers}
        citation_by_paper_id = {citation.paper_id: citation for citation in citations if citation.paper_id}
        fallback_paper = papers[0] if papers else None
        for target in contract.targets:
            target = str(target or "").strip()
            if not target:
                continue
            key = normalize_lookup_text(target)
            if not key:
                continue
            paper: CandidatePaper | None = None
            evidence_ids: list[str] = []
            for claim in claims:
                claim_target = str(claim.entity or target).strip()
                if claim_target and normalize_lookup_text(claim_target) not in {key, ""}:
                    continue
                if claim.paper_ids:
                    paper = paper_by_id.get(claim.paper_ids[0]) or self._candidate_from_paper_id(claim.paper_ids[0])
                evidence_ids = list(claim.evidence_ids[:4])
                if paper is not None:
                    break
            if paper is None:
                citation = citation_by_paper_id.get(target) or (citations[0] if citations else None)
                if citation is not None and citation.paper_id:
                    paper = paper_by_id.get(citation.paper_id) or self._candidate_from_paper_id(citation.paper_id)
            if paper is None:
                paper = fallback_paper
            if paper is None:
                continue
            support_titles = list(dict.fromkeys([paper.title, *[item.title for item in citations if item.title]]))
            bindings[key] = {
                "target": target,
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "relation": contract.relation,
                "requested_fields": list(contract.requested_fields),
                "required_modalities": list(contract.required_modalities),
                "clean_query": contract.clean_query,
                "answer_preview": self._truncate_context_text(answer, limit=900),
                "evidence_ids": evidence_ids or [item.doc_id for item in evidence if item.paper_id == paper.paper_id][:4],
                "support_titles": support_titles[:4],
            }
        memory["target_bindings"] = bindings
        memory["last_successful_research"] = {
            "relation": contract.relation,
            "targets": list(contract.targets),
            "requested_fields": list(contract.requested_fields),
            "titles": [paper.title for paper in papers[:4]],
            "clean_query": contract.clean_query,
            "answer_preview": self._truncate_context_text(answer, limit=1200),
        }
        if any(claim.claim_type == "followup_research" for claim in claims):
            relationship = self._followup_relationship_memory(contract=contract, claims=claims, papers=papers, answer=answer)
            if relationship:
                memory["last_followup_relationship"] = relationship
        session.working_memory = memory

    def _followup_relationship_memory(
        self,
        *,
        contract: QueryContract,
        claims: list[Claim],
        papers: list[CandidatePaper],
        answer: str,
    ) -> dict[str, Any]:
        claim = next((item for item in claims if item.claim_type == "followup_research"), None)
        if claim is None:
            return {}
        structured = dict(claim.structured_data or {})
        candidate_title = str(structured.get("selected_candidate_title", "") or "").strip()
        rows = [dict(item or {}) for item in list(structured.get("followup_titles", []) or []) if isinstance(item, dict)]
        selected_row: dict[str, Any] = {}
        if candidate_title:
            selected_key = normalize_lookup_text(candidate_title)
            selected_row = next(
                (
                    row
                    for row in rows
                    if selected_key
                    and (
                        selected_key in normalize_lookup_text(str(row.get("title", "")))
                        or normalize_lookup_text(str(row.get("title", ""))) in selected_key
                    )
                ),
                rows[0] if rows else {},
            )
        elif rows:
            selected_row = rows[0]
            candidate_title = str(selected_row.get("title", "") or "").strip()
        if not candidate_title:
            return {}
        seed = next((dict(item or {}) for item in list(structured.get("seed_papers", []) or []) if isinstance(item, dict)), {})
        seed_target = contract.targets[0] if contract.targets else str(claim.entity or "")
        return {
            "seed_target": seed_target,
            "seed_title": str(seed.get("title", "") or "").strip(),
            "seed_paper_id": str(seed.get("paper_id", "") or "").strip(),
            "candidate_title": candidate_title,
            "candidate_paper_id": str(selected_row.get("paper_id", "") or "").strip(),
            "relationship_strength": str(selected_row.get("relationship_strength", "") or "").strip(),
            "relation_type": str(selected_row.get("relation_type", "") or "").strip(),
            "strict_followup": bool(selected_row.get("strict_followup", False)),
            "clean_query": contract.clean_query,
            "answer_preview": self._truncate_context_text(answer, limit=900),
        }

    def _remember_compound_outcome(
        self,
        *,
        session: SessionContext,
        clean_query: str,
        subtask_results: list[dict[str, Any]],
    ) -> None:
        subtasks: list[dict[str, Any]] = []
        for result in subtask_results:
            contract = result.get("contract")
            if not isinstance(contract, QueryContract):
                continue
            claims = [item for item in list(result.get("claims", []) or []) if isinstance(item, Claim)]
            evidence = [item for item in list(result.get("evidence", []) or []) if isinstance(item, EvidenceBlock)]
            citations = [item for item in list(result.get("citations", []) or []) if isinstance(item, AssistantCitation)]
            paper_ids = list(dict.fromkeys(pid for claim in claims for pid in claim.paper_ids))
            papers = [paper for paper_id in paper_ids if (paper := self._candidate_from_paper_id(paper_id)) is not None]
            if not papers:
                papers = [paper for citation in citations if (paper := self._candidate_from_paper_id(citation.paper_id)) is not None]
            self._remember_research_outcome(
                session=session,
                contract=contract,
                answer=str(result.get("answer", "")),
                claims=claims,
                papers=papers,
                evidence=evidence,
                citations=citations,
            )
            subtasks.append(
                {
                    "relation": contract.relation,
                    "targets": list(contract.targets),
                    "requested_fields": list(contract.requested_fields),
                    "clean_query": contract.clean_query,
                    "answer_preview": self._truncate_context_text(str(result.get("answer", "")), limit=900),
                    "citation_titles": [citation.title for citation in citations[:4]],
                }
            )
        memory = dict(session.working_memory or {})
        memory["last_compound_query"] = {
            "query": clean_query,
            "subtasks": subtasks,
        }
        session.working_memory = memory

    def _remember_conversation_tool_result(
        self,
        *,
        session: SessionContext,
        contract: QueryContract,
        tool: str,
        query: str,
        answer: str,
        artifact: dict[str, Any] | None = None,
    ) -> None:
        memory = dict(session.working_memory or {})
        results = [item for item in list(memory.get("tool_results", []) or []) if isinstance(item, dict)]
        record = {
            "tool": tool,
            "query": query,
            "relation": contract.relation,
            "targets": list(contract.targets),
            "requested_fields": list(contract.requested_fields),
            "answer_shape": contract.answer_shape,
            "answer_preview": self._truncate_context_text(answer, limit=1800),
        }
        if isinstance(artifact, dict) and artifact:
            record["artifact"] = artifact
            if isinstance(artifact.get("items"), list):
                list_artifact = dict(artifact)
                list_artifact.setdefault("query", query)
                list_artifact.setdefault("tool", tool)
                memory["last_displayed_list"] = list_artifact
            if tool == "query_library_metadata":
                memory["last_library_metadata_result"] = artifact
        results.append(record)
        memory["tool_results"] = results[-12:]
        memory["last_tool_result"] = record
        session.working_memory = memory

    def _conversation_tool_result_artifact(self, *, tool: str, result: dict[str, Any]) -> dict[str, Any]:
        if tool != "query_library_metadata" or not isinstance(result, dict):
            return {}
        rows = [dict(item) for item in list(result.get("rows", []) or []) if isinstance(item, dict)]
        items: list[dict[str, Any]] = []
        for index, row in enumerate(rows[:80], start=1):
            compact_row: dict[str, Any] = {}
            for key, value in row.items():
                if value is None or isinstance(value, (int, float)):
                    compact_row[str(key)] = value
                    continue
                compact_row[str(key)] = self._truncate_context_text(str(value), limit=900)
            item = {
                "ordinal": index,
                "row": compact_row,
            }
            for key in ["paper_id", "title", "year", "year_int", "authors", "author"]:
                if key in compact_row:
                    item[key] = compact_row[key]
            items.append(item)
        return {
            "type": "tabular_sql_result",
            "tool": tool,
            "sql": self._truncate_context_text(str(result.get("sql", "") or ""), limit=1200),
            "columns": [str(item) for item in list(result.get("columns", []) or [])],
            "row_count": int(result.get("row_count", len(rows)) or 0),
            "truncated": bool(result.get("truncated", False)),
            "items": items,
        }

    def _target_binding_from_memory(self, *, session: SessionContext, target: str) -> dict[str, Any] | None:
        key = normalize_lookup_text(target)
        if not key:
            return None
        bindings = dict((session.working_memory or {}).get("target_bindings", {}) or {})
        binding = bindings.get(key)
        return dict(binding) if isinstance(binding, dict) else None

    def _apply_conversation_memory_to_contract(self, *, contract: QueryContract, session: SessionContext) -> QueryContract:
        if contract.interaction_mode != "research" or not contract.targets:
            return contract
        target_bindings = {
            target: binding
            for target in contract.targets
            if (binding := self._target_binding_from_memory(session=session, target=target))
        }
        topic_state = contract_topic_state(contract)
        goals = research_plan_goals(contract)
        if contract.relation == "origin_lookup" or "origin" in contract_answer_slots(contract) or goals & {"paper_title", "year"}:
            return contract
        allow_explicit_target_binding = bool(target_bindings) and topic_state != "switch"
        if "formula" in goals and topic_state != "continue":
            allow_explicit_target_binding = False
        if (
            not contract_allows_active_context_override(contract)
            and not allow_explicit_target_binding
        ):
            return contract
        if "exclude_previous_focus" in contract.notes or is_negative_correction_query(contract.clean_query):
            return contract
        if self._selected_clarification_paper_id(contract):
            return contract
        notes = list(contract.notes)
        for target in contract.targets:
            binding = target_bindings.get(target)
            if not binding:
                continue
            paper_id = str(binding.get("paper_id", "") or "").strip()
            title = str(binding.get("title", "") or "").strip()
            if not paper_id:
                continue
            notes = list(dict.fromkeys([*notes, "resolved_from_conversation_memory", f"selected_paper_id={paper_id}"]))
            if title:
                notes.append("memory_title=" + title)
            return contract.model_copy(update={"continuation_mode": "followup", "notes": notes})
        return contract

    def _formula_answer_correction_contract(self, *, contract: QueryContract, session: SessionContext) -> QueryContract:
        active = session.effective_active_research()
        title = active.titles[0] if active.titles else ""
        paper = self._paper_from_query_hint(title) if title else None
        notes = list(
            dict.fromkeys(
                [
                    *contract.notes,
                    "formula_answer_correction",
                    "prefer_scalar_objective",
                    "answer_slot=formula",
                ]
            )
        )
        if paper is not None:
            notes = list(dict.fromkeys([*notes, f"selected_paper_id={paper.paper_id}", "memory_title=" + paper.title]))
        target = active.targets[0] if active.targets else (contract.targets[0] if contract.targets else "当前目标")
        scope = f"限定在论文《{paper.title}》中" if paper is not None else "沿用上一轮论文上下文"
        return QueryContract(
            clean_query=f"{target} 的公式是什么？{scope}重新查找目标函数或损失函数；上一条候选公式可能是梯度/推导式，不要优先返回梯度公式。",
            interaction_mode="research",
            relation="formula_lookup",
            targets=list(active.targets or contract.targets or [target]),
            requested_fields=["formula", "variable_explanation", "source"],
            required_modalities=["page_text", "table"],
            answer_shape="bullets",
            precision_requirement="exact",
            continuation_mode="followup",
            allow_web_search=contract.allow_web_search,
            notes=notes,
        )

    def _resolve_formula_answer_correction_contract(self, *, contract: QueryContract, session: SessionContext) -> QueryContract:
        active = session.effective_active_research()
        active_formula = active.relation == "formula_lookup" or "formula" in {str(field) for field in active.requested_fields}
        if not active_formula or not active.targets:
            return contract
        if not looks_like_formula_answer_correction(contract.clean_query):
            return contract
        return self._formula_answer_correction_contract(contract=contract, session=session)

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
        target = self._formula_followup_target(contract=contract, session=session, paper=paper)
        if not target:
            return contract
        notes = list(contract.notes)
        notes = list(
            dict.fromkeys(
                [
                    *notes,
                    "formula_location_followup",
                    "resolved_from_user_paper_hint",
                    f"selected_paper_id={paper.paper_id}",
                    "memory_title=" + paper.title,
                    "answer_slot=formula",
                ]
            )
        )
        return QueryContract(
            clean_query=f"{target} 的公式是什么？限定在论文《{paper.title}》中查找。",
            interaction_mode="research",
            relation="formula_lookup",
            targets=[target],
            requested_fields=["formula", "variable_explanation", "source"],
            required_modalities=["page_text", "table"],
            answer_shape="bullets",
            precision_requirement="exact",
            continuation_mode="followup",
            allow_web_search=contract.allow_web_search,
            notes=notes,
        )

    def _resolve_formula_contextual_paper_contract(self, *, contract: QueryContract, session: SessionContext) -> QueryContract:
        goals = research_plan_goals(contract)
        if contract.interaction_mode != "research" or "formula" not in goals or not contract.targets:
            return contract
        if self._selected_clarification_paper_id(contract) or "exclude_previous_focus" in contract.notes:
            return contract
        active = session.effective_active_research()
        context_text = " ".join([*active.titles, *active.targets]).strip()
        if not context_text:
            return contract
        paper = self._paper_from_query_hint(context_text)
        if paper is None:
            return contract
        if not self._formula_query_allows_active_paper_context(contract=contract, session=session, paper=paper):
            return contract
        target = str(contract.targets[0] or "").strip()
        if not target or not self._paper_context_supports_formula_target(paper=paper, target=target):
            return contract
        notes = list(
            dict.fromkeys(
                [
                    *contract.notes,
                    "formula_contextual_paper_binding",
                    f"selected_paper_id={paper.paper_id}",
                    "memory_title=" + paper.title,
                ]
            )
        )
        return contract.model_copy(
            update={
                "clean_query": f"{target} 的公式是什么？限定在论文《{paper.title}》中查找。",
                "relation": "formula_lookup",
                "requested_fields": ["formula", "variable_explanation", "source"],
                "required_modalities": ["page_text", "table"],
                "answer_shape": "bullets",
                "precision_requirement": "exact",
                "continuation_mode": "followup",
                "notes": notes,
            }
        )

    def _resolve_paper_scope_correction_contract(self, *, contract: QueryContract, session: SessionContext) -> QueryContract:
        if contract.interaction_mode != "research" or self._selected_clarification_paper_id(contract):
            return contract
        active = session.effective_active_research()
        if not active.has_content() or not active.targets:
            return contract
        if not looks_like_paper_scope_correction(contract.clean_query):
            return contract
        paper = self._paper_from_query_hint(contract.clean_query)
        if paper is None:
            return contract
        inherited_query = active.clean_query or contract.clean_query
        notes = self._active_paper_reference_notes(
            notes=contract.notes,
            paper=paper,
            marker="paper_scope_correction",
        )
        rewritten = f"限定在论文《{paper.title}》中回答：{inherited_query}"
        scoped = QueryContract(
            clean_query=rewritten,
            interaction_mode="research",
            relation=active.relation or contract.relation,
            targets=list(active.targets),
            answer_slots=list(contract.answer_slots),
            requested_fields=list(active.requested_fields or contract.requested_fields or ["answer"]),
            required_modalities=list(active.required_modalities or contract.required_modalities or ["page_text"]),
            answer_shape=active.answer_shape or contract.answer_shape,
            precision_requirement=active.precision_requirement or contract.precision_requirement,
            continuation_mode="followup",
            allow_web_search=contract.allow_web_search,
            notes=notes,
        )
        return self._promote_contextual_metric_contract(scoped)

    def _resolve_contextual_active_paper_contract(self, *, contract: QueryContract, session: SessionContext) -> QueryContract:
        if contract.interaction_mode != "research" or self._selected_clarification_paper_id(contract):
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
        notes = self._active_paper_reference_notes(
            notes=contract.notes,
            paper=paper,
            marker="active_paper_reference",
        )
        clean_query = contract.clean_query
        if self._normalize_entity_key(paper.title) not in self._normalize_entity_key(clean_query):
            clean_query = f"限定在论文《{paper.title}》中回答：{clean_query}"
        scoped = contract.model_copy(
            update={
                "clean_query": clean_query,
                "continuation_mode": "followup",
                "notes": notes,
            }
        )
        return self._promote_contextual_metric_contract(scoped)

    @staticmethod
    def _active_paper_reference_notes(*, notes: list[str], paper: CandidatePaper, marker: str) -> list[str]:
        return list(
            dict.fromkeys(
                [
                    *notes,
                    marker,
                    "resolved_from_conversation_memory",
                    f"selected_paper_id={paper.paper_id}",
                    "memory_title=" + paper.title,
                ]
            )
        )

    def _promote_contextual_metric_contract(self, contract: QueryContract) -> QueryContract:
        if contract.relation == "metric_value_lookup":
            return contract
        if not looks_like_contextual_metric_query(
            contract.clean_query,
            targets=list(contract.targets),
            is_short_acronym=is_short_acronym,
        ):
            return contract
        requested_fields = list(dict.fromkeys([*contract.requested_fields, "metric_value", "setting", "evidence"]))
        required_modalities = list(dict.fromkeys([*contract.required_modalities, "table", "caption", "page_text"]))
        answer_slots = list(dict.fromkeys([*contract.answer_slots, "metric_value"]))
        notes = list(dict.fromkeys([*contract.notes, "contextual_metric_query", "answer_slot=metric_value"]))
        return contract.model_copy(
            update={
                "relation": "metric_value_lookup",
                "answer_slots": answer_slots,
                "requested_fields": requested_fields,
                "required_modalities": required_modalities,
                "answer_shape": "narrative",
                "precision_requirement": "exact",
                "notes": notes,
            }
        )

    def _formula_query_allows_active_paper_context(
        self,
        *,
        contract: QueryContract,
        session: SessionContext,
        paper: CandidatePaper,
    ) -> bool:
        active_names = [*session.effective_active_research().targets, *self._paper_hint_names(paper)]
        return formula_query_allows_active_paper_context(
            contract.clean_query,
            active_names=active_names,
            normalize_entity_key=self._normalize_entity_key,
        )

    def _paper_context_supports_formula_target(self, *, paper: CandidatePaper, target: str) -> bool:
        target = str(target or "").strip()
        if not target:
            return False
        for doc in self.retriever.block_documents_for_paper(paper.paper_id, limit=256):
            text = str(doc.page_content or "")
            if not matches_target(text, target):
                continue
            meta = dict(doc.metadata or {})
            lowered = text.lower()
            if int(meta.get("formula_hint", 0) or 0):
                return True
            if any(token in lowered for token in ["objective", "loss", "formula", "log σ", "log sigma", "lpba", "l pba"]):
                return True
        return False

    def _formula_followup_target(
        self,
        *,
        contract: QueryContract,
        session: SessionContext,
        paper: CandidatePaper,
    ) -> str:
        paper_names = self._paper_hint_names(paper)
        paper_name_keys = {self._normalize_entity_key(name) for name in paper_names if name}
        active = session.effective_active_research()
        active_keys = {normalize_lookup_text(item) for item in active.targets}
        for target in contract.targets:
            candidate = str(target or "").strip()
            if not candidate:
                continue
            if self._normalize_entity_key(candidate) in paper_name_keys:
                continue
            if is_short_acronym(candidate) or normalize_lookup_text(candidate) in active_keys:
                return candidate
        return str(active.targets[0] or "").strip()

    def _paper_from_query_hint(self, query: str) -> CandidatePaper | None:
        query_text = str(query or "").strip()
        if not query_text:
            return None
        query_key = self._normalize_entity_key(query_text)
        query_words = normalize_lookup_text(query_text)
        query_hints = [
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", query_text)
            if any(ch.isupper() for ch in token[1:]) or any(ch.isdigit() for ch in token)
        ]
        scored: list[tuple[int, CandidatePaper]] = []
        for doc in self.retriever.paper_documents():
            meta = dict(doc.metadata or {})
            paper_id = str(meta.get("paper_id", "") or "").strip()
            if not paper_id:
                continue
            paper = self._candidate_from_paper_id(paper_id)
            if paper is None:
                continue
            best = 0
            for name in self._paper_hint_names(paper):
                name = str(name or "").strip()
                if not name:
                    continue
                name_key = self._normalize_entity_key(name)
                if len(name_key) < 4:
                    continue
                if name_key in query_key:
                    best = max(best, min(200, len(name_key)))
                    continue
                if matches_target(query_words, name.lower()):
                    best = max(best, min(160, len(name_key)))
            paper_context = "\n".join(
                [
                    paper.title,
                    str(paper.metadata.get("aliases", "")),
                    str(paper.metadata.get("abstract_note", "")),
                    str(paper.metadata.get("generated_summary", "")),
                    str(paper.metadata.get("paper_card_text", "")),
                    str(doc.page_content or ""),
                ]
            )
            for hint in query_hints:
                if matches_target(paper_context, hint):
                    best = max(best, 80 + min(40, len(hint)))
            if best:
                scored.append((best, paper))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], -len(item[1].title), item[1].title))
        return scored[0][1]

    @staticmethod
    def _paper_hint_names(paper: CandidatePaper) -> list[str]:
        names: list[str] = []
        title = str(paper.title or "").strip()
        aliases = [alias.strip() for alias in str(paper.metadata.get("aliases", "")).split("||") if alias.strip()]
        for item in [title, *aliases]:
            if item and item not in names:
                names.append(item)
        if title:
            for separator in [":", " - ", " — ", " – "]:
                if separator in title:
                    head = title.split(separator, 1)[0].strip()
                    if head and head not in names:
                        names.append(head)
        return names

    @staticmethod
    def _normalize_entity_key(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())

    def _contract_from_pending_clarification(
        self,
        *,
        clean_query: str,
        session: SessionContext,
        clarification_choice: dict[str, Any] | None = None,
    ) -> QueryContract | None:
        if session.pending_clarification_type != "ambiguity" or not session.pending_clarification_options:
            return None
        selected = self._option_from_clarification_choice(clarification_choice, session.pending_clarification_options)
        if selected is None:
            selected = self._select_pending_clarification_option(
                clean_query=clean_query,
                options=session.pending_clarification_options,
            )
        if selected is None:
            return None
        target = session.pending_clarification_target or str(selected.get("target", "") or "").strip()
        if not target:
            target = " ".join(extract_targets(clean_query)[:1])
        return self._contract_from_selected_clarification_option(
            clean_query=clean_query,
            target=target,
            selected=selected,
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
        selected_target = str(selected.get("target", "") or "").strip()
        target = target or selected_target
        meaning = str(selected.get("meaning", "") or selected.get("label", "") or target).strip()
        title = str(selected.get("title", "") or "").strip()
        notes = [
            resolution_note,
            "selected_ambiguity_option=" + json.dumps(selected, ensure_ascii=False),
        ]
        notes.extend(notes_extra or [])
        paper_id = str(selected.get("paper_id", "") or "").strip()
        if paper_id:
            notes.append(f"selected_paper_id={paper_id}")
        raw_requested = selected.get("source_requested_fields", [])
        source_requested = [str(item).strip() for item in raw_requested if str(item).strip()] if isinstance(raw_requested, list) else []
        raw_slots = selected.get("source_answer_slots", [])
        source_answer_slots = [str(item).strip() for item in raw_slots if str(item).strip()] if isinstance(raw_slots, list) else []
        source_relation = str(selected.get("source_relation", "") or selected.get("relation", "") or "").strip()
        is_formula_choice = source_relation == "formula_lookup" or "formula" in source_requested or "formula" in source_answer_slots
        if is_formula_choice:
            answer_slots = source_answer_slots or (["formula"] if "formula" in source_requested else [])
            rewritten = f"{target} 的公式是什么？{resolution_subject} {meaning}"
            if title:
                rewritten += f"，来源论文是《{title}》"
            return QueryContract(
                clean_query=rewritten,
                interaction_mode="research",
                relation="formula_lookup",
                targets=[target] if target else [],
                answer_slots=answer_slots,
                requested_fields=["formula", "variable_explanation", "source"],
                required_modalities=["page_text", "table"],
                answer_shape="bullets",
                precision_requirement="exact",
                continuation_mode="followup",
                notes=notes,
            )
        rewritten = f"{target} 是什么？{resolution_subject} {meaning}"
        if title:
            rewritten += f"，来源论文是《{title}》"
        return QueryContract(
            clean_query=rewritten,
            interaction_mode="research",
            relation="entity_definition",
            targets=[target] if target else [],
            requested_fields=["definition", "mechanism", "role_in_context"],
            required_modalities=["page_text", "paper_card", "table"],
            answer_shape="narrative",
            precision_requirement="high",
            continuation_mode="followup",
            notes=notes,
        )

    def _next_clarification_attempt(
        self,
        *,
        session: SessionContext,
        contract: QueryContract,
        verification: VerificationReport,
    ) -> int:
        key = self._clarification_key(contract=contract, verification=verification)
        if key and key == session.last_clarification_key:
            return session.clarification_attempts + 1
        return 1

    def _remember_clarification_attempt(
        self,
        *,
        session: SessionContext,
        contract: QueryContract,
        verification: VerificationReport,
    ) -> None:
        key = self._clarification_key(contract=contract, verification=verification)
        if key and key == session.last_clarification_key:
            session.clarification_attempts += 1
        else:
            session.last_clarification_key = key
            session.clarification_attempts = 1

    @staticmethod
    def _reset_clarification_tracking(session: SessionContext) -> None:
        session.last_clarification_key = ""
        session.clarification_attempts = 0

    def _clarification_key(self, *, contract: QueryContract, verification: VerificationReport) -> str:
        options = self._clarification_options(contract)
        option_key = "|".join(
            str(option.get("option_id") or option.get("paper_id") or option.get("meaning") or option.get("title") or "")
            for option in options[:4]
        )
        target_key = ",".join(normalize_lookup_text(item) for item in contract.targets if item)
        missing_key = ",".join(str(item) for item in verification.missing_fields)
        return "|".join(
            [
                contract.relation,
                target_key,
                verification.recommended_action,
                missing_key,
                option_key,
            ]
        )

    @staticmethod
    def _selected_clarification_paper_id(contract: QueryContract) -> str:
        for note in contract.notes:
            raw = str(note or "")
            if raw.startswith("selected_paper_id="):
                return raw.split("=", 1)[1].strip()
            if not raw.startswith("selected_ambiguity_option="):
                continue
            try:
                payload = json.loads(raw.split("=", 1)[1])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                paper_id = str(payload.get("paper_id", "") or "").strip()
                if paper_id:
                    return paper_id
        return ""

    @staticmethod
    def _option_from_clarification_choice(
        choice: dict[str, Any] | None,
        options: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not isinstance(choice, dict) or not options:
            return None
        option_id = str(choice.get("option_id", "") or "").strip()
        if option_id:
            for option in options:
                if str(option.get("option_id", "") or "").strip() == option_id:
                    return option
        raw_index = choice.get("index")
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            index = -1
        if 0 <= index < len(options):
            return options[index]
        paper_id = str(choice.get("paper_id", "") or "").strip()
        meaning = str(choice.get("meaning", "") or "").strip().lower()
        label = str(choice.get("label", "") or "").strip().lower()
        for option in options:
            if paper_id and str(option.get("paper_id", "") or "").strip() == paper_id:
                return option
            if meaning and str(option.get("meaning", "") or "").strip().lower() == meaning:
                return option
            if label and str(option.get("label", "") or "").strip().lower() == label:
                return option
        return None

    def _select_pending_clarification_option(
        self,
        *,
        clean_query: str,
        options: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        index = pending_clarification_selection_index(clean_query)
        if index is not None and 0 <= index < len(options):
            return options[index]
        normalized_query = normalize_lookup_text(clean_query)
        if not normalized_query:
            return None
        for option in options:
            meaning = normalize_lookup_text(str(option.get("meaning", "")))
            label = normalize_lookup_text(str(option.get("label", "")))
            title = normalize_lookup_text(str(option.get("title", "")))
            if meaning and normalized_query == meaning:
                return option
            if label and normalized_query == label:
                return option
            if (
                meaning
                and len(meaning) >= 10
                and meaning in normalized_query
                and looks_like_clarification_choice_text(normalized_query)
            ):
                return option
            if (
                label
                and len(label) >= 10
                and label in normalized_query
                and looks_like_clarification_choice_text(normalized_query)
            ):
                return option
            if title and len(normalized_query) >= 6 and normalized_query in title:
                return option
        return None

    def _disambiguation_options_from_evidence(
        self,
        *,
        contract: QueryContract,
        session: SessionContext,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> list[dict[str, Any]]:
        if not self._contract_needs_evidence_disambiguation(contract):
            return []
        if "resolved_human_choice" in contract.notes or self._selected_clarification_paper_id(contract):
            return []
        target = str(contract.targets[0] or "").strip()
        if not is_negative_correction_query(contract.clean_query) and "exclude_previous_focus" not in contract.notes:
            if self._target_binding_from_memory(session=session, target=target):
                return []
        options = self._acronym_options_from_evidence(target=target, papers=papers, evidence=evidence)
        goals = research_plan_goals(contract)
        if len(options) < 2 and "formula" in goals:
            broad_evidence = self.retriever.search_concept_evidence(
                query=target,
                contract=contract,
                limit=max(self.settings.evidence_limit_default, 96),
            )
            broad_options = self._acronym_options_from_evidence(target=target, papers=papers, evidence=broad_evidence)
            if len(broad_options) > len(options):
                options = broad_options
        if len(options) < 2 and "formula" in goals:
            corpus_evidence = self._acronym_evidence_from_corpus(target=target, limit=160)
            corpus_options = self._acronym_options_from_evidence(target=target, papers=papers, evidence=corpus_evidence)
            if len(corpus_options) > len(options):
                options = corpus_options
        excluded_titles = self._excluded_focus_titles(session=session, contract=contract)
        if excluded_titles:
            options = [
                option
                for option in options
                if self._normalize_title_key(str(option.get("title", ""))) not in excluded_titles
            ]
        if len(options) < 2:
            return []
        context_targets = [item for item in contract.targets[1:] if str(item).strip()]
        if context_targets:
            matched = [option for option in options if self._ambiguity_option_matches_context(option=option, context_targets=context_targets)]
            if len(matched) <= 1:
                return []
            options = matched
        if "exclude_previous_focus" in contract.notes and len(options) <= 1:
            return []
        return self._normalize_clarification_options(
            options[:4],
            contract=contract,
            target=target,
            kind="acronym_meaning",
            source="evidence_disambiguation",
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
            system_prompt=(
                "你是通用论文/实体候选消歧裁判器。"
                "你的唯一任务是在用户 query、QueryContract 和候选 metadata/snippet 之间判断哪个候选最符合用户真实意图。"
                "不要硬编码任何具体缩写、方法名、论文标题或 paper_id 的默认答案；只能基于输入字段做判断。"
                "不要生成最终研究答案，不要补充外部知识，只决定是否可自动绑定候选。"
                "当 query 明确指向某个候选的标题、上下文、方法原始提出/直接定义证据，且其他候选只是引用、应用、比较或弱相关时，可以 auto_resolve。"
                "输入中的 ranking_signals 是由候选自身 title/snippet/metadata 计算出的通用线索，不是某个缩写的白名单；"
                "当 direct_definition_or_origin、strong_title_or_alias_alignment 明显集中在同一个候选，"
                "而其他候选主要是 related_usage_or_citation 时，不要因为候选数量多而保守降到 0.70，应给出 >=0.85 的自动消歧。"
                "如果证据不足、候选关系接近、query 可能指向多篇论文，必须 ask_human。"
                "输出必须是 JSON，字段为 decision(auto_resolve|ask_human), selected_option_id, selected_paper_id, confidence, reason, rejected_options。"
                "confidence >= 0.85 才表示可自动消歧；0.65 到 0.85 只表示可作为推荐项；低于 0.65 不要默认推荐。"
            ),
            human_prompt=json.dumps(
                {
                    "user_query": contract.clean_query,
                    "query_contract": {
                        "relation": contract.relation,
                        "targets": contract.targets,
                        "answer_slots": contract_answer_slots(contract),
                        "requested_fields": contract.requested_fields,
                        "required_modalities": contract.required_modalities,
                        "answer_shape": contract.answer_shape,
                        "precision_requirement": contract.precision_requirement,
                        "continuation_mode": contract.continuation_mode,
                        "notes": [
                            note
                            for note in contract.notes
                            if not str(note).startswith("ambiguity_option=")
                        ][:12],
                    },
                    "candidate_options": [
                        self._disambiguation_judge_option_payload(option=option)
                        for option in options[:8]
                    ],
                    "output_schema": {
                        "decision": "auto_resolve | ask_human",
                        "selected_option_id": "string|null",
                        "selected_paper_id": "string|null",
                        "confidence": "number in [0,1]",
                        "reason": "short reason based only on provided metadata/snippets",
                        "rejected_options": [{"option_id": "string", "reason": "string"}],
                    },
                },
                ensure_ascii=False,
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

    def _disambiguation_judge_option_payload(self, *, option: dict[str, Any]) -> dict[str, Any]:
        paper_id = str(option.get("paper_id", "") or "").strip()
        paper = self._candidate_from_paper_id(paper_id) if paper_id else None
        metadata = dict(paper.metadata or {}) if paper is not None else {}
        signals = self._disambiguation_ranking_signals(option=option, paper=paper)
        return {
            "option_id": str(option.get("option_id", "") or "").strip(),
            "index": option.get("index"),
            "kind": str(option.get("kind", "") or "").strip(),
            "target": str(option.get("target", "") or "").strip(),
            "label": str(option.get("label", "") or "").strip(),
            "meaning": str(option.get("meaning", "") or "").strip(),
            "paper_id": paper_id,
            "title": str(option.get("title", "") or "").strip(),
            "year": str(option.get("year", "") or "").strip(),
            "snippet": self._truncate_context_text(str(option.get("snippet", "") or ""), limit=420),
            "match_reason": str(option.get("match_reason", "") or (paper.match_reason if paper is not None else "") or "").strip(),
            "evidence_relation": str(option.get("source_relation", "") or option.get("source", "") or "").strip(),
            "source_requested_fields": self._string_list(option.get("source_requested_fields")),
            "source_answer_slots": self._string_list(option.get("source_answer_slots")),
            "paper_aliases": self._truncate_context_text(str(metadata.get("aliases", "") or ""), limit=220),
            "paper_summary": self._truncate_context_text(
                str(
                    metadata.get("paper_card_text", "")
                    or metadata.get("generated_summary", "")
                    or metadata.get("abstract_note", "")
                    or ""
                ),
                limit=420,
            ),
            "ranking_signals": signals,
        }

    def _disambiguation_ranking_signals(
        self,
        *,
        option: dict[str, Any],
        paper: CandidatePaper | None,
    ) -> dict[str, Any]:
        title = str(option.get("title", "") or "").strip()
        target = str(option.get("target", "") or "").strip()
        label = str(option.get("label", "") or "").strip()
        meaning = str(option.get("meaning", "") or "").strip()
        snippet = str(option.get("snippet", "") or "").strip()
        aliases = str((paper.metadata or {}).get("aliases", "") or "") if paper is not None else ""
        summary = (
            str((paper.metadata or {}).get("paper_card_text", "") or "")
            or str((paper.metadata or {}).get("generated_summary", "") or "")
            or str((paper.metadata or {}).get("abstract_note", "") or "")
            if paper is not None
            else ""
        )
        context = "\n".join([title, aliases, label, meaning, snippet, summary])
        title_alias_text = "\n".join([title, aliases])
        title_alignment = self._candidate_title_alignment_score(
            target=target,
            label=label,
            meaning=meaning,
            title_alias_text=title_alias_text,
        )
        origin_score = self._candidate_origin_signal_score(context)
        usage_score = self._candidate_usage_signal_score(context)
        role = "ambiguous"
        if title_alignment >= 0.75 and origin_score >= 0.35:
            role = "direct_definition_or_origin"
        elif usage_score >= max(0.35, origin_score):
            role = "related_usage_or_citation"
        elif title_alignment >= 0.75:
            role = "strong_title_or_alias_alignment"
        return {
            "title_or_alias_alignment": round(title_alignment, 3),
            "origin_or_direct_definition_signal": round(origin_score, 3),
            "usage_or_citation_signal": round(usage_score, 3),
            "candidate_role_hint": role,
        }

    @classmethod
    def _candidate_title_alignment_score(
        cls,
        *,
        target: str,
        label: str,
        meaning: str,
        title_alias_text: str,
    ) -> float:
        title_key = normalize_lookup_text(title_alias_text)
        if not title_key:
            return 0.0
        probes = [target, label, meaning]
        scores: list[float] = []
        for probe in probes:
            probe_key = normalize_lookup_text(probe)
            if not probe_key:
                continue
            if probe_key and probe_key in title_key:
                scores.append(1.0)
                continue
            probe_tokens = cls._disambiguation_content_tokens(probe_key)
            if not probe_tokens:
                continue
            title_tokens = set(cls._disambiguation_content_tokens(title_key))
            if not title_tokens:
                continue
            overlap = len([token for token in probe_tokens if token in title_tokens])
            if overlap:
                scores.append(overlap / max(1, len(probe_tokens)))
        return max(scores or [0.0])

    @staticmethod
    def _disambiguation_content_tokens(text: str) -> list[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "by",
            "for",
            "from",
            "in",
            "is",
            "it",
            "its",
            "main",
            "of",
            "on",
            "or",
            "our",
            "the",
            "this",
            "to",
            "we",
            "with",
        }
        tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
        return [token for token in tokens if len(token) > 1 and token not in stopwords]

    @staticmethod
    def _candidate_origin_signal_score(text: str) -> float:
        lowered = str(text or "").lower()
        patterns = [
            r"\bour\s+main\s+contribution\b",
            r"\bwe\s+(?:propose|proposed|present|introduce|introduced|derive|develop)\b",
            r"\bthis\s+paper\s+(?:proposes|introduces|presents|derives|develops)\b",
            r"\bmain\s+contribution\b",
            r"\bpropose\s+(?:a|an|the)?\b",
            r"\bintroduce\s+(?:a|an|the)?\b",
        ]
        score = 0.0
        for pattern in patterns:
            if re.search(pattern, lowered):
                score += 0.35
        return min(score, 1.0)

    @staticmethod
    def _candidate_usage_signal_score(text: str) -> float:
        lowered = str(text or "").lower()
        patterns = [
            r"\badopt(?:s|ed|ing)?\b",
            r"\buse(?:s|d|ing)?\b",
            r"\bfollowing\b",
            r"\bbased\s+on\b",
            r"\bextends?\b",
            r"\bvariant\s+of\b",
            r"\bin\s+recent\s+work\b",
            r"\bproposed\s+by\b",
            r"\bet\s+al\.\s+(?:proposed|introduced)\b",
            r"\binclude(?:s|d|ing)?\b",
        ]
        score = 0.0
        for pattern in patterns:
            if re.search(pattern, lowered):
                score += 0.25
        return min(score, 1.0)

    @staticmethod
    def _selected_option_from_judge_decision(
        *,
        decision: DisambiguationJudgeDecision | None,
        options: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if decision is None:
            return None
        selected_option_id = str(decision.selected_option_id or "").strip()
        if selected_option_id:
            for option in options:
                if str(option.get("option_id", "") or "").strip() == selected_option_id:
                    return option
        selected_paper_id = str(decision.selected_paper_id or "").strip()
        if selected_paper_id:
            for option in options:
                if str(option.get("paper_id", "") or "").strip() == selected_paper_id:
                    return option
        return None

    def _judge_allows_auto_resolve(self, decision: DisambiguationJudgeDecision | None) -> bool:
        return (
            decision is not None
            and decision.decision == "auto_resolve"
            and float(decision.confidence) >= self.agent_settings.disambiguation_auto_resolve_threshold
        )

    def _apply_disambiguation_judge_recommendation(
        self,
        *,
        options: list[dict[str, Any]],
        decision: DisambiguationJudgeDecision | None,
    ) -> list[dict[str, Any]]:
        selected = self._selected_option_from_judge_decision(decision=decision, options=options)
        if (
            selected is None
            or decision is None
            or float(decision.confidence) < self.agent_settings.disambiguation_recommend_threshold
        ):
            return options
        rejected_reasons = {
            str(item.option_id or "").strip(): str(item.reason or "").strip()
            for item in decision.rejected_options
            if str(item.option_id or "").strip()
        }
        selected_id = str(selected.get("option_id", "") or "").strip()
        annotated: list[dict[str, Any]] = []
        for option in options:
            payload = dict(option)
            option_id = str(payload.get("option_id", "") or "").strip()
            payload["display_title"] = str(payload.get("display_title", "") or payload.get("title", "") or "").strip()
            if option_id == selected_id:
                payload["display_label"] = str(payload.get("display_label", "") or "推荐候选").strip()
                payload["display_reason"] = self._truncate_context_text(str(decision.reason or ""), limit=180)
                payload["judge_recommended"] = True
                payload["disambiguation_confidence"] = round(float(decision.confidence), 3)
            elif option_id in rejected_reasons:
                payload["display_reason"] = self._truncate_context_text(rejected_reasons[option_id], limit=180)
            annotated.append(payload)
        annotated.sort(
            key=lambda item: (
                str(item.get("option_id", "") or "") != selected_id,
                int(item.get("index", 9999) if isinstance(item.get("index"), int) else 9999),
            )
        )
        return annotated

    @staticmethod
    def _disambiguation_judge_summary(
        *,
        options: list[dict[str, Any]],
        judge_decision: DisambiguationJudgeDecision | None,
    ) -> str:
        if judge_decision is None:
            return f"options={len(options)}, judge=unavailable"
        return (
            f"options={len(options)}, judge={judge_decision.decision}, "
            f"confidence={float(judge_decision.confidence):.2f}"
        )

    def _contract_with_auto_resolved_ambiguity(
        self,
        *,
        contract: QueryContract,
        selected: dict[str, Any],
        decision: DisambiguationJudgeDecision | None,
    ) -> QueryContract:
        notes = [
            str(note).strip()
            for note in contract.notes
            if str(note).strip()
            and not str(note).startswith("ambiguity_option=")
            and not str(note).startswith("selected_ambiguity_option=")
            and not str(note).startswith("selected_paper_id=")
            and not str(note).startswith("disambiguation_judge_")
        ]
        selected_payload = clarification_option_public_payload(selected)
        notes.append("auto_resolved_by_llm_judge")
        notes.append("selected_ambiguity_option=" + json.dumps(selected_payload, ensure_ascii=False))
        paper_id = str(selected.get("paper_id", "") or "").strip()
        if paper_id:
            notes.append(f"selected_paper_id={paper_id}")
        if decision is not None:
            notes.append(f"disambiguation_judge_confidence={float(decision.confidence):.3f}")
            reason = self._truncate_context_text(str(decision.reason or ""), limit=220)
            if reason:
                notes.append(f"disambiguation_judge_reason={reason}")
        return contract.model_copy(update={"notes": list(dict.fromkeys(notes))})

    def _refresh_state_for_selected_ambiguity(
        self,
        *,
        state: dict[str, Any],
        selected: dict[str, Any],
        emit: Callable[[str, dict[str, Any]], None],
        execution_steps: list[dict[str, Any]],
    ) -> None:
        paper_id = str(selected.get("paper_id", "") or "").strip()
        if not paper_id:
            return
        contract: QueryContract = state["contract"]
        plan: ResearchPlan = state["plan"]
        excluded_titles: set[str] = state["excluded_titles"]
        candidate_pool: list[CandidatePaper] = [
            *list(state.get("screened_papers") or []),
            *list(state.get("candidate_papers") or []),
        ]
        selected_papers = [paper for paper in candidate_pool if paper.paper_id == paper_id]
        if not selected_papers:
            paper = self._candidate_from_paper_id(paper_id)
            selected_papers = [paper] if paper is not None else []
        if selected_papers:
            state["screened_papers"] = selected_papers[:1]
            emit("screened_papers", {"count": len(state["screened_papers"]), "items": [item.model_dump() for item in state["screened_papers"]]})
        evidence = [item for item in list(state.get("evidence") or []) if item.paper_id == paper_id]
        if not evidence:
            evidence_query = evidence_query_text(contract)
            if should_use_concept_evidence(contract):
                evidence = self.retriever.search_concept_evidence(
                    query=evidence_query,
                    contract=contract,
                    paper_ids=[paper_id],
                    limit=plan.evidence_limit,
                )
                if not evidence:
                    evidence = self.retriever.expand_evidence(
                        paper_ids=[paper_id],
                        query=evidence_query,
                        contract=contract,
                        limit=plan.evidence_limit,
                    )
            else:
                evidence = self.retriever.expand_evidence(
                    paper_ids=[paper_id],
                    query=evidence_query,
                    contract=contract,
                    limit=plan.evidence_limit,
                )
            if excluded_titles:
                evidence = self._filter_evidence_by_excluded_titles(evidence, excluded_titles=excluded_titles)
            self._record_agent_observation(
                emit=emit,
                execution_steps=execution_steps,
                tool="search_corpus",
                summary=f"auto_resolved_evidence={len(evidence)}",
                payload={"stage": "search_evidence", "selected_paper_id": paper_id, "evidence_count": len(evidence)},
            )
        state["evidence"] = evidence
        emit("evidence", {"count": len(evidence), "items": [item.model_dump() for item in evidence]})

    def _contract_needs_evidence_disambiguation(self, contract: QueryContract) -> bool:
        if not contract.targets:
            return False
        target = str(contract.targets[0] or "").strip()
        if not is_short_acronym(target):
            return False
        if note_values(notes=contract.notes, prefix="ambiguous_slot="):
            return True
        return bool(research_plan_goals(contract) & self._disambiguation_goal_markers())

    @staticmethod
    def _disambiguation_goal_markers() -> set[str]:
        return {"definition", "entity_type", "role_in_context", "mechanism", "formula"}

    def _disambiguation_missing_fields(self, contract: QueryContract) -> list[str]:
        ambiguous_slots = note_values(notes=contract.notes, prefix="ambiguous_slot=")
        return ambiguous_slots or ["disambiguation"]

    def _acronym_options_from_evidence(
        self,
        *,
        target: str,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> list[dict[str, Any]]:
        paper_by_id = {item.paper_id: item for item in papers}
        buckets: dict[str, dict[str, Any]] = {}
        target_key = target.lower()
        for item in evidence:
            if not any(matches_target(haystack, target) for haystack in [item.snippet, item.caption, item.title] if haystack):
                continue
            paper = paper_by_id.get(item.paper_id) or self._candidate_from_paper_id(item.paper_id)
            if paper is None:
                continue
            text = " ".join([item.snippet, item.caption, item.title])
            expansion = self._extract_acronym_expansion_from_text(text=text, acronym=target)
            option_key = self._normalize_acronym_meaning(expansion) if expansion else normalize_lookup_text(f"{target_key}:{paper.paper_id}")
            if not option_key:
                option_key = f"{target_key}:{paper.paper_id}"
            bucket = buckets.setdefault(
                option_key,
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "year": paper.year,
                    "meaning": expansion or target,
                    "snippet": "",
                    "score": 0.0,
                    "paper_ids": [],
                    "titles": [],
                },
            )
            bucket["score"] = float(bucket.get("score", 0.0)) + float(item.score) + (5.0 if expansion else 0.0)
            if not bucket.get("snippet"):
                bucket["snippet"] = " ".join(item.snippet.split())[:220]
            if paper.paper_id not in bucket["paper_ids"]:
                bucket["paper_ids"].append(paper.paper_id)
            if paper.title not in bucket["titles"]:
                bucket["titles"].append(paper.title)
            if expansion and len(expansion) > len(str(bucket.get("meaning", ""))):
                bucket["meaning"] = expansion
        options = list(buckets.values())
        expanded_papers = {
            str(option.get("paper_id", ""))
            for option in options
            if str(option.get("meaning", "")).strip().lower() != target.lower()
        }
        options = [
            option
            for option in options
            if not (
                str(option.get("paper_id", "")) in expanded_papers
                and str(option.get("meaning", "")).strip().lower() == target.lower()
            )
        ]
        if any(str(option.get("meaning", "")).strip().lower() != target.lower() for option in options):
            options = [option for option in options if str(option.get("meaning", "")).strip().lower() != target.lower()]
        for option in options:
            option["context_text"] = self._ambiguity_option_context_text(option)
        options.sort(key=lambda item: (-float(item.get("score", 0.0)), str(item.get("title", ""))))
        return options

    def _acronym_evidence_from_corpus(self, *, target: str, limit: int) -> list[EvidenceBlock]:
        evidence: list[EvidenceBlock] = []
        for paper_doc in self.retriever.paper_documents():
            paper_id = str((paper_doc.metadata or {}).get("paper_id", "") or "").strip()
            if not paper_id:
                continue
            for doc in self.retriever.block_documents_for_paper(paper_id, limit=320):
                meta = dict(doc.metadata or {})
                text = str(doc.page_content or "")
                if not matches_target(text, target):
                    continue
                score = 1.0
                expansion = self._extract_acronym_expansion_from_text(text=text, acronym=target)
                if expansion:
                    score += 6.0
                if int(meta.get("formula_hint", 0) or 0):
                    score += 2.0
                evidence.append(
                    EvidenceBlock(
                        doc_id=str(meta.get("doc_id", "")),
                        paper_id=paper_id,
                        title=str(meta.get("title", "")),
                        file_path=str(meta.get("file_path", "")),
                        page=int(meta.get("page", 0) or 0),
                        block_type=str(meta.get("block_type", "")),
                        caption=str(meta.get("caption", "")),
                        bbox=str(meta.get("bbox", "")),
                        snippet=text[:900],
                        score=score,
                        metadata=meta,
                    )
                )
                if len(evidence) >= limit:
                    return evidence
        evidence.sort(key=lambda item: (-item.score, item.title, item.page))
        return evidence[:limit]

    @staticmethod
    def _extract_acronym_expansion_from_text(*, text: str, acronym: str) -> str:
        compact = " ".join(str(text or "").split())
        if not compact or not acronym:
            return ""
        patterns = [
            rf"([A-Za-z][A-Za-z\-/]+(?:\s+[A-Za-z][A-Za-z\-/]+){{1,8}})\s*\(\s*{re.escape(acronym)}\s*\)",
            rf"{re.escape(acronym)}\s*(?:stands for|means|refers to|is short for)\s*([A-Za-z][A-Za-z\-/]+(?:\s+[A-Za-z][A-Za-z\-/]+){{1,8}})",
        ]
        stopwords = {"and", "or", "the", "with", "from", "into", "using", "based"}
        for pattern in patterns:
            match = re.search(pattern, compact, flags=re.IGNORECASE)
            if not match:
                continue
            expansion = " ".join(match.group(1).strip(" ,.;:-").split())
            expansion = re.sub(r"^and(?=[A-Z])", "", expansion).strip()
            expansion = re.sub(r"^and\s+", "", expansion, flags=re.IGNORECASE).strip()
            words = expansion.split()
            while words and words[0].lower() in stopwords:
                words.pop(0)
            expansion = " ".join(words)
            if len(expansion) >= 6:
                return expansion
        return ""

    @staticmethod
    def _normalize_acronym_meaning(text: str) -> str:
        normalized = str(text or "").lower().replace("behaviour", "behavior")
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return " ".join(normalized.split())

    def _ambiguity_option_context_text(self, option: dict[str, Any]) -> str:
        paper_id = str(option.get("paper_id", "") or "")
        paper = self._candidate_from_paper_id(paper_id) if paper_id else None
        parts = [
            str(option.get("meaning", "")),
            str(option.get("title", "")),
            str(option.get("snippet", "")),
        ]
        if paper is not None:
            parts.extend(
                [
                    str(paper.metadata.get("aliases", "")),
                    str(paper.metadata.get("paper_card_text", "")),
                    str(paper.metadata.get("generated_summary", "")),
                    str(paper.metadata.get("abstract_note", "")),
                ]
            )
        return "\n".join(part for part in parts if part)

    def _ambiguity_option_matches_context(self, *, option: dict[str, Any], context_targets: list[str]) -> bool:
        text = str(option.get("context_text", "")) or self._ambiguity_option_context_text(option)
        return any(matches_target(text, str(target)) for target in context_targets if str(target).strip())

    @staticmethod
    def _contract_with_ambiguity_options(*, contract: QueryContract, options: list[dict[str, Any]]) -> QueryContract:
        notes = [note for note in contract.notes if not str(note).startswith("ambiguity_option=")]
        for option in options[:4]:
            payload = clarification_option_public_payload(option)
            notes.append("ambiguity_option=" + json.dumps(payload, ensure_ascii=False))
        return contract.model_copy(update={"notes": notes})

    def _clarification_options(self, contract: QueryContract) -> list[dict[str, Any]]:
        options = ambiguity_options_from_notes(contract.notes)
        target = contract.targets[0] if contract.targets else ""
        return self._normalize_clarification_options(
            options,
            contract=contract,
            target=target,
            kind="acronym_meaning",
            source="contract_notes",
        )

    def _normalize_clarification_options(
        self,
        options: list[dict[str, Any]],
        *,
        contract: QueryContract,
        target: str = "",
        kind: str = "paper_choice",
        source: str = "clarification",
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, option in enumerate(options):
            normalized.append(
                self._normalize_clarification_option(
                    option,
                    index=index,
                    contract=contract,
                    target=target,
                    kind=kind,
                    source=source,
                )
            )
        return normalized

    def _normalize_clarification_option(
        self,
        option: dict[str, Any],
        *,
        index: int,
        contract: QueryContract,
        target: str = "",
        kind: str = "paper_choice",
        source: str = "clarification",
    ) -> dict[str, Any]:
        payload = dict(option)
        resolved_target = str(payload.get("target", "") or target or (contract.targets[0] if contract.targets else "") or "").strip()
        resolved_kind = str(payload.get("kind", "") or kind or "paper_choice").strip()
        meaning = str(payload.get("meaning", "") or "").strip()
        title = str(payload.get("title", "") or "").strip()
        year = str(payload.get("year", "") or "").strip()
        label = str(payload.get("label", "") or meaning or title or resolved_target or f"option {index + 1}").strip()
        description = str(payload.get("description", "") or "").strip()
        if not description:
            description = self._clarification_option_description(payload, title=title, year=year)
        payload["schema_version"] = CLARIFICATION_OPTION_SCHEMA_VERSION
        payload["index"] = index
        payload["kind"] = resolved_kind
        payload["target"] = resolved_target
        payload["label"] = label
        payload["description"] = self._truncate_context_text(description, limit=260) if description else ""
        payload.setdefault("meaning", meaning or label)
        payload.setdefault("title", title)
        payload.setdefault("year", year)
        payload["display_title"] = str(payload.get("display_title", "") or title).strip()
        payload["display_label"] = str(payload.get("display_label", "") or "").strip()
        payload["display_reason"] = self._truncate_context_text(str(payload.get("display_reason", "") or ""), limit=220)
        if "disambiguation_confidence" in payload:
            try:
                payload["disambiguation_confidence"] = round(float(payload.get("disambiguation_confidence") or 0.0), 3)
            except (TypeError, ValueError):
                payload.pop("disambiguation_confidence", None)
        payload.setdefault("source", source)
        payload["source_relation"] = str(payload.get("source_relation", "") or contract.relation)
        payload["source_requested_fields"] = self._string_list(payload.get("source_requested_fields") or contract.requested_fields)
        payload["source_required_modalities"] = self._string_list(payload.get("source_required_modalities") or contract.required_modalities)
        payload["source_answer_slots"] = self._string_list(payload.get("source_answer_slots") or contract.answer_slots)
        payload["paper_ids"] = self._string_list(payload.get("paper_ids"))
        payload["titles"] = self._string_list(payload.get("titles"))
        payload["evidence_ids"] = self._string_list(payload.get("evidence_ids"))
        payload["option_id"] = str(payload.get("option_id", "") or "").strip() or self._clarification_option_id(
            kind=resolved_kind,
            target=resolved_target,
            label=label,
            paper_id=str(payload.get("paper_id", "") or ""),
            title=title,
            index=index,
        )
        return payload

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, tuple | set):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value or "").strip()
        return [text] if text else []

    @staticmethod
    def _clarification_option_description(option: dict[str, Any], *, title: str, year: str) -> str:
        meta = " · ".join(item for item in [title, year] if item)
        context = str(option.get("context_text", "") or option.get("snippet", "") or "").strip()
        context = " ".join(context.split())
        return context or meta

    @staticmethod
    def _clarification_option_id(
        *,
        kind: str,
        target: str,
        label: str,
        paper_id: str,
        title: str,
        index: int,
    ) -> str:
        seed = json.dumps(
            {
                "kind": kind,
                "target": target,
                "label": label,
                "paper_id": paper_id,
                "title": title,
                "index": index,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        prefix = re.sub(r"[^a-z0-9]+", "-", f"{kind}-{target}".lower()).strip("-") or "clarification"
        return f"{prefix}-{digest}"

    def _store_pending_clarification(self, *, session: SessionContext, contract: QueryContract) -> None:
        options = self._clarification_options(contract)
        if options:
            session.pending_clarification_type = "ambiguity"
            session.pending_clarification_target = contract.targets[0] if contract.targets else ""
            session.pending_clarification_options = options
        else:
            self._clear_pending_clarification(session)

    @staticmethod
    def _clear_pending_clarification(session: SessionContext) -> None:
        session.pending_clarification_type = ""
        session.pending_clarification_target = ""
        session.pending_clarification_options = []

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
        precision = precision_requirement if precision_requirement in {"exact", "high", "normal"} else "normal"
        active = ActiveResearch(
            relation=relation,
            targets=targets,
            titles=titles,
            requested_fields=requested_fields,
            required_modalities=required_modalities,
            answer_shape=answer_shape,
            precision_requirement=precision,  # type: ignore[arg-type]
            clean_query=clean_query,
        )
        if not active.last_topic_signature:
            active.last_topic_signature = active.topic_signature()
        return active

    def _excluded_focus_titles(self, *, session: SessionContext, contract: QueryContract) -> set[str]:
        if "exclude_previous_focus" not in contract.notes and not is_negative_correction_query(contract.clean_query):
            return set()
        titles: list[str] = []
        titles.extend(session.effective_active_research().titles)
        if session.turns:
            titles.extend(session.turns[-1].titles)
        return {self._normalize_title_key(title) for title in titles if self._normalize_title_key(title)}

    @staticmethod
    def _filter_candidate_papers_by_excluded_titles(
        candidates: list[CandidatePaper],
        *,
        excluded_titles: set[str],
    ) -> list[CandidatePaper]:
        if not excluded_titles:
            return candidates
        return [item for item in candidates if ResearchAssistantAgentV4._normalize_title_key(item.title) not in excluded_titles]

    @staticmethod
    def _filter_evidence_by_excluded_titles(
        evidence: list[EvidenceBlock],
        *,
        excluded_titles: set[str],
    ) -> list[EvidenceBlock]:
        if not excluded_titles:
            return evidence
        return [item for item in evidence if ResearchAssistantAgentV4._normalize_title_key(item.title) not in excluded_titles]

    @staticmethod
    def _normalize_title_key(title: str) -> str:
        return " ".join(str(title or "").lower().split())

    def _entity_evidence_limit(self, *, contract: QueryContract, plan: ResearchPlan, excluded_titles: set[str]) -> int:
        goals = research_plan_goals(contract)
        if goals & {"entity_type", "role_in_context"} and contract.targets and is_short_acronym(contract.targets[0]):
            return max(plan.evidence_limit, 96 if excluded_titles else 72)
        return plan.evidence_limit

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
        repeated_excluded = bool(excluded_titles & {self._normalize_title_key(title) for title in focus_titles})
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
        if self._contract_needs_evidence_disambiguation(contract):
            if self._target_binding_from_memory(session=session, target=contract.targets[0]) and "exclude_previous_focus" not in contract.notes:
                option_count = 1
            else:
                option_count = len(self._acronym_options_from_evidence(target=contract.targets[0], papers=papers, evidence=evidence))
            if option_count > 1 and not claims and not ambiguity_options_from_notes(contract.notes):
                return {
                    "decision": "clarify",
                    "reason": "Multiple acronym meanings remain unresolved.",
                    "missing_fields": self._disambiguation_missing_fields(contract),
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

    @staticmethod
    def _clean_contract_target_text(text: str) -> str:
        return clean_contract_target_text(text)

    @staticmethod
    def _is_structural_target_reference(text: str) -> bool:
        return is_structural_target_reference(text)

    @staticmethod
    def _normalize_modalities(modalities: list[str], *, relation: str) -> list[str]:
        return normalize_modalities(modalities, relation=relation)

    def _build_research_plan(self, contract: QueryContract) -> ResearchPlan:
        return build_research_plan(contract=contract, settings=self.settings)

    def _compress_session_history_if_needed(self, session: SessionContext) -> None:
        if self.clients.chat is None:
            return
        if len(session.turns) < max(6, self.settings.agent_history_max_turns - 1):
            return
        retained_turns = max(4, self.settings.agent_history_max_turns // 2)
        older_turns = session.turns[:-retained_turns]
        if not older_turns:
            return
        compressed = self.clients.invoke_text(
            system_prompt=(
                "你是研究助手的会话记忆压缩器。"
                "请把较早的对话压缩成简洁中文摘要，保留："
                "1. 主要研究主题和实体；"
                "2. 已经回答过的问题类型（如公式、定义、实验结果、图表）；"
                "3. 仍然可能被继续追问的开放上下文。"
                "不要编造。输出 3-6 句纯文本摘要。"
            ),
            human_prompt=json.dumps(
                {
                    "existing_summary": session.summary,
                    "older_turns": [
                        {
                            "query": turn.query,
                            "relation": turn.relation,
                            "interaction_mode": turn.interaction_mode,
                            "targets": turn.targets,
                            "requested_fields": turn.requested_fields,
                            "answer_shape": turn.answer_shape,
                            "answer": turn.answer[:320],
                        }
                        for turn in older_turns
                    ],
                },
                ensure_ascii=False,
            ),
            fallback=session.summary,
        ).strip()
        if compressed:
            session.summary = compressed
        session.turns = session.turns[-retained_turns:]
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
        if not use_web_search or not self.web_search.is_configured:
            return []
        search_query = str(query_override or "").strip() or web_query_text(contract)
        return self.web_search.search(
            query=search_query,
            max_results=max_web_results,
            topic=web_search_topic(search_query or contract.clean_query),
            include_domains=web_include_domains(contract),
        )

    @staticmethod
    def _build_web_research_claim(*, contract: QueryContract, web_evidence: list[EvidenceBlock]) -> Claim:
        return build_web_research_claim(contract=contract, web_evidence=web_evidence)

    def _claim_focus_titles(self, *, claims: list[Claim], papers: list[CandidatePaper]) -> list[str]:
        titles: list[str] = []
        by_id = {item.paper_id: item.title for item in papers}
        for claim in claims:
            for paper_id in claim.paper_ids:
                title = by_id.get(paper_id)
                if not title:
                    doc = self.retriever.paper_doc_by_id(paper_id)
                    if doc is not None:
                        title = str((doc.metadata or {}).get("title", ""))
                if title and title not in titles:
                    titles.append(title)
        return titles[:3] or [item.title for item in papers[:3]]

    def _resolve_followup_seed_papers(
        self,
        *,
        contract: QueryContract,
        candidates: list[CandidatePaper],
        session: SessionContext,
    ) -> list[CandidatePaper]:
        if not candidates:
            return []
        by_id = {item.paper_id: item for item in candidates}
        if self.clients.chat is not None:
            payload = self.clients.invoke_json(
                system_prompt=(
                    "你是论文关系求解器中的种子论文定位器。"
                    "请根据当前问题、目标实体和候选论文，找出用户真正想追踪其后续工作的原始/种子论文。"
                    "如果目标是数据集、方法或模型，优先选择“引入/提出/定义该对象”的论文，而不是后续扩展。"
                    "只输出 JSON，字段为 seed_paper_ids 和 rationale。"
                ),
                human_prompt=json.dumps(
                    {
                        "query": contract.clean_query,
                        "targets": contract.targets,
                        "active_titles": session.effective_active_research().titles,
                        "candidates": [self._paper_brief(item) for item in candidates[:8]],
                    },
                    ensure_ascii=False,
                ),
                fallback={},
            )
            seed_ids = payload.get("seed_paper_ids", []) if isinstance(payload, dict) else []
            if isinstance(seed_ids, list):
                selected = [by_id[str(item)] for item in seed_ids if str(item) in by_id]
                if selected:
                    return selected[:2]
        ranked = sorted(
            candidates,
            key=lambda item: (-self._followup_seed_score(contract=contract, paper=item, session=session), item.title),
        )
        return ranked[:1]

    def _expand_followup_candidate_pool(
        self,
        *,
        contract: QueryContract,
        seed_papers: list[CandidatePaper],
        initial_candidates: list[CandidatePaper],
    ) -> list[CandidatePaper]:
        pool = {item.paper_id: item for item in initial_candidates}
        query_parts = [contract.clean_query, *contract.targets]
        for paper in seed_papers[:2]:
            query_parts.append(self._paper_anchor_text(paper))
            query_parts.append(self._followup_expansion_terms(paper))
        search_query = " ".join(part.strip() for part in query_parts if str(part).strip())
        if search_query:
            expanded = self.retriever.search_papers(
                query=search_query,
                contract=contract.model_copy(update={"continuation_mode": "fresh"}),
                limit=max(16, self.settings.paper_limit_default + 10),
            )
            for item in expanded:
                pool.setdefault(item.paper_id, item)
        ranked = list(pool.values())
        ranked.sort(key=lambda item: (-item.score, safe_year(item.year), item.title))
        return ranked

    def _rank_followup_candidates(
        self,
        *,
        contract: QueryContract,
        seed_papers: list[CandidatePaper],
        candidates: list[CandidatePaper],
        evidence: list[EvidenceBlock] | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        seed_ids = {item.paper_id for item in seed_papers}
        filtered = [item for item in candidates if item.paper_id not in seed_ids]
        if not filtered:
            return []
        selected_title = self._selected_followup_candidate_title(contract)
        if selected_title:
            selected_filtered = [item for item in filtered if self._candidate_title_matches(item, selected_title)]
            if selected_filtered:
                return [
                    {
                        "paper": paper,
                        **self._selected_followup_candidate_assessment(
                            contract=contract,
                            seed_papers=seed_papers,
                            paper=paper,
                            evidence=evidence or [],
                        ),
                    }
                    for paper in selected_filtered[:3]
                ]
        by_id = {item.paper_id: item for item in filtered}
        if self.clients.chat is not None:
            payload = self.clients.invoke_json(
                system_prompt=(
                    "你是论文关系分析器。"
                    "请判断哪些候选论文是种子论文的后续研究、扩展工作、迁移工作或直接延续。"
                    "后续工作必须与种子论文的对象、问题设定或方法线索直接相关；只在同一大领域但关系松散的论文不要选。"
                    "绝对不要把种子论文本身选进去。"
                    "只输出 JSON，字段为 followups。followups 中每项包含 paper_id, relation_type, reason, confidence。"
                ),
                human_prompt=json.dumps(
                    {
                        "query": contract.clean_query,
                        "targets": contract.targets,
                        "seed_papers": [self._paper_brief(item) for item in seed_papers[:2]],
                        "candidates": [self._paper_brief(item) for item in filtered[:10]],
                    },
                    ensure_ascii=False,
                ),
                fallback={},
            )
            raw_followups = payload.get("followups", []) if isinstance(payload, dict) else []
            if isinstance(raw_followups, list):
                selected: list[dict[str, Any]] = []
                for item in raw_followups:
                    if not isinstance(item, dict):
                        continue
                    paper_id = str(item.get("paper_id", "")).strip()
                    paper = by_id.get(paper_id)
                    if paper is None:
                        continue
                    assessment = self._followup_relationship_assessment(
                        contract=contract,
                        seed_papers=seed_papers,
                        paper=paper,
                    )
                    if assessment["score"] < 0.3:
                        continue
                    selected.append(
                        {
                            "paper": paper,
                            "relation_type": str(assessment["relation_type"]),
                            "reason": str(assessment["reason"]),
                            "confidence": min(
                                self._coerce_confidence(item.get("confidence", 0.82)),
                                float(assessment["confidence"]),
                            ),
                            "relationship_strength": str(assessment["strength"]),
                        }
                    )
                if selected:
                    fallback_selected = self._rank_followup_candidates_fallback(
                        contract=contract,
                        seed_papers=seed_papers,
                        candidates=filtered,
                    )
                    return self._merge_followup_rankings(primary=selected, secondary=fallback_selected)[:10]
        return self._rank_followup_candidates_fallback(contract=contract, seed_papers=seed_papers, candidates=filtered)

    @staticmethod
    def _selected_followup_candidate_title(contract: QueryContract) -> str:
        for note in contract.notes:
            raw = str(note or "")
            if raw.startswith("candidate_title="):
                return raw.split("=", 1)[1].strip()
        return ""

    def _candidate_title_matches(self, paper: CandidatePaper, selected_title: str) -> bool:
        selected_key = normalize_lookup_text(selected_title)
        if not selected_key:
            return False
        title_key = normalize_lookup_text(paper.title)
        aliases = normalize_lookup_text(str(paper.metadata.get("aliases", "")))
        return selected_key in title_key or title_key in selected_key or selected_key in aliases

    def _selected_followup_candidate_assessment(
        self,
        *,
        contract: QueryContract,
        seed_papers: list[CandidatePaper],
        paper: CandidatePaper,
        evidence: list[EvidenceBlock],
    ) -> dict[str, Any]:
        relationship_evidence = self._followup_relationship_evidence(
            contract=contract,
            seed_papers=seed_papers,
            paper=paper,
            evidence=evidence,
        )
        llm_assessment = self._llm_validate_followup_candidate(
            contract=contract,
            seed_papers=seed_papers,
            paper=paper,
            relationship_evidence=relationship_evidence,
        )
        if llm_assessment:
            return llm_assessment
        fallback = self._followup_relationship_assessment(contract=contract, seed_papers=seed_papers, paper=paper)
        if float(fallback.get("score", 0.0)) < 0.3:
            return {
                "relation_type": "证据不足",
                "reason": "当前本地证据没有显示候选论文明确使用、继承、引用或评测种子论文/数据集。",
                "confidence": 0.55,
                "relationship_strength": "not_enough_evidence",
                "strict_followup": False,
                "evidence_ids": [item.doc_id for item in relationship_evidence[:4]],
            }
        return {
            "relation_type": str(fallback["relation_type"]),
            "reason": str(fallback["reason"]),
            "confidence": float(fallback["confidence"]),
            "relationship_strength": str(fallback["strength"]),
            "strict_followup": str(fallback["strength"]) == "direct",
            "evidence_ids": [item.doc_id for item in relationship_evidence[:4]],
        }

    def _followup_relationship_evidence(
        self,
        *,
        contract: QueryContract,
        seed_papers: list[CandidatePaper],
        paper: CandidatePaper,
        evidence: list[EvidenceBlock],
    ) -> list[EvidenceBlock]:
        seed_ids = [item.paper_id for item in seed_papers[:2]]
        pair_ids = list(dict.fromkeys([*seed_ids, paper.paper_id]))
        selected = [item for item in evidence if item.paper_id in set(pair_ids)]
        if len(selected) < 6:
            query_parts = [
                contract.clean_query,
                " ".join(contract.targets),
                paper.title,
                "uses evaluates benchmark extends cites dataset method follow-up",
            ]
            expanded = self.retriever.expand_evidence(
                paper_ids=pair_ids,
                query=" ".join(part for part in query_parts if part),
                contract=contract.model_copy(update={"required_modalities": ["page_text", "paper_card"]}),
                limit=12,
            )
            by_id = {item.doc_id: item for item in selected}
            for item in expanded:
                by_id.setdefault(item.doc_id, item)
            selected = list(by_id.values())
        selected.sort(key=lambda item: (0 if item.paper_id == paper.paper_id else 1, -item.score, item.page, item.doc_id))
        return selected[:12]

    def _llm_validate_followup_candidate(
        self,
        *,
        contract: QueryContract,
        seed_papers: list[CandidatePaper],
        paper: CandidatePaper,
        relationship_evidence: list[EvidenceBlock],
    ) -> dict[str, Any]:
        if self.clients.chat is None or not seed_papers:
            return {}
        payload = self.clients.invoke_json(
            system_prompt=(
                "你是论文关系验证器。"
                "任务是判断候选论文是否是种子论文/数据集/方法的严格后续工作。"
                "严格后续只在证据明确显示候选论文使用、继承、引用、复现、评测或直接扩展种子论文/数据集/方法时成立。"
                "仅主题相似、作者重合、关键词相同，不能判为严格后续，只能判为 related_continuation 或 not_enough_evidence。"
                "只能基于输入的 seed_papers、candidate_paper 和 relationship_evidence 判断，不要使用外部记忆补事实。"
                "relationship_evidence 中 role=candidate 的片段必须出现明确引用/使用/评测/继承/扩展 seed 或其数据集/方法，才可以判 strict_followup 或 direct_use_or_evaluation。"
                "只输出 JSON：classification, strict_followup, relation_type, relationship_strength, reason, confidence, evidence_ids。"
                "classification 只能是 strict_followup, direct_use_or_evaluation, related_continuation, not_enough_evidence, unrelated。"
                "relationship_strength 只能是 direct, strong_related, not_enough_evidence, unrelated。"
            ),
            human_prompt=json.dumps(
                {
                    "query": contract.clean_query,
                    "targets": contract.targets,
                    "seed_papers": [self._paper_relationship_brief(item) for item in seed_papers[:2]],
                    "candidate_paper": self._paper_relationship_brief(paper),
                    "relationship_evidence": [
                        {
                            "doc_id": item.doc_id,
                            "paper_id": item.paper_id,
                            "role": "candidate" if item.paper_id == paper.paper_id else "seed",
                            "title": item.title,
                            "page": item.page,
                            "block_type": item.block_type,
                            "snippet": item.snippet[:900],
                        }
                        for item in relationship_evidence
                    ],
                },
                ensure_ascii=False,
            ),
            fallback={},
        )
        if not isinstance(payload, dict) or not payload:
            return {}
        classification = str(payload.get("classification", "") or "").strip()
        strength = str(payload.get("relationship_strength", "") or "").strip()
        if strength not in {"direct", "strong_related", "not_enough_evidence", "unrelated"}:
            if classification in {"strict_followup", "direct_use_or_evaluation"}:
                strength = "direct"
            elif classification == "related_continuation":
                strength = "strong_related"
            elif classification == "unrelated":
                strength = "unrelated"
            else:
                strength = "not_enough_evidence"
        strict = bool(payload.get("strict_followup", False)) and strength == "direct"
        relation_type = str(payload.get("relation_type", "") or "").strip()
        if not relation_type:
            relation_type = "严格后续/直接使用证据" if strict else ("强相关延续候选" if strength == "strong_related" else "证据不足")
        reason = " ".join(str(payload.get("reason", "") or "").split())
        if not reason:
            reason = "关系验证器没有找到足够明确的严格后续证据。"
        return {
            "relation_type": relation_type,
            "reason": reason,
            "confidence": self._coerce_confidence(payload.get("confidence", 0.68)),
            "relationship_strength": strength,
            "strict_followup": strict,
            "classification": classification,
            "evidence_ids": self._relationship_evidence_ids_from_payload(
                payload=payload,
                relationship_evidence=relationship_evidence,
            ),
        }

    @staticmethod
    def _relationship_evidence_ids_from_payload(
        *,
        payload: dict[str, Any],
        relationship_evidence: list[EvidenceBlock],
    ) -> list[str]:
        available = {item.doc_id for item in relationship_evidence}
        raw_ids = payload.get("evidence_ids", [])
        selected: list[str] = []
        if isinstance(raw_ids, list):
            selected = [str(item).strip() for item in raw_ids if str(item).strip() in available]
        if selected:
            return selected[:6]
        return [item.doc_id for item in relationship_evidence[:4]]

    def _paper_relationship_brief(self, paper: CandidatePaper) -> dict[str, Any]:
        return {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "year": paper.year,
            "authors": str(paper.metadata.get("authors", "")),
            "aliases": str(paper.metadata.get("aliases", "")),
            "summary": self._paper_summary_text(paper.paper_id),
            "paper_card_text": str(paper.metadata.get("paper_card_text", ""))[:1800],
            "tags": str(paper.metadata.get("tags", "")),
        }

    def _rank_followup_candidates_fallback(
        self,
        *,
        contract: QueryContract,
        seed_papers: list[CandidatePaper],
        candidates: list[CandidatePaper],
    ) -> list[dict[str, Any]]:
        seed_keywords = self._paper_keyword_set(seed_papers)
        seed_author_tokens = self._paper_author_tokens(seed_papers)
        target_text = " ".join(contract.targets)
        seed_year = min((safe_year(item.year) for item in seed_papers), default=9999)
        seed_ids = {item.paper_id for item in seed_papers}
        scored: list[tuple[float, CandidatePaper, dict[str, Any]]] = []
        for paper in self._filter_followup_candidates(contract=contract, candidates=candidates):
            if paper.paper_id in seed_ids:
                continue
            score = paper.score
            summary = self._paper_summary_text(paper.paper_id)
            haystack = f"{paper.title}\n{summary}\n{paper.metadata.get('paper_card_text', '')}"
            if target_text and matches_target(haystack.lower(), target_text.lower()):
                score += 1.2
            if seed_year < 9999:
                year = safe_year(paper.year)
                if year >= seed_year:
                    score += 0.4 + min(0.5, max(0, year - seed_year) * 0.1)
            overlap = len(seed_keywords & self._paper_keyword_set([paper]))
            if overlap:
                score += min(1.2, overlap * 0.18)
            author_overlap = len(seed_author_tokens & self._paper_author_tokens([paper]))
            if author_overlap:
                score += min(0.8, author_overlap * 0.25)
            if has_followup_soft_relation_signal(haystack):
                score += 0.35
            score += followup_relevance_score(haystack)
            assessment = self._followup_relationship_assessment(contract=contract, seed_papers=seed_papers, paper=paper)
            if assessment["score"] < 0.3:
                continue
            score += float(assessment["score"])
            scored.append((score, paper, assessment))
        ranked = [
            (paper, assessment)
            for _, paper, assessment in sorted(scored, key=lambda item: (-item[0], safe_year(item[1].year), item[1].title))
        ]
        results: list[dict[str, Any]] = []
        for paper, assessment in ranked[:10]:
            results.append(
                {
                    "paper": paper,
                    "relation_type": str(assessment["relation_type"]),
                    "reason": str(assessment["reason"]),
                    "confidence": float(assessment["confidence"]),
                    "relationship_strength": str(assessment["strength"]),
                }
            )
        return results

    def _followup_relationship_assessment(
        self,
        *,
        contract: QueryContract,
        seed_papers: list[CandidatePaper],
        paper: CandidatePaper,
    ) -> dict[str, Any]:
        target_aliases = self._followup_target_aliases(contract=contract, seed_papers=seed_papers)
        seed_keywords = self._paper_keyword_set(seed_papers)
        candidate_keywords = self._paper_keyword_set([paper])
        seed_phrases: set[str] = set()
        for seed in seed_papers:
            seed_text = f"{seed.title}\n{self._paper_summary_text(seed.paper_id)}\n{seed.metadata.get('paper_card_text', '')}"
            seed_phrases.update(self._extract_followup_keyphrases(seed_text))
        seed_author_tokens = self._paper_author_tokens(seed_papers)
        summary = self._paper_summary_text(paper.paper_id)
        haystack = f"{paper.title}\n{summary}\n{paper.metadata.get('paper_card_text', '')}\n{paper.metadata.get('abstract_note', '')}"
        lowered = haystack.lower()
        score = 0.0
        explicit_direct_signals: list[str] = []
        support_signals: list[str] = []
        target_seen = ""
        for alias in target_aliases:
            if alias and matches_target(haystack, alias):
                target_seen = alias
                if target_relation_cue_near_text(text=haystack, target=alias):
                    score += 3.2
                    explicit_direct_signals.append(f"候选摘要/元数据明确提到并使用、评测或扩展 {alias}")
                else:
                    score += 1.2
                    support_signals.append(f"候选摘要/元数据提到 {alias}")
                break
        candidate_phrases = set(self._extract_followup_keyphrases(haystack))
        phrase_overlap = {
            phrase
            for phrase in seed_phrases & candidate_phrases
            if " " in phrase and phrase not in {"large language models"}
        }
        if phrase_overlap:
            score += min(1.2, len(phrase_overlap) * 0.35)
            support_signals.append("共享研究线索：" + "、".join(sorted(phrase_overlap)[:4]))
        overlap = seed_keywords & candidate_keywords
        displayable_topic_terms = {
            "behavioral",
            "signal",
            "signals",
            "persona",
            "decoding",
            "transfer",
            "transferable",
            "conditioned",
            "generation",
            "profile",
            "profiles",
            "hypothesis",
            "preference-inference",
            "user-level",
        }
        topical_overlap = {
            token
            for token in overlap
            if token in displayable_topic_terms
        }
        if topical_overlap and not phrase_overlap:
            score += min(1.0, len(topical_overlap) * 0.16)
            support_signals.append("共享部分主题词：" + "、".join(sorted(topical_overlap)[:4]))
        author_overlap = seed_author_tokens & self._paper_author_tokens([paper])
        if author_overlap:
            score += min(1.0, len(author_overlap) * 0.35)
            support_signals.append("存在作者重合")
        if has_followup_support_relation_signal(lowered):
            score += 0.45
            support_signals.append("包含扩展、使用、评测或迁移类关系词")
        if has_followup_domain_signal(lowered):
            score += 0.35
            support_signals.append("主题属于 personalized preference / alignment 相邻方向")
        if explicit_direct_signals:
            strength = "direct"
            relation_type = self._infer_followup_relation_type(paper, strict=True)
            reason_bits = explicit_direct_signals + support_signals[:2]
            confidence = 0.86
        elif score >= 1.4 and (target_seen or support_signals):
            strength = "strong_related"
            relation_type = "强相关延续候选"
            reason_bits = support_signals
            confidence = 0.66
        else:
            strength = "weak_related"
            relation_type = "同主题待确认候选"
            reason_bits = support_signals
            confidence = 0.48
        reason = "；".join(dict.fromkeys(reason_bits)) if reason_bits else self._followup_reason_fallback(seed_papers=seed_papers, paper=paper)
        if strength != "direct" and reason:
            reason = f"{reason}；目前证据不足以确认它严格继承、引用或使用种子论文/数据集。"
        return {
            "score": score,
            "strength": strength,
            "relation_type": relation_type,
            "reason": reason,
            "confidence": confidence,
        }

    def _followup_target_aliases(self, *, contract: QueryContract, seed_papers: list[CandidatePaper]) -> list[str]:
        aliases: list[str] = []
        for target in contract.targets:
            target = str(target or "").strip()
            if target:
                aliases.append(target)
        for paper in seed_papers[:2]:
            raw_aliases = str(paper.metadata.get("aliases", ""))
            for alias in re.split(r"\|\||[,;/]", raw_aliases):
                alias = alias.strip()
                if alias and len(alias) <= 48:
                    aliases.append(alias)
            anchor = self._paper_anchor_text(paper)
            if anchor and len(anchor) <= 48:
                aliases.append(anchor)
        normalized: list[str] = []
        seen: set[str] = set()
        for alias in aliases:
            key = alias.lower()
            if key and key not in seen:
                seen.add(key)
                normalized.append(alias)
        return normalized

    def _filter_followup_candidates(self, *, contract: QueryContract, candidates: list[CandidatePaper]) -> list[CandidatePaper]:
        if not candidates:
            return []
        target = contract.targets[0].lower() if contract.targets else ""
        filtered: list[CandidatePaper] = []
        for item in candidates:
            title_lower = item.title.lower()
            card_text = str(item.metadata.get("paper_card_text", "")).lower()
            summary_lower = str(item.metadata.get("generated_summary", "") or self._paper_summary_text(item.paper_id)).lower()
            if (
                target
                and target not in title_lower
                and target not in summary_lower
                and target not in str(item.metadata.get("abstract_note", "")).lower()
                and target not in card_text
                and not has_followup_domain_signal(title_lower + "\n" + card_text + "\n" + summary_lower)
            ):
                continue
            filtered.append(item)
        return filtered or candidates[: min(8, len(candidates))]

    @staticmethod
    def _merge_followup_rankings(
        *,
        primary: list[dict[str, Any]],
        secondary: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source in (primary, secondary):
            for item in source:
                paper = item.get("paper")
                paper_id = getattr(paper, "paper_id", "")
                if not paper_id or paper_id in seen:
                    continue
                seen.add(paper_id)
                merged.append(item)
        return merged

    def _followup_seed_score(
        self,
        *,
        contract: QueryContract,
        paper: CandidatePaper,
        session: SessionContext,
    ) -> float:
        score = paper.score
        summary = self._paper_summary_text(paper.paper_id)
        haystack = f"{paper.title}\n{summary}\n{paper.metadata.get('paper_card_text', '')}".lower()
        if paper.title in session.effective_active_research().titles:
            score += 2.5
        for target in contract.targets:
            if target and matches_target(haystack, target.lower()):
                score += 1.1
        if has_followup_seed_intro_signal(haystack):
            score += 1.2
        year = safe_year(paper.year)
        if year < 9999:
            score += max(0.0, (2100 - year) / 1000.0)
        return score

    @staticmethod
    def _paper_anchor_text(paper: CandidatePaper) -> str:
        title = str(paper.title or "").strip()
        if not title:
            return ""
        for separator in [":", " - ", " — ", " – "]:
            if separator in title:
                return title.split(separator, 1)[0].strip()
        return title

    def _followup_expansion_terms(self, paper: CandidatePaper) -> str:
        text = f"{paper.title}\n{self._paper_summary_text(paper.paper_id)}\n{paper.metadata.get('paper_card_text', '')}".lower()
        terms = self._extract_followup_keyphrases(text)
        if has_followup_domain_signal(text):
            terms.extend(["follow-up", "extension", "downstream", "benchmark", "transfer", "personalization", "preference"])
        return " ".join(dict.fromkeys(item for item in terms if item))[:600]

    @staticmethod
    def _extract_followup_keyphrases(text: str) -> list[str]:
        lowered = str(text or "").lower()
        phrase_bank = [
            "user-level alignment",
            "personalized preference",
            "personalized alignment",
            "preference inference",
            "conditioned generation",
            "transferable personalization",
            "modular personalization",
            "user preference",
            "preference summary",
            "personalization",
            "alignment",
            "benchmark",
            "dataset",
        ]
        phrases = [phrase for phrase in phrase_bank if phrase in lowered]
        title_like = re.sub(r"[^a-z0-9\s-]", " ", lowered)
        words = [
            word
            for word in title_like.split()
            if len(word) >= 5 and word not in {"large", "language", "models", "paper", "using", "through", "across"}
        ]
        frequent: list[str] = []
        for word in words:
            if words.count(word) >= 2 and word not in frequent:
                frequent.append(word)
        return [*phrases, *frequent[:8]]

    def _paper_brief(self, paper: CandidatePaper) -> dict[str, Any]:
        return {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "year": paper.year,
            "authors": str(paper.metadata.get("authors", "")),
            "aliases": str(paper.metadata.get("aliases", "")),
            "summary": self._paper_summary_text(paper.paper_id),
        }

    def _paper_keyword_set(self, papers: list[CandidatePaper]) -> set[str]:
        keywords: set[str] = set()
        stopwords = {
            "that",
            "with",
            "from",
            "this",
            "their",
            "into",
            "through",
            "using",
            "large",
            "language",
            "models",
            "model",
            "paper",
            "across",
            "approach",
            "approaches",
            "average",
            "different",
            "diverse",
            "demonstrate",
            "demonstrates",
            "method",
            "methods",
            "task",
            "tasks",
            "result",
            "results",
            "performance",
            "application",
            "applications",
        }
        for paper in papers:
            text = f"{paper.title} {self._paper_summary_text(paper.paper_id)}"
            for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", text.lower()):
                if token.endswith("ies") and len(token) > 5:
                    token = token[:-3] + "y"
                elif token.endswith("s") and len(token) > 6:
                    token = token[:-1]
                if token not in stopwords:
                    keywords.add(token)
        return keywords

    @staticmethod
    def _paper_author_tokens(papers: list[CandidatePaper]) -> set[str]:
        tokens: set[str] = set()
        for paper in papers:
            authors = str(paper.metadata.get("authors", ""))
            for token in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", authors.lower()):
                if token not in {"and", "et", "al"}:
                    tokens.add(token)
        return tokens

    def _followup_reason_fallback(self, *, seed_papers: list[CandidatePaper], paper: CandidatePaper) -> str:
        seed_titles = ", ".join(item.title for item in seed_papers[:1])
        summary = " ".join(self._paper_summary_text(paper.paper_id).split())
        if len(summary) > 120:
            summary = summary[:117].rstrip() + "..."
        if seed_titles:
            return f"它延续了《{seed_titles}》相关主题，重点在于：{summary or '与该方向直接相关。'}"
        return summary or "与当前研究方向直接相关。"

    def _infer_followup_relation_type(self, paper: CandidatePaper, *, strict: bool = False) -> str:
        summary = self._paper_summary_text(paper.paper_id).lower()
        if strict and any(token in summary for token in ["uses", "using", "evaluate", "evaluates", "benchmark", "dataset"]):
            return "直接使用/评测证据"
        if strict:
            return "直接后续/扩展证据"
        if any(token in summary for token in ["dataset", "benchmark", "evaluation"]):
            return "dataset/benchmark continuation"
        if any(token in summary for token in ["transfer", "cross-task", "cross model"]):
            return "transfer extension"
        if any(token in summary for token in ["reasoning", "behavioral", "preference inference"]):
            return "method/model extension"
        return "related continuation"

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
        pdf_path = str(file_path or "").strip()
        if not pdf_path or page <= 0:
            return ""
        cache_key = (pdf_path, page)
        cached = self._rendered_page_data_url_cache.get(cache_key)
        if cached is not None:
            return cached
        source = Path(pdf_path)
        if not source.exists():
            return ""
        try:
            with TemporaryDirectory(prefix="zprag_v4_fig_") as temp_dir:
                output_prefix = Path(temp_dir) / "page"
                command = [
                    "pdftoppm",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-singlefile",
                    "-png",
                    "-r",
                    str(max(72, int(self.settings.pdf_render_dpi))),
                    str(source),
                    str(output_prefix),
                ]
                if not _subprocess_command_allowed(command):
                    logger.warning("blocked non-whitelisted subprocess command: %s", command[0] if command else "")
                    return ""
                subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=max(5.0, float(self.settings.figure_vlm_timeout_seconds)),
                )
                image_path = output_prefix.with_suffix(".png")
                if not image_path.exists():
                    return ""
                encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
                data_url = f"data:image/png;base64,{encoded}"
                self._rendered_page_data_url_cache[cache_key] = data_url
                return data_url
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to render figure page image: file=%s page=%s err=%s", pdf_path, page, exc)
            return ""

    def _clarification_question(self, contract: QueryContract, session: SessionContext) -> str:
        ambiguity_options = ambiguity_options_from_notes(contract.notes)
        if (
            not ambiguity_options
            and session.pending_clarification_type == "ambiguity"
            and session.pending_clarification_options
            and (
                not contract.targets
                or not session.pending_clarification_target
                or normalize_lookup_text(session.pending_clarification_target)
                in {normalize_lookup_text(target) for target in contract.targets}
            )
        ):
            ambiguity_options = list(session.pending_clarification_options)
        if ambiguity_options:
            target = contract.targets[0] if contract.targets else "这个缩写"
            ambiguity_options = self._normalize_clarification_options(
                ambiguity_options,
                contract=contract,
                target=target,
                kind="acronym_meaning",
                source="clarification_question",
            )
            lines = [f"`{target}` 在本地论文库里有多个可能含义，我不应该继续猜。你想问哪一个？"]
            for index, option in enumerate(ambiguity_options, start=1):
                display_label = str(option.get("display_label", "") or "").strip()
                base_label = str(option.get("label", "") or option.get("meaning", "") or target).strip()
                meaning = f"{display_label}：{base_label}" if display_label and base_label else (display_label or base_label)
                title = str(option.get("display_title", "") or option.get("title", "")).strip()
                year = str(option.get("year", "")).strip()
                suffix = f"（{year}）" if year else ""
                reason = str(option.get("display_reason", "") or "").strip()
                reason_suffix = f"：{reason}" if reason else ""
                lines.append(f"{index}. {meaning}，见《{title}》{suffix}{reason_suffix}")
            return "\n".join(lines)
        requested = {str(item) for item in contract.requested_fields}
        query_lower = str(contract.clean_query or "").lower()
        targets = [str(item).strip() for item in contract.targets if str(item).strip()]
        if "formula" in requested and is_negative_correction_query(query_lower):
            target_text = " / ".join(targets) if targets else "当前目标"
            return (
                f"你说得对，上一条候选公式不能直接当作 `{target_text}` 的公式。"
                "我这边需要重新定位目标含义和对应论文证据；如果本地 PDF 里只有文字说明而没有公式，我会明确说未找到，而不是继续套用不匹配的公式。"
            )
        if "formula" in requested and targets:
            target_text = " / ".join(targets)
            return (
                f"我还不能确认 `{target_text}` 的目标函数或公式。"
                "请指定它对应的论文、方法全称或上下文；如果本地 PDF 里没有明确公式，我会直接说明未找到。"
            )
        if self.clients.chat is not None:
            response_text = self.clients.invoke_text(
                system_prompt=(
                    "你是论文研究助手的研究澄清问题生成器。"
                    "请根据当前 query_contract 和会话上下文，生成一句自然、具体、不生硬的中文回复。"
                    "如果当前像是在质疑上一轮回答，就先承认需要重新核对，再给出 1-2 个具体追问方向。"
                    "如果当前缺的是目标实体或论文来源，也要直接点明缺口，但不要只说“请明确你的问题”。"
                ),
                human_prompt=json.dumps(
                    {
                        "query": contract.clean_query,
                        "interaction_mode": contract.interaction_mode,
                        "continuation_mode": contract.continuation_mode,
                        "targets": contract.targets,
                        "requested_fields": contract.requested_fields,
                        "required_modalities": contract.required_modalities,
                        "answer_slots": contract_answer_slots(contract),
                        "conversation_context": self._session_conversation_context(session),
                        "active_research_context": session.active_research_context_payload(),
                        "recent_turns": [
                            {
                                "query": turn.query,
                                "targets": turn.targets,
                                "requested_fields": turn.requested_fields,
                            }
                            for turn in session.turns[-2:]
                        ],
                    },
                    ensure_ascii=False,
                ),
                fallback="",
            ).strip()
            if response_text:
                return response_text
        if contract.continuation_mode == "followup" and not session.effective_active_research().targets:
            return "我需要确认你是在延续上一轮的哪篇论文或哪个主题。"
        goals = research_plan_goals(contract)
        if contract.targets and goals & {"definition", "entity_type", "mechanism", "figure_conclusion", "answer", "general_answer"}:
            return f"当前语料里还没有稳定定位到与 `{contract.targets[0]}` 直接相关的证据。你可以指定论文、上下文，或换一种问法再试一次。"
        return "我需要更多上下文来确定你当前要继续的研究任务。"
