from __future__ import annotations

import json
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.domain.models import QueryContract, SessionContext
from app.services.conversation_intents import compact_conversation_query, protected_conversation_intent, smalltalk_relation_from_slots
from app.services.intent_contract_adapter import (
    research_relation_from_slots,
    research_requirements_from_slots,
)
from app.services.intent_fallback_helpers import non_research_fallback_intent, research_fallback_intent
from app.services.intent_legacy_adapter import legacy_contract_payload_to_intent_payload
from app.services.intent_llm_prompt import intent_router_system_prompt
from app.services.memory_intents import (
    is_pdf_agent_topology_design_query,
    is_short_followup,
    looks_like_memory_reference,
    looks_like_recent_tool_result_reference,
)
from app.services.research_intents import (
    looks_like_metric_value_query,
    looks_like_origin_lookup_query,
    looks_like_summary_results_query,
)
from app.services.query_shaping import fallback_query_targets, fallback_target_aliases

ConversationContextFn = Callable[[SessionContext], dict[str, Any]]
ConversationMessagesFn = Callable[[SessionContext], list[dict[str, str]]]
NormalizeTargetsFn = Callable[[list[str], list[str]], list[str]]


IntentKind = Literal["smalltalk", "meta_library", "research", "memory_op"]
TopicState = Literal["continue", "switch", "new"]
AnswerSlot = Literal[
    "greeting",
    "self_identity",
    "capability",
    "clarify",
    "library_status",
    "library_recommendation",
    "citation_ranking",
    "previous_rationale",
    "comparison",
    "definition",
    "entity_definition",
    "concept_definition",
    "formula",
    "origin",
    "followup_research",
    "paper_summary",
    "metric_value",
    "figure",
    "paper_recommendation",
    "topology_discovery",
    "topology_recommendation",
    "training_component",
    "general_answer",
]

class Intent(BaseModel):
    intent_kind: IntentKind = "research"
    topic_state: TopicState = "new"
    active_topic: str = ""
    needs_local_corpus: bool = True
    needs_web: bool = False
    refers_previous_turn: bool = False
    target_entities: list[str] = Field(default_factory=list)
    target_aliases: list[str] = Field(default_factory=list)
    user_goal: str = ""
    answer_slots: list[AnswerSlot] = Field(default_factory=lambda: ["general_answer"])
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    ambiguous_slots: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("target_entities", "target_aliases", "ambiguous_slots", "notes", mode="before")
    @classmethod
    def _normalize_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("answer_slots", mode="before")
    @classmethod
    def _normalize_slots(cls, value: Any) -> list[str]:
        if value is None:
            return ["general_answer"]
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return ["general_answer"]
        slots = [str(item).strip() for item in value if str(item).strip()]
        return slots or ["general_answer"]

    @model_validator(mode="after")
    def _sync_topic_state_compatibility(self) -> "Intent":
        if self.refers_previous_turn and self.topic_state == "new":
            self.topic_state = "continue"
        if self.topic_state == "continue":
            self.refers_previous_turn = True
        elif self.topic_state in {"switch", "new"}:
            self.refers_previous_turn = False
        return self


class IntentRecognizer:
    """Structured intent layer.

    The model is asked for orthogonal dimensions and slots, not a 22-way
    relation. QueryContract is still produced as a compatibility adapter for
    the existing retriever / solver / composer stack.
    """

    def __init__(
        self,
        *,
        clients: Any,
        conversation_context: ConversationContextFn,
        normalize_targets: NormalizeTargetsFn,
        conversation_messages: ConversationMessagesFn | None = None,
    ) -> None:
        self.clients = clients
        self.conversation_context = conversation_context
        self.conversation_messages = conversation_messages or (lambda session: [])
        self.normalize_targets = normalize_targets

    def contract_for_query(
        self,
        *,
        clean_query: str,
        session: SessionContext,
        extracted_targets: list[str],
    ) -> QueryContract:
        intent = self.recognize(
            clean_query=clean_query,
            session=session,
            extracted_targets=extracted_targets,
        )
        return self.to_query_contract(
            intent=intent,
            clean_query=clean_query,
            session=session,
            extracted_targets=extracted_targets,
        )

    def recognize(
        self,
        *,
        clean_query: str,
        session: SessionContext,
        extracted_targets: list[str],
    ) -> Intent:
        protected = self._protected_local_intent(clean_query, session=session)
        if protected is not None:
            return protected
        llm_intent = self._llm_intent(
            clean_query=clean_query,
            session=session,
            extracted_targets=extracted_targets,
        )
        if llm_intent is not None:
            return llm_intent
        return self._fallback_intent(
            clean_query=clean_query,
            session=session,
            extracted_targets=extracted_targets,
        )

    def to_query_contract(
        self,
        *,
        intent: Intent,
        clean_query: str,
        session: SessionContext,
        extracted_targets: list[str],
    ) -> QueryContract:
        slots = list(dict.fromkeys(intent.answer_slots or ["general_answer"]))
        requested_fields: list[str] = []
        required_modalities: list[str] = []
        relation = "general_question"
        interaction_mode: Literal["conversation", "research"] = "research"
        answer_shape = "narrative"
        precision_requirement: Literal["exact", "high", "normal"] = "high"
        target_aliases = list(dict.fromkeys(intent.target_aliases or []))
        targets = self.normalize_targets(
            list(intent.target_entities or extracted_targets),
            requested_fields,
        )
        active = session.effective_active_research()
        if not targets and intent.topic_state == "continue":
            targets = self.normalize_targets(list(active.targets), requested_fields)
        continuation_mode: Literal["fresh", "followup", "context_switch"] = {
            "continue": "followup",
            "switch": "context_switch",
            "new": "fresh",
        }.get(intent.topic_state, "fresh")  # type: ignore[assignment]
        notes = list(
            dict.fromkeys(
                [
                    "structured_intent",
                    f"intent_kind={intent.intent_kind}",
                    f"intent_confidence={intent.confidence:.2f}",
                    f"topic_state={intent.topic_state}",
                    *(["active_topic=" + intent.active_topic] if intent.active_topic else []),
                    *[f"answer_slot={slot}" for slot in slots],
                    *[f"target_alias={alias}" for alias in target_aliases],
                    *[f"ambiguous_slot={slot}" for slot in intent.ambiguous_slots],
                    *intent.notes,
                ]
            )
        )
        if intent.needs_web:
            notes.append("requires_external_tool")

        if intent.intent_kind == "smalltalk":
            interaction_mode = "conversation"
            relation = smalltalk_relation_from_slots(slots)
            requested_fields = []
            required_modalities = []
            answer_shape = "bullets" if relation == "capability" else "narrative"
            precision_requirement = "normal"
            continuation_mode = "fresh"
            targets = []
        elif intent.intent_kind == "meta_library":
            interaction_mode = "conversation"
            requested_fields = []
            required_modalities = []
            precision_requirement = "exact" if "library_status" in slots else "normal"
            answer_shape = "table" if "citation_ranking" in slots else "bullets"
            if "citation_ranking" in slots:
                relation = "library_citation_ranking"
                requested_fields = ["citation_count_ranking"]
                continuation_mode = "followup" if session.turns else "fresh"
                intent.needs_web = True
                notes.extend(["external_metric", "citation_count_requires_web"])
            elif "library_recommendation" in slots:
                relation = "library_recommendation"
            else:
                relation = "library_status"
        elif intent.intent_kind == "memory_op" and not intent.needs_local_corpus:
            interaction_mode = "conversation"
            required_modalities = []
            precision_requirement = "normal"
            if "comparison" in slots:
                relation = "memory_synthesis"
                requested_fields = ["comparison", "synthesis"]
                answer_shape = "table"
            else:
                relation = "memory_followup"
                requested_fields = ["previous_tool_basis", "answer"]
                answer_shape = "narrative"
            continuation_mode = "followup"
        else:
            relation = research_relation_from_slots(slots=slots, clean_query=clean_query, targets=targets)
            requested_fields, required_modalities, answer_shape, precision_requirement = research_requirements_from_slots(
                slots=slots,
                targets=targets,
                clean_query=clean_query,
            )
            if intent.intent_kind == "memory_op":
                continuation_mode = "followup"
                notes.append("memory_resolved_research")

        needs_clarification = intent.intent_kind != "smalltalk" and (
            intent.confidence < 0.6 or bool(intent.ambiguous_slots)
        )
        if needs_clarification:
            clarification_notes = ["intent_needs_clarification", *intent.ambiguous_slots]
            if intent.confidence < 0.6:
                clarification_notes.append("low_intent_confidence")
            return QueryContract(
                clean_query=clean_query,
                interaction_mode="conversation",
                relation="clarify_user_intent",
                targets=targets,
                answer_slots=slots,
                requested_fields=[],
                required_modalities=[],
                answer_shape="narrative",
                precision_requirement="normal",
                continuation_mode="fresh",
                allow_web_search=False,
                notes=list(dict.fromkeys([*notes, *clarification_notes])),
            )

        return QueryContract(
            clean_query=clean_query,
            interaction_mode=interaction_mode,
            relation=relation,
            targets=targets,
            answer_slots=slots,
            requested_fields=requested_fields,
            required_modalities=required_modalities,
            answer_shape=answer_shape,
            precision_requirement=precision_requirement,
            continuation_mode=continuation_mode,
            allow_web_search=bool(intent.needs_web),
            notes=list(dict.fromkeys(notes)),
        )

    def _llm_intent(
        self,
        *,
        clean_query: str,
        session: SessionContext,
        extracted_targets: list[str],
    ) -> Intent | None:
        if getattr(self.clients, "chat", None) is None:
            return None
        payload = {
            "active_research_context": session.active_research_context_payload(),
            "last_relation": session.last_relation,
            "extracted_targets": extracted_targets,
            "schema": Intent.model_json_schema(),
        }
        context_json = json.dumps(payload, ensure_ascii=False)
        system_prompt = intent_router_system_prompt(context_json)
        invoke_json_messages = getattr(self.clients, "invoke_json_messages", None)
        try:
            if callable(invoke_json_messages):
                raw = invoke_json_messages(
                    system_prompt=system_prompt,
                    messages=[
                        *self.conversation_messages(session),
                        {"role": "user", "content": clean_query},
                    ],
                    fallback={},
                )
            else:
                raw = self.clients.invoke_json(
                    system_prompt=system_prompt,
                    human_prompt=(
                        "用户原文：\n"
                        f"{clean_query}\n\n"
                        "非语言上下文 JSON：\n"
                        f"{context_json}"
                    ),
                    fallback={},
                )
        except Exception:  # noqa: BLE001
            if not callable(invoke_json_messages):
                return None
            try:
                raw = invoke_json_messages(
                    system_prompt=system_prompt,
                    messages=[
                        *self.conversation_messages(session),
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "current_query": clean_query,
                                    "conversation_context": self.conversation_context(session),
                                    **payload,
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    fallback={},
                )
            except Exception:  # noqa: BLE001
                return None
        if not isinstance(raw, dict) or not raw:
            return None
        if "intent_kind" not in raw and "relation" in raw:
            return self._intent_from_legacy_contract_payload(raw, clean_query=clean_query)
        intent = self._intent_from_payload(raw)
        if intent is not None:
            return intent
        return self._intent_from_legacy_contract_payload(raw, clean_query=clean_query)

    @staticmethod
    def _intent_from_payload(payload: dict[str, Any]) -> Intent | None:
        try:
            return Intent.model_validate(payload)
        except ValidationError:
            return None

    @staticmethod
    def _intent_from_legacy_contract_payload(payload: dict[str, Any], *, clean_query: str = "") -> Intent | None:
        legacy = legacy_contract_payload_to_intent_payload(payload, clean_query=clean_query)
        return Intent(**legacy.as_intent_kwargs()) if legacy is not None else None  # type: ignore[arg-type]

    def _protected_local_intent(self, query: str, *, session: SessionContext | None = None) -> Intent | None:
        protected_conversation = protected_conversation_intent(query)
        if protected_conversation is not None:
            return Intent(
                intent_kind="smalltalk",
                topic_state="new",
                needs_local_corpus=False,
                target_entities=[],
                user_goal=protected_conversation.user_goal,
                answer_slots=protected_conversation.answer_slots,  # type: ignore[arg-type]
                confidence=protected_conversation.confidence,
                ambiguous_slots=protected_conversation.ambiguous_slots,
                notes=protected_conversation.notes,
            )
        normalized = compact_conversation_query(query)
        lowered = " ".join(str(query or "").strip().lower().split())
        if looks_like_origin_lookup_query(query):
            targets = fallback_query_targets(query)
            if targets:
                return Intent(
                    intent_kind="research",
                    topic_state="new",
                    active_topic=f"查找 {targets[0]} 最早提出论文",
                    needs_local_corpus=True,
                    target_entities=targets,
                    target_aliases=fallback_target_aliases(targets),
                    user_goal=query,
                    answer_slots=["origin"],
                    confidence=0.92,
                    notes=["local_protected_origin_lookup"],
                )
        if looks_like_recent_tool_result_reference(query, session=session):
            return Intent(
                intent_kind="memory_op",
                topic_state="continue",
                needs_local_corpus=False,
                refers_previous_turn=True,
                target_entities=[],
                user_goal="回答用户对上一轮工具结果中某一项的追问。",
                answer_slots=["previous_rationale"],
                confidence=0.9,
                notes=["local_protected_tool_result_reference"],
            )
        explicit_targets = fallback_query_targets(query)
        if explicit_targets:
            if looks_like_metric_value_query(query):
                return Intent(
                    intent_kind="research",
                    topic_state="new",
                    active_topic=query,
                    needs_local_corpus=True,
                    target_entities=explicit_targets,
                    target_aliases=fallback_target_aliases(explicit_targets),
                    user_goal=query,
                    answer_slots=["metric_value"],
                    confidence=0.86,
                    notes=["local_protected_explicit_target_metric"],
                )
            if looks_like_summary_results_query(query):
                return Intent(
                    intent_kind="research",
                    topic_state="new",
                    active_topic=query,
                    needs_local_corpus=True,
                    target_entities=explicit_targets,
                    target_aliases=fallback_target_aliases(explicit_targets),
                    user_goal=query,
                    answer_slots=["paper_summary"],
                    confidence=0.86,
                    notes=["local_protected_explicit_target_summary"],
                )
        if is_pdf_agent_topology_design_query(lowered=lowered, compact=normalized):
            return Intent(
                intent_kind="research",
                topic_state="continue" if session is not None and session.turns else "new",
                active_topic="PDF-Agent multi-agent topology design",
                needs_local_corpus=True,
                target_entities=["agent topology"],
                target_aliases=["DAG", "chain", "tree", "mesh", "irregular/random"],
                user_goal="为 PDF-Agent 的 multi-agent 内部协作选择拓扑结构。",
                answer_slots=["topology_recommendation"],
                confidence=0.9,
                notes=["local_protected_pdf_agent_topology_design"],
            )
        return None

    def _fallback_intent(
        self,
        *,
        clean_query: str,
        session: SessionContext,
        extracted_targets: list[str],
    ) -> Intent:
        lowered = " ".join(clean_query.lower().split())
        compact = compact_conversation_query(clean_query)
        targets = list(extracted_targets)
        active = session.effective_active_research()
        refers_previous = looks_like_memory_reference(clean_query) or (
            bool(active.targets) and is_short_followup(clean_query)
        )
        non_research = non_research_fallback_intent(
            clean_query=clean_query,
            lowered=lowered,
            session_has_turns=bool(session.turns),
            active_targets=list(active.targets),
            extracted_targets=targets,
        )
        if non_research is not None:
            return Intent(**non_research.as_intent_kwargs())  # type: ignore[arg-type]

        research_fallback = research_fallback_intent(
            clean_query=clean_query,
            lowered=lowered,
            compact=compact,
            active_relation=active.relation,
            active_targets=list(active.targets),
            extracted_targets=targets,
            refers_previous=refers_previous,
        )
        return Intent(**research_fallback.as_intent_kwargs())  # type: ignore[arg-type]
