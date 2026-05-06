from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any, Literal

from app.domain.models import QueryContract, SessionContext
from app.services.intents.contract_adapter import (
    answer_slots_from_relation,
    research_relation_from_slots,
    research_requirements_from_slots,
)
from app.services.planning.query_shaping import query_target_candidates
from app.services.intents.research import normalized_query_text, research_answer_slots

RouterAction = Literal["answer_directly", "need_conversation_tool", "need_corpus_search", "need_web", "need_clarify"]
NormalizeTargetsFn = Callable[[list[str], list[str]], list[str]]


ROUTER_TOOLS: list[dict[str, Any]] = [
    {
        "name": "answer_directly",
        "description": "Use when the user asks ordinary conversation or the answer does not need local PDF/web evidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "answer_style": {
                    "type": "string",
                    "description": "Optional direct-answer subtype such as greeting, self_identity, capability, or general_answer.",
                },
            },
            "required": ["rationale", "confidence"],
            "additionalProperties": False,
        },
    },
    {
        "name": "need_conversation_tool",
        "description": "Use for local conversation tools such as library status, library recommendations, citation ranking, or memory follow-ups.",
        "input_schema": {
            "type": "object",
            "properties": {
                "relation": {"type": "string"},
                "targets": {"type": "array", "items": {"type": "string"}},
                "requested_fields": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "continuation_mode": {"type": "string"},
                "answer_shape": {"type": "string"},
                "precision_requirement": {"type": "string"},
                "notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["relation", "rationale", "confidence"],
            "additionalProperties": False,
        },
    },
    {
        "name": "need_corpus_search",
        "description": "Use when the user asks about local papers, PDF content, formulas, tables, figures, methods, results, or citations. Also suggest the initial tool actions (planned_actions) to skip the separate planning step.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "relation": {"type": "string"},
                "targets": {"type": "array", "items": {"type": "string"}},
                "requested_fields": {"type": "array", "items": {"type": "string"}},
                "required_modalities": {"type": "array", "items": {"type": "string"}},
                "answer_shape": {"type": "string"},
                "precision_requirement": {"type": "string"},
                "continuation_mode": {"type": "string"},
                "notes": {"type": "array", "items": {"type": "string"}},
                "planned_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Initial tool actions for the agent loop, e.g. [\"read_memory\", \"search_corpus\", \"compose\"].",
                },
                "rationale": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["query", "rationale", "confidence"],
            "additionalProperties": False,
        },
    },
    {
        "name": "need_web",
        "description": "Use when the user asks for current, external, or web-only facts that may change over time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "rationale": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["query", "rationale", "confidence"],
            "additionalProperties": False,
        },
    },
    {
        "name": "need_clarify",
        "description": "Use when a required target, paper, or user intent slot cannot be resolved confidently.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["question", "reason", "confidence"],
            "additionalProperties": False,
        },
    },
]


@dataclass(frozen=True, slots=True)
class RouterDecision:
    action: RouterAction
    confidence: float
    args: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    tags: list[str] = field(default_factory=list)

    def notes(self) -> list[str]:
        notes = [f"router_action={self.action}", f"router_confidence={self.confidence:.2f}"]
        for tag in self.tags:
            notes.append(f"router_tag={tag}")
        return notes


class LLMIntentRouter:
    def __init__(
        self,
        *,
        clients: Any,
        conversation_context: Any,
        conversation_messages: Any | None = None,
    ) -> None:
        self.clients = clients
        self.conversation_context = conversation_context
        self.conversation_messages = conversation_messages or (lambda session: [])

    def route(self, *, query: str, session: SessionContext) -> RouterDecision:
        if getattr(self.clients, "chat", None) is None:
            return unavailable_router_decision("router_unavailable")
        system_prompt = (
            "你是 PDF-RAG 助手的 intent router。"
            "只通过 tool call 表达路由决策，不要直接回答用户。"
            "根据用户话语、会话上下文和工具 description 判断下一步。"
            "如果低置信或目标不清，调用 need_clarify。"
            "示例：用户问'DPO 的原始论文是哪篇' → 调用 need_corpus_search，"
            "continuation_mode=context_switch（即使上下文中已有 DPO 讨论，"
            "因为上下文引用可能是综述或二次引用而非原始论文）。"
        )
        context_payload = {
            "conversation_context": self.conversation_context(session),
            "active_research_context": session.active_research_context_payload(),
        }
        messages = [
            *self.conversation_messages(session),
            {"role": "user", "content": query},
        ]
        planner_messages = getattr(self.clients, "invoke_tool_plan_messages", None)
        if callable(planner_messages):
            payload = planner_messages(
                system_prompt=system_prompt,
                messages=messages,
                tools=ROUTER_TOOLS,
                fallback={},
            )
        else:
            planner = getattr(self.clients, "invoke_tool_plan", None)
            if not callable(planner):
                return unavailable_router_decision("router_unavailable")
            payload = planner(
                system_prompt=system_prompt,
                human_prompt=json.dumps({"query": query, **context_payload}, ensure_ascii=False),
                tools=ROUTER_TOOLS,
                fallback={},
            )
        return self._decision_from_payload(payload=payload, query=query) or unavailable_router_decision(
            "router_invalid_payload"
        )

    @staticmethod
    def _decision_from_payload(*, payload: Any, query: str) -> RouterDecision | None:
        if not isinstance(payload, dict):
            return None
        actions = [str(item) for item in list(payload.get("actions", []) or [])]
        if not actions:
            return None
        action = actions[0]
        if action not in {tool["name"] for tool in ROUTER_TOOLS}:
            return None
        args = _args_for_action(payload=payload, action=action)
        if action in {"need_corpus_search", "need_web"} and not str(args.get("query", "") or "").strip():
            args["query"] = query
        confidence = _bounded_confidence(args.get("confidence", 0.0))
        rationale = str(args.get("rationale", "") or args.get("reason", "") or payload.get("thought", "") or "").strip()
        return RouterDecision(
            action=action,  # type: ignore[arg-type]
            confidence=confidence,
            args=args,
            rationale=rationale,
            tags=_decision_tags(action=action, args=args),
        )


def query_contract_from_router_decision(
    *,
    decision: RouterDecision,
    clean_query: str,
    session: SessionContext,
    extracted_targets: list[str],
    normalize_targets: NormalizeTargetsFn,
    confidence_floor: float = 0.6,
) -> QueryContract | None:
    if "router_unavailable" in decision.tags:
        return None
    base_notes = _router_contract_notes(decision)
    low_confidence = decision.confidence < _bounded_confidence(confidence_floor)
    research_recovery_candidate = decision.action in {"need_clarify", "answer_directly"} or low_confidence
    if decision.action == "need_clarify" or low_confidence:
        recovered = _recover_research_slot_contract(
            decision=decision,
            clean_query=clean_query,
            session=session,
            extracted_targets=extracted_targets,
            normalize_targets=normalize_targets,
            base_notes=base_notes,
            confidence_floor=confidence_floor,
        ) if research_recovery_candidate else None
        if recovered is not None:
            return recovered
        raw_options = decision.args.get("options", [])
        options = [str(item).strip() for item in raw_options if str(item).strip()] if isinstance(raw_options, list) else []
        notes = [*base_notes, "intent_needs_clarification"]
        if low_confidence:
            notes.append("low_intent_confidence")
        for option in options[:6]:
            notes.append("clarification_option=" + option)
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
            notes=list(dict.fromkeys(notes)),
        )
    if decision.action == "answer_directly":
        recovered = _recover_research_slot_contract(
            decision=decision,
            clean_query=clean_query,
            session=session,
            extracted_targets=extracted_targets,
            normalize_targets=normalize_targets,
            base_notes=base_notes,
            confidence_floor=confidence_floor,
        )
        if recovered is not None:
            return recovered
        relation, answer_slot = _direct_answer_relation_and_slot(decision.args.get("answer_style"))
        return QueryContract(
            clean_query=clean_query,
            interaction_mode="conversation",
            relation=relation,
            targets=[],
            answer_slots=[answer_slot],
            requested_fields=[],
            required_modalities=[],
            answer_shape="narrative",
            precision_requirement="normal",
            continuation_mode="fresh",
            allow_web_search=False,
            notes=list(dict.fromkeys([*base_notes, "intent_kind=smalltalk", f"answer_slot={answer_slot}"])),
        )
    if decision.action == "need_conversation_tool":
        relation = _conversation_tool_relation(decision.args.get("relation"))
        requested_fields = _string_list(decision.args.get("requested_fields"))
        targets = normalize_targets(_string_list(decision.args.get("targets")), requested_fields)
        answer_slots = answer_slots_from_relation(relation)
        notes = [
            *base_notes,
            "intent_kind=conversation_tool",
            *[f"answer_slot={slot}" for slot in answer_slots],
            *_note_list(decision.args.get("notes")),
        ]
        return QueryContract(
            clean_query=clean_query,
            interaction_mode="conversation",
            relation=relation,
            targets=targets,
            answer_slots=answer_slots,
            requested_fields=requested_fields,
            required_modalities=_string_list(decision.args.get("required_modalities")),
            answer_shape=_string_value(decision.args.get("answer_shape"), default="narrative"),
            precision_requirement=_sanitize_precision_requirement(decision.args.get("precision_requirement")),
            continuation_mode=_sanitize_continuation_mode(decision.args.get("continuation_mode"), default="fresh"),
            allow_web_search=False,
            notes=list(dict.fromkeys(notes)),
        )
    query = str(decision.args.get("query", "") or clean_query).strip() or clean_query
    raw_targets = decision.args.get("targets", [])
    router_targets = [str(item).strip() for item in raw_targets if str(item).strip()] if isinstance(raw_targets, list) else []
    requested_fields = _string_list(decision.args.get("requested_fields"))
    targets = normalize_targets(router_targets or extracted_targets or query_target_candidates(query), requested_fields)
    lowered, compact = normalized_query_text(query)
    slots = research_answer_slots(
        clean_query=query,
        lowered=lowered,
        compact=compact,
        active_relation=session.effective_active_research().relation,
    )
    relation = _research_relation(decision.args.get("relation")) or research_relation_from_slots(slots=slots, clean_query=query, targets=targets)
    answer_slots = _string_list(decision.args.get("answer_slots")) or (answer_slots_from_relation(relation) if decision.args.get("relation") else slots)
    default_fields, default_modalities, default_shape, default_precision = research_requirements_from_slots(
        slots=answer_slots,
        targets=targets,
        clean_query=query,
    )
    requested_fields = requested_fields or default_fields
    required_modalities = _string_list(decision.args.get("required_modalities")) or default_modalities
    answer_shape = _string_value(decision.args.get("answer_shape"), default=default_shape)
    precision_requirement = _sanitize_precision_requirement(decision.args.get("precision_requirement"), default=default_precision)
    notes = [
        *base_notes,
        "intent_kind=research",
        "topic_state=new",
        *[f"answer_slot={slot}" for slot in answer_slots],
        *_note_list(decision.args.get("notes")),
    ]
    if decision.action == "need_web":
        notes.append("requires_external_tool")
    planned = _string_list(decision.args.get("planned_actions"))
    if planned:
        notes.append("router_planned_actions=" + ",".join(planned))
    return QueryContract(
        clean_query=query,
        interaction_mode="research",
        relation=relation,
        targets=targets,
        answer_slots=answer_slots,
        requested_fields=requested_fields,
        required_modalities=required_modalities,
        answer_shape=answer_shape,
        precision_requirement=precision_requirement,
        continuation_mode=_sanitize_continuation_mode(decision.args.get("continuation_mode"), default="fresh"),
        allow_web_search=decision.action == "need_web",
        notes=list(dict.fromkeys(notes)),
    )


def _recover_research_slot_contract(
    *,
    decision: RouterDecision,
    clean_query: str,
    session: SessionContext,
    extracted_targets: list[str],
    normalize_targets: NormalizeTargetsFn,
    base_notes: list[str],
    confidence_floor: float = 0.6,
) -> QueryContract | None:
    query = str(decision.args.get("query", "") or clean_query).strip() or clean_query
    lowered, compact = normalized_query_text(query)
    slots = research_answer_slots(
        clean_query=query,
        lowered=lowered,
        compact=compact,
        active_relation=session.effective_active_research().relation,
    )
    if slots == ["general_answer"]:
        return None
    default_fields, _, _, _ = research_requirements_from_slots(
        slots=slots,
        targets=[],
        clean_query=query,
    )
    router_targets = _string_list(decision.args.get("targets"))
    targets = normalize_targets(router_targets or extracted_targets or query_target_candidates(query), default_fields)
    target_optional_slots = {"paper_recommendation", "topology_recommendation", "topology_discovery"}
    if not targets and not (set(slots) & target_optional_slots):
        return None
    relation = research_relation_from_slots(slots=slots, clean_query=query, targets=targets)
    answer_slots = slots
    requested_fields, required_modalities, answer_shape, precision_requirement = research_requirements_from_slots(
        slots=answer_slots,
        targets=targets,
        clean_query=query,
    )
    active = session.effective_active_research()
    notes = [
        *base_notes,
        "intent_kind=research",
        "topic_state=new",
        "router_recovered_research_slot",
        *[f"answer_slot={slot}" for slot in answer_slots],
    ]
    if decision.action == "need_clarify":
        notes.append("clarify_recovered_research_slot")
        if "origin" in answer_slots:
            notes.append("clarify_recovered_origin_lookup")
        notes.append("clarify_recovered_from_router")
    if decision.action == "answer_directly":
        notes.append("direct_answer_recovered_research_slot")
    if decision.confidence < _bounded_confidence(confidence_floor):
        notes.append("low_intent_confidence")
        notes.append("low_confidence_recovered_research_slot")
    return QueryContract(
        clean_query=query,
        interaction_mode="research",
        relation=relation,
        targets=targets,
        answer_slots=answer_slots,
        requested_fields=requested_fields,
        required_modalities=required_modalities,
        answer_shape=answer_shape,
        precision_requirement=precision_requirement,
        continuation_mode="context_switch" if active.has_content() else "fresh",
        allow_web_search=False,
        notes=list(dict.fromkeys(notes)),
    )


def router_miss_clarification_contract(*, clean_query: str) -> QueryContract:
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
            "legacy_intent_fallback_removed",
            "intent_needs_clarification",
            "low_intent_confidence",
        ],
    )


def unavailable_router_decision(reason: str = "router_unavailable") -> RouterDecision:
    normalized_reason = str(reason or "router_unavailable").strip() or "router_unavailable"
    tags = ["router_unavailable"]
    if normalized_reason != "router_unavailable":
        tags.append(normalized_reason)
    return RouterDecision(
        action="need_clarify",
        confidence=0.0,
        args={"question": "请补充你想查询的论文或问题目标。", "reason": normalized_reason},
        rationale=normalized_reason,
        tags=tags,
    )


def _router_contract_notes(decision: RouterDecision) -> list[str]:
    notes = [
        "structured_intent",
        "llm_tool_router",
        f"intent_confidence={decision.confidence:.2f}",
        *decision.notes(),
    ]
    if decision.rationale:
        notes.append("router_rationale=" + decision.rationale[:180])
    return notes


def _args_for_action(*, payload: dict[str, Any], action: str) -> dict[str, Any]:
    for item in list(payload.get("tool_call_args", []) or []):
        if not isinstance(item, dict) or str(item.get("name", "") or "") != action:
            continue
        args = item.get("args", {})
        return dict(args) if isinstance(args, dict) else {}
    return {}


# Map router answer_style values → (relation, answer_slot).
_DIRECT_ANSWER_STYLE_MAP: dict[frozenset[str], tuple[str, str]] = {
    frozenset({"greeting", "hello", "smalltalk"}): ("greeting", "greeting"),
    frozenset({"self_identity", "identity", "who_are_you"}): ("self_identity", "self_identity"),
    frozenset({"capability", "abilities", "help", "what_can_you_do"}): ("capability", "capability"),
}


def _direct_answer_relation_and_slot(answer_style: Any) -> tuple[str, str]:
    normalized = "_".join(str(answer_style or "").strip().lower().replace("-", "_").split())
    for styles, mapping in _DIRECT_ANSWER_STYLE_MAP.items():
        if normalized in styles:
            return mapping
    return "general_question", "general_answer"


# Known relation types.  The LLM router may suggest any of these; unknown
# values are normalised to the closest safe fallback.  To support a new
# relation, add it here, then ensure answer_slots_from_relation and the
# conversation / research tool registries know how to compose it.
_CONVERSATION_RELATIONS = frozenset({
    "library_status",
    "library_recommendation",
    "library_citation_ranking",
    "memory_followup",
    "memory_synthesis",
    "clarify_user_intent",
    "general_question",
})

_RESEARCH_RELATIONS = frozenset({
    "origin_lookup",
    "formula_lookup",
    "followup_research",
    "figure_question",
    "metric_value_lookup",
    "paper_summary_results",
    "paper_recommendation",
    "topology_recommendation",
    "topology_discovery",
    "entity_definition",
    "concept_definition",
    "general_question",
})


def _conversation_tool_relation(value: Any) -> str:
    relation = str(value or "").strip()
    if relation in _CONVERSATION_RELATIONS:
        return relation
    return "general_question"


def _research_relation(value: Any) -> str:
    relation = str(value or "").strip()
    if relation in _RESEARCH_RELATIONS:
        return relation
    return ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _note_list(value: Any) -> list[str]:
    return _string_list(value)


def _string_value(value: Any, *, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _sanitize_continuation_mode(raw: Any, *, default: str = "fresh") -> str:
    """Map LLM-generated continuation_mode to valid Literal values."""
    text = str(raw or "").strip().lower()
    # Normalize common LLM hallucinations
    if text in {"context_continuation", "context_continue", "contextual"}:
        return "context_switch"
    if text in {"fresh", "followup", "context_switch"}:
        return text
    return default


def _bounded_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return max(0.0, min(1.0, parsed))


_VALID_PRECISION_REQUIREMENTS: frozenset[str] = frozenset({"exact", "high", "normal"})


def _sanitize_precision_requirement(value: Any, default: str = "normal") -> str:
    text = str(value or "").strip()
    if text in _VALID_PRECISION_REQUIREMENTS:
        return text
    lowered = text.lower()
    if any(kw in lowered for kw in ["精确", "exact", "strict", "准确", "确切", "严格", "精准"]):
        return "exact"
    if any(kw in lowered for kw in ["high", "precise", "高精度", "高", "详细", "详尽"]):
        return "high"
    return default


def _decision_tags(*, action: str, args: dict[str, Any]) -> list[str]:
    tags = [action]
    for target in list(args.get("targets", []) or []):
        text = str(target).strip()
        if text:
            tags.append(f"target:{text}")
    return tags
