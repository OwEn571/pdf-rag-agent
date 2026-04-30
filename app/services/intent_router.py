from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any, Literal

from app.domain.models import QueryContract, SessionContext
from app.services.intent import IntentRecognizer
from app.services.research_intents import normalized_query_text

RouterAction = Literal["answer_directly", "need_corpus_search", "need_web", "need_clarify"]
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
                "answer_style": {"type": "string"},
            },
            "required": ["rationale", "confidence"],
            "additionalProperties": False,
        },
    },
    {
        "name": "need_corpus_search",
        "description": "Use when the user asks about local papers, PDF content, formulas, tables, figures, methods, results, or citations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "targets": {"type": "array", "items": {"type": "string"}},
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
        fallback = RouterDecision(
            action="need_clarify",
            confidence=0.0,
            args={"question": "请补充你想查询的论文或问题目标。", "reason": "router_unavailable"},
            rationale="router_unavailable",
            tags=["router_unavailable"],
        )
        if getattr(self.clients, "chat", None) is None:
            return fallback
        system_prompt = (
            "你是 PDF-RAG 助手的 intent router。"
            "只通过 tool call 表达路由决策，不要直接回答用户。"
            "不要使用固定关键词规则；根据用户话语、会话上下文和工具 description 判断下一步。"
            "如果低置信或目标不清，调用 need_clarify。"
        )
        context_payload = {
            "conversation_context": self.conversation_context(session),
            "active_research_context": session.active_research_context_payload(),
        }
        messages = [
            *self.conversation_messages(session),
            {
                "role": "user",
                "content": query,
            },
        ]
        planner_messages = getattr(self.clients, "invoke_tool_plan_messages", None)
        if callable(planner_messages):
            payload = planner_messages(
                system_prompt=system_prompt + "\n\n上下文：\n" + json.dumps(context_payload, ensure_ascii=False),
                messages=messages,
                tools=ROUTER_TOOLS,
                fallback={},
            )
        else:
            planner = getattr(self.clients, "invoke_tool_plan", None)
            if not callable(planner):
                return fallback
            payload = planner(
                system_prompt=system_prompt,
                human_prompt=json.dumps({"query": query, **context_payload}, ensure_ascii=False),
                tools=ROUTER_TOOLS,
                fallback={},
            )
        return self._decision_from_payload(payload=payload, query=query) or fallback

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
) -> QueryContract | None:
    if "router_unavailable" in decision.tags:
        return None
    base_notes = _router_contract_notes(decision)
    if decision.action == "need_clarify" or decision.confidence < 0.6:
        raw_options = decision.args.get("options", [])
        options = [str(item).strip() for item in raw_options if str(item).strip()] if isinstance(raw_options, list) else []
        notes = [*base_notes, "intent_needs_clarification"]
        if decision.confidence < 0.6:
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
        return QueryContract(
            clean_query=clean_query,
            interaction_mode="conversation",
            relation="general_question",
            targets=[],
            answer_slots=["general_answer"],
            requested_fields=[],
            required_modalities=[],
            answer_shape="narrative",
            precision_requirement="normal",
            continuation_mode="fresh",
            allow_web_search=False,
            notes=list(dict.fromkeys([*base_notes, "intent_kind=smalltalk", "answer_slot=general_answer"])),
        )
    query = str(decision.args.get("query", "") or clean_query).strip() or clean_query
    raw_targets = decision.args.get("targets", [])
    router_targets = [str(item).strip() for item in raw_targets if str(item).strip()] if isinstance(raw_targets, list) else []
    targets = normalize_targets(router_targets or extracted_targets, [])
    lowered, compact = normalized_query_text(query)
    slots = IntentRecognizer._research_slots(
        clean_query=query,
        lowered=lowered,
        compact=compact,
        session=session,
    )
    relation = IntentRecognizer._research_relation(slots=slots, clean_query=query, targets=targets)
    requested_fields, required_modalities, answer_shape, precision_requirement = IntentRecognizer._research_requirements(
        slots=slots,
        targets=targets,
        clean_query=query,
    )
    notes = [
        *base_notes,
        "intent_kind=research",
        "topic_state=new",
        *[f"answer_slot={slot}" for slot in slots],
    ]
    if decision.action == "need_web":
        notes.append("requires_external_tool")
    return QueryContract(
        clean_query=query,
        interaction_mode="research",
        relation=relation,
        targets=targets,
        answer_slots=list(slots),
        requested_fields=requested_fields,
        required_modalities=required_modalities,
        answer_shape=answer_shape,
        precision_requirement=precision_requirement,
        continuation_mode="fresh",
        allow_web_search=decision.action == "need_web",
        notes=list(dict.fromkeys(notes)),
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


def _bounded_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return max(0.0, min(1.0, parsed))


def _decision_tags(*, action: str, args: dict[str, Any]) -> list[str]:
    tags = [action]
    for target in list(args.get("targets", []) or []):
        text = str(target).strip()
        if text:
            tags.append(f"target:{text}")
    return tags
