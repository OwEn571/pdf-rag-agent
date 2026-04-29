from __future__ import annotations

from typing import Any, Callable

from app.domain.models import QueryContract
from app.services.agent_tools import (
    agent_tool_manifest,
    conversation_tool_sequence,
    normalize_plan_actions,
    research_tool_sequence,
)
from app.services.confidence import confidence_from_contract, should_ask_human


NegativeCorrectionFn = Callable[[str], bool]


def planner_state_summary(state: dict[str, Any]) -> dict[str, Any]:
    verification = state.get("verification")
    verification_payload: dict[str, Any] | None = None
    if verification is not None and hasattr(verification, "model_dump"):
        verification_payload = verification.model_dump()
    return {
        "candidate_papers": len(state.get("candidate_papers", []) or []),
        "screened_papers": len(state.get("screened_papers", []) or []),
        "evidence": len(state.get("evidence", []) or []),
        "web_evidence": len(state.get("web_evidence", []) or []),
        "claims": len(state.get("claims", []) or []),
        "has_answer": bool(state.get("answer")),
        "verification": verification_payload,
    }


def planner_intent_payload(contract: QueryContract) -> dict[str, Any]:
    notes = [str(item) for item in contract.notes]
    answer_slots = [
        str(item).strip()
        for item in list(getattr(contract, "answer_slots", []) or [])
        if str(item).strip()
    ]
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
        "research" if contract.interaction_mode == "research" else "smalltalk",
    )
    ambiguous_slots = [
        note.split("=", 1)[1]
        for note in notes
        if note.startswith("ambiguous_slot=") and "=" in note
    ]
    confidence = next(
        (
            note.split("=", 1)[1]
            for note in notes
            if note.startswith("intent_confidence=") and "=" in note
        ),
        "",
    )
    return {
        "kind": intent_kind,
        "confidence": confidence,
        "ambiguous_slots": ambiguous_slots,
        "interaction_mode": contract.interaction_mode,
        "continuation_mode": contract.continuation_mode,
        "requested_fields": contract.requested_fields,
        "required_modalities": contract.required_modalities,
        "answer_shape": contract.answer_shape,
        "answer_slots": answer_slots,
        "allow_web_search": contract.allow_web_search,
    }


def planner_context_payload(
    *,
    contract: QueryContract,
    active_research_context: dict[str, Any],
    use_web_search: bool,
    include_available_tools: bool,
) -> dict[str, Any]:
    payload = {
        "intent": planner_intent_payload(contract),
        "targets": contract.targets,
        "notes": contract.notes,
        "active_research_context": active_research_context,
        "web_enabled": use_web_search,
    }
    if include_available_tools:
        payload["available_tools"] = agent_tool_manifest()
    return payload


def fallback_plan(
    *,
    contract: QueryContract,
    use_web_search: bool,
    settings: Any,
    is_negative_correction_query: NegativeCorrectionFn,
) -> dict[str, Any]:
    if should_fallback_to_human(contract=contract, settings=settings):
        fallback_actions = ["ask_human"]
    elif contract.interaction_mode == "conversation":
        fallback_actions = conversation_tool_sequence(relation=contract.relation, planned_actions=[])
    else:
        fallback_actions = research_tool_sequence(
            planned_actions=[],
            use_web_search=use_web_search,
            needs_reflection="exclude_previous_focus" in contract.notes
            or is_negative_correction_query(contract.clean_query),
        )
    return {
        "thought": "Use tools through the agent loop, observe the result, then compose or ask for clarification.",
        "actions": fallback_actions,
        "stop_conditions": ["answer_is_grounded", "ambiguity_requires_human_choice"],
    }


def should_fallback_to_human(*, contract: QueryContract, settings: Any) -> bool:
    return should_ask_human(confidence_from_contract(contract), settings)


def normalize_plan_payload(*, payload: Any, fallback: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    allowed = {str(tool.get("name")) for tool in agent_tool_manifest()}
    normalized_actions = normalize_plan_actions(actions=payload.get("actions", []), allowed=allowed)
    if not normalized_actions:
        return None
    return {
        "thought": str(payload.get("thought") or fallback["thought"]),
        "actions": normalized_actions,
        "stop_conditions": (
            payload.get("stop_conditions")
            if isinstance(payload.get("stop_conditions"), list)
            else fallback["stop_conditions"]
        ),
        "tool_call_args": (
            payload.get("tool_call_args")
            if isinstance(payload.get("tool_call_args"), list)
            else []
        ),
    }
