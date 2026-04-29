from __future__ import annotations

import json
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

JSON_PLANNER_SYSTEM_PROMPT = (
    "你是论文助手的工具循环控制器。"
    "请只选择当前最有用的一小组工具，不要回答问题。"
    "工具能力和边界以 available_tools 的 description 为唯一依据；不要使用隐藏的 relation->固定流水线规则。"
    "如果现有记忆或证据已经足够，可以直接选择 compose；如果缺关键槽位，选择 ask_human。"
    "只输出 JSON：thought, actions, stop_conditions。"
    "actions 只能从 available_tools 的 name 中选择。"
)

TOOL_CALL_PLANNER_SYSTEM_PROMPT = (
    "你是论文助手的工具选择器。"
    "你不能直接回答用户，只能通过 tool calls 选择下一步工具。"
    "工具描述是唯一的能力说明；不要假设固定流水线。"
    "每次根据 intent、上下文和已有 observation 决定：读记忆、搜本地语料、搜外部、请求用户澄清，或 compose。"
    "只返回 tool calls，不要输出普通回答。"
)

NEXT_ACTION_SYSTEM_PROMPT = (
    "你是 observation-driven 工具循环的下一步选择器。"
    "根据当前 intent、已执行工具和 state 摘要，只选择一个下一步工具。"
    "如果已经足够回答，选择 compose；如果必须由用户消歧，选择 ask_human。"
    "不要输出普通回答。"
)


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


def planner_prompt_with_context(*, system_prompt: str, context_json: str) -> str:
    return system_prompt + "\n\n以下非语言上下文只用于工具选择，不是用户新问题：\n" + context_json


def planner_context_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def planner_messages_with_user(
    *,
    conversation_messages: list[dict[str, str]],
    contract: QueryContract,
) -> list[dict[str, str]]:
    return [*conversation_messages, {"role": "user", "content": contract.clean_query}]


def json_planner_system_prompt(context_payload: dict[str, Any]) -> str:
    return planner_prompt_with_context(
        system_prompt=JSON_PLANNER_SYSTEM_PROMPT,
        context_json=planner_context_json(context_payload),
    )


def json_planner_human_prompt(
    *,
    contract: QueryContract,
    conversation_context: dict[str, Any],
    context_payload: dict[str, Any],
) -> str:
    return planner_context_json(
        {
            "query": contract.clean_query,
            "conversation_context": conversation_context,
            **context_payload,
        }
    )


def tool_call_planner_system_prompt(context_payload: dict[str, Any]) -> str:
    return planner_prompt_with_context(
        system_prompt=TOOL_CALL_PLANNER_SYSTEM_PROMPT,
        context_json=planner_context_json(context_payload),
    )


def tool_call_planner_human_prompt(
    *,
    contract: QueryContract,
    conversation_context: dict[str, Any],
    context_payload: dict[str, Any],
) -> str:
    return planner_context_json(
        {
            "query": contract.clean_query,
            "conversation_context": conversation_context,
            **context_payload,
        }
    )


def next_action_human_prompt(
    *,
    contract: QueryContract,
    state: dict[str, Any],
    executed_actions: list[str],
    conversation_context: dict[str, Any],
) -> str:
    return planner_context_json(
        {
            "query": contract.clean_query,
            "intent": planner_intent_payload(contract),
            "targets": contract.targets,
            "notes": contract.notes,
            "executed_actions": executed_actions,
            "state_summary": planner_state_summary(state),
            "conversation_context": conversation_context,
        }
    )


def first_unexecuted_planned_action(
    *,
    payload: dict[str, Any],
    allowed_tools: set[str],
    executed_actions: list[str],
) -> str | None:
    actions = normalize_plan_actions(actions=payload.get("actions", []), allowed=allowed_tools)
    executed = set(executed_actions)
    for action in actions:
        if action not in executed:
            return action
    return None


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
