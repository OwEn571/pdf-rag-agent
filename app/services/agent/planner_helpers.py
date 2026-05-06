from __future__ import annotations

import json
from typing import Any, Callable

from app.domain.models import QueryContract
from app.services.agent.tools import (
    agent_tool_manifest,
    conversation_tool_sequence,
    normalize_plan_actions,
    research_tool_sequence,
)
from app.services.infra.confidence import confidence_from_contract, should_ask_human
from app.services.contracts.context import (
    contract_answer_slots,
    contract_has_note,
    contract_note_value,
    contract_note_values,
    contract_notes,
)


NegativeCorrectionFn = Callable[[str], bool]

JSON_PLANNER_SYSTEM_PROMPT = (
    "你是论文助手的工具循环控制器。"
    "请只选择当前最有用的一小组工具，不要回答问题。"
    "工具能力和边界以 available_tools 的 description 为唯一依据；不要使用隐藏的 relation->固定流水线规则。"
    "需要检索本地论文时，优先选 search_corpus（它已融合 BM25+向量+筛选），不要拆分调用原子检索工具。"
    "如果现有记忆或证据已经足够，可以直接选择 compose；如果缺关键槽位，选择 ask_human。"
    "只输出 JSON：thought, actions, stop_conditions。"
    "actions 只能从 available_tools 的 name 中选择。"
)

TOOL_CALL_PLANNER_SYSTEM_PROMPT = (
    "你是论文助手的工具选择器。"
    "你不能直接回答用户，只能通过 tool calls 选择下一步工具。"
    "工具描述是唯一的能力说明；不要假设固定流水线。"
    "每次根据 intent、上下文和已有 observation 决定：读记忆、搜本地语料、搜外部、请求用户澄清，或 compose。"
    "对于需要检索本地论文的问题，优先使用 search_corpus 作为主要检索工具——它内部已融合 BM25+向量+筛选，不需要单独调用 bm25_search / vector_search / hybrid_search。"
    "只在需要单一路检索策略做对比或精确控制时才使用原子检索工具。"
    "对于已经归一化为 research 且带有可检索对象/答案槽位的问题，即使 notes 里保留了低置信度或 need_clarify 的旧信号，也要先检索或读记忆；只有观察结果仍无法消歧时才 ask_human。"
    "只返回 tool calls，不要输出普通回答。"
)

NEXT_ACTION_SYSTEM_PROMPT = (
    "你是 observation-driven 工具循环的下一步选择器。"
    "根据当前 intent、已执行工具和 state 摘要，只选择一个下一步工具。"
    "如果已经足够回答，选择 compose；如果必须由用户消歧，选择 ask_human。"
    "tool_history 记录了每次工具调用的名字和参数，tool_results 是每次调用的输出摘要。"
    "不要以相同参数重复调用已成功的工具；但可以用不同参数重新检索。"
    "如果 tool_results 显示证据不足，尝试换 query 或换检索方式。"
    "不要输出普通回答。"
)


def planner_state_summary(state: dict[str, Any]) -> dict[str, Any]:
    verification = state.get("verification")
    verification_payload: dict[str, Any] | None = None
    if verification is not None and hasattr(verification, "model_dump"):
        verification_payload = verification.model_dump()
    execution_log = state.get("_execution_log")
    result_previews = state.get("_tool_result_previews")
    return {
        "candidate_papers": len(state.get("candidate_papers", []) or []),
        "screened_papers": len(state.get("screened_papers", []) or []),
        "evidence": len(state.get("evidence", []) or []),
        "web_evidence": len(state.get("web_evidence", []) or []),
        "claims": len(state.get("claims", []) or []),
        "has_answer": bool(state.get("answer")),
        "verification": verification_payload,
        "tool_history": list(execution_log) if isinstance(execution_log, list) else [],
        "tool_results": list(result_previews) if isinstance(result_previews, list) else [],
    }


def planner_intent_payload(contract: QueryContract) -> dict[str, Any]:
    answer_slots = contract_answer_slots(contract)
    intent_kind = contract_note_value(contract, prefix="intent_kind=") or (
        "research" if contract.interaction_mode == "research" else "smalltalk"
    )
    ambiguous_slots = contract_note_values(contract, prefix="ambiguous_slot=")
    confidence = contract_note_value(contract, prefix="intent_confidence=")
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
    available_tools: list[dict[str, Any]] | None = None,
    # P1-4: Compound subtask context for inter-subtask awareness
    compound_subtask_index: int = 0,
    compound_subtask_total: int = 1,
    peer_relations: list[str] | None = None,
    peer_targets: list[str] | None = None,
) -> dict[str, Any]:
    payload = {
        "intent": planner_intent_payload(contract),
        "targets": contract.targets,
        "notes": contract_notes(contract),
        "active_research_context": active_research_context,
        "web_enabled": use_web_search,
    }
    if include_available_tools:
        payload["available_tools"] = available_tools if available_tools is not None else agent_tool_manifest()
    # P1-4: Add compound context so planner knows "I'm subtask 2/3, peers did X"
    if compound_subtask_total > 1:
        payload["compound_context"] = {
            "subtask_index": compound_subtask_index,
            "subtask_total": compound_subtask_total,
            "peer_relations": peer_relations or [],
            "peer_targets": peer_targets or [],
        }
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
            "notes": contract_notes(contract),
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


def _all_unexecuted_planned_actions(
    *,
    payload: dict[str, Any],
    allowed_tools: set[str],
    executed_actions: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    """Return unexecuted (action_name, arguments) pairs from the LLM payload."""
    from app.services.tools.registry_helpers import tool_inputs_by_name

    tool_inputs = tool_inputs_by_name(payload)
    actions = normalize_plan_actions(actions=payload.get("actions", []), allowed=allowed_tools)
    executed = set(executed_actions)
    result: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for action in actions:
        if action not in executed and action not in seen:
            seen.add(action)
            result.append((action, dict(tool_inputs.get(action, {}))))
    return result


def research_contract_should_try_tools_before_human(contract: QueryContract) -> bool:
    # P2-2: Extend protection to conversation-mode compound subtasks
    # (library_status, library_recommendation, comparison_synthesis) so
    # a premature ask_human doesn't trigger global compound rollback.
    if contract.interaction_mode == "conversation":
        if contract.relation in {"library_status", "library_recommendation", "comparison_synthesis"}:
            return contract.targets or bool(contract.requested_fields)
        return False
    if contract.interaction_mode != "research":
        return False
    if contract.continuation_mode == "followup":
        return True
    if contract.targets or contract_answer_slots(contract):
        return True
    if contract.requested_fields and contract.requested_fields != ["answer"]:
        return True
    if contract.required_modalities and contract.required_modalities != ["page_text"]:
        return True
    return any(
        contract_has_note(contract, note)
        for note in {
            "router_recovered_research_slot",
            "clarify_recovered_research_slot",
            "direct_answer_recovered_research_slot",
            "low_confidence_recovered_research_slot",
            "memory_resolved_research",
            "resolved_from_conversation_memory",
            "exclude_previous_focus",
        }
    )


def plan_prefers_premature_human_clarification(*, contract: QueryContract, plan: dict[str, Any]) -> bool:
    actions = [str(item) for item in list(plan.get("actions", []) or [])]
    if not actions or actions[0] != "ask_human":
        return False
    return research_contract_should_try_tools_before_human(contract)


def defer_premature_research_clarification(
    *,
    contract: QueryContract,
    plan: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    if not plan_prefers_premature_human_clarification(contract=contract, plan=plan):
        return plan
    return {
        **fallback,
        "thought": "Research contract has actionable targets or slots; try tools before asking for human clarification.",
        "actions": [action for action in list(fallback.get("actions", []) or []) if action != "ask_human"],
    }


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
        fallback_actions = conversation_tool_sequence(planned_actions=[])
    else:
        fallback_actions = research_tool_sequence(planned_actions=[])
    return {
        "thought": "Use tools through the agent loop, observe the result, then compose or ask for clarification.",
        "actions": fallback_actions,
        "stop_conditions": ["answer_is_grounded", "ambiguity_requires_human_choice"],
    }


def should_fallback_to_human(*, contract: QueryContract, settings: Any) -> bool:
    if research_contract_should_try_tools_before_human(contract):
        return False
    return should_ask_human(confidence_from_contract(contract), settings)


def normalize_plan_payload(
    *,
    payload: Any,
    fallback: dict[str, Any],
    allowed_names: set[str] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    allowed = allowed_names if allowed_names is not None else {str(tool.get("name")) for tool in agent_tool_manifest()}
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
