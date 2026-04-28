from __future__ import annotations

import json
from typing import Any, Callable

from app.domain.models import QueryContract, SessionContext
from app.services.agent_tools import (
    agent_tool_manifest,
    conversation_tool_sequence,
    normalize_plan_actions,
    research_tool_sequence,
)

ConversationContextFn = Callable[[SessionContext], dict[str, Any]]
ConversationMessagesFn = Callable[[SessionContext], list[dict[str, str]]]
NegativeCorrectionFn = Callable[[str], bool]


class AgentPlanner:
    def __init__(
        self,
        *,
        clients: Any,
        conversation_context: ConversationContextFn,
        is_negative_correction_query: NegativeCorrectionFn,
        conversation_messages: ConversationMessagesFn | None = None,
    ) -> None:
        self.clients = clients
        self.conversation_context = conversation_context
        self.conversation_messages = conversation_messages or (lambda session: [])
        self.is_negative_correction_query = is_negative_correction_query

    def plan_actions(
        self,
        *,
        contract: QueryContract,
        session: SessionContext,
        use_web_search: bool,
    ) -> dict[str, Any]:
        fallback = self._fallback_plan(
            contract=contract,
            use_web_search=use_web_search,
        )
        if self.clients.chat is None:
            return fallback
        tool_plan = self.plan_with_tool_calls(
            contract=contract,
            session=session,
            use_web_search=use_web_search,
        )
        normalized_tool_plan = self.normalize_plan_payload(
            payload=tool_plan,
            fallback=fallback,
        )
        if normalized_tool_plan is not None:
            return normalized_tool_plan
        context_payload = {
            "intent": self._intent_payload(contract),
            "targets": contract.targets,
            "notes": contract.notes,
            "available_tools": agent_tool_manifest(),
            "active_research_context": session.active_research_context_payload(),
            "web_enabled": use_web_search,
        }
        system_prompt = (
                "你是论文助手的工具循环控制器。"
                "请只选择当前最有用的一小组工具，不要回答问题。"
                "工具能力和边界以 available_tools 的 description 为唯一依据；不要使用隐藏的 relation->固定流水线规则。"
                "如果现有记忆或证据已经足够，可以直接选择 compose；如果缺关键槽位，选择 ask_human。"
                "只输出 JSON：thought, actions, stop_conditions。"
                "actions 只能从 available_tools 的 name 中选择。"
        )
        context_json = json.dumps(context_payload, ensure_ascii=False)
        invoke_json_messages = getattr(self.clients, "invoke_json_messages", None)
        if callable(invoke_json_messages):
            payload = invoke_json_messages(
                system_prompt=(
                    system_prompt
                    + "\n\n以下非语言上下文只用于工具选择，不是用户新问题：\n"
                    + context_json
                ),
                messages=[
                    *self.conversation_messages(session),
                    {"role": "user", "content": contract.clean_query},
                ],
                fallback=fallback,
            )
        else:
            payload = self.clients.invoke_json(
                system_prompt=system_prompt,
                human_prompt=json.dumps(
                    {
                        "query": contract.clean_query,
                        "conversation_context": self.conversation_context(session),
                        **context_payload,
                    },
                    ensure_ascii=False,
                ),
                fallback=fallback,
            )
        normalized_json_plan = self.normalize_plan_payload(
            payload=payload,
            fallback=fallback,
        )
        if normalized_json_plan is not None:
            return normalized_json_plan
        return fallback

    def plan_with_tool_calls(
        self,
        *,
        contract: QueryContract,
        session: SessionContext,
        use_web_search: bool,
    ) -> dict[str, Any]:
        planner = getattr(self.clients, "invoke_tool_plan", None)
        planner_messages = getattr(self.clients, "invoke_tool_plan_messages", None)
        if not callable(planner) and not callable(planner_messages):
            return {}
        context_payload = {
            "intent": self._intent_payload(contract),
            "targets": contract.targets,
            "notes": contract.notes,
            "active_research_context": session.active_research_context_payload(),
            "web_enabled": use_web_search,
        }
        system_prompt = (
                "你是论文助手的工具选择器。"
                "你不能直接回答用户，只能通过 tool calls 选择下一步工具。"
                "工具描述是唯一的能力说明；不要假设固定流水线。"
                "每次根据 intent、上下文和已有 observation 决定：读记忆、搜本地语料、搜外部、请求用户澄清，或 compose。"
                "只返回 tool calls，不要输出普通回答。"
        )
        if callable(planner_messages):
            return planner_messages(
                system_prompt=(
                    system_prompt
                    + "\n\n以下非语言上下文只用于工具选择，不是用户新问题：\n"
                    + json.dumps(context_payload, ensure_ascii=False)
                ),
                messages=[
                    *self.conversation_messages(session),
                    {"role": "user", "content": contract.clean_query},
                ],
                tools=agent_tool_manifest(),
                fallback={},
            )
        return planner(
            system_prompt=system_prompt,
            human_prompt=json.dumps(
                {
                    "query": contract.clean_query,
                    "conversation_context": self.conversation_context(session),
                    **context_payload,
                },
                ensure_ascii=False,
            ),
            tools=agent_tool_manifest(),
            fallback={},
        )

    def choose_next_action(
        self,
        *,
        contract: QueryContract,
        session: SessionContext,
        state: dict[str, Any],
        executed_actions: list[str],
        allowed_tools: set[str],
    ) -> str | None:
        planner = getattr(self.clients, "invoke_tool_plan", None)
        if self.clients.chat is None or not callable(planner):
            return None
        available_tools = [tool for tool in agent_tool_manifest() if str(tool.get("name")) in allowed_tools]
        if not available_tools:
            return None
        payload = planner(
            system_prompt=(
                "你是 observation-driven 工具循环的下一步选择器。"
                "根据当前 intent、已执行工具和 state 摘要，只选择一个下一步工具。"
                "如果已经足够回答，选择 compose；如果必须由用户消歧，选择 ask_human。"
                "不要输出普通回答。"
            ),
            human_prompt=json.dumps(
                {
                    "query": contract.clean_query,
                    "intent": self._intent_payload(contract),
                    "targets": contract.targets,
                    "notes": contract.notes,
                    "executed_actions": executed_actions,
                    "state_summary": self._state_summary(state),
                    "conversation_context": self.conversation_context(session),
                },
                ensure_ascii=False,
            ),
            tools=available_tools,
            fallback={},
        )
        actions = normalize_plan_actions(actions=payload.get("actions", []), allowed=allowed_tools)
        executed = set(executed_actions)
        for action in actions:
            if action not in executed:
                return action
        return None

    @staticmethod
    def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
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

    @staticmethod
    def _intent_payload(contract: QueryContract) -> dict[str, Any]:
        notes = [str(item) for item in contract.notes]
        answer_slots = [str(item).strip() for item in list(getattr(contract, "answer_slots", []) or []) if str(item).strip()]
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

    def _fallback_plan(self, *, contract: QueryContract, use_web_search: bool) -> dict[str, Any]:
        if self._contract_needs_human_clarification(contract):
            fallback_actions = ["ask_human"]
        elif contract.interaction_mode == "conversation":
            fallback_actions = conversation_tool_sequence(relation=contract.relation, planned_actions=[])
        else:
            fallback_actions = research_tool_sequence(
                planned_actions=[],
                use_web_search=use_web_search,
                needs_reflection="exclude_previous_focus" in contract.notes
                or self.is_negative_correction_query(contract.clean_query),
            )
        return {
            "thought": "Use tools through the agent loop, observe the result, then compose or ask for clarification.",
            "actions": fallback_actions,
            "stop_conditions": ["answer_is_grounded", "ambiguity_requires_human_choice"],
        }

    @staticmethod
    def _contract_needs_human_clarification(contract: QueryContract) -> bool:
        notes = {str(item) for item in contract.notes}
        if "low_intent_confidence" in notes or "intent_needs_clarification" in notes:
            return True
        return any(note.startswith("ambiguous_slot=") for note in notes)

    @staticmethod
    def normalize_plan_payload(
        *,
        payload: Any,
        fallback: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        allowed = {str(tool.get("name")) for tool in agent_tool_manifest()}
        normalized_actions = normalize_plan_actions(actions=payload.get("actions", []), allowed=allowed)
        if not normalized_actions:
            return None
        return {
            "thought": str(payload.get("thought") or fallback["thought"]),
            "actions": normalized_actions,
            "stop_conditions": payload.get("stop_conditions") if isinstance(payload.get("stop_conditions"), list) else fallback["stop_conditions"],
            "tool_call_args": payload.get("tool_call_args") if isinstance(payload.get("tool_call_args"), list) else [],
        }
