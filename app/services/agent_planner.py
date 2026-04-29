from __future__ import annotations

import json
from typing import Any, Callable

from app.domain.models import QueryContract, SessionContext
from app.services.agent_tools import agent_tool_manifest, normalize_plan_actions
from app.services.agent_planner_helpers import (
    fallback_plan,
    normalize_plan_payload,
    planner_context_payload,
    planner_intent_payload,
    planner_state_summary,
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
        confidence_floor: float = 0.6,
    ) -> None:
        self.clients = clients
        self.conversation_context = conversation_context
        self.conversation_messages = conversation_messages or (lambda session: [])
        self.is_negative_correction_query = is_negative_correction_query
        self.confidence_floor = confidence_floor

    def plan_actions(
        self,
        *,
        contract: QueryContract,
        session: SessionContext,
        use_web_search: bool,
    ) -> dict[str, Any]:
        fallback = fallback_plan(
            contract=contract,
            use_web_search=use_web_search,
            settings=self,
            is_negative_correction_query=self.is_negative_correction_query,
        )
        if self.clients.chat is None:
            return fallback
        tool_plan = self.plan_with_tool_calls(
            contract=contract,
            session=session,
            use_web_search=use_web_search,
        )
        normalized_tool_plan = normalize_plan_payload(
            payload=tool_plan,
            fallback=fallback,
        )
        if normalized_tool_plan is not None:
            return normalized_tool_plan
        context_payload = planner_context_payload(
            contract=contract,
            active_research_context=session.active_research_context_payload(),
            use_web_search=use_web_search,
            include_available_tools=True,
        )
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
        normalized_json_plan = normalize_plan_payload(
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
        context_payload = planner_context_payload(
            contract=contract,
            active_research_context=session.active_research_context_payload(),
            use_web_search=use_web_search,
            include_available_tools=False,
        )
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
                    "intent": planner_intent_payload(contract),
                    "targets": contract.targets,
                    "notes": contract.notes,
                    "executed_actions": executed_actions,
                    "state_summary": planner_state_summary(state),
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
