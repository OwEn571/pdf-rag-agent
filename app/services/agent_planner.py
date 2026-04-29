from __future__ import annotations

import json
from typing import Any, Callable

from app.domain.models import QueryContract, SessionContext
from app.services.agent_tools import (
    agent_tool_manifest,
    agent_tool_manifest_for_names,
    normalize_plan_actions,
)
from app.services.agent_planner_helpers import (
    JSON_PLANNER_SYSTEM_PROMPT,
    NEXT_ACTION_SYSTEM_PROMPT,
    TOOL_CALL_PLANNER_SYSTEM_PROMPT,
    fallback_plan,
    normalize_plan_payload,
    planner_context_payload,
    planner_intent_payload,
    planner_prompt_with_context,
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
        context_json = json.dumps(context_payload, ensure_ascii=False)
        invoke_json_messages = getattr(self.clients, "invoke_json_messages", None)
        if callable(invoke_json_messages):
            payload = invoke_json_messages(
                system_prompt=planner_prompt_with_context(
                    system_prompt=JSON_PLANNER_SYSTEM_PROMPT,
                    context_json=context_json,
                ),
                messages=[
                    *self.conversation_messages(session),
                    {"role": "user", "content": contract.clean_query},
                ],
                fallback=fallback,
            )
        else:
            payload = self.clients.invoke_json(
                system_prompt=JSON_PLANNER_SYSTEM_PROMPT,
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
        if callable(planner_messages):
            return planner_messages(
                system_prompt=planner_prompt_with_context(
                    system_prompt=TOOL_CALL_PLANNER_SYSTEM_PROMPT,
                    context_json=json.dumps(context_payload, ensure_ascii=False),
                ),
                messages=[
                    *self.conversation_messages(session),
                    {"role": "user", "content": contract.clean_query},
                ],
                tools=agent_tool_manifest(),
                fallback={},
            )
        return planner(
            system_prompt=TOOL_CALL_PLANNER_SYSTEM_PROMPT,
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
        available_tools = agent_tool_manifest_for_names(allowed_tools)
        if not available_tools:
            return None
        payload = planner(
            system_prompt=NEXT_ACTION_SYSTEM_PROMPT,
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
