from __future__ import annotations

from typing import Any, Callable

from app.domain.models import QueryContract, SessionContext
from app.services.contracts.context import contract_note_value
from app.services.agent.tools import (
    agent_tool_manifest,
    agent_tool_manifest_for_names,
)
from app.services.agent.planner_helpers import (
    NEXT_ACTION_SYSTEM_PROMPT,
    _all_unexecuted_planned_actions,
    defer_premature_research_clarification,
    fallback_plan,
    first_unexecuted_planned_action,
    json_planner_human_prompt,
    json_planner_system_prompt,
    next_action_human_prompt,
    normalize_plan_payload,
    planner_context_payload,
    planner_messages_with_user,
    research_contract_should_try_tools_before_human,
    tool_call_planner_human_prompt,
    tool_call_planner_system_prompt,
)
from app.services.tools.registry_helpers import tool_inputs_by_name

ConversationContextFn = Callable[[SessionContext], dict[str, Any]]
ConversationMessagesFn = Callable[[SessionContext], list[dict[str, str]]]
NegativeCorrectionFn = Callable[[str], bool]
ToolManifestFn = Callable[[], list[dict[str, Any]]]


class AgentPlanner:
    def __init__(
        self,
        *,
        clients: Any,
        conversation_context: ConversationContextFn,
        is_negative_correction_query: NegativeCorrectionFn,
        conversation_messages: ConversationMessagesFn | None = None,
        confidence_floor: float = 0.6,
        dynamic_tool_manifest: ToolManifestFn | None = None,
    ) -> None:
        self.clients = clients
        self.conversation_context = conversation_context
        self.conversation_messages = conversation_messages or (lambda session: [])
        self.is_negative_correction_query = is_negative_correction_query
        self.confidence_floor = confidence_floor
        self.dynamic_tool_manifest = dynamic_tool_manifest or (lambda: [])

    def tool_manifest(self) -> list[dict[str, Any]]:
        manifest: list[dict[str, Any]] = []
        seen: set[str] = set()
        for tool in [*agent_tool_manifest(), *list(self.dynamic_tool_manifest() or [])]:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            manifest.append(dict(tool))
        return manifest

    def tool_names(self) -> set[str]:
        return {str(tool.get("name")) for tool in self.tool_manifest()}

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
        # Router-provided plan: skip LLM planner entirely
        router_plan = contract_note_value(contract, prefix="router_planned_actions=")
        if router_plan:
            actions = [a.strip() for a in router_plan.split(",") if a.strip()]
            actions = [a for a in actions if a in self.tool_names()]
            if actions:
                return {
                    "thought": "Using router-provided plan.",
                    "actions": actions,
                    "stop_conditions": ["answer_is_grounded", "ambiguity_requires_human_choice"],
                }
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
            allowed_names=self.tool_names(),
        )
        if normalized_tool_plan is not None:
            return defer_premature_research_clarification(
                contract=contract,
                plan=normalized_tool_plan,
                fallback=fallback,
            )
        context_payload = planner_context_payload(
            contract=contract,
            active_research_context=session.active_research_context_payload(),
            use_web_search=use_web_search,
            include_available_tools=True,
            available_tools=self.tool_manifest(),
        )
        invoke_json_messages = getattr(self.clients, "invoke_json_messages", None)
        if callable(invoke_json_messages):
            payload = invoke_json_messages(
                system_prompt=json_planner_system_prompt(context_payload),
                messages=planner_messages_with_user(
                    conversation_messages=self.conversation_messages(session),
                    contract=contract,
                ),
                fallback=fallback,
            )
        else:
            payload = self.clients.invoke_json(
                system_prompt=json_planner_system_prompt(context_payload),
                human_prompt=json_planner_human_prompt(
                    contract=contract,
                    conversation_context=self.conversation_context(session),
                    context_payload=context_payload,
                ),
                fallback=fallback,
            )
        normalized_json_plan = normalize_plan_payload(
            payload=payload,
            fallback=fallback,
            allowed_names=self.tool_names(),
        )
        if normalized_json_plan is not None:
            return defer_premature_research_clarification(
                contract=contract,
                plan=normalized_json_plan,
                fallback=fallback,
            )
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
        tools = self.tool_manifest()
        if callable(planner_messages):
            return planner_messages(
                system_prompt=tool_call_planner_system_prompt(context_payload),
                messages=planner_messages_with_user(
                    conversation_messages=self.conversation_messages(session),
                    contract=contract,
                ),
                tools=tools,
                fallback={},
            )
        return planner(
            system_prompt=tool_call_planner_system_prompt(context_payload),
            human_prompt=tool_call_planner_human_prompt(
                contract=contract,
                conversation_context=self.conversation_context(session),
                context_payload=context_payload,
            ),
            tools=tools,
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
        actions = self.choose_next_actions(
            contract=contract,
            session=session,
            state=state,
            executed_actions=executed_actions,
            allowed_tools=allowed_tools,
        )
        return actions[0][0] if actions else None

    def choose_next_actions(
        self,
        *,
        contract: QueryContract,
        session: SessionContext,
        state: dict[str, Any],
        executed_actions: list[str],
        allowed_tools: set[str],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Return (action_name, arguments) pairs from LLM tool-call response."""
        planner_fn = getattr(self.clients, "invoke_tool_plan", None)
        if self.clients.chat is None or not callable(planner_fn):
            return []
        available_tools = agent_tool_manifest_for_names(allowed_tools, extra_tools=self.dynamic_tool_manifest())
        if not available_tools:
            return []
        payload = planner_fn(
            system_prompt=NEXT_ACTION_SYSTEM_PROMPT,
            human_prompt=next_action_human_prompt(
                contract=contract,
                state=state,
                executed_actions=executed_actions,
                conversation_context=self.conversation_context(session),
            ),
            tools=available_tools,
            fallback={},
        )
        # Store LLM-provided args in state for backwards compat
        tool_inputs = tool_inputs_by_name(payload)
        if tool_inputs:
            state_tool_inputs = state.setdefault("tool_inputs", {})
            if isinstance(state_tool_inputs, dict):
                for name, arguments in tool_inputs.items():
                    if name in allowed_tools:
                        state_tool_inputs[name] = arguments
        actions = _all_unexecuted_planned_actions(
            payload=payload,
            allowed_tools=allowed_tools,
            executed_actions=executed_actions,
        )
        if not actions:
            return []
        filtered: list[tuple[str, dict[str, Any]]] = []
        for action_name, action_args in actions:
            if action_name == "ask_human" and self._should_defer_next_action_clarification(
                contract=contract,
                state=state,
                executed_actions=executed_actions,
            ):
                continue
            filtered.append((action_name, action_args))
        return filtered

    @staticmethod
    def _should_defer_next_action_clarification(
        *,
        contract: QueryContract,
        state: dict[str, Any],
        executed_actions: list[str],
    ) -> bool:
        if not research_contract_should_try_tools_before_human(contract):
            return False
        observed = set(executed_actions)
        if observed & {
            "read_memory",
            "search_corpus",
            "bm25_search",
            "vector_search",
            "hybrid_search",
            "grep_corpus",
            "read_pdf_page",
            "web_search",
        }:
            return False
        return not bool(state.get("evidence") or state.get("screened_papers") or state.get("web_evidence"))
