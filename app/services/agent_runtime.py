from __future__ import annotations

from typing import Any, Callable

from app.domain.models import QueryContract, SessionContext, VerificationReport
from app.services.agent_tool_registries import (
    build_conversation_tool_registry,
    build_research_tool_registry,
)
from app.services.agent_tools import (
    AgentToolExecutor,
    conversation_execution_tool_names,
    research_execution_tool_names,
)
from app.services.agent_runtime_helpers import (
    conversation_runtime_actions,
    conversation_runtime_state,
    execute_tool_loop,
    finalize_research_runtime,
    next_conversation_action,
    next_research_action,
    record_tool_loop_ready,
    research_runtime_actions,
    research_runtime_state,
    tool_loop_ready_tool,
)
from app.services.followup_intents import is_negative_correction_query
from app.services.research_planning import build_research_plan

EmitFn = Callable[[str, dict[str, Any]], None]


class AgentRuntime:
    def __init__(self, *, agent: Any) -> None:
        self.agent = agent

    def execute_conversation_tools(
        self,
        *,
        contract: QueryContract,
        query: str,
        session: SessionContext,
        agent_plan: dict[str, Any],
        max_web_results: int,
        emit: EmitFn,
        execution_steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        actions = conversation_runtime_actions(contract=contract, agent_plan=agent_plan)
        state = conversation_runtime_state(contract=contract, agent_plan=agent_plan)
        record_tool_loop_ready(
            emit=emit,
            execution_steps=execution_steps,
            tool="compose",
            actions=actions,
            tool_inputs=state["tool_inputs"],
        )
        tools = build_conversation_tool_registry(
            agent=self.agent,
            state=state,
            contract=contract,
            query=query,
            session=session,
            max_web_results=max_web_results,
            emit=emit,
            execution_steps=execution_steps,
        )
        executor = AgentToolExecutor(tools)
        execute_tool_loop(
            agent=self.agent,
            contract=contract,
            session=session,
            state=state,
            executor=executor,
            planned_actions=actions,
            allowed_tools=conversation_execution_tool_names(),
            emit=emit,
            fallback_next=lambda executed: next_conversation_action(
                contract=contract,
                state=state,
                executed=executed,
                agent_settings=getattr(self.agent, "agent_settings", None),
            ),
            stop_condition=lambda executed: bool(state.get("answer")),
        )
        if not state.get("answer"):
            executor.run("compose")
        return state

    def run_research_agent_loop(
        self,
        *,
        contract: QueryContract,
        session: SessionContext,
        agent_plan: dict[str, Any],
        web_enabled: bool,
        explicit_web_search: bool,
        max_web_results: int,
        emit: EmitFn,
        execution_steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        plan = build_research_plan(contract=contract, settings=self.agent.settings)
        excluded_titles = self.agent._excluded_focus_titles(session=session, contract=contract)
        state = research_runtime_state(
            contract=contract,
            plan=plan,
            excluded_titles=excluded_titles,
            agent_plan=agent_plan,
        )
        emit("plan", plan.model_dump())
        execution_steps.append({"node": "agent_tool:build_research_plan", "summary": ",".join(plan.solver_sequence)})

        actions = research_runtime_actions(
            contract=contract,
            agent_plan=agent_plan,
            web_enabled=web_enabled,
            is_negative_correction_query=is_negative_correction_query,
        )
        record_tool_loop_ready(
            emit=emit,
            execution_steps=execution_steps,
            tool=tool_loop_ready_tool(actions),
            actions=actions,
            tool_inputs=state["tool_inputs"],
        )

        tools = build_research_tool_registry(
            agent=self.agent,
            state=state,
            session=session,
            web_enabled=web_enabled,
            explicit_web_search=explicit_web_search,
            max_web_results=max_web_results,
            emit=emit,
            execution_steps=execution_steps,
        )
        executor = AgentToolExecutor(tools)
        execute_tool_loop(
            agent=self.agent,
            contract=contract,
            session=session,
            state=state,
            executor=executor,
            planned_actions=actions,
            allowed_tools=research_execution_tool_names(),
            emit=emit,
            fallback_next=lambda executed: next_research_action(
                contract=state["contract"],
                state=state,
                executed=executed,
                web_enabled=web_enabled,
                agent_settings=getattr(self.agent, "agent_settings", None),
            ),
            stop_condition=lambda executed: isinstance(state.get("verification"), VerificationReport)
            and state["verification"].status in {"pass", "clarify"},
        )

        if state["verification"] is None:
            executor.run("compose")

        finalize_research_runtime(
            agent=self.agent,
            state=state,
            session=session,
            emit=emit,
            execution_steps=execution_steps,
        )
        return state
