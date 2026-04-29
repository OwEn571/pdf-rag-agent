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
    agent_loop_summary,
    configured_max_steps,
    conversation_runtime_actions,
    conversation_runtime_state,
    dequeue_action,
    finalize_research_verification,
    next_conversation_action,
    next_research_action,
    planner_next_action,
    research_runtime_actions,
    research_runtime_state,
    tool_loop_ready_tool,
    verification_execution_step,
)
from app.services.tool_registry_helpers import (
    tool_input_from_state,
    tool_loop_ready_observation,
)

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
        emit(
            "observation",
            tool_loop_ready_observation(tool="compose", actions=actions, tool_inputs=state["tool_inputs"]),
        )
        execution_steps.append({"node": "agent_loop", "summary": agent_loop_summary(actions)})
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
        self._execute_tool_loop(
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
        plan = self.agent._build_research_plan(contract)
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
            is_negative_correction_query=self.agent._is_negative_correction_query,
        )
        emit(
            "observation",
            tool_loop_ready_observation(
                tool=tool_loop_ready_tool(actions),
                actions=actions,
                tool_inputs=state["tool_inputs"],
            ),
        )
        execution_steps.append({"node": "agent_loop", "summary": agent_loop_summary(actions)})

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
        self._execute_tool_loop(
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

        self.agent._agent_reflect(
            state=state,
            session=session,
            emit=emit,
            execution_steps=execution_steps,
        )
        verification, confidence = finalize_research_verification(state)
        emit("verification", verification.model_dump())
        emit("confidence", confidence)
        execution_steps.append(verification_execution_step(verification))
        return state

    def _execute_tool_loop(
        self,
        *,
        contract: QueryContract,
        session: SessionContext,
        state: dict[str, Any],
        executor: AgentToolExecutor,
        planned_actions: list[str],
        allowed_tools: set[str],
        emit: EmitFn,
        fallback_next: Callable[[set[str]], str | None],
        stop_condition: Callable[[set[str]], bool],
        max_steps: int = 8,
    ) -> None:
        queue = [action for action in planned_actions if action in allowed_tools]
        executed_order: list[str] = []
        max_step_count = configured_max_steps(
            getattr(self.agent, "agent_settings", None),
            fallback=max_steps,
        )
        for index in range(1, max_step_count + 1):
            action = dequeue_action(queue=queue, executed=executor.executed)
            if action is None:
                action = planner_next_action(
                    agent=self.agent,
                    contract=contract,
                    session=session,
                    state=state,
                    executed_actions=executed_order,
                    allowed_tools=allowed_tools,
                )
            if action is None:
                action = fallback_next(executor.executed)
            if action is None or action not in allowed_tools:
                break
            tool_input = tool_input_from_state(state, action)
            state["current_tool_input"] = tool_input
            self.agent._emit_agent_step(
                emit=emit,
                index=index,
                action=action,
                contract=state.get("contract", contract),
                state=state,
                arguments=tool_input,
            )
            should_stop = executor.run(action)
            executed_order.append(action)
            if should_stop or stop_condition(executor.executed):
                break
