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
    conversation_tool_sequence,
    research_execution_tool_names,
    research_tool_sequence,
)
from app.services.agent_runtime_helpers import (
    configured_max_steps,
    next_conversation_action,
    next_research_action,
)
from app.services.confidence import (
    confidence_from_verification_report,
    confidence_payload,
)
from app.services.tool_registry_helpers import (
    tool_input_from_state,
    tool_inputs_by_name,
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
        planned_actions = [str(item) for item in list(agent_plan.get("actions", []) or [])]
        actions = conversation_tool_sequence(relation=contract.relation, planned_actions=planned_actions)
        state: dict[str, Any] = {
            "contract": contract,
            "answer": "",
            "citations": [],
            "verification_report": {"status": "pass", "recommended_action": "conversation_tool_answer"},
            "citation_candidates": [],
            "citation_lookup": {},
            "tool_inputs": tool_inputs_by_name(agent_plan),
            "current_tool_input": {},
        }
        emit(
            "observation",
            tool_loop_ready_observation(tool="compose", actions=actions, tool_inputs=state["tool_inputs"]),
        )
        execution_steps.append({"node": "agent_loop", "summary": " -> ".join(actions)})
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
        state: dict[str, Any] = {
            "contract": contract,
            "plan": plan,
            "candidate_papers": [],
            "screened_papers": [],
            "precomputed_evidence": None,
            "evidence": [],
            "web_evidence": [],
            "claims": [],
            "verification": None,
            "reflection": {},
            "excluded_titles": excluded_titles,
            "tool_inputs": tool_inputs_by_name(agent_plan),
            "current_tool_input": {},
        }
        emit("plan", plan.model_dump())
        execution_steps.append({"node": "agent_tool:build_research_plan", "summary": ",".join(plan.solver_sequence)})

        raw_actions = agent_plan.get("actions", []) if isinstance(agent_plan, dict) else []
        actions = research_tool_sequence(
            planned_actions=raw_actions if isinstance(raw_actions, list) else [],
            use_web_search=web_enabled,
            needs_reflection="exclude_previous_focus" in contract.notes
            or self.agent._is_negative_correction_query(contract.clean_query),
        )
        ready_tool = "search_corpus" if "search_corpus" in actions else "compose"
        emit(
            "observation",
            tool_loop_ready_observation(tool=ready_tool, actions=actions, tool_inputs=state["tool_inputs"]),
        )
        execution_steps.append({"node": "agent_loop", "summary": " -> ".join(actions)})

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
        verification = state.get("verification")
        if not isinstance(verification, VerificationReport):
            verification = VerificationReport(
                status="clarify",
                missing_fields=["verified_claims"],
                recommended_action="clarify_after_reflection",
            )
            state["verification"] = verification
        emit("verification", verification.model_dump())
        runtime_confidence = confidence_from_verification_report(verification)
        state["confidence"] = confidence_payload(runtime_confidence)
        emit("confidence", state["confidence"])
        execution_steps.append({"node": "agent_tool:verify_claim", "summary": verification.status})
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
            action = self._dequeue_action(queue=queue, executed=executor.executed)
            if action is None:
                action = self._planner_next_action(
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

    @staticmethod
    def _dequeue_action(*, queue: list[str], executed: set[str]) -> str | None:
        while queue:
            action = queue.pop(0)
            if action not in executed:
                return action
        return None

    def _planner_next_action(
        self,
        *,
        contract: QueryContract,
        session: SessionContext,
        state: dict[str, Any],
        executed_actions: list[str],
        allowed_tools: set[str],
    ) -> str | None:
        planner = getattr(self.agent, "planner", None)
        choose_next = getattr(planner, "choose_next_action", None)
        if not callable(choose_next):
            return None
        return choose_next(
            contract=state.get("contract", contract),
            session=session,
            state=state,
            executed_actions=executed_actions,
            allowed_tools=allowed_tools,
        )
