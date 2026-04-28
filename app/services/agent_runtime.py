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
from app.services.confidence import confidence_from_contract, should_ask_human

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
            "tool_inputs": self._tool_inputs_by_name(agent_plan),
            "current_tool_input": {},
        }
        emit(
            "observation",
            {
                "tool": "compose",
                "summary": "tool_loop_ready",
                "payload": {"actions": actions, "tool_inputs": state["tool_inputs"]},
            },
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
            fallback_next=lambda executed: self._next_conversation_action(
                contract=contract,
                state=state,
                executed=executed,
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
            "tool_inputs": self._tool_inputs_by_name(agent_plan),
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
        emit(
            "observation",
            {
                "tool": "search_corpus" if "search_corpus" in actions else "compose",
                "summary": "tool_loop_ready",
                "payload": {"actions": actions, "tool_inputs": state["tool_inputs"]},
            },
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
            fallback_next=lambda executed: self._next_research_action(
                contract=state["contract"],
                state=state,
                executed=executed,
                web_enabled=web_enabled,
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
        configured_max_steps = self._configured_max_steps(fallback=max_steps)
        for index in range(1, configured_max_steps + 1):
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
            tool_input = self._tool_input_for_action(state=state, action=action)
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

    def _configured_max_steps(self, *, fallback: int) -> int:
        agent_settings = getattr(self.agent, "agent_settings", None)
        value = getattr(agent_settings, "max_agent_steps", fallback)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = fallback
        return max(1, parsed)

    @staticmethod
    def _tool_inputs_by_name(agent_plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
        raw_items = agent_plan.get("tool_call_args", []) if isinstance(agent_plan, dict) else []
        if not isinstance(raw_items, list):
            return {}
        tool_inputs: dict[str, dict[str, Any]] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip()
            args = item.get("args", {})
            if not name or not isinstance(args, dict):
                continue
            tool_inputs.setdefault(name, dict(args))
        return tool_inputs

    @staticmethod
    def _tool_input_for_action(*, state: dict[str, Any], action: str) -> dict[str, Any]:
        tool_inputs = state.get("tool_inputs", {})
        if not isinstance(tool_inputs, dict):
            return {}
        payload = tool_inputs.get(action, {})
        return dict(payload) if isinstance(payload, dict) else {}

    def _next_conversation_action(
        self,
        *,
        contract: QueryContract,
        state: dict[str, Any],
        executed: set[str],
    ) -> str | None:
        notes = {str(item) for item in contract.notes}
        fields = {str(item) for item in contract.requested_fields}
        is_memory_turn = (
            "intent_kind=memory_op" in notes
            or bool(fields & {"comparison", "synthesis", "previous_tool_basis"})
            or contract.continuation_mode == "followup"
        )
        is_citation_turn = "citation_count_ranking" in fields or "citation_count_requires_web" in notes
        if self._contract_needs_human_clarification(contract) and "ask_human" not in executed:
            return "ask_human"
        if (is_memory_turn or is_citation_turn) and "read_memory" not in executed:
            return "read_memory"
        if is_citation_turn and "web_search" not in executed:
            return "web_search"
        if contract.relation == "library_status" and "query_library_metadata" not in executed:
            return "query_library_metadata"
        if not state.get("answer") and "compose" not in executed:
            return "compose"
        return None

    def _next_research_action(
        self,
        *,
        contract: QueryContract,
        state: dict[str, Any],
        executed: set[str],
        web_enabled: bool,
    ) -> str | None:
        if self._contract_needs_human_clarification(contract) and "ask_human" not in executed:
            return "ask_human"
        if (
            contract.continuation_mode == "followup"
            or "memory_resolved_research" in contract.notes
            or "resolved_from_conversation_memory" in contract.notes
            or "exclude_previous_focus" in contract.notes
        ) and "read_memory" not in executed:
            return "read_memory"
        has_evidence = bool(state.get("evidence"))
        has_papers = bool(state.get("screened_papers"))
        if (not has_evidence or not has_papers) and "search_corpus" not in executed:
            return "search_corpus"
        if web_enabled and "web_search" not in executed:
            return "web_search"
        if "compose" not in executed:
            return "compose"
        return None

    def _contract_needs_human_clarification(self, contract: QueryContract) -> bool:
        return should_ask_human(
            confidence_from_contract(contract),
            getattr(self.agent, "agent_settings", None),
        )
