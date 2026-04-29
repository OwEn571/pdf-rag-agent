from __future__ import annotations

from typing import Any, Callable

from app.domain.models import QueryContract, SessionContext, VerificationReport
from app.services.agent_tools import conversation_tool_sequence, research_tool_sequence
from app.services.confidence import (
    confidence_from_contract,
    confidence_from_verification_report,
    confidence_payload,
    should_ask_human,
)
from app.services.tool_registry_helpers import tool_input_from_state, tool_inputs_by_name

NegativeCorrectionFn = Callable[[str], bool]
EmitFn = Callable[[str, dict[str, Any]], None]
FallbackNextFn = Callable[[set[str]], str | None]
StopConditionFn = Callable[[set[str]], bool]


def conversation_runtime_state(*, contract: QueryContract, agent_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": contract,
        "answer": "",
        "citations": [],
        "verification_report": {"status": "pass", "recommended_action": "conversation_tool_answer"},
        "citation_candidates": [],
        "citation_lookup": {},
        "tool_inputs": tool_inputs_by_name(agent_plan),
        "current_tool_input": {},
    }


def conversation_runtime_actions(*, contract: QueryContract, agent_plan: dict[str, Any]) -> list[str]:
    raw_actions = agent_plan.get("actions", []) if isinstance(agent_plan, dict) else []
    planned_actions = [str(item) for item in list(raw_actions or [])]
    return conversation_tool_sequence(relation=contract.relation, planned_actions=planned_actions)


def research_runtime_state(
    *,
    contract: QueryContract,
    plan: Any,
    excluded_titles: set[str],
    agent_plan: dict[str, Any],
) -> dict[str, Any]:
    return {
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


def research_runtime_actions(
    *,
    contract: QueryContract,
    agent_plan: dict[str, Any],
    web_enabled: bool,
    is_negative_correction_query: NegativeCorrectionFn,
) -> list[str]:
    raw_actions = agent_plan.get("actions", []) if isinstance(agent_plan, dict) else []
    return research_tool_sequence(
        planned_actions=raw_actions if isinstance(raw_actions, list) else [],
        use_web_search=web_enabled,
        needs_reflection="exclude_previous_focus" in contract.notes
        or is_negative_correction_query(contract.clean_query),
    )


def agent_loop_summary(actions: list[str]) -> str:
    return " -> ".join(actions)


def tool_loop_ready_tool(actions: list[str]) -> str:
    return "search_corpus" if "search_corpus" in actions else "compose"


def finalize_research_verification(state: dict[str, Any]) -> tuple[VerificationReport, dict[str, Any]]:
    verification = state.get("verification")
    if not isinstance(verification, VerificationReport):
        verification = VerificationReport(
            status="clarify",
            missing_fields=["verified_claims"],
            recommended_action="clarify_after_reflection",
        )
        state["verification"] = verification
    confidence = confidence_payload(confidence_from_verification_report(verification))
    state["confidence"] = confidence
    return verification, confidence


def verification_execution_step(verification: VerificationReport) -> dict[str, str]:
    return {"node": "agent_tool:verify_claim", "summary": verification.status}


def configured_max_steps(agent_settings: Any, *, fallback: int) -> int:
    value = getattr(agent_settings, "max_agent_steps", fallback)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(1, parsed)


def dequeue_action(*, queue: list[str], executed: set[str]) -> str | None:
    while queue:
        action = queue.pop(0)
        if action not in executed:
            return action
    return None


def planner_next_action(
    *,
    agent: Any,
    contract: QueryContract,
    session: SessionContext,
    state: dict[str, Any],
    executed_actions: list[str],
    allowed_tools: set[str],
) -> str | None:
    planner = getattr(agent, "planner", None)
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


def execute_tool_loop(
    *,
    agent: Any,
    contract: QueryContract,
    session: SessionContext,
    state: dict[str, Any],
    executor: Any,
    planned_actions: list[str],
    allowed_tools: set[str],
    emit: EmitFn,
    fallback_next: FallbackNextFn,
    stop_condition: StopConditionFn,
    max_steps: int = 8,
) -> None:
    queue = [action for action in planned_actions if action in allowed_tools]
    executed_order: list[str] = []
    max_step_count = configured_max_steps(
        getattr(agent, "agent_settings", None),
        fallback=max_steps,
    )
    for index in range(1, max_step_count + 1):
        action = dequeue_action(queue=queue, executed=executor.executed)
        if action is None:
            action = planner_next_action(
                agent=agent,
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
        agent._emit_agent_step(
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


def contract_needs_human_clarification(contract: QueryContract, agent_settings: Any) -> bool:
    return should_ask_human(confidence_from_contract(contract), agent_settings)


def next_conversation_action(
    *,
    contract: QueryContract,
    state: dict[str, Any],
    executed: set[str],
    agent_settings: Any,
) -> str | None:
    notes = {str(item) for item in contract.notes}
    fields = {str(item) for item in contract.requested_fields}
    is_memory_turn = (
        "intent_kind=memory_op" in notes
        or bool(fields & {"comparison", "synthesis", "previous_tool_basis"})
        or contract.continuation_mode == "followup"
    )
    is_citation_turn = "citation_count_ranking" in fields or "citation_count_requires_web" in notes
    if contract_needs_human_clarification(contract, agent_settings) and "ask_human" not in executed:
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


def next_research_action(
    *,
    contract: QueryContract,
    state: dict[str, Any],
    executed: set[str],
    web_enabled: bool,
    agent_settings: Any,
) -> str | None:
    if contract_needs_human_clarification(contract, agent_settings) and "ask_human" not in executed:
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
