from __future__ import annotations

from typing import Any, Callable

from app.domain.models import QueryContract, SessionContext

EmitFn = Callable[[str, dict[str, Any]], None]


def task_plan_with_allow_list(plan: dict[str, Any], tools_allowed: list[str]) -> dict[str, Any]:
    if not tools_allowed:
        return plan
    allowed = {str(item) for item in tools_allowed if str(item).strip()}
    actions = [str(item) for item in list(plan.get("actions", []) or []) if str(item) in allowed]
    tool_call_args = [
        item
        for item in list(plan.get("tool_call_args", []) or [])
        if isinstance(item, dict) and str(item.get("name", "") or "") in allowed
    ]
    return {**plan, "actions": actions, "tool_call_args": tool_call_args}


def run_task_subagent(
    *,
    agent: Any,
    prompt: str,
    description: str,
    tools_allowed: list[str],
    max_steps: Any,
    session: SessionContext,
    max_web_results: int,
    emit: EmitFn,
    execution_steps: list[dict[str, Any]],
    contract: QueryContract | None = None,
) -> dict[str, Any]:
    sub_contract = contract or agent._extract_query_contract(
        query=prompt,
        session=session,
        mode="auto",
        clarification_choice=None,
    )
    sub_plan = agent.planner.plan_actions(contract=sub_contract, session=session, use_web_search=False)
    sub_plan = task_plan_with_allow_list(sub_plan, tools_allowed)
    emit("agent_plan", {"subtask": prompt, "task_tool": True, **sub_plan})
    agent._emit_agent_tool_call(
        emit=emit,
        tool="Task",
        arguments={
            "description": description,
            "prompt": prompt,
            "tools_allowed": tools_allowed,
            "max_steps": max_steps,
        },
    )
    if sub_contract.interaction_mode == "conversation":
        sub_state = agent.runtime.execute_conversation_tools(
            contract=sub_contract,
            query=prompt,
            session=session,
            agent_plan=sub_plan,
            max_web_results=max_web_results,
            emit=emit,
            execution_steps=execution_steps,
        )
        answer = str(sub_state.get("answer", "") or "")
        citations = list(sub_state.get("citations", []) or [])
        verification_payload = dict(sub_state.get("verification_report", {}) or {"status": "pass"})
        return {
            "prompt": prompt,
            "answer": answer,
            "citations": citations,
            "verification": verification_payload,
            "verification_obj": None,
            "contract": sub_contract.model_dump(),
            "contract_obj": sub_contract,
            "claims": [],
            "evidence": [],
        }

    sub_state = agent.runtime.run_research_agent_loop(
        contract=sub_contract,
        session=session,
        agent_plan=sub_plan,
        web_enabled=False,
        explicit_web_search=False,
        max_web_results=max_web_results,
        emit=emit,
        execution_steps=execution_steps,
    )
    verification = sub_state.get("verification")
    answer, citations = agent._compose_answer(
        contract=sub_state["contract"],
        claims=sub_state["claims"],
        evidence=sub_state["evidence"],
        papers=sub_state["screened_papers"],
        verification=verification,
        session=session,
    )
    return {
        "prompt": prompt,
        "answer": answer,
        "citations": citations,
        "verification": verification.model_dump() if hasattr(verification, "model_dump") else {},
        "verification_obj": verification,
        "contract": sub_state["contract"].model_dump(),
        "contract_obj": sub_state["contract"],
        "claims": list(sub_state.get("claims", []) or []),
        "evidence": list(sub_state.get("evidence", []) or []),
        "papers": list(sub_state.get("screened_papers", []) or []),
    }
