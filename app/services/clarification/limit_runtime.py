from __future__ import annotations

from typing import Any, Callable

from app.domain.models import QueryContract, SessionContext
from app.services.agent.runtime_helpers import (
    clarification_limit_decision,
    promote_best_effort_state_after_clarification_limit,
)
from app.services.agent.tool_events import record_agent_observation as record_agent_observation_event
from app.services.clarification.intents import (
    clarification_options_from_contract_notes,
    clarification_tracking_key,
    next_clarification_attempt,
)


EmitFn = Callable[[str, dict[str, Any]], None]
CLARIFICATION_LIMIT_STAGE = "clarification_limit"


def force_best_effort_after_clarification_limit(
    *,
    state: dict[str, Any],
    session: SessionContext,
    web_enabled: bool,
    explicit_web_search: bool,
    max_web_results: int,
    emit: EmitFn,
    execution_steps: list[dict[str, Any]],
    runtime: Any,
    max_clarification_attempts: int,
) -> dict[str, Any] | None:
    contract: QueryContract = state["contract"]
    verification = state.get("verification")
    clarification_options = clarification_options_from_contract_notes(contract)
    clarification_key = clarification_tracking_key(
        contract=contract,
        verification=verification,
        options=clarification_options,
    )
    next_attempt = next_clarification_attempt(session=session, key=clarification_key)
    decision = clarification_limit_decision(
        contract=contract,
        verification=verification,
        next_attempt=next_attempt,
        max_attempts=max_clarification_attempts,
        options=clarification_options,
    )
    if decision is None:
        return None

    record_agent_observation_event(
        emit=emit,
        execution_steps=execution_steps,
        tool="ask_human",
        summary=decision.summary,
        payload={"stage": CLARIFICATION_LIMIT_STAGE, **decision.observation_payload},
    )

    forced_state = runtime.run_research_agent_loop(
        contract=decision.forced_contract,
        session=session,
        agent_plan=decision.forced_plan,
        web_enabled=web_enabled,
        explicit_web_search=explicit_web_search,
        max_web_results=max_web_results,
        emit=emit,
        execution_steps=execution_steps,
    )
    return promote_best_effort_state_after_clarification_limit(forced_state)
