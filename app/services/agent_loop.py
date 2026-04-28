from __future__ import annotations

from typing import Any

from app.domain.models import (
    AssistantCitation,
    AssistantResponse,
    QueryContract,
    SessionTurn,
    VerificationReport,
)
from app.services.agent_context import AgentRunContext


def run_conversation_turn(
    *,
    agent: Any,
    run_context: AgentRunContext,
    query: str,
    contract: QueryContract,
    agent_plan: dict[str, Any],
    max_web_results: int,
) -> dict[str, Any]:
    session = run_context.session
    execution_steps = run_context.execution_steps
    conversation_state = agent.runtime.execute_conversation_tools(
        contract=contract,
        query=query,
        session=session,
        agent_plan=agent_plan,
        max_web_results=max_web_results,
        emit=run_context.emit,
        execution_steps=execution_steps,
    )
    answer = str(conversation_state.get("answer", ""))
    citations = [
        item
        for item in list(conversation_state.get("citations", []) or [])
        if isinstance(item, AssistantCitation)
    ]
    verification_payload = dict(conversation_state.get("verification_report", {}) or {"status": "pass"})
    conversation_needs_human = verification_payload.get("status") == "clarify"
    session.last_relation = contract.relation
    citation_titles = [item.title for item in citations if item.title]
    active_research = None
    if agent._conversation_relation_updates_research_context(contract.relation):
        active_research = agent._make_active_research(
            relation=contract.relation,
            targets=list(contract.targets),
            titles=citation_titles,
            requested_fields=list(contract.requested_fields),
            required_modalities=list(contract.required_modalities),
            answer_shape=contract.answer_shape,
            precision_requirement=contract.precision_requirement,
            clean_query=contract.clean_query,
        )
    if conversation_needs_human:
        verification = VerificationReport(
            status="clarify",
            missing_fields=_string_list(verification_payload.get("missing_fields")),
            unsupported_claims=_string_list(verification_payload.get("unsupported_claims")),
            contradictory_claims=_string_list(verification_payload.get("contradictory_claims")),
            recommended_action=str(verification_payload.get("recommended_action", "") or "ask_human"),
        )
        agent._store_pending_clarification(session=session, contract=contract)
        agent._remember_clarification_attempt(session=session, contract=contract, verification=verification)
    else:
        agent._clear_pending_clarification(session)
        agent._reset_clarification_tracking(session)
    agent.sessions.commit_turn(
        session,
        SessionTurn.from_contract(
            query=query,
            answer=answer,
            contract=contract,
            interaction_mode="conversation",
            titles=citation_titles,
        ),
        active=active_research,
    )
    response = AssistantResponse(
        session_id=run_context.session_id,
        interaction_mode="conversation",
        answer=answer,
        citations=citations,
        query_contract=contract.model_dump(),
        research_plan_summary=agent_plan,
        runtime_summary=agent._runtime_summary(
            contract=contract,
            session=session,
            tool_plan=agent_plan,
            execution_steps=execution_steps,
            verification_report=verification_payload,
            citations=citations,
        ),
        execution_steps=execution_steps,
        verification_report=verification_payload,
        needs_human=conversation_needs_human,
        clarification_question=agent._clarification_question(contract, session) if conversation_needs_human else "",
        clarification_options=agent._clarification_options(contract) if conversation_needs_human else [],
    )
    return response.model_dump()


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []
