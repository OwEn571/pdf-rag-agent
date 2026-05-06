from __future__ import annotations

import logging
from typing import Any, Callable
from uuid import uuid4

from app.services.agent.context import AgentRunContext
from app.services.agent.loop import finish_agent_turn, run_compound_turn_if_needed, run_standard_turn
from app.services.contracts.session_context import compress_session_history_if_needed


logger = logging.getLogger(__name__)


def run_agent_chat_turn(
    *,
    agent: Any,
    query: str,
    session_id: str | None,
    use_web_search: bool,
    max_web_results: int,
    clarification_choice: dict[str, Any] | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved_session_id = session_id or uuid4().hex[:12]
    session = agent.sessions.get(resolved_session_id)
    compress_session_history_if_needed(
        session=session,
        clients=agent.clients,
        settings=agent.settings,
        sessions=agent.sessions,
    )
    run_context = AgentRunContext.create(
        session_id=resolved_session_id,
        session=session,
        event_callback=event_callback,
    )
    emit = run_context.emit

    emit("session", {"session_id": resolved_session_id})
    compound_result = run_compound_turn_if_needed(
        agent=agent,
        run_context=run_context,
        query=query,
        clarification_choice=clarification_choice,
    )
    if compound_result is not None:
        return finish_agent_turn(
            settings=agent.settings,
            run_context=run_context,
            final_payload=compound_result,
            logger=logger,
        )

    payload = run_standard_turn(
        agent=agent,
        run_context=run_context,
        query=query,
        use_web_search=use_web_search,
        max_web_results=max_web_results,
        clarification_choice=clarification_choice,
        stream_answer=event_callback is not None,
    )
    return finish_agent_turn(
        settings=agent.settings,
        run_context=run_context,
        final_payload=payload,
        logger=logger,
    )
