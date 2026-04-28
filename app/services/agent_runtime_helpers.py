from __future__ import annotations

from typing import Any

from app.domain.models import QueryContract
from app.services.confidence import confidence_from_contract, should_ask_human


def configured_max_steps(agent_settings: Any, *, fallback: int) -> int:
    value = getattr(agent_settings, "max_agent_steps", fallback)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(1, parsed)


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
