from __future__ import annotations

from typing import Any

from app.domain.models import QueryContract


RESEARCH_CONTEXT_CONVERSATION_RELATIONS = {
    "library_status",
    "library_recommendation",
    "memory_followup",
    "library_citation_ranking",
    "memory_synthesis",
}

LEGACY_TOOL_NAME_ALIASES = {
    "understand_user_intent": "read_memory",
    "reflect_previous_answer": "read_memory",
    "read_conversation_memory": "read_memory",
    "answer_from_memory": "read_memory",
    "synthesize_previous_results": "read_memory",
    "recover_previous_recommendation_candidates": "read_memory",
    "web_citation_lookup": "web_search",
    "rank_by_verified_citation_count": "web_search",
    "answer_conversation": "compose",
    "get_library_status": "compose",
    "get_library_recommendation": "compose",
    "resolve_ambiguity": "compose",
    "compose_or_ask_human": "compose",
    "detect_ambiguity": "ask_human",
    "clarification_limit": "ask_human",
    "retry_research": "search_corpus",
}


def conversation_relation_updates_research_context(relation: str) -> bool:
    return relation in RESEARCH_CONTEXT_CONVERSATION_RELATIONS


def note_values(*, notes: list[str], prefix: str) -> list[str]:
    return [item.removeprefix(prefix) for item in notes if item.startswith(prefix)]


def note_value(*, notes: list[str], prefix: str) -> str:
    values = note_values(notes=notes, prefix=prefix)
    return values[0] if values else ""


def note_float(*, notes: list[str], prefix: str) -> float | None:
    value = note_value(notes=notes, prefix=prefix)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def contract_answer_slots(contract: QueryContract) -> list[str]:
    slots = [str(item).strip() for item in list(getattr(contract, "answer_slots", []) or []) if str(item).strip()]
    if slots:
        return list(dict.fromkeys(slots))
    notes = [str(item) for item in list(contract.notes or [])]
    return note_values(notes=notes, prefix="answer_slot=")


def contract_topic_state(contract: QueryContract) -> str:
    notes = [str(item) for item in list(contract.notes or [])]
    value = note_value(notes=notes, prefix="topic_state=")
    if value in {"continue", "switch", "new"}:
        return value
    if contract.continuation_mode == "followup":
        return "continue"
    if contract.continuation_mode == "context_switch":
        return "switch"
    return "new"


def contract_allows_active_context_override(contract: QueryContract) -> bool:
    return contract_topic_state(contract) == "continue"


def observed_tool_names(execution_steps: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for step in execution_steps:
        node = str(step.get("node", "") if isinstance(step, dict) else "")
        if node.startswith("agent_tool:"):
            names.append(node.split(":", 1)[1])
        elif node in {"query_contract_extractor", "agent_planner", "compound_planner", "citation_rank_planner"}:
            continue
        elif node.startswith("compound_task:"):
            names.append("compose")
    return list(dict.fromkeys(name for name in names if name))


def canonical_tools(*, raw_tools: list[Any], aliases: dict[str, str], canonical_names: set[str]) -> list[str]:
    canonical: list[str] = []
    for raw in raw_tools:
        tool = str(raw)
        mapped = aliases.get(tool, tool)
        if mapped in canonical_names and mapped not in canonical:
            canonical.append(mapped)
    return canonical


def canonical_agent_tool(*, tool: str, aliases: dict[str, str], canonical_names: set[str]) -> str:
    if tool in canonical_names:
        return tool
    return aliases.get(tool, "compose")


def intent_kind_from_contract(contract: QueryContract) -> str:
    if contract.interaction_mode == "research":
        return "research"
    if contract.relation.startswith("library"):
        return "meta_library"
    if contract.relation.startswith("memory"):
        return "memory_op"
    return "smalltalk"
