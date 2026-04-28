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


def intent_kind_from_contract(contract: QueryContract) -> str:
    if contract.interaction_mode == "research":
        return "research"
    if contract.relation.startswith("library"):
        return "meta_library"
    if contract.relation.startswith("memory"):
        return "memory_op"
    return "smalltalk"
