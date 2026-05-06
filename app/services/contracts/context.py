from __future__ import annotations

import json
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


def note_json_values(*, notes: list[str], prefix: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for value in note_values(notes=notes, prefix=prefix):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            values.append(payload)
    return values


def note_json_value(*, notes: list[str], prefix: str) -> dict[str, Any]:
    values = note_json_values(notes=notes, prefix=prefix)
    return values[0] if values else {}


def notes_without_prefixes(*, notes: list[Any], prefixes: set[str]) -> list[str]:
    cleaned: list[str] = []
    for note in notes:
        text = str(note or "").strip()
        if not text:
            continue
        if any(text.startswith(prefix) for prefix in prefixes):
            continue
        cleaned.append(text)
    return cleaned


def contract_notes(contract: QueryContract) -> list[str]:
    return [str(item) for item in list(contract.notes or []) if str(item).strip()]


def contract_note_values(contract: QueryContract, *, prefix: str) -> list[str]:
    return note_values(notes=contract_notes(contract), prefix=prefix)


def contract_note_value(contract: QueryContract, *, prefix: str) -> str:
    return note_value(notes=contract_notes(contract), prefix=prefix)


def contract_note_json_value(contract: QueryContract, *, prefix: str) -> dict[str, Any]:
    return note_json_value(notes=contract_notes(contract), prefix=prefix)


def contract_note_float(contract: QueryContract, *, prefix: str) -> float | None:
    return note_float(notes=contract_notes(contract), prefix=prefix)


def contract_notes_without_prefixes(contract: QueryContract, *, prefixes: set[str]) -> list[str]:
    return notes_without_prefixes(notes=contract_notes(contract), prefixes=prefixes)


def has_note(*, notes: list[str], value: str) -> bool:
    expected = str(value or "").strip()
    if not expected:
        return False
    return expected in {str(item).strip() for item in notes if str(item).strip()}


def contract_has_note(contract: QueryContract, value: str) -> bool:
    return has_note(notes=contract_notes(contract), value=value)


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
    return contract_note_values(contract, prefix="answer_slot=")


def contract_topic_state(contract: QueryContract) -> str:
    value = contract_note_value(contract, prefix="topic_state=")
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
