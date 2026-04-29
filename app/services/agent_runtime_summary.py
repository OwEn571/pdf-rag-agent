from __future__ import annotations

from typing import Any

from app.domain.models import AssistantCitation, Claim, QueryContract
from app.services.agent_tools import all_agent_tool_names
from app.services.confidence import confidence_from_verification_report, confidence_payload
from app.services.contract_context import (
    LEGACY_TOOL_NAME_ALIASES,
    canonical_tools,
    contract_answer_slots,
    contract_topic_state,
    intent_kind_from_contract,
    note_float,
    note_value,
    note_values,
    observed_tool_names,
)


def build_runtime_summary(
    *,
    contract: QueryContract,
    active_research_context: dict[str, Any] | None = None,
    tool_plan: dict[str, Any] | None = None,
    research_plan: dict[str, Any] | None = None,
    execution_steps: list[dict[str, Any]] | None = None,
    verification_report: dict[str, Any] | None = None,
    claims: list[Claim] | None = None,
    citations: list[AssistantCitation] | None = None,
) -> dict[str, Any]:
    notes = [str(item) for item in list(contract.notes or [])]
    answer_slots = contract_answer_slots(contract)
    ambiguous_slots = note_values(notes=notes, prefix="ambiguous_slot=")
    topic_state = contract_topic_state(contract)
    planned_raw = list((tool_plan or {}).get("actions", []) or [])
    observed_raw = observed_tool_names(execution_steps or [])
    canonical_tool_names = all_agent_tool_names()
    planned = canonical_tools(
        raw_tools=planned_raw,
        aliases=LEGACY_TOOL_NAME_ALIASES,
        canonical_names=canonical_tool_names,
    )
    observed = canonical_tools(
        raw_tools=observed_raw,
        aliases=LEGACY_TOOL_NAME_ALIASES,
        canonical_names=canonical_tool_names,
    )
    verification_status = str((verification_report or {}).get("status") or "")
    verifier_confidence = confidence_payload(confidence_from_verification_report(verification_report or {}))
    selected_paper_id = note_value(notes=notes, prefix="selected_paper_id=")
    selected_title = note_value(notes=notes, prefix="memory_title=")
    binding_sources = [
        note
        for note in notes
        if note
        in {
            "resolved_from_conversation_memory",
            "resolved_from_user_paper_hint",
            "formula_contextual_paper_binding",
            "formula_location_followup",
            "exclude_previous_focus",
        }
    ]
    contract_context = {
        "topic_state": topic_state,
        "target_aliases": note_values(notes=notes, prefix="target_alias="),
        "selected_paper_id": selected_paper_id,
        "selected_title": selected_title,
        "binding_sources": binding_sources,
        "needs_clarification": "intent_needs_clarification" in notes,
        "clarification_reasons": list(
            dict.fromkeys([*ambiguous_slots, *[note for note in notes if note in {"low_intent_confidence"}]])
        ),
    }
    claim_source_counts: dict[str, int] = {}
    for claim in list(claims or []):
        if not isinstance(claim, Claim):
            continue
        source = str(dict(claim.structured_data or {}).get("source") or "legacy_solver")
        claim_source_counts[source] = claim_source_counts.get(source, 0) + 1
    summary = {
        "intent": {
            "kind": note_value(notes=notes, prefix="intent_kind=") or intent_kind_from_contract(contract),
            "confidence": note_float(notes=notes, prefix="intent_confidence="),
            "goal": contract.clean_query,
            "mode": contract.interaction_mode,
            "relation": contract.relation,
            "targets": list(contract.targets),
            "answer_slots": answer_slots,
            "ambiguous_slots": ambiguous_slots,
            "needs_local_corpus": contract.interaction_mode == "research",
            "needs_web": bool(contract.allow_web_search),
            "refers_previous_turn": contract.continuation_mode == "followup",
            "topic_state": topic_state,
            "active_topic": note_value(notes=notes, prefix="active_topic="),
        },
        "tool_loop": {
            "mode": "react_loop",
            "planned_tools": planned,
            "observed_tools": observed,
            "raw_planned_tools": [str(item) for item in planned_raw],
            "raw_observed_tools": observed_raw,
            "legacy_tools": [
                tool
                for tool in observed_raw
                if tool not in canonical_tool_names and tool not in {"agent_loop", "conversation_agent_loop"}
            ],
        },
        "grounding": {
            "verification_status": verification_status or "pending",
            "confidence": verifier_confidence,
            "claim_count": len(claims or []),
            "citation_count": len(citations or []),
            "claim_sources": claim_source_counts,
            "research_solver_sequence": list((research_plan or {}).get("solver_sequence", []) or []),
        },
        "contract_context": contract_context,
    }
    if active_research_context is not None:
        summary["active_research_context"] = active_research_context
    return summary
