from __future__ import annotations

from typing import Any

from app.domain.models import QueryContract, SessionContext
from app.services.contract_context import (
    contract_allows_active_context_override,
    contract_answer_slots,
    contract_topic_state,
)
from app.services.contract_normalization import normalize_lookup_text
from app.services.followup_intents import is_negative_correction_query
from app.services.research_planning import research_plan_goals


def target_binding_from_memory(*, session: SessionContext, target: str) -> dict[str, Any] | None:
    key = normalize_lookup_text(target)
    if not key:
        return None
    bindings = dict((session.working_memory or {}).get("target_bindings", {}) or {})
    binding = bindings.get(key)
    return dict(binding) if isinstance(binding, dict) else None


def active_memory_bindings(session: SessionContext) -> list[dict[str, Any]]:
    bindings = dict((session.working_memory or {}).get("target_bindings", {}) or {})
    selected: list[dict[str, Any]] = []
    for target in session.effective_active_research().targets:
        binding = bindings.get(normalize_lookup_text(target))
        if isinstance(binding, dict):
            selected.append(dict(binding))
    if len(selected) >= 2:
        return selected
    for binding in bindings.values():
        if isinstance(binding, dict) and binding not in selected:
            selected.append(dict(binding))
        if len(selected) >= 4:
            break
    return selected


def memory_binding_doc_ids(bindings: list[dict[str, Any]]) -> list[str]:
    doc_ids: list[str] = []
    for binding in bindings:
        for doc_id in list(binding.get("evidence_ids", []) or [])[:2]:
            if str(doc_id).strip():
                doc_ids.append(str(doc_id).strip())
        paper_id = str(binding.get("paper_id", "") or "").strip()
        if paper_id:
            doc_ids.append(f"paper::{paper_id}")
    return list(dict.fromkeys(doc_ids))


def apply_conversation_memory_to_contract(
    *,
    contract: QueryContract,
    session: SessionContext,
    selected_clarification_paper_id: str = "",
) -> QueryContract:
    if contract.interaction_mode != "research" or not contract.targets:
        return contract
    target_bindings = {
        target: binding
        for target in contract.targets
        if (binding := target_binding_from_memory(session=session, target=target))
    }
    topic_state = contract_topic_state(contract)
    goals = research_plan_goals(contract)
    if contract.relation == "origin_lookup" or "origin" in contract_answer_slots(contract) or goals & {"paper_title", "year"}:
        return contract
    allow_explicit_target_binding = bool(target_bindings) and topic_state != "switch"
    if "formula" in goals and topic_state != "continue":
        allow_explicit_target_binding = False
    if not contract_allows_active_context_override(contract) and not allow_explicit_target_binding:
        return contract
    if "exclude_previous_focus" in contract.notes or is_negative_correction_query(contract.clean_query):
        return contract
    if selected_clarification_paper_id:
        return contract
    notes = list(contract.notes)
    for target in contract.targets:
        binding = target_bindings.get(target)
        if not binding:
            continue
        paper_id = str(binding.get("paper_id", "") or "").strip()
        title = str(binding.get("title", "") or "").strip()
        if not paper_id:
            continue
        notes = list(dict.fromkeys([*notes, "resolved_from_conversation_memory", f"selected_paper_id={paper_id}"]))
        if title:
            notes.append("memory_title=" + title)
        return contract.model_copy(update={"continuation_mode": "followup", "notes": notes})
    return contract
