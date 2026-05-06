from __future__ import annotations

import re
from collections.abc import Callable

from app.domain.models import QueryContract, SessionContext
from app.services.contracts.context import contract_allows_active_context_override, contract_notes_without_prefixes
from app.services.contracts.normalization import normalize_lookup_text
from app.services.intents.followup_relationship import followup_relationship_recheck_requested
from app.services.planning.research import research_plan_context_from_contract

NormalizeTargets = Callable[[list[str], list[str]], list[str]]


def normalize_followup_direction_contract(
    *,
    contract: QueryContract,
    normalize_targets: NormalizeTargets,
) -> QueryContract:
    clean_query = " ".join(str(contract.clean_query or "").split())
    lowered = clean_query.lower()
    if not clean_query or ("后续" not in clean_query and "follow" not in lowered and "successor" not in lowered):
        return contract
    direction_match = re.search(
        r"^(?P<candidate>.+?)\s*(?:真是|是否是|是不是|是|算是|属于|is|are)\s*(?P<seed>.+?)\s*的?\s*(?:严格)?\s*(?:后续|扩展|继承|follow[- ]?up|successor)",
        clean_query,
        flags=re.IGNORECASE,
    )
    if direction_match is None:
        return contract
    candidate_title = " ".join(str(direction_match.group("candidate") or "").strip(" ，,。？?").split())
    candidate_title = re.sub(r"(?:真|真的|是否|是不是)\s*$", "", candidate_title).strip()
    seed_text = " ".join(str(direction_match.group("seed") or "").strip(" ，,。？?").split())
    seed_targets = normalize_targets([seed_text], contract.requested_fields)
    if not seed_targets:
        return contract
    notes = contract_notes_without_prefixes(contract, prefixes={"candidate_title="})
    if candidate_title:
        notes.append(f"candidate_title={candidate_title}")
    notes.append("followup_direction_resolved")
    requested_fields = list(dict.fromkeys([*contract.requested_fields, "candidate_relationship", "evidence"]))
    required_modalities = list(dict.fromkeys([*contract.required_modalities, "paper_card", "page_text"]))
    answer_shape = contract.answer_shape if contract.answer_shape in {"bullets", "table"} else "bullets"
    return contract.model_copy(
        update={
            "interaction_mode": "research",
            "relation": "followup_research",
            "targets": seed_targets,
            "requested_fields": requested_fields,
            "required_modalities": required_modalities,
            "answer_shape": answer_shape,
            "precision_requirement": "high",
            "notes": list(dict.fromkeys(notes)),
        }
    )


def inherit_followup_relationship_contract(
    *,
    contract: QueryContract,
    session: SessionContext,
    normalize_targets: NormalizeTargets,
) -> QueryContract:
    memory = dict(session.working_memory or {})
    relationship = dict(memory.get("last_followup_relationship", {}) or {})
    if not relationship:
        return contract
    clean_query = str(contract.clean_query or "")
    normalized_query = normalize_lookup_text(clean_query)
    if not followup_relationship_recheck_requested(clean_query, normalized_query):
        return contract
    if not contract_allows_active_context_override(contract) and contract.relation != "followup_research":
        return contract
    goals = set(research_plan_context_from_contract(contract).goals)
    relation_like = bool(goals & {"followup_papers", "candidate_relationship", "summary", "results", "answer", "general_answer"})
    if not relation_like and contract.continuation_mode != "followup":
        return contract
    seed_target = str(relationship.get("seed_target", "") or "").strip()
    candidate_title = str(relationship.get("candidate_title", "") or "").strip()
    if not seed_target or not candidate_title:
        return contract
    targets = normalize_targets([seed_target], contract.requested_fields) or [seed_target]
    notes = contract_notes_without_prefixes(contract, prefixes={"candidate_title="})
    notes.extend(
        [
            f"candidate_title={candidate_title}",
            "inherited_followup_relationship",
            "strict_followup_validation",
        ]
    )
    return contract.model_copy(
        update={
            "clean_query": f"{candidate_title} 是否是 {seed_target} 的严格后续工作？",
            "interaction_mode": "research",
            "relation": "followup_research",
            "targets": targets,
            "requested_fields": ["candidate_relationship", "strict_followup", "evidence"],
            "required_modalities": ["paper_card", "page_text"],
            "answer_shape": "bullets",
            "precision_requirement": "high",
            "continuation_mode": "followup",
            "notes": list(dict.fromkeys(notes)),
        }
    )
