from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from app.domain.models import QueryContract, VerificationReport
from app.services.contract_normalization import normalize_contract_targets, normalize_lookup_text, normalize_modalities


TargetNormalizer = Callable[[list[str], list[str]], list[str]]

ALLOWED_COMPOUND_SUBTASK_RELATIONS = {
    "library_status",
    "library_recommendation",
    "origin_lookup",
    "formula_lookup",
    "followup_research",
    "entity_definition",
    "topology_discovery",
    "topology_recommendation",
    "figure_question",
    "paper_summary_results",
    "metric_value_lookup",
    "concept_definition",
    "paper_recommendation",
    "general_question",
    "comparison_synthesis",
}


def default_compound_target_normalizer(targets: list[str], requested_fields: list[str]) -> list[str]:
    return normalize_contract_targets(
        targets=targets,
        requested_fields=requested_fields,
        canonicalize_targets=lambda values: values,
    )


def compound_task_label(contract: QueryContract) -> str:
    if contract.relation == "library_status":
        return "查看论文库概览和文章预览"
    if contract.relation == "library_recommendation":
        return "从库内给出默认推荐"
    if contract.relation == "formula_lookup":
        target = contract.targets[0] if contract.targets else "目标对象"
        return f"查询 {target} 公式"
    if contract.relation == "comparison_synthesis":
        target_text = " 和 ".join(contract.targets) if contract.targets else "前面结果"
        return f"比较 {target_text}"
    return contract.clean_query


def compound_subtask_contract_from_payload(
    payload: object,
    *,
    fallback_query: str,
    index: int,
    target_normalizer: TargetNormalizer = default_compound_target_normalizer,
) -> QueryContract | None:
    if not isinstance(payload, dict):
        return None
    continuation_mode = str(payload.get("continuation_mode", "") or "").strip().lower()
    if continuation_mode not in {"fresh", "followup", "context_switch"}:
        continuation_mode = "fresh" if index == 0 else "followup"
    clean_query = " ".join(str(payload.get("clean_query", "") or fallback_query).strip().split())
    raw_targets = payload.get("targets", [])
    targets = [str(item).strip() for item in raw_targets if str(item).strip()] if isinstance(raw_targets, list) else []
    raw_answer_slots = payload.get("answer_slots", [])
    if isinstance(raw_answer_slots, str):
        raw_answer_slots = [raw_answer_slots]
    answer_slots = [str(item).strip() for item in raw_answer_slots if str(item).strip()] if isinstance(raw_answer_slots, list) else []
    raw_requested_fields = payload.get("requested_fields", [])
    requested_fields = [str(item).strip() for item in raw_requested_fields if str(item).strip()] if isinstance(raw_requested_fields, list) else []
    targets = target_normalizer(targets, requested_fields)
    relation = str(payload.get("relation", "") or "").strip()
    if relation not in ALLOWED_COMPOUND_SUBTASK_RELATIONS:
        relation = compound_subtask_relation_from_slots(
            answer_slots=answer_slots,
            requested_fields=requested_fields,
            targets=targets,
        )
    if relation not in ALLOWED_COMPOUND_SUBTASK_RELATIONS:
        return None
    interaction_mode = str(payload.get("interaction_mode", "") or "").strip().lower()
    if interaction_mode not in {"conversation", "research"}:
        interaction_mode = "conversation" if relation in {"library_status", "library_recommendation", "comparison_synthesis"} else "research"
    if relation in {"library_status", "library_recommendation", "comparison_synthesis"}:
        interaction_mode = "conversation"
    if relation in {"library_status", "library_recommendation"}:
        targets = []
        requested_fields = []
    raw_required_modalities = payload.get("required_modalities", [])
    required_modalities = normalize_modalities(
        [str(item).strip() for item in raw_required_modalities if str(item).strip()] if isinstance(raw_required_modalities, list) else [],
        relation=relation,
    )
    if relation == "formula_lookup":
        requested_fields = [*requested_fields, *[field for field in ["formula", "variable_explanation"] if field not in requested_fields]]
        required_modalities = [*required_modalities, *[modality for modality in ["page_text", "table"] if modality not in required_modalities]]
        interaction_mode = "research"
    if interaction_mode == "conversation":
        required_modalities = []
    elif not required_modalities:
        required_modalities = ["page_text", "paper_card"]
    if interaction_mode == "research" and not requested_fields:
        requested_fields = ["answer"]
    answer_shape = str(payload.get("answer_shape", "") or "").strip().lower()
    if answer_shape not in {"bullets", "narrative", "table"}:
        answer_shape = "table" if relation == "comparison_synthesis" else "narrative"
    precision_requirement = str(payload.get("precision_requirement", "") or "").strip().lower()
    if precision_requirement not in {"exact", "high", "normal"}:
        precision_requirement = "exact" if relation in {"formula_lookup", "metric_value_lookup"} else "high"
    raw_notes = payload.get("notes", [])
    notes = [str(item).strip() for item in raw_notes if str(item).strip()] if isinstance(raw_notes, list) else []
    notes = list(dict.fromkeys([*notes, "compound_subtask", *[f"answer_slot={slot}" for slot in answer_slots], f"subtask_{relation}"]))
    return QueryContract(
        clean_query=clean_query,
        interaction_mode=interaction_mode,
        relation=relation,
        targets=targets,
        answer_slots=answer_slots,
        requested_fields=requested_fields,
        required_modalities=required_modalities,
        answer_shape=answer_shape,
        precision_requirement=precision_requirement,  # type: ignore[arg-type]
        continuation_mode=continuation_mode,  # type: ignore[arg-type]
        notes=notes,
    )


def compound_subtask_relation_from_slots(
    *,
    answer_slots: list[str],
    requested_fields: list[str],
    targets: list[str],
) -> str:
    slots = {"_".join(str(item or "").strip().lower().replace("-", "_").split()) for item in answer_slots}
    fields = {"_".join(str(item or "").strip().lower().replace("-", "_").split()) for item in requested_fields}
    tokens = slots | fields
    if "library_status" in tokens:
        return "library_status"
    if "library_recommendation" in tokens:
        return "library_recommendation"
    if "comparison" in tokens or "synthesis" in tokens:
        return "comparison_synthesis"
    if "origin" in tokens or {"paper_title", "year"} <= tokens:
        return "origin_lookup"
    if "formula" in tokens:
        return "formula_lookup"
    if "followup_research" in tokens or "followup_papers" in tokens:
        return "followup_research"
    if "figure" in tokens or "figure_conclusion" in tokens:
        return "figure_question"
    if "metric_value" in tokens:
        return "metric_value_lookup"
    if "paper_summary" in tokens or "summary" in tokens or "results" in tokens:
        return "paper_summary_results"
    if "paper_recommendation" in tokens or "recommended_papers" in tokens:
        return "paper_recommendation"
    if "topology_recommendation" in tokens or "best_topology" in tokens:
        return "topology_recommendation"
    if "topology_discovery" in tokens or "relevant_papers" in tokens:
        return "topology_discovery"
    if "entity_definition" in tokens or "entity_type" in tokens or ("definition" in tokens and targets):
        return "entity_definition"
    if "concept_definition" in tokens or "definition" in tokens:
        return "concept_definition"
    return "general_question"


def merge_redundant_field_subtasks(subcontracts: list[QueryContract]) -> list[QueryContract]:
    mergeable_relations = {
        "paper_summary_results",
        "metric_value_lookup",
        "entity_definition",
        "concept_definition",
        "formula_lookup",
        "figure_question",
        "general_question",
        "followup_research",
    }
    merged: list[QueryContract] = []
    by_key: dict[tuple[str, str, tuple[str, ...]], int] = {}
    precision_rank = {"normal": 0, "high": 1, "exact": 2}
    for contract in subcontracts:
        normalized_targets = tuple(normalize_lookup_text(target) for target in contract.targets if target)
        key = (contract.interaction_mode, contract.relation, normalized_targets)
        if contract.relation not in mergeable_relations or key not in by_key:
            by_key[key] = len(merged)
            merged.append(contract)
            continue
        existing_index = by_key[key]
        existing = merged[existing_index]
        requested_fields = list(dict.fromkeys([*existing.requested_fields, *contract.requested_fields]))
        required_modalities = list(dict.fromkeys([*existing.required_modalities, *contract.required_modalities]))
        notes = list(dict.fromkeys([*existing.notes, *contract.notes, "merged_same_target_fields"]))
        clean_query = existing.clean_query
        if contract.clean_query and contract.clean_query not in clean_query:
            clean_query = f"{clean_query}；{contract.clean_query}"
        precision = (
            contract.precision_requirement
            if precision_rank.get(contract.precision_requirement, 0) > precision_rank.get(existing.precision_requirement, 0)
            else existing.precision_requirement
        )
        merged[existing_index] = existing.model_copy(
            update={
                "clean_query": clean_query,
                "requested_fields": requested_fields or existing.requested_fields,
                "required_modalities": required_modalities or existing.required_modalities,
                "precision_requirement": precision,
                "notes": notes,
            }
        )
    return merged


def compound_section_heading(*, contract: QueryContract, index: int) -> str:
    return f"## {index}. {compound_task_label(contract)}"


def compound_research_progress_markdown(*, contract: QueryContract, index: int) -> str:
    heading = compound_section_heading(contract=contract, index=index)
    if contract.relation == "formula_lookup":
        target = contract.targets[0] if contract.targets else "目标对象"
        return f"{heading}\n\n好的，我现在去查询 **{target}** 的公式。"
    return heading


def demote_markdown_headings(answer: str) -> str:
    return re.sub(r"^(#{1,5})\s+", lambda match: "#" + match.group(1) + " ", str(answer or "").strip(), flags=re.M)


def format_compound_section(*, contract: QueryContract, answer: str, index: int) -> str:
    normalized = demote_markdown_headings(str(answer or "").strip())
    return f"{compound_section_heading(contract=contract, index=index)}\n\n{normalized}".strip()


def compound_task_result_from_task_payload(
    task_result: dict[str, Any],
    *,
    fallback_contract: QueryContract,
) -> dict[str, Any]:
    contract = task_result.get("contract_obj")
    if not isinstance(contract, QueryContract):
        raw_contract = task_result.get("contract")
        if isinstance(raw_contract, dict):
            try:
                contract = QueryContract.model_validate(raw_contract)
            except Exception:  # noqa: BLE001
                contract = fallback_contract
        else:
            contract = fallback_contract
    verification = task_result.get("verification_obj")
    if not isinstance(verification, VerificationReport):
        raw_verification = task_result.get("verification")
        if isinstance(raw_verification, dict):
            try:
                verification = VerificationReport.model_validate(raw_verification)
            except Exception:  # noqa: BLE001
                verification = VerificationReport(status="pass", recommended_action="task_subagent")
        else:
            verification = VerificationReport(status="pass", recommended_action="task_subagent")
    return {
        "contract": contract,
        "answer": str(task_result.get("answer", "") or ""),
        "citations": list(task_result.get("citations", []) or []),
        "claims": list(task_result.get("claims", []) or []),
        "evidence": list(task_result.get("evidence", []) or []),
        "verification": verification,
    }
