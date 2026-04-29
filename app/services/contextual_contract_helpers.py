from __future__ import annotations

from app.domain.models import ActiveResearch, CandidatePaper, QueryContract
from app.services.followup_intents import looks_like_contextual_metric_query
from app.services.query_shaping import is_short_acronym


def active_paper_reference_notes(*, notes: list[str], paper: CandidatePaper, marker: str) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *notes,
                marker,
                "resolved_from_conversation_memory",
                f"selected_paper_id={paper.paper_id}",
                "memory_title=" + paper.title,
            ]
        )
    )


def promote_contextual_metric_contract(contract: QueryContract) -> QueryContract:
    if contract.relation == "metric_value_lookup":
        return contract
    if not looks_like_contextual_metric_query(
        contract.clean_query,
        targets=list(contract.targets),
        is_short_acronym=is_short_acronym,
    ):
        return contract
    requested_fields = list(dict.fromkeys([*contract.requested_fields, "metric_value", "setting", "evidence"]))
    required_modalities = list(dict.fromkeys([*contract.required_modalities, "table", "caption", "page_text"]))
    answer_slots = list(dict.fromkeys([*contract.answer_slots, "metric_value"]))
    notes = list(dict.fromkeys([*contract.notes, "contextual_metric_query", "answer_slot=metric_value"]))
    return contract.model_copy(
        update={
            "relation": "metric_value_lookup",
            "answer_slots": answer_slots,
            "requested_fields": requested_fields,
            "required_modalities": required_modalities,
            "answer_shape": "narrative",
            "precision_requirement": "exact",
            "notes": notes,
        }
    )


def formula_answer_correction_contract(
    *,
    contract: QueryContract,
    active: ActiveResearch,
    paper: CandidatePaper | None,
) -> QueryContract:
    notes = list(
        dict.fromkeys(
            [
                *contract.notes,
                "formula_answer_correction",
                "prefer_scalar_objective",
                "answer_slot=formula",
            ]
        )
    )
    if paper is not None:
        notes = list(dict.fromkeys([*notes, f"selected_paper_id={paper.paper_id}", "memory_title=" + paper.title]))
    target = active.targets[0] if active.targets else (contract.targets[0] if contract.targets else "当前目标")
    scope = f"限定在论文《{paper.title}》中" if paper is not None else "沿用上一轮论文上下文"
    return QueryContract(
        clean_query=f"{target} 的公式是什么？{scope}重新查找目标函数或损失函数；上一条候选公式可能是梯度/推导式，不要优先返回梯度公式。",
        interaction_mode="research",
        relation="formula_lookup",
        targets=list(active.targets or contract.targets or [target]),
        requested_fields=["formula", "variable_explanation", "source"],
        required_modalities=["page_text", "table"],
        answer_shape="bullets",
        precision_requirement="exact",
        continuation_mode="followup",
        allow_web_search=contract.allow_web_search,
        notes=notes,
    )


def formula_location_followup_contract(
    *,
    contract: QueryContract,
    paper: CandidatePaper,
    target: str,
) -> QueryContract:
    notes = list(
        dict.fromkeys(
            [
                *contract.notes,
                "formula_location_followup",
                "resolved_from_user_paper_hint",
                f"selected_paper_id={paper.paper_id}",
                "memory_title=" + paper.title,
                "answer_slot=formula",
            ]
        )
    )
    return QueryContract(
        clean_query=f"{target} 的公式是什么？限定在论文《{paper.title}》中查找。",
        interaction_mode="research",
        relation="formula_lookup",
        targets=[target],
        requested_fields=["formula", "variable_explanation", "source"],
        required_modalities=["page_text", "table"],
        answer_shape="bullets",
        precision_requirement="exact",
        continuation_mode="followup",
        allow_web_search=contract.allow_web_search,
        notes=notes,
    )


def formula_contextual_paper_contract(
    *,
    contract: QueryContract,
    paper: CandidatePaper,
    target: str,
) -> QueryContract:
    notes = list(
        dict.fromkeys(
            [
                *contract.notes,
                "formula_contextual_paper_binding",
                f"selected_paper_id={paper.paper_id}",
                "memory_title=" + paper.title,
            ]
        )
    )
    return contract.model_copy(
        update={
            "clean_query": f"{target} 的公式是什么？限定在论文《{paper.title}》中查找。",
            "relation": "formula_lookup",
            "requested_fields": ["formula", "variable_explanation", "source"],
            "required_modalities": ["page_text", "table"],
            "answer_shape": "bullets",
            "precision_requirement": "exact",
            "continuation_mode": "followup",
            "notes": notes,
        }
    )
