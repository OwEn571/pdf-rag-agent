from __future__ import annotations

from app.domain.models import CandidatePaper, QueryContract
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
