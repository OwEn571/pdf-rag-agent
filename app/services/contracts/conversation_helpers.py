from __future__ import annotations

from collections.abc import Callable

from app.domain.models import CandidatePaper, QueryContract, SessionContext
from app.services.contracts.contextual_helpers import formula_answer_correction_contract
from app.services.contracts.context import contract_notes
from app.services.contracts.normalization import normalize_lookup_text
from app.services.contracts.conversation_memory import active_memory_bindings
from app.services.intents.followup import (
    is_formula_interpretation_followup_query,
    is_language_preference_followup,
    is_metric_definition_followup_query,
    is_memory_synthesis_query,
    looks_like_formula_answer_correction,
)
from app.services.intents.library import (
    citation_ranking_has_library_context,
    is_citation_ranking_query,
    is_library_count_query,
    is_library_status_query,
    is_scoped_library_recommendation_query,
    library_recommendation_contract,
    library_status_contract,
)
from app.services.intents.memory import is_memory_comparison_query


PaperHintLookupFn = Callable[[str], CandidatePaper | None]

CONVERSATION_TOOL_RELATIONS = {
    "greeting",
    "self_identity",
    "capability",
    "library_status",
    "library_recommendation",
    "memory_followup",
    "clarify_user_intent",
    "correction_without_context",
    "memory_synthesis",
    "library_citation_ranking",
}


def normalize_conversation_tool_contract(
    *,
    contract: QueryContract,
    clean_query: str,
    session: SessionContext,
    paper_from_query_hint: PaperHintLookupFn,
) -> QueryContract:
    if is_citation_ranking_query(clean_query) and citation_ranking_has_library_context(
        clean_query=clean_query,
        session=session,
    ):
        return QueryContract(
            clean_query=clean_query,
            interaction_mode="conversation",
            relation="library_citation_ranking",
            targets=[],
            requested_fields=["citation_count_ranking"],
            required_modalities=[],
            answer_shape="table",
            precision_requirement="normal",
            continuation_mode="followup" if session.turns else "fresh",
            allow_web_search=True,
            notes=["agent_tool", "external_metric", "citation_count_requires_web"],
        )
    active = session.effective_active_research()
    active_formula = active.relation == "formula_lookup" or "formula" in {str(field) for field in active.requested_fields}
    if active_formula and active.targets and looks_like_formula_answer_correction(clean_query):
        title = active.titles[0] if active.titles else ""
        paper = paper_from_query_hint(title) if title else None
        return formula_answer_correction_contract(contract=contract, active=active, paper=paper)
    had_formula_context = active_formula or any(
        turn.relation == "formula_lookup"
        or "formula" in {str(item) for item in list(turn.requested_fields or [])}
        or "formula" in {str(item) for item in list(turn.answer_slots or [])}
        for turn in session.turns[-3:]
    )
    if is_formula_interpretation_followup_query(clean_query, had_formula_context=had_formula_context):
        return QueryContract(
            clean_query=clean_query,
            interaction_mode="conversation",
            relation="memory_followup",
            targets=list(active.targets),
            requested_fields=["formula_interpretation"],
            required_modalities=[],
            answer_shape="narrative",
            precision_requirement="normal",
            continuation_mode="followup",
            notes=["agent_tool", "formula_interpretation_followup"],
        )
    if is_language_preference_followup(clean_query, has_turns=bool(session.turns)):
        return QueryContract(
            clean_query=clean_query,
            interaction_mode="conversation",
            relation="memory_followup",
            targets=list(active.targets),
            requested_fields=["answer_language_preference"],
            required_modalities=[],
            answer_shape="narrative",
            precision_requirement="normal",
            continuation_mode="followup",
            notes=["agent_tool", "answer_language_preference"],
        )
    active_metric_context = active.relation == "metric_value_lookup" or bool(
        {"metric_value", "setting", "results"} & {str(field) for field in active.requested_fields}
    )
    if is_metric_definition_followup_query(clean_query, has_metric_context=active_metric_context):
        targets = list(active.targets) or list(contract.targets)
        rewritten_query = clean_query
        if targets:
            rewritten_query = f"{', '.join(targets)} 的准确度/指标在论文中是怎么定义或计算的？"
        requested_fields = ["metric_value", "metric_definition", "setting", "evidence"]
        return QueryContract(
            clean_query=rewritten_query,
            interaction_mode="research",
            relation="metric_value_lookup",
            targets=targets,
            requested_fields=requested_fields,
            required_modalities=["table", "caption", "page_text"],
            answer_shape="bullets",
            precision_requirement="exact",
            continuation_mode="followup",
            allow_web_search=contract.allow_web_search,
            notes=list(
                dict.fromkeys(
                    [
                        *contract_notes(contract),
                        "metric_definition_followup",
                        "resolved_from_active_metric_context",
                        *[f"answer_slot={field}" for field in requested_fields],
                    ]
                )
            ),
        )
    if is_memory_synthesis_query(clean_query) and (
        len(active_memory_bindings(session)) >= 2
        or len(list(dict((session.working_memory or {}).get("last_compound_query", {}) or {}).get("subtasks", []) or [])) >= 2
    ):
        targets = list(dict.fromkeys(session.effective_active_research().targets))
        return QueryContract(
            clean_query=clean_query,
            interaction_mode="conversation",
            relation="memory_synthesis",
            targets=targets,
            requested_fields=["comparison", "synthesis"],
            required_modalities=[],
            answer_shape="table" if is_memory_comparison_query(normalize_lookup_text(clean_query)) else "narrative",
            precision_requirement="high",
            continuation_mode="followup",
            notes=["agent_tool", "conversation_memory_synthesis"],
        )
    if is_scoped_library_recommendation_query(clean_query) and not is_library_count_query(clean_query):
        return library_recommendation_contract(clean_query).model_copy(
            update={
                "notes": list(
                    dict.fromkeys([*contract_notes(contract), "agent_tool", "dynamic_library_recommendation"])
                )
            }
        )
    if is_library_status_query(clean_query):
        return library_status_contract(clean_query).model_copy(
            update={"notes": list(dict.fromkeys([*contract_notes(contract), "agent_tool", "dynamic_library_stats"]))}
        )
    if contract.relation in CONVERSATION_TOOL_RELATIONS:
        return contract.model_copy(
            update={
                "interaction_mode": "conversation",
                "required_modalities": [],
                "notes": list(dict.fromkeys([*contract_notes(contract), "agent_tool"])),
            }
        )
    return contract
