from __future__ import annotations

from dataclasses import dataclass

from app.domain.models import Claim, QueryContract, ResearchPlan
from app.services.contracts.context import contract_answer_slots
from app.services.intents.marker_matching import (
    MarkerProfile,
    normalized_query_text,
    query_matches_any,
)
from app.services.intents.research import ORIGIN_LOOKUP_MARKERS, RESEARCH_SLOT_MARKERS


SOLVER_GOAL_MARKERS: dict[str, MarkerProfile] = {
    "origin": (*ORIGIN_LOOKUP_MARKERS, "提出的"),
    "formula": RESEARCH_SLOT_MARKERS["formula"],
    "followup": (*RESEARCH_SLOT_MARKERS["followup_research"], "successor"),
    "figure": RESEARCH_SLOT_MARKERS["figure"],
    "summary": RESEARCH_SLOT_MARKERS["paper_summary"],
    "metric": (*RESEARCH_SLOT_MARKERS["metric_value"], "win rate"),
    "recommendation": ("推荐", "值得", "入门", "recommend"),
    "definition_targeted": ("是什么", "什么意思", "定义"),
    "definition_english": ("what is", "what are"),
}


@dataclass(frozen=True)
class ClaimGoalContext:
    clean_query: str = ""
    targets: tuple[str, ...] = ()
    required_claims: tuple[str, ...] = ()
    answer_slots: tuple[str, ...] = ()
    requested_fields: tuple[str, ...] = ()
    required_modalities: tuple[str, ...] = ()


def append_unique_claims(claims: list[Claim], new_claims: list[Claim]) -> None:
    existing = {(claim.claim_type, claim.entity, claim.value) for claim in claims}
    for claim in new_claims:
        key = (claim.claim_type, claim.entity, claim.value)
        if key not in existing:
            existing.add(key)
            claims.append(claim)


def claim_goals(*, contract: QueryContract, plan: ResearchPlan) -> set[str]:
    return claim_goals_for_context(claim_goal_context_from_contract_plan(contract=contract, plan=plan))


def claim_goal_context_from_contract_plan(*, contract: QueryContract, plan: ResearchPlan) -> ClaimGoalContext:
    return ClaimGoalContext(
        clean_query=str(contract.clean_query or ""),
        targets=tuple(str(item).strip() for item in list(contract.targets or []) if str(item).strip()),
        required_claims=tuple(str(item).strip() for item in list(plan.required_claims or []) if str(item).strip()),
        answer_slots=tuple(contract_answer_slots(contract)),
        requested_fields=tuple(str(item).strip() for item in list(contract.requested_fields or []) if str(item).strip()),
        required_modalities=tuple(str(item).strip() for item in list(contract.required_modalities or []) if str(item).strip()),
    )


def claim_goals_for_context(context: ClaimGoalContext) -> set[str]:
    goals = {
        str(item).strip()
        for item in [
            *context.required_claims,
            *context.answer_slots,
            *context.requested_fields,
        ]
        if str(item).strip()
    }
    if not goals or goals <= {"answer"}:
        goals.update(fallback_goals_from_query(context.clean_query, targets=list(context.targets)))
    for modality in context.required_modalities:
        if modality == "figure":
            goals.add("figure_conclusion")
        elif modality in {"table", "caption"} and looks_like_metric_goal(context.clean_query, goals):
            goals.add("metric_value")
    if "formula" in goals:
        goals.add("source")
    if "training_component" in goals:
        goals.update({"mechanism", "reward_model_requirement", "evidence"})
    return goals


def looks_like_metric_goal(query: str, goals: set[str]) -> bool:
    if goals & {"metric_value", "setting"}:
        return True
    normalized, compact = normalized_query_text(query)
    return query_matches_any(normalized, compact, SOLVER_GOAL_MARKERS["metric"])


def fallback_goals_from_query(query: str, *, targets: list[str]) -> set[str]:
    raw_query = str(query or "")
    normalized, compact = normalized_query_text(raw_query)
    goals: set[str] = set()
    if query_matches_any(normalized, compact, SOLVER_GOAL_MARKERS["origin"]):
        goals.update({"paper_title", "year"})
    if query_matches_any(normalized, compact, SOLVER_GOAL_MARKERS["formula"]):
        goals.add("formula")
    if query_matches_any(normalized, compact, SOLVER_GOAL_MARKERS["followup"]):
        goals.add("followup_papers")
    if query_matches_any(normalized, compact, SOLVER_GOAL_MARKERS["figure"]):
        goals.add("figure_conclusion")
    if query_matches_any(normalized, compact, SOLVER_GOAL_MARKERS["summary"]):
        goals.update({"summary", "results"})
    if looks_like_metric_goal(query, goals):
        goals.add("metric_value")
    if query_matches_any(normalized, compact, SOLVER_GOAL_MARKERS["recommendation"]):
        goals.add("recommended_papers")
    has_targeted_definition = bool(targets) and query_matches_any(
        normalized,
        raw_query,
        SOLVER_GOAL_MARKERS["definition_targeted"],
    )
    has_english_definition = query_matches_any(
        normalized,
        "",
        SOLVER_GOAL_MARKERS["definition_english"],
    )
    if has_targeted_definition or has_english_definition:
        goals.update({"entity_type", "definition", "mechanism"})
    return goals or {"answer"}
