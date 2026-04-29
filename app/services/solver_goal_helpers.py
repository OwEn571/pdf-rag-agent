from __future__ import annotations

from app.domain.models import Claim, QueryContract, ResearchPlan
from app.services.intent_marker_matching import (
    MarkerProfile,
    normalized_query_text,
    query_matches_any,
)
from app.services.research_intents import ORIGIN_LOOKUP_MARKERS, RESEARCH_SLOT_MARKERS


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


def append_unique_claims(claims: list[Claim], new_claims: list[Claim]) -> None:
    existing = {(claim.claim_type, claim.entity, claim.value) for claim in claims}
    for claim in new_claims:
        key = (claim.claim_type, claim.entity, claim.value)
        if key not in existing:
            existing.add(key)
            claims.append(claim)


def claim_goals(*, contract: QueryContract, plan: ResearchPlan) -> set[str]:
    goals = {
        str(item).strip()
        for item in [
            *list(plan.required_claims or []),
            *list(getattr(contract, "answer_slots", []) or []),
            *list(contract.requested_fields or []),
            *[
                str(note).split("=", 1)[1]
                for note in contract.notes
                if str(note).startswith("answer_slot=") and "=" in str(note)
            ],
        ]
        if str(item).strip()
    }
    if not goals or goals <= {"answer"}:
        goals.update(fallback_goals_from_query(contract.clean_query, targets=contract.targets))
    for modality in contract.required_modalities:
        if modality == "figure":
            goals.add("figure_conclusion")
        elif modality in {"table", "caption"} and looks_like_metric_goal(contract.clean_query, goals):
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
