from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.domain.models import QueryContract, ResearchPlan
from app.services.contracts.context import contract_answer_slots
from app.services.planning.solver_goals import fallback_goals_from_query, looks_like_metric_goal


@dataclass(frozen=True)
class ResearchPlanContext:
    clean_query: str = ""
    targets: tuple[str, ...] = ()
    requested_fields: tuple[str, ...] = ()
    goals: frozenset[str] = frozenset()
    required_modalities: tuple[str, ...] = ("page_text",)


def build_research_plan(*, contract: QueryContract, settings: Any) -> ResearchPlan:
    return build_research_plan_from_context(
        context=research_plan_context_from_contract(contract),
        settings=settings,
    )


def build_research_plan_from_context(*, context: ResearchPlanContext, settings: Any) -> ResearchPlan:
    goals = set(context.goals)
    return ResearchPlan(
        paper_recall_mode=paper_recall_mode_for_context(context),
        paper_limit=paper_limit_for_goals(goals, default=int(settings.paper_limit_default)),
        evidence_limit=evidence_limit_for_goals(goals, default=int(settings.evidence_limit_default)),
        solver_sequence=solver_sequence_for_goals(goals, list(context.required_modalities)),
        required_claims=required_claims_for_goals(goals),
        retry_budget=int(settings.llm_retry_budget),
    )


def research_plan_context_from_contract(contract: QueryContract) -> ResearchPlanContext:
    return ResearchPlanContext(
        clean_query=str(contract.clean_query or ""),
        targets=tuple(str(item).strip() for item in list(contract.targets or []) if str(item).strip()),
        requested_fields=tuple(str(item).strip() for item in list(contract.requested_fields or []) if str(item).strip()),
        goals=frozenset(research_plan_goals(contract)),
        required_modalities=tuple(str(item).strip() for item in list(contract.required_modalities or []) if str(item).strip())
        or ("page_text",),
    )


def paper_recall_mode_for_context(context: ResearchPlanContext) -> str:
    return "anchor_first" if context.targets else "broad"


# ============================================================================
# Research planning configuration tables.
# These dicts/sets map goals, relations, and answer-slot names to resource
# limits, solver sequences, and claim priorities.  They are data, not code —
# adding a new goal type is a one-line entry in the appropriate table.
# ============================================================================

# goal → minimum paper limit (higher = broader retrieval)
_GOAL_PAPER_LIMITS: dict[frozenset[str], int] = {
    frozenset({"paper_title", "year", "origin"}): 16,
    frozenset({"followup_papers", "recommended_papers"}): 10,
    frozenset({
        "summary", "results", "metric_value", "figure_conclusion",
        "relevant_papers", "best_topology", "topology_types", "langgraph_recommendation",
    }): 8,
}

# goal → minimum evidence limit
_GOAL_EVIDENCE_LIMITS: dict[frozenset[str], int] = {
    frozenset({"paper_title", "year", "origin", "summary", "results", "key_findings"}): 36,
    frozenset({"metric_value", "setting"}): 32,
    frozenset({"figure_conclusion", "caption"}): 30,
    frozenset({
        "followup_papers", "candidate_relationship", "strict_followup",
        "relevant_papers", "best_topology", "topology_types", "langgraph_recommendation",
    }): 28,
    frozenset({
        "formula", "definition", "entity_type", "role_in_context",
        "mechanism", "examples", "answer", "general_answer",
    }): 24,
}

# goal set → solver stage to prepend
_GOAL_SOLVER_STAGES: dict[frozenset[str], str] = {
    frozenset({"followup_papers", "candidate_relationship", "strict_followup"}): "followup_solver",
    frozenset({"formula", "variable_explanation"}): "formula_solver",
}

# Goals that require the text solver
_TEXT_SOLVER_GOALS: frozenset[str] = frozenset({
    "paper_title", "year", "origin", "summary", "results", "key_findings",
    "recommended_papers", "definition", "entity_type", "role_in_context",
    "mechanism", "examples", "relevant_papers", "best_topology",
    "topology_types", "langgraph_recommendation", "answer", "general_answer",
    "reward_model_requirement",
})

# Goals that activate the table / figure solvers
_TABLE_SOLVER_GOALS: frozenset[str] = frozenset({"metric_value", "setting", "results"})
_FIGURE_SOLVER_GOALS: frozenset[str] = frozenset({"figure_conclusion", "caption"})

# Ordered claim priority — claims earlier in the list appear first in output
_REQUIRED_CLAIMS_PRIORITY: tuple[str, ...] = (
    "paper_title", "year", "formula", "variable_explanation",
    "followup_papers", "recommended_papers", "best_topology",
    "langgraph_recommendation", "relevant_papers", "topology_types",
    "figure_conclusion", "metric_value", "setting", "summary", "results",
    "definition", "entity_type", "role_in_context", "mechanism",
    "reward_model_requirement", "evidence", "answer",
)

# answer_slot name → expanded goal set
_RESEARCH_GOAL_ALIASES: dict[str, frozenset[str]] = {
    "general_answer": frozenset({"answer"}),
    "origin": frozenset({"origin", "paper_title", "year"}),
    "source": frozenset({"source"}),
    "paper_title": frozenset({"paper_title"}),
    "year": frozenset({"year"}),
    "formula": frozenset({"formula"}),
    "variable_explanation": frozenset({"variable_explanation"}),
    "followup_research": frozenset({"followup_papers", "candidate_relationship", "evidence"}),
    "relationship": frozenset({"candidate_relationship"}),
    "paper_summary": frozenset({"summary", "results", "evidence"}),
    "summary": frozenset({"summary"}),
    "results": frozenset({"results"}),
    "key_findings": frozenset({"key_findings"}),
    "figure": frozenset({"figure_conclusion", "caption", "evidence"}),
    "figure_conclusion": frozenset({"figure_conclusion"}),
    "caption": frozenset({"caption"}),
    "metric_value": frozenset({"metric_value", "setting"}),
    "metric_definition": frozenset({"metric_value", "setting", "definition"}),
    "paper_recommendation": frozenset({"recommended_papers", "rationale"}),
    "recommended_papers": frozenset({"recommended_papers"}),
    "topology_discovery": frozenset({"relevant_papers", "topology_types"}),
    "topology_recommendation": frozenset({"best_topology", "langgraph_recommendation"}),
    "definition": frozenset({"definition", "mechanism"}),
    "entity_definition": frozenset({"entity_type", "definition", "mechanism", "role_in_context"}),
    "concept_definition": frozenset({"definition", "mechanism", "examples"}),
    "entity_type": frozenset({"entity_type"}),
    "role_in_context": frozenset({"role_in_context"}),
    "mechanism": frozenset({"mechanism"}),
    "examples": frozenset({"examples"}),
    "training_component": frozenset({"training_component", "mechanism", "reward_model_requirement", "evidence"}),
    "reward_model_requirement": frozenset({"reward_model_requirement"}),
    "evidence": frozenset({"evidence"}),
    "answer": frozenset({"answer"}),
}

# relation → goals inherited from legacy contract compatibility
_RELATION_GOAL_COMPATIBILITY: dict[str, frozenset[str]] = {
    "origin_lookup": frozenset({"paper_title", "year"}),
    "formula_lookup": frozenset({"formula", "variable_explanation", "source"}),
    "paper_summary_results": frozenset({"summary", "results", "evidence"}),
    "paper_recommendation": frozenset({"recommended_papers", "rationale"}),
    "followup_research": frozenset({"followup_papers", "candidate_relationship", "evidence"}),
    "entity_definition": frozenset({"entity_type", "definition", "mechanism", "role_in_context"}),
    "concept_definition": frozenset({"definition", "mechanism", "examples"}),
    "topology_discovery": frozenset({"relevant_papers", "topology_types"}),
    "topology_recommendation": frozenset({"best_topology", "langgraph_recommendation"}),
    "figure_question": frozenset({"figure_conclusion", "caption", "evidence"}),
    "metric_value_lookup": frozenset({"metric_value", "setting", "evidence"}),
}


# ============================================================================
# Functions that consume the tables above
# ============================================================================


def paper_limit_for_goals(goals: set[str], *, default: int) -> int:
    limit = default
    for goal_set, min_limit in _GOAL_PAPER_LIMITS.items():
        if goals & set(goal_set):
            limit = max(limit, min_limit)
    return limit


def evidence_limit_for_goals(goals: set[str], *, default: int) -> int:
    limit = default
    for goal_set, min_limit in _GOAL_EVIDENCE_LIMITS.items():
        if goals & set(goal_set):
            limit = max(limit, min_limit)
    return limit


def solver_sequence_for_goals(goals: set[str], modalities: list[str]) -> list[str]:
    sequence: list[str] = []
    for goal_set, solver_name in _GOAL_SOLVER_STAGES.items():
        if goals & set(goal_set) and solver_name not in sequence:
            sequence.append(solver_name)
    if goals & _TEXT_SOLVER_GOALS:
        sequence.append("text_solver")
    if "table" in modalities or goals & _TABLE_SOLVER_GOALS:
        sequence.append("table_solver")
    if "figure" in modalities or goals & _FIGURE_SOLVER_GOALS:
        sequence.append("figure_solver")
    if not sequence:
        sequence.append("text_solver")
    return list(dict.fromkeys(sequence))


def required_claims_for_goals(goals: set[str]) -> list[str]:
    claims = [item for item in _REQUIRED_CLAIMS_PRIORITY if item in goals]
    return claims or ["answer"]


def research_plan_goals(contract: QueryContract) -> set[str]:
    notes = [str(note) for note in contract.notes]
    values: list[str] = [
        *contract_answer_slots(contract),
        *list(contract.requested_fields or []),
    ]
    goals: set[str] = set()
    for value in values:
        goals.update(normalize_research_goal(value))
    legacy_goals = goals_from_relation_compatibility(contract.relation)
    if "structured_intent" not in notes and legacy_goals:
        goals.update(legacy_goals)
    if not goals or goals <= {"answer", "general_answer"}:
        goals.update(fallback_goals_from_query(contract.clean_query, targets=contract.targets))
    if not goals or goals <= {"answer", "general_answer"}:
        goals.update(legacy_goals)
    for modality in contract.required_modalities:
        if modality == "figure":
            goals.add("figure_conclusion")
        elif modality in {"table", "caption"} and looks_like_metric_goal(
            contract.clean_query,
            goals,
        ):
            goals.add("metric_value")
    if "formula" in goals:
        goals.add("source")
    if "origin" in goals:
        goals.update({"paper_title", "year"})
    if "paper_summary" in goals:
        goals.update({"summary", "results"})
    if "training_component" in goals:
        goals.update({"mechanism", "reward_model_requirement", "evidence"})
    return goals or {"answer"}


def normalize_research_goal(value: str) -> set[str]:
    key = "_".join(str(value or "").strip().lower().replace("-", "_").split())
    entry = _RESEARCH_GOAL_ALIASES.get(key)
    return set(entry) if entry is not None else ({key} if key else set())


def goals_from_relation_compatibility(relation: str) -> set[str]:
    entry = _RELATION_GOAL_COMPATIBILITY.get(str(relation or ""))
    return set(entry) if entry is not None else set()
