from __future__ import annotations

from typing import Any

from app.domain.models import QueryContract, ResearchPlan
from app.services.solver_goal_helpers import fallback_goals_from_query, looks_like_metric_goal


def build_research_plan(*, contract: QueryContract, settings: Any) -> ResearchPlan:
    goals = research_plan_goals(contract)
    return ResearchPlan(
        paper_recall_mode="anchor_first" if contract.targets else "broad",
        paper_limit=paper_limit_for_goals(goals, default=int(settings.paper_limit_default)),
        evidence_limit=evidence_limit_for_goals(goals, default=int(settings.evidence_limit_default)),
        solver_sequence=solver_sequence_for_goals(goals, contract.required_modalities),
        required_claims=required_claims_for_goals(goals),
        retry_budget=int(settings.llm_retry_budget),
    )


def paper_limit_for_goals(goals: set[str], *, default: int) -> int:
    limit = default
    if goals & {"paper_title", "year", "origin"}:
        limit = max(limit, 16)
    if goals & {"followup_papers", "recommended_papers"}:
        limit = max(limit, 10)
    if goals & {
        "summary",
        "results",
        "metric_value",
        "figure_conclusion",
        "relevant_papers",
        "best_topology",
        "topology_types",
        "langgraph_recommendation",
    }:
        limit = max(limit, 8)
    return limit


def evidence_limit_for_goals(goals: set[str], *, default: int) -> int:
    limit = default
    if goals & {"paper_title", "year", "origin", "summary", "results", "key_findings"}:
        limit = max(limit, 36)
    if goals & {"metric_value", "setting"}:
        limit = max(limit, 32)
    if goals & {"figure_conclusion", "caption"}:
        limit = max(limit, 30)
    if goals & {
        "followup_papers",
        "candidate_relationship",
        "strict_followup",
        "relevant_papers",
        "best_topology",
        "topology_types",
        "langgraph_recommendation",
    }:
        limit = max(limit, 28)
    if goals & {
        "formula",
        "definition",
        "entity_type",
        "role_in_context",
        "mechanism",
        "examples",
        "answer",
        "general_answer",
    }:
        limit = max(limit, 24)
    return limit


def solver_sequence_for_goals(goals: set[str], modalities: list[str]) -> list[str]:
    sequence: list[str] = []
    if goals & {"followup_papers", "candidate_relationship", "strict_followup"}:
        sequence.append("followup_solver")
    if goals & {"formula", "variable_explanation"}:
        sequence.append("formula_solver")
    wants_text = bool(
        goals
        & {
            "paper_title",
            "year",
            "origin",
            "summary",
            "results",
            "key_findings",
            "recommended_papers",
            "definition",
            "entity_type",
            "role_in_context",
            "mechanism",
            "examples",
            "relevant_papers",
            "best_topology",
            "topology_types",
            "langgraph_recommendation",
            "answer",
            "general_answer",
            "reward_model_requirement",
        }
    )
    wants_table = "table" in modalities or bool(goals & {"metric_value", "setting", "results"})
    wants_figure = "figure" in modalities or bool(goals & {"figure_conclusion", "caption"})
    if wants_text:
        sequence.append("text_solver")
    if wants_table:
        sequence.append("table_solver")
    if wants_figure:
        sequence.append("figure_solver")
    if not sequence:
        sequence.append("text_solver")
    return list(dict.fromkeys(sequence))


def required_claims_for_goals(goals: set[str]) -> list[str]:
    ordered = [
        "paper_title",
        "year",
        "formula",
        "variable_explanation",
        "followup_papers",
        "recommended_papers",
        "best_topology",
        "langgraph_recommendation",
        "relevant_papers",
        "topology_types",
        "figure_conclusion",
        "metric_value",
        "setting",
        "summary",
        "results",
        "definition",
        "entity_type",
        "role_in_context",
        "mechanism",
        "reward_model_requirement",
        "evidence",
        "answer",
    ]
    claims = [item for item in ordered if item in goals]
    return claims or ["answer"]


def research_plan_goals(contract: QueryContract) -> set[str]:
    notes = [str(note) for note in contract.notes]
    values: list[str] = [
        *list(getattr(contract, "answer_slots", []) or []),
        *list(contract.requested_fields or []),
        *[
            str(note).split("=", 1)[1]
            for note in notes
            if str(note).startswith("answer_slot=") and "=" in str(note)
        ],
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
    aliases = {
        "general_answer": {"answer"},
        "origin": {"origin", "paper_title", "year"},
        "source": {"source"},
        "paper_title": {"paper_title"},
        "year": {"year"},
        "formula": {"formula"},
        "variable_explanation": {"variable_explanation"},
        "followup_research": {"followup_papers", "candidate_relationship", "evidence"},
        "relationship": {"candidate_relationship"},
        "paper_summary": {"summary", "results", "evidence"},
        "summary": {"summary"},
        "results": {"results"},
        "key_findings": {"key_findings"},
        "figure": {"figure_conclusion", "caption", "evidence"},
        "figure_conclusion": {"figure_conclusion"},
        "caption": {"caption"},
        "metric_value": {"metric_value", "setting"},
        "metric_definition": {"metric_value", "setting", "definition"},
        "paper_recommendation": {"recommended_papers", "rationale"},
        "recommended_papers": {"recommended_papers"},
        "topology_discovery": {"relevant_papers", "topology_types"},
        "topology_recommendation": {"best_topology", "langgraph_recommendation"},
        "definition": {"definition", "mechanism"},
        "entity_definition": {"entity_type", "definition", "mechanism", "role_in_context"},
        "concept_definition": {"definition", "mechanism", "examples"},
        "entity_type": {"entity_type"},
        "role_in_context": {"role_in_context"},
        "mechanism": {"mechanism"},
        "examples": {"examples"},
        "training_component": {"training_component", "mechanism", "reward_model_requirement", "evidence"},
        "reward_model_requirement": {"reward_model_requirement"},
        "evidence": {"evidence"},
        "answer": {"answer"},
    }
    return set(aliases.get(key, {key} if key else set()))


def goals_from_relation_compatibility(relation: str) -> set[str]:
    compatibility = {
        "origin_lookup": {"paper_title", "year"},
        "formula_lookup": {"formula", "variable_explanation", "source"},
        "paper_summary_results": {"summary", "results", "evidence"},
        "paper_recommendation": {"recommended_papers", "rationale"},
        "followup_research": {"followup_papers", "candidate_relationship", "evidence"},
        "entity_definition": {"entity_type", "definition", "mechanism", "role_in_context"},
        "concept_definition": {"definition", "mechanism", "examples"},
        "topology_discovery": {"relevant_papers", "topology_types"},
        "topology_recommendation": {"best_topology", "langgraph_recommendation"},
        "figure_question": {"figure_conclusion", "caption", "evidence"},
        "metric_value_lookup": {"metric_value", "setting", "evidence"},
    }
    return set(compatibility.get(str(relation or ""), set()))
