from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverDispatchContext:
    goals: frozenset[str] = frozenset()
    required_modalities: tuple[str, ...] = ()


# Ordered dispatch table: each entry is (goal_set, modality_set, stage_name).
# Walked top-to-bottom; the first matching entry appends the stage.
# "entity_definition" and "concept_definition" are mutually exclusive
# (entity_definition wins when entity_type/role_in_context is present).
_DETERMINISTIC_STAGE_TABLE: list[tuple[frozenset[str], frozenset[str], str]] = [
    (frozenset({"paper_title", "year", "origin"}), frozenset(), "origin_lookup"),
    (frozenset({"formula"}), frozenset(), "formula"),
    (frozenset({"followup_papers", "candidate_relationship", "strict_followup"}), frozenset(), "followup_research"),
    (frozenset({"figure_conclusion"}), frozenset({"figure"}), "figure"),
    (frozenset({"recommended_papers"}), frozenset(), "paper_recommendation"),
    (frozenset({"best_topology", "langgraph_recommendation"}), frozenset(), "topology_recommendation"),
    (frozenset({"relevant_papers", "topology_types"}), frozenset(), "topology_discovery"),
    (frozenset({"summary", "results", "key_findings"}), frozenset(), "paper_summary_results"),
    (frozenset({"reward_model_requirement"}), frozenset(), "default_text"),
    (frozenset({"entity_type", "role_in_context"}), frozenset(), "entity_definition"),
    (frozenset({"definition", "mechanism", "examples"}), frozenset(), "concept_definition"),
    (frozenset({"metric_value", "setting"}), frozenset(), "table_metric"),
]


def deterministic_solver_stages(*, goals: set[str], required_modalities: list[str]) -> list[str]:
    return deterministic_solver_stages_for_context(
        SolverDispatchContext(
            goals=frozenset(goals),
            required_modalities=tuple(required_modalities),
        )
    )


def deterministic_solver_stages_for_context(context: SolverDispatchContext) -> list[str]:
    goals = set(context.goals)
    modalities = set(context.required_modalities)
    stages: list[str] = []
    entity_stage_seen = False
    for goal_set, modality_set, stage_name in _DETERMINISTIC_STAGE_TABLE:
        if goals & set(goal_set) or modalities & set(modality_set):
            # entity_definition and concept_definition are mutually exclusive
            if stage_name in {"entity_definition", "concept_definition"}:
                if entity_stage_seen:
                    continue
                entity_stage_seen = True
            stages.append(stage_name)
    return stages
