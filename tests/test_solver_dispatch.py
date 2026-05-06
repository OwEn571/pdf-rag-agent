from __future__ import annotations

from app.services.planning.solver_dispatch import SolverDispatchContext, deterministic_solver_stages, deterministic_solver_stages_for_context


def test_deterministic_solver_stages_preserve_pipeline_order() -> None:
    stages = deterministic_solver_stages(
        goals={
            "paper_title",
            "formula",
            "metric_value",
            "recommended_papers",
            "summary",
            "entity_type",
            "best_topology",
            "relevant_papers",
        },
        required_modalities=["figure"],
    )

    assert stages == [
        "origin_lookup",
        "formula",
        "figure",
        "paper_recommendation",
        "topology_recommendation",
        "topology_discovery",
        "paper_summary_results",
        "entity_definition",
        "table_metric",
    ]


def test_deterministic_solver_stages_choose_concept_only_without_entity_goal() -> None:
    assert deterministic_solver_stages(goals={"definition", "mechanism"}, required_modalities=[]) == [
        "concept_definition"
    ]


def test_deterministic_solver_stages_accept_typed_dispatch_context() -> None:
    context = SolverDispatchContext(
        goals=frozenset({"formula", "metric_value"}),
        required_modalities=("table",),
    )

    assert deterministic_solver_stages_for_context(context) == ["formula", "table_metric"]


def test_deterministic_solver_stages_do_not_duplicate_topology_recommendation() -> None:
    assert deterministic_solver_stages(
        goals={"best_topology", "langgraph_recommendation"},
        required_modalities=[],
    ) == ["topology_recommendation"]
