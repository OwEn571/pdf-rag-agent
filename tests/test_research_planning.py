from __future__ import annotations

from types import SimpleNamespace

from app.domain.models import QueryContract
from app.services.planning.research import (
    ResearchPlanContext,
    build_research_plan,
    build_research_plan_from_context,
    goals_from_relation_compatibility,
    paper_recall_mode_for_context,
    research_plan_context_from_contract,
    research_plan_goals,
    solver_sequence_for_goals,
)


def test_research_planning_builds_larger_budget_for_summary_results() -> None:
    plan = build_research_plan(
        contract=QueryContract(
            clean_query="这篇论文的核心结论和实验结果是什么？",
            relation="paper_summary_results",
            targets=["AlignX"],
            requested_fields=["summary", "results"],
        ),
        settings=SimpleNamespace(paper_limit_default=4, evidence_limit_default=8, llm_retry_budget=2),
    )

    assert plan.paper_recall_mode == "anchor_first"
    assert plan.paper_limit >= 8
    assert plan.evidence_limit >= 36
    assert plan.solver_sequence == ["text_solver", "table_solver"]
    assert {"summary", "results"} <= set(plan.required_claims)


def test_research_planning_builds_from_typed_context_without_contract_control_flow() -> None:
    context = ResearchPlanContext(
        clean_query="AlignX 的核心结论和实验结果是什么？",
        targets=("AlignX",),
        goals=frozenset({"summary", "results", "evidence"}),
        required_modalities=("page_text", "table"),
    )

    plan = build_research_plan_from_context(
        context=context,
        settings=SimpleNamespace(paper_limit_default=4, evidence_limit_default=8, llm_retry_budget=2),
    )

    assert paper_recall_mode_for_context(context) == "anchor_first"
    assert plan.paper_recall_mode == "anchor_first"
    assert plan.solver_sequence == ["text_solver", "table_solver"]
    assert {"summary", "results"} <= set(plan.required_claims)


def test_research_plan_context_from_contract_freezes_planning_tags() -> None:
    contract = QueryContract(
        clean_query="PBA 的公式是什么？",
        relation="formula_lookup",
        targets=["PBA", ""],
        required_modalities=["page_text", "table"],
    )

    context = research_plan_context_from_contract(contract)

    assert context.clean_query == "PBA 的公式是什么？"
    assert context.targets == ("PBA",)
    assert context.requested_fields == ("answer",)
    assert {"formula", "variable_explanation", "source"} <= set(context.goals)
    assert context.required_modalities == ("page_text", "table")


def test_research_planning_maps_formula_and_followup_goals_to_specialized_solvers() -> None:
    formula_goals = research_plan_goals(
        QueryContract(clean_query="PBA 的公式是什么？", relation="formula_lookup", targets=["PBA"])
    )
    followup_goals = research_plan_goals(
        QueryContract(clean_query="AlignX 有严格后续工作吗？", relation="followup_research", targets=["AlignX"])
    )

    assert {"formula", "variable_explanation", "source"} <= formula_goals
    formula_sequence = solver_sequence_for_goals(formula_goals, ["page_text", "table"])
    assert formula_sequence[0] == "formula_solver"
    assert "table_solver" in formula_sequence
    assert solver_sequence_for_goals(followup_goals, ["page_text"])[0] == "followup_solver"


def test_research_planning_relation_compatibility_is_centralized() -> None:
    assert goals_from_relation_compatibility("origin_lookup") == {"paper_title", "year"}
    assert "metric_value" in goals_from_relation_compatibility("metric_value_lookup")


def test_research_planning_metric_definition_uses_text_and_table_solvers() -> None:
    plan = build_research_plan(
        contract=QueryContract(
            clean_query="ICA、PBA 的准确度/指标在论文中是怎么定义或计算的？",
            relation="metric_value_lookup",
            targets=["ICA", "PBA"],
            requested_fields=["metric_value", "metric_definition", "setting", "evidence"],
            required_modalities=["table", "caption", "page_text"],
        ),
        settings=SimpleNamespace(paper_limit_default=4, evidence_limit_default=8, llm_retry_budget=2),
    )

    assert plan.solver_sequence == ["text_solver", "table_solver"]
    assert {"metric_value", "setting", "definition"} <= set(plan.required_claims)
    assert plan.evidence_limit >= 32
