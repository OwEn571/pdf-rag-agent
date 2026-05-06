from __future__ import annotations

from app.domain.models import Claim, QueryContract, ResearchPlan
from app.services.intents.marker_matching import query_matches_any
from app.services.planning.solver_goals import (
    ClaimGoalContext,
    SOLVER_GOAL_MARKERS,
    append_unique_claims,
    claim_goal_context_from_contract_plan,
    claim_goals,
    claim_goals_for_context,
    fallback_goals_from_query,
    looks_like_metric_goal,
)


def test_solver_goal_markers_use_centralized_profiles() -> None:
    assert query_matches_any("最早", "最早", SOLVER_GOAL_MARKERS["origin"])
    assert query_matches_any("win rate", "winrate", SOLVER_GOAL_MARKERS["metric"])
    assert not query_matches_any("闲聊", "闲聊", SOLVER_GOAL_MARKERS["formula"])


def test_append_unique_claims_deduplicates_by_type_entity_and_value() -> None:
    claims = [Claim(claim_type="summary", entity="AlignX", value="main result")]

    append_unique_claims(
        claims,
        [
            Claim(claim_type="summary", entity="AlignX", value="main result"),
            Claim(claim_type="summary", entity="AlignX", value="second result"),
        ],
    )

    assert [claim.value for claim in claims] == ["main result", "second result"]


def test_fallback_goals_detects_origin_lookup() -> None:
    goals = fallback_goals_from_query("AlignX 最早提出论文是哪篇？", targets=["AlignX"])

    assert {"paper_title", "year"} <= goals


def test_fallback_goals_detects_formula_and_entity_definition() -> None:
    goals = fallback_goals_from_query("PBA 的公式和梯度是什么？", targets=["PBA"])

    assert {"formula", "entity_type", "definition", "mechanism"} <= goals


def test_fallback_goals_detects_result_metrics() -> None:
    goals = fallback_goals_from_query("给我实验结果准确率", targets=[])

    assert {"summary", "results", "metric_value"} <= goals


def test_metric_goal_detection_uses_goal_set_and_query_terms() -> None:
    assert looks_like_metric_goal("这个准确度是多少？", set())
    assert looks_like_metric_goal("请总结主要发现", {"setting"})
    assert not looks_like_metric_goal("请总结主要发现", set())


def test_claim_goals_adds_figure_goal_from_modality() -> None:
    goals = claim_goals(
        contract=QueryContract(clean_query="Figure 1 展示什么？", required_modalities=["figure"]),
        plan=ResearchPlan(required_claims=["answer"]),
    )

    assert "figure_conclusion" in goals


def test_claim_goal_context_from_contract_plan_freezes_solver_tags() -> None:
    context = claim_goal_context_from_contract_plan(
        contract=QueryContract(
            clean_query="PBA 的公式是什么？",
            targets=["PBA"],
            answer_slots=["formula"],
            requested_fields=["answer"],
            required_modalities=["page_text"],
        ),
        plan=ResearchPlan(required_claims=["formula"]),
    )

    assert context.clean_query == "PBA 的公式是什么？"
    assert context.targets == ("PBA",)
    assert context.answer_slots == ("formula",)
    assert "source" in claim_goals_for_context(context)


def test_claim_goals_adds_metric_goal_from_table_modality() -> None:
    goals = claim_goals(
        contract=QueryContract(clean_query="实验准确率是多少？", required_modalities=["table"]),
        plan=ResearchPlan(required_claims=["answer"]),
    )

    assert "metric_value" in goals


def test_claim_goals_for_context_adds_metric_goal_from_table_modality() -> None:
    goals = claim_goals_for_context(
        ClaimGoalContext(
            clean_query="实验准确率是多少？",
            required_claims=("answer",),
            requested_fields=("answer",),
            required_modalities=("table",),
        )
    )

    assert "metric_value" in goals
