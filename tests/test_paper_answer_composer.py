from __future__ import annotations

from app.domain.models import Claim, QueryContract
from app.services.answers.paper import (
    compose_metric_value_answer,
    compose_paper_summary_results_answer,
    metric_lines_from_claims,
    paper_result_core_points,
)


def test_metric_lines_from_claims_deduplicates_preserving_order() -> None:
    claims = [
        Claim(claim_type="metric_value", structured_data={"metric_lines": [" ICA 57.80 ", "PBA 59.66"]}),
        Claim(claim_type="metric_value", structured_data={"metric_lines": ["ica 57.80", "", "PBA 60.00"]}),
    ]

    assert metric_lines_from_claims(claims) == ["ICA 57.80", "PBA 59.66", "PBA 60.00"]


def test_compose_metric_value_answer_keeps_raw_table_lines() -> None:
    answer = compose_metric_value_answer(
        contract=QueryContract(clean_query="PBA 准确率", targets=["PBA"]),
        claims=[Claim(claim_type="metric_value", entity="PBA", structured_data={"metric_lines": ["PBA on PPAIR: 59.66"]})],
    )

    assert "PBA 的表现需要按表格证据来读" in answer
    assert "- PBA on PPAIR: 59.66" in answer


def test_compose_paper_summary_results_uses_core_points_and_metric_fallback() -> None:
    answer = compose_paper_summary_results_answer(
        contract=QueryContract(clean_query="AlignX 主要结论", targets=["AlignX"]),
        claims=[
            Claim(
                claim_type="paper_summary_results",
                entity="AlignX",
                value="AlignX studies preference inference and conditioned generation with a modular design.",
                structured_data={"metric_lines": ["Average accuracy improves by 17.06%"]},
            )
        ],
    )

    assert "preference inference" in answer
    assert "conditioned generation" in answer
    assert "modular" in answer
    assert "Average accuracy improves by 17.06%" in answer


def test_paper_result_core_points_falls_back_to_support_excerpt() -> None:
    points = paper_result_core_points(target="Paper", support_text="A concise result statement.")

    assert points == ["A concise result statement."]
