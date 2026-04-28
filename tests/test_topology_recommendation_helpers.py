from __future__ import annotations

from app.services.topology_recommendation_helpers import (
    fallback_topology_recommendation,
    is_unusable_topology_recommendation_text,
)


def test_unusable_topology_recommendation_text_detects_empty_and_negative_answers() -> None:
    assert is_unusable_topology_recommendation_text("")
    assert is_unusable_topology_recommendation_text("The evidence does not contain a specific comparison.")
    assert is_unusable_topology_recommendation_text("无法确定哪一种 topology 最好")
    assert not is_unusable_topology_recommendation_text("DAG is better when dependencies must be explicit.")


def test_fallback_topology_recommendation_uses_terms_or_default() -> None:
    recommendation = fallback_topology_recommendation(["chain", "DAG"])
    default_recommendation = fallback_topology_recommendation([])

    assert recommendation["engineering_best"] == "DAG"
    assert "chain / DAG" in recommendation["summary"]
    assert "chain / tree / mesh / DAG" in default_recommendation["summary"]
