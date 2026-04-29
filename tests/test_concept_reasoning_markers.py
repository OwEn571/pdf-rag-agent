from __future__ import annotations

from app.services.agent_mixins.concept_reasoning import CONCEPT_REASONING_MARKERS
from app.services.intent_marker_matching import query_matches_any


def test_concept_reasoning_markers_cover_category_profiles() -> None:
    assert query_matches_any(
        "reinforcement learning with reward model",
        "",
        CONCEPT_REASONING_MARKERS["category_rl"],
    )
    assert query_matches_any("benchmark dataset", "", CONCEPT_REASONING_MARKERS["category_dataset"])
    assert query_matches_any("agent framework", "", CONCEPT_REASONING_MARKERS["category_framework"])
    assert query_matches_any("training objective", "", CONCEPT_REASONING_MARKERS["category_objective"])


def test_concept_reasoning_markers_cover_detail_profiles() -> None:
    assert query_matches_any("human feedback", "", CONCEPT_REASONING_MARKERS["detail_human_feedback"])
    assert query_matches_any("policy optimization", "", CONCEPT_REASONING_MARKERS["detail_policy_optimization"])
    assert query_matches_any(
        "retrieval-augmented generation",
        "",
        CONCEPT_REASONING_MARKERS["detail_retrieval_generation"],
    )
