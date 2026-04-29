from __future__ import annotations

from app.services.agent_mixins.entity_definition import ENTITY_DEFINITION_MARKERS
from app.services.intent_marker_matching import query_matches_any


def test_entity_definition_markers_cover_type_classifier_profiles() -> None:
    assert query_matches_any(
        "policy optimization objective",
        "",
        ENTITY_DEFINITION_MARKERS["algorithm_type"],
    )
    assert query_matches_any("benchmark dataset", "", ENTITY_DEFINITION_MARKERS["dataset_type"])
    assert query_matches_any("agent framework", "", ENTITY_DEFINITION_MARKERS["framework_type"])
    assert query_matches_any("language model", "", ENTITY_DEFINITION_MARKERS["model_type"])


def test_entity_definition_markers_cover_grpo_explanation_profiles() -> None:
    assert query_matches_any(
        "group scores and relative rewards",
        "",
        ENTITY_DEFINITION_MARKERS["group_comparison"],
    )
    assert query_matches_any("foregoes the critic", "", ENTITY_DEFINITION_MARKERS["critic_free"])
    assert query_matches_any(
        "sample a group of outputs",
        "",
        ENTITY_DEFINITION_MARKERS["workflow_sampling"],
    )
    assert query_matches_any(
        "update the policy with kl penalty",
        "",
        ENTITY_DEFINITION_MARKERS["workflow_policy_update"],
    )
