from __future__ import annotations

from app.services.intent_legacy_adapter import legacy_contract_payload_to_intent_payload


def test_legacy_contract_payload_adapter_rejects_invalid_payload() -> None:
    assert legacy_contract_payload_to_intent_payload({"relation": "general_question"}, clean_query="x") is None
    assert legacy_contract_payload_to_intent_payload({"interaction_mode": "research"}, clean_query="x") is None


def test_legacy_contract_payload_adapter_preserves_library_and_metric_relations() -> None:
    library = legacy_contract_payload_to_intent_payload(
        {"interaction_mode": "conversation", "relation": "library_citation_ranking", "allow_web_search": True},
        clean_query="按引用数排序",
    )
    metric = legacy_contract_payload_to_intent_payload(
        {"interaction_mode": "research", "relation": "metric_value_lookup", "targets": ["PBA"]},
        clean_query="PBA 准确率多少",
    )

    assert library is not None
    assert library.intent_kind == "meta_library"
    assert library.answer_slots == ["citation_ranking"]
    assert library.needs_web
    assert metric is not None
    assert metric.intent_kind == "research"
    assert metric.answer_slots == ["metric_value"]
    assert metric.target_entities == ["PBA"]


def test_legacy_contract_payload_adapter_overrides_origin_like_clarification() -> None:
    payload = legacy_contract_payload_to_intent_payload(
        {
            "interaction_mode": "conversation",
            "relation": "clarify_user_intent",
            "targets": ["DPO"],
        },
        clean_query="DPO 最早是哪篇论文提出的？",
    )

    assert payload is not None
    assert payload.intent_kind == "memory_op"
    assert payload.topic_state == "new"
    assert payload.answer_slots == ["origin"]
    assert "local_origin_lookup_override" in payload.notes
