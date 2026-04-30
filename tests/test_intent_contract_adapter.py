from __future__ import annotations

from app.services.intent_contract_adapter import (
    research_profile_slots,
    research_relation_from_slots,
    research_requirements_from_slots,
)


def test_intent_contract_adapter_maps_research_slots_to_contract_requirements() -> None:
    assert research_relation_from_slots(slots=["metric_value"], clean_query="PBA 准确率多少", targets=["PBA"]) == (
        "metric_value_lookup"
    )
    fields, modalities, shape, precision = research_requirements_from_slots(
        slots=["metric_value"],
        clean_query="PBA 准确率多少",
        targets=["PBA"],
    )

    assert fields == ["metric_value", "setting", "evidence"]
    assert modalities == ["table", "caption", "page_text"]
    assert shape == "narrative"
    assert precision == "exact"


def test_intent_contract_adapter_resolves_definition_slot_by_target_context() -> None:
    assert research_profile_slots(slots=["definition"], clean_query="PBA 是什么？", targets=["PBA"]) == [
        "entity_definition"
    ]
    assert research_profile_slots(slots=["definition"], clean_query="什么是 RAG？", targets=["RAG"]) == [
        "concept_definition"
    ]
