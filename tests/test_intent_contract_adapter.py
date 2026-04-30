from __future__ import annotations

from app.services.intent_contract_adapter import (
    answer_slots_from_relation,
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


def test_intent_contract_adapter_maps_legacy_relations_to_answer_slots() -> None:
    assert answer_slots_from_relation("library_citation_ranking") == ["citation_ranking"]
    assert answer_slots_from_relation("metric_value_lookup") == ["metric_value"]
    assert answer_slots_from_relation("unknown_relation") == ["general_answer"]


def test_intent_contract_adapter_resolves_definition_slot_by_target_context() -> None:
    assert research_profile_slots(slots=["definition"], clean_query="PBA 是什么？", targets=["PBA"]) == [
        "entity_definition"
    ]
    assert research_profile_slots(slots=["definition"], clean_query="什么是 RAG？", targets=["RAG"]) == [
        "concept_definition"
    ]
