from __future__ import annotations

from app.domain.models import QueryContract
from app.services.contract_context import (
    canonical_tools,
    contract_allows_active_context_override,
    contract_answer_slots,
    contract_topic_state,
    conversation_relation_updates_research_context,
    intent_kind_from_contract,
    note_float,
    note_value,
    note_values,
    observed_tool_names,
)


def test_contract_context_reads_notes_and_topic_state() -> None:
    contract = QueryContract(
        clean_query="继续",
        relation="memory_followup",
        interaction_mode="conversation",
        answer_slots=[],
        continuation_mode="fresh",
        notes=[
            "answer_slot=summary",
            "answer_slot=results",
            "topic_state=continue",
            "intent_confidence=0.73",
            "active_topic=AlignX",
        ],
    )

    assert note_value(notes=contract.notes, prefix="active_topic=") == "AlignX"
    assert note_values(notes=contract.notes, prefix="answer_slot=") == ["summary", "results"]
    assert note_float(notes=contract.notes, prefix="intent_confidence=") == 0.73
    assert contract_answer_slots(contract) == ["summary", "results"]
    assert contract_topic_state(contract) == "continue"
    assert contract_allows_active_context_override(contract)


def test_contract_context_prefers_explicit_answer_slots_and_relation_kind() -> None:
    contract = QueryContract(
        clean_query="库里有什么？",
        interaction_mode="conversation",
        relation="library_status",
        answer_slots=["summary", "summary"],
        notes=["answer_slot=ignored"],
    )

    assert contract_answer_slots(contract) == ["summary", "ignored"]
    assert intent_kind_from_contract(contract) == "meta_library"
    assert conversation_relation_updates_research_context("memory_synthesis")
    assert not conversation_relation_updates_research_context("smalltalk")


def test_contract_context_normalizes_observed_and_canonical_tools() -> None:
    observed = observed_tool_names(
        [
            {"node": "query_contract_extractor"},
            {"node": "agent_tool:search_corpus"},
            {"node": "agent_tool:search_corpus"},
            {"node": "compound_task:1"},
            {"node": "ignored"},
        ]
    )
    canonical = canonical_tools(
        raw_tools=["web_citation_lookup", "search_corpus", "unknown"],
        aliases={"web_citation_lookup": "web_search"},
        canonical_names={"search_corpus", "web_search"},
    )

    assert observed == ["search_corpus", "compose"]
    assert canonical == ["web_search", "search_corpus"]
