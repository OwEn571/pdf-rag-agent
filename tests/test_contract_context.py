from __future__ import annotations

from app.domain.models import QueryContract
from app.services.contracts.context import (
    canonical_agent_tool,
    canonical_tools,
    contract_allows_active_context_override,
    contract_answer_slots,
    contract_has_note,
    contract_note_float,
    contract_note_json_value,
    contract_note_value,
    contract_note_values,
    contract_notes,
    contract_notes_without_prefixes,
    contract_topic_state,
    conversation_relation_updates_research_context,
    has_note,
    intent_kind_from_contract,
    note_float,
    note_json_value,
    note_json_values,
    note_value,
    note_values,
    notes_without_prefixes,
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
            "selected_ambiguity_option={\"paper_id\":\"p1\",\"title\":\"Paper\"}",
            "selected_ambiguity_option=not json",
        ],
    )

    assert note_value(notes=contract.notes, prefix="active_topic=") == "AlignX"
    assert contract_note_value(contract, prefix="active_topic=") == "AlignX"
    assert note_values(notes=contract.notes, prefix="answer_slot=") == ["summary", "results"]
    assert contract_note_values(contract, prefix="answer_slot=") == ["summary", "results"]
    assert contract_notes(contract)[:2] == ["answer_slot=summary", "answer_slot=results"]
    assert has_note(notes=contract.notes, value="topic_state=continue")
    assert contract_has_note(contract, "topic_state=continue")
    assert note_float(notes=contract.notes, prefix="intent_confidence=") == 0.73
    assert contract_note_float(contract, prefix="intent_confidence=") == 0.73
    assert note_json_value(notes=contract.notes, prefix="selected_ambiguity_option=")["paper_id"] == "p1"
    assert contract_note_json_value(contract, prefix="selected_ambiguity_option=")["paper_id"] == "p1"
    assert note_json_values(notes=contract.notes, prefix="selected_ambiguity_option=") == [
        {"paper_id": "p1", "title": "Paper"}
    ]
    assert notes_without_prefixes(notes=contract.notes, prefixes={"selected_ambiguity_option="}) == [
        "answer_slot=summary",
        "answer_slot=results",
        "topic_state=continue",
        "intent_confidence=0.73",
        "active_topic=AlignX",
    ]
    assert contract_notes_without_prefixes(contract, prefixes={"selected_ambiguity_option="}) == [
        "answer_slot=summary",
        "answer_slot=results",
        "topic_state=continue",
        "intent_confidence=0.73",
        "active_topic=AlignX",
    ]
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
    assert canonical_agent_tool(
        tool="compose",
        aliases={},
        canonical_names={"ask_human", "compose"},
    ) == "compose"
    assert canonical_agent_tool(
        tool="missing_tool",
        aliases={},
        canonical_names={"read_memory", "compose"},
    ) == "compose"
