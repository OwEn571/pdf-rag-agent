from __future__ import annotations

import json

from app.domain.models import CandidatePaper, DisambiguationJudgeDecision, QueryContract
from app.services.clarification_intents import (
    CLARIFICATION_OPTION_SCHEMA_VERSION,
    ambiguity_option_context_text,
    ambiguity_option_matches_context,
    ambiguity_options_from_notes,
    candidate_origin_signal_score,
    candidate_title_alignment_score,
    candidate_usage_signal_score,
    clarification_option_description,
    clarification_option_id,
    clarification_option_public_payload,
    clarification_options_from_contract_notes,
    clarification_string_list,
    contract_from_selected_clarification_option,
    contract_with_ambiguity_options,
    disambiguation_judge_option_payload,
    disambiguation_judge_summary,
    disambiguation_missing_fields,
    extract_acronym_expansion_from_text,
    looks_like_clarification_choice_text,
    normalize_acronym_meaning,
    normalize_clarification_options,
    option_from_clarification_choice,
    pending_clarification_selection_index,
    select_pending_clarification_option,
    selected_option_from_judge_decision,
    selected_clarification_paper_id,
)


def test_clarification_choice_text_detects_selection_cues() -> None:
    assert looks_like_clarification_choice_text("我说的是第二个选项")
    assert looks_like_clarification_choice_text("choose the one about alignx")
    assert not looks_like_clarification_choice_text("alignx paper")


def test_pending_clarification_selection_index_detects_digits_and_ordinals() -> None:
    assert pending_clarification_selection_index("选 2") == 1
    assert pending_clarification_selection_index("第二个") == 1
    assert pending_clarification_selection_index("the third") == 2
    assert pending_clarification_selection_index("没有明确选择") is None


def test_clarification_option_public_payload_preserves_protocol_fields() -> None:
    payload = clarification_option_public_payload(
        {
            "option_id": "opt-1",
            "kind": "acronym_meaning",
            "title": "AlignX",
            "display_reason": "best match",
            "judge_recommended": True,
            "debug_only": "drop",
        }
    )

    assert payload["schema_version"] == CLARIFICATION_OPTION_SCHEMA_VERSION
    assert payload["option_id"] == "opt-1"
    assert payload["display_reason"] == "best match"
    assert payload["judge_recommended"] is True
    assert "debug_only" not in payload


def test_clarification_string_list_coerces_common_shapes() -> None:
    assert clarification_string_list([" A ", "", "B"]) == ["A", "B"]
    assert sorted(clarification_string_list({"B", "A"})) == ["A", "B"]
    assert clarification_string_list(" value ") == ["value"]
    assert clarification_string_list(None) == []


def test_clarification_option_description_and_id_are_stable() -> None:
    assert clarification_option_description({"snippet": "  alpha   beta "}, title="T", year="2026") == "alpha beta"
    assert clarification_option_description({}, title="T", year="2026") == "T · 2026"

    first = clarification_option_id(
        kind="acronym_meaning",
        target="DPO",
        label="Direct Preference Optimization",
        paper_id="p1",
        title="Paper",
        index=0,
    )
    second = clarification_option_id(
        kind="acronym_meaning",
        target="DPO",
        label="Direct Preference Optimization",
        paper_id="p1",
        title="Paper",
        index=0,
    )
    assert first == second
    assert first.startswith("acronym-meaning-dpo-")


def test_ambiguity_options_from_notes_reads_valid_payloads_only() -> None:
    valid = {"title": "Preference Bridged Alignment", "option_id": "pba"}
    notes = [
        "plain note",
        "ambiguity_option=not-json",
        "ambiguity_option=" + json.dumps({"option_id": "missing-title"}),
        "ambiguity_option=" + json.dumps(valid),
    ]

    assert ambiguity_options_from_notes(notes) == [valid]


def test_selected_clarification_paper_id_reads_direct_and_payload_notes() -> None:
    direct = QueryContract(clean_query="x", notes=["selected_paper_id=p1"])
    payload = QueryContract(
        clean_query="x",
        notes=["selected_ambiguity_option=" + json.dumps({"paper_id": "p2"})],
    )

    assert selected_clarification_paper_id(direct) == "p1"
    assert selected_clarification_paper_id(payload) == "p2"
    assert selected_clarification_paper_id(QueryContract(clean_query="x", notes=["selected_ambiguity_option=bad"])) == ""


def test_option_from_clarification_choice_matches_id_index_and_fields() -> None:
    options = [
        {"option_id": "a", "paper_id": "p1", "meaning": "Alpha", "label": "First"},
        {"option_id": "b", "paper_id": "p2", "meaning": "Beta", "label": "Second"},
    ]

    assert option_from_clarification_choice({"option_id": "b"}, options) == options[1]
    assert option_from_clarification_choice({"index": 0}, options) == options[0]
    assert option_from_clarification_choice({"paper_id": "p2"}, options) == options[1]
    assert option_from_clarification_choice({"meaning": "alpha"}, options) == options[0]
    assert option_from_clarification_choice({"label": "second"}, options) == options[1]
    assert option_from_clarification_choice({"option_id": "missing"}, options) is None


def test_select_pending_clarification_option_matches_textual_choice() -> None:
    options = [
        {"meaning": "Direct Preference Optimization", "label": "DPO", "title": "DPO Paper"},
        {"meaning": "Preference Bridged Alignment", "label": "PBA", "title": "PBA Paper"},
    ]

    assert select_pending_clarification_option(clean_query="第二个", options=options) == options[1]
    assert select_pending_clarification_option(clean_query="我说的是 Direct Preference Optimization", options=options) == options[0]
    assert select_pending_clarification_option(clean_query="PBA Paper", options=options) == options[1]
    assert select_pending_clarification_option(clean_query="没有选择", options=options) is None


def test_contract_with_ambiguity_options_replaces_old_option_notes() -> None:
    contract = QueryContract(clean_query="x", notes=["keep", "ambiguity_option=" + json.dumps({"title": "Old"})])
    option = {"option_id": "new", "kind": "acronym_meaning", "title": "New", "debug": "drop"}

    updated = contract_with_ambiguity_options(contract=contract, options=[option])

    assert updated.notes[0] == "keep"
    assert len([note for note in updated.notes if note.startswith("ambiguity_option=")]) == 1
    payload = json.loads(updated.notes[1].split("=", 1)[1])
    assert payload["title"] == "New"
    assert "debug" not in payload


def test_normalize_clarification_options_fills_protocol_fields() -> None:
    contract = QueryContract(
        clean_query="PBA是什么",
        relation="entity_definition",
        targets=["PBA"],
        requested_fields=["definition"],
        required_modalities=["page_text"],
        answer_slots=["definition"],
    )
    options = normalize_clarification_options(
        [{"paper_id": "p1", "title": "AlignX", "year": "2025", "meaning": "Preference Bridged Alignment"}],
        contract=contract,
        kind="acronym_meaning",
        source="unit_test",
    )

    assert options[0]["schema_version"] == CLARIFICATION_OPTION_SCHEMA_VERSION
    assert options[0]["target"] == "PBA"
    assert options[0]["label"] == "Preference Bridged Alignment"
    assert options[0]["source"] == "unit_test"
    assert options[0]["source_requested_fields"] == ["definition"]
    assert options[0]["source_required_modalities"] == ["page_text"]
    assert options[0]["source_answer_slots"] == ["definition"]
    assert options[0]["option_id"].startswith("acronym-meaning-pba-")


def test_clarification_options_from_contract_notes_normalizes_payloads() -> None:
    contract = contract_with_ambiguity_options(
        contract=QueryContract(clean_query="PBA是什么", relation="entity_definition", targets=["PBA"]),
        options=[{"paper_id": "p1", "title": "AlignX", "meaning": "Preference Bridged Alignment"}],
    )

    options = clarification_options_from_contract_notes(contract)

    assert options[0]["kind"] == "acronym_meaning"
    assert "source" in options[0]
    assert options[0]["target"] == "PBA"


def test_contract_from_selected_clarification_option_preserves_formula_slots() -> None:
    contract = contract_from_selected_clarification_option(
        clean_query="选第二个",
        target="PBA",
        selected={
            "meaning": "Preference Bridged Alignment",
            "title": "AlignX",
            "paper_id": "paper-1",
            "source_relation": "formula_lookup",
            "source_answer_slots": ["formula"],
        },
    )

    assert contract.relation == "formula_lookup"
    assert contract.answer_slots == ["formula"]
    assert contract.requested_fields == ["formula", "variable_explanation", "source"]
    assert "selected_paper_id=paper-1" in contract.notes


def test_disambiguation_ranking_signals_distinguish_origin_from_usage() -> None:
    origin_score = candidate_origin_signal_score("Our main contribution is Direct Preference Optimization (DPO).")
    usage_score = candidate_usage_signal_score("This paper adopts Direct Preference Optimization for tuning.")

    assert origin_score >= 0.35
    assert usage_score >= 0.25
    assert candidate_title_alignment_score(
        target="DPO",
        label="Direct Preference Optimization",
        meaning="Direct Preference Optimization",
        title_alias_text="Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
    ) == 1.0


def test_disambiguation_judge_payload_includes_metadata_and_signals() -> None:
    paper = CandidatePaper(
        paper_id="DPO",
        title="Direct Preference Optimization",
        match_reason="title match",
        metadata={
            "aliases": "DPO",
            "generated_summary": "This paper proposes Direct Preference Optimization.",
        },
    )
    option = {
        "option_id": "dpo",
        "target": "DPO",
        "meaning": "Direct Preference Optimization",
        "paper_id": "DPO",
        "title": "Direct Preference Optimization",
        "snippet": "Our main contribution is Direct Preference Optimization (DPO).",
        "source_requested_fields": ("formula", "source"),
    }

    payload = disambiguation_judge_option_payload(option=option, paper=paper)

    assert payload["paper_aliases"] == "DPO"
    assert payload["match_reason"] == "title match"
    assert payload["source_requested_fields"] == ["formula", "source"]
    assert payload["ranking_signals"]["candidate_role_hint"] == "direct_definition_or_origin"


def test_selected_option_and_judge_summary_are_stable() -> None:
    options = [{"option_id": "a", "paper_id": "p1"}, {"option_id": "b", "paper_id": "p2"}]
    decision = DisambiguationJudgeDecision(
        decision="ask_human",
        selected_option_id="b",
        selected_paper_id="p2",
        confidence=0.72,
    )

    assert selected_option_from_judge_decision(decision=decision, options=options) == options[1]
    assert disambiguation_judge_summary(options=options, judge_decision=decision) == "options=2, judge=ask_human, confidence=0.72"
    assert disambiguation_judge_summary(options=options, judge_decision=None) == "options=2, judge=unavailable"


def test_acronym_helpers_extract_normalize_and_match_context() -> None:
    expansion = extract_acronym_expansion_from_text(
        text="Preference-Bridged Alignment (PBA) is used for personalized alignment.",
        acronym="PBA",
    )
    paper = CandidatePaper(
        paper_id="pba",
        title="AlignX",
        metadata={"aliases": "PBA", "generated_summary": "Preference Bridged Alignment is a method."},
    )
    option = {"meaning": expansion, "title": "AlignX", "snippet": "PBA is evaluated on PPAIR."}
    option["context_text"] = ambiguity_option_context_text(option, paper=paper)

    assert expansion == "Preference-Bridged Alignment"
    assert normalize_acronym_meaning("Behaviour-based Alignment") == "behavior based alignment"
    assert ambiguity_option_matches_context(option=option, context_targets=["PPAIR"])


def test_disambiguation_missing_fields_uses_ambiguous_slots() -> None:
    contract = QueryContract(clean_query="PBA是什么", notes=["ambiguous_slot=target"])

    assert disambiguation_missing_fields(contract) == ["target"]
    assert disambiguation_missing_fields(QueryContract(clean_query="PBA是什么")) == ["disambiguation"]
