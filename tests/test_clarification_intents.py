from __future__ import annotations

import json

from app.services.clarification_intents import (
    CLARIFICATION_OPTION_SCHEMA_VERSION,
    ambiguity_options_from_notes,
    clarification_option_description,
    clarification_option_id,
    clarification_option_public_payload,
    clarification_string_list,
    looks_like_clarification_choice_text,
    pending_clarification_selection_index,
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
