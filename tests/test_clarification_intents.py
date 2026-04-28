from __future__ import annotations

from app.services.clarification_intents import (
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
