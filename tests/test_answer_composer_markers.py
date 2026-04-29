from __future__ import annotations

from app.services.agent_mixins.answer_composer import (
    ANSWER_COMPOSER_MARKERS,
    AnswerComposerMixin,
)


def test_clean_topology_public_text_uses_blocked_marker_profile() -> None:
    assert "blocked_topology_text" in ANSWER_COMPOSER_MARKERS
    assert AnswerComposerMixin._clean_topology_public_text("Cannot determine the best topology.") == ""
    assert AnswerComposerMixin._clean_topology_public_text("DAG works when dependencies matter") == (
        "DAG works when dependencies matter。"
    )


def test_format_formula_symbol_uses_math_symbol_marker_profile() -> None:
    assert "formula_math_symbol" in ANSWER_COMPOSER_MARKERS
    assert AnswerComposerMixin._format_formula_symbol(r"\pi_\theta") == r"$\pi_\theta$"
    assert AnswerComposerMixin._format_formula_symbol("reward") == "`reward`"
