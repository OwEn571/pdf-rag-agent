from __future__ import annotations

from app.services.formula_text_helpers import (
    best_formula_window,
    formula_block_score,
    formula_query_wants_gradient,
    normalize_extracted_formula_text,
    normalize_formula_variables,
)


def test_normalize_extracted_formula_text_converts_compact_unicode_math() -> None:
    normalized = normalize_extracted_formula_text("LDPO = - logσ(β log πθ(yw|x) / πref(yw|x))")

    assert r"L_{\mathrm{DPO}}" in normalized
    assert r"\log \sigma" in normalized
    assert r"\pi_{\theta}" in normalized
    assert r"\pi_{\mathrm{ref}}" in normalized


def test_normalize_formula_variables_deduplicates_and_normalizes_symbols() -> None:
    variables = normalize_formula_variables(
        [
            {"symbol": "∇θLDPO", "description": "policy gradient"},
            {"name": "∇θLDPO", "meaning": "policy gradient"},
        ]
    )

    assert variables == [{"symbol": r"\nabla_{\theta}\mathcal{L}_{\mathrm{DPO}}", "description": "policy gradient"}]


def test_best_formula_window_prefers_formula_line_over_explanatory_noise() -> None:
    window = best_formula_window(
        "\n".join(
            [
                "What does the method do in section 3?",
                "Mechanistic understanding is discussed here.",
                r"L_PBA = -\log \sigma(\beta \Delta)",
                "Variables are explained below.",
            ]
        )
    )

    assert "L_PBA" in window
    assert "Mechanistic understanding" in window


def test_formula_block_score_penalizes_gradient_when_query_wants_objective() -> None:
    weights = {"formula": 1.0, "gradient": 1.0}

    objective_score = formula_block_score("formula objective loss", query="PBA 公式是什么？", token_weights=weights)
    gradient_score = formula_block_score("formula gradient ∇θ", query="PBA 公式是什么？", token_weights=weights)

    assert objective_score > gradient_score
    assert formula_query_wants_gradient("PBA 梯度怎么更新？")
