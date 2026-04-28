from __future__ import annotations

from app.services.formula_text_helpers import (
    best_formula_window,
    formula_block_score,
    formula_query_wants_gradient,
    llm_formula_payload_from_response,
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


def test_llm_formula_payload_from_response_filters_evidence_and_normalizes_terms() -> None:
    payload = llm_formula_payload_from_response(
        {
            "formulas": [
                {"formula_text": "missing", "evidence_ids": ["missing"]},
                {
                    "formula_latex": r"L_DPO = -log σ(β log πθ)",
                    "evidence_ids": "ev-1",
                    "variables": [{"symbol": "πθ", "description": "policy"}],
                    "confidence": "high",
                },
            ]
        },
        allowed_evidence_ids={"ev-1"},
        term_extractor=lambda text: ["policy"] if "policy" in text else [],
    )

    assert payload["formula_format"] == "latex"
    assert payload["evidence_ids"] == ["ev-1"]
    assert payload["terms"] == ["policy"]
    assert payload["source"] == "llm_formula_extractor"
