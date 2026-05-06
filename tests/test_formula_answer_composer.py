from __future__ import annotations

from app.domain.models import Claim, QueryContract
from app.services.answers.formula import (
    auto_resolved_candidate_notice,
    compose_formula_answer,
    format_formula_description,
    formula_term_lines,
    normalize_markdown_math_artifacts,
)


def test_compose_formula_answer_renders_latex_and_terms() -> None:
    claim = Claim(
        claim_type="formula",
        entity="PBA",
        value=r"\mathcal{L} = - \log \sigma(\beta r)",
        structured_data={"formula_format": "latex", "terms": ["beta", "log_sigma"]},
    )

    answer = compose_formula_answer(claims=[claim])

    assert "$$\n\\mathcal{L} = - \\log \\sigma(\\beta r)\n$$" in answer
    assert "$\\beta$" in answer
    assert "$\\log \\sigma$" in answer


def test_formula_variable_lines_take_precedence_over_term_fallback() -> None:
    claim = Claim(
        claim_type="formula",
        value="reward",
        structured_data={
            "terms": ["beta"],
            "variables": [{"symbol": "r(x, y)", "description": r"reward with \pi_theta"}],
        },
    )

    lines = formula_term_lines(claim)

    assert lines == [r"- $r(x, y)$：reward with $\pi_theta$"]


def test_formula_notice_and_math_artifact_normalization() -> None:
    contract = QueryContract(
        clean_query="这个公式是什么",
        notes=[
            "auto_resolved_by_llm_judge",
            'selected_ambiguity_option={"display_title":"AlignX"}',
        ],
    )

    assert auto_resolved_candidate_notice(contract) == "我按最匹配的候选《AlignX》来回答。"
    assert normalize_markdown_math_artifacts("$pi_theta(x) beta$") == r"$\pi_{\theta}(x) \beta$"
    assert format_formula_description(r"uses \pi_ref and y_w") == r"uses $\pi_ref$ and $y_w$"
