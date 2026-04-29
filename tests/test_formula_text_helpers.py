from __future__ import annotations

from app.services.formula_text_helpers import (
    best_formula_window,
    fallback_formula_payload,
    formula_block_score,
    formula_claim_from_payload,
    formula_extractor_human_prompt,
    formula_extractor_system_prompt,
    formula_matched_targets,
    formula_query_wants_gradient,
    formula_target_terms,
    llm_formula_payload_from_response,
    normalize_extracted_formula_text,
    normalize_formula_variables,
    select_formula_blocks,
)
from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract


def _evidence(doc_id: str, *, snippet: str, page: int = 1) -> EvidenceBlock:
    return EvidenceBlock(
        doc_id=doc_id,
        paper_id="paper-1",
        title="Formula Paper",
        file_path="/tmp/formula.pdf",
        page=page,
        block_type="page_text",
        snippet=snippet,
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


def test_formula_target_matching_uses_paper_and_evidence_context() -> None:
    contract = QueryContract(clean_query="PBA 公式是什么？", targets=["PBA", "DPO", ""])
    paper = CandidatePaper(paper_id="paper-1", title="Preference-bridged Alignment")
    evidence = [_evidence("ev-1", snippet="We define the PBA objective in Equation 1.")]

    targets = formula_target_terms(contract)
    matched = formula_matched_targets(
        paper=paper,
        evidence=evidence,
        target_terms=targets,
        target_matcher=lambda text, target: target in text,
    )

    assert targets == ["PBA", "DPO"]
    assert matched == ["PBA"]


def test_select_formula_blocks_prefers_strong_scored_blocks_then_keyword_fallback() -> None:
    weak = _evidence("weak", snippet="formula details", page=2)
    strong = _evidence("strong", snippet="objective equation", page=1)
    selected = select_formula_blocks(
        [weak, strong],
        block_scorer=lambda text: 4.0 if "objective" in text else 1.0,
    )
    fallback = select_formula_blocks(
        [weak],
        block_scorer=lambda text: 0.0,
    )

    assert selected == [strong]
    assert fallback == [weak]


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


def test_formula_extractor_prompts_include_contract_and_evidence_payload() -> None:
    evidence = [
        EvidenceBlock(
            doc_id="ev-1",
            paper_id="p1",
            title="Formula Paper",
            file_path="/tmp/formula.pdf",
            page=2,
            block_type="page_text",
            snippet="L_DPO = -log σ(β log πθ)",
        )
    ]
    contract = QueryContract(clean_query="DPO 公式是什么？", targets=["DPO"], requested_fields=["formula"])

    system_prompt = formula_extractor_system_prompt()
    human_prompt = formula_extractor_human_prompt(contract=contract, evidence=evidence)

    assert "论文公式抽取器" in system_prompt
    assert "不要凭常识补全" in system_prompt
    assert '"query": "DPO 公式是什么？"' in human_prompt
    assert '"targets": ["DPO"]' in human_prompt
    assert '"doc_id": "ev-1"' in human_prompt
    assert "L_DPO" in human_prompt


def test_fallback_formula_payload_extracts_window_and_evidence_ids() -> None:
    evidence = [
        EvidenceBlock(
            doc_id="ev-1",
            paper_id="p1",
            title="Formula Paper",
            file_path="/tmp/formula.pdf",
            page=1,
            block_type="page_text",
            snippet="Intro text\nL_DPO = -log σ(β log πθ)\nVariables follow.",
        )
    ]

    payload = fallback_formula_payload(evidence, term_extractor=lambda text: ["beta"] if "beta" in text or "β" in text else [])

    assert payload["evidence_ids"] == ["ev-1"]
    assert payload["source"] == "formula_window_extractor"
    assert payload["confidence"] == 0.74
    assert r"L_{\mathrm{DPO}}" in payload["formula_text"]


def test_formula_claim_from_payload_builds_structured_claim() -> None:
    paper = CandidatePaper(paper_id="p1", title="Formula Paper", year="2025")
    block = EvidenceBlock(
        doc_id="ev-1",
        paper_id="p1",
        title="Formula Paper",
        file_path="/tmp/formula.pdf",
        page=1,
        block_type="page_text",
        snippet="formula text",
    )

    claim = formula_claim_from_payload(
        contract=QueryContract(clean_query="DPO 公式", targets=["DPO"]),
        paper=paper,
        matched_targets=["DPO"],
        formula_payload={
            "formula_text": r"L_{\mathrm{DPO}} = -\log \sigma(\beta \Delta)",
            "formula_format": "latex",
            "evidence_ids": ["ev-1"],
            "variables": [{"symbol": r"\beta", "description": "temperature"}],
            "terms": ["log_sigma"],
            "confidence": "high",
        },
        formula_blocks=[block],
        fallback_evidence_ids=["fallback"],
        fallback_term_text="temperature",
        term_extractor=lambda text: ["beta"] if "temperature" in text or "beta" in text or r"\beta" in text else [],
    )

    assert claim is not None
    assert claim.claim_type == "formula"
    assert claim.entity == "DPO"
    assert claim.evidence_ids == ["ev-1"]
    assert {"log_sigma", "beta"} <= set(claim.structured_data["terms"])
    assert claim.confidence == 0.88
