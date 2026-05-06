from __future__ import annotations

import re

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan
from app.services.claims.verification_helpers import (
    claim_value_looks_like_formula,
    formula_claim_matches_target,
    paper_identity_matches_targets,
    targets_supported,
    verification_goals,
)


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def _evidence(snippet: str, *, title: str = "DPO Paper") -> EvidenceBlock:
    return EvidenceBlock(
        doc_id="ev-1",
        paper_id="paper-1",
        title=title,
        file_path="/tmp/paper.pdf",
        page=4,
        block_type="page_text",
        snippet=snippet,
    )


def test_verification_goals_adds_modality_derived_goals() -> None:
    goals = verification_goals(
        contract=QueryContract(
            clean_query="Figure 2 的结论是什么，表里的 accuracy 是多少？",
            requested_fields=["answer"],
            required_modalities=["figure", "table"],
        ),
        plan=ResearchPlan(required_claims=["definition"]),
    )

    assert "figure_conclusion" in goals
    assert "metric_value" in goals
    assert "definition" in goals


def test_formula_claim_matches_target_uses_formula_value_and_evidence() -> None:
    assert claim_value_looks_like_formula(r"L_DPO = -\\log \\sigma(\\beta r)")

    claim = Claim(
        claim_type="formula",
        entity="DPO",
        value=r"L_DPO = -\\log \\sigma(\\beta r)",
        evidence_ids=["ev-1"],
        paper_ids=["paper-1"],
    )

    assert formula_claim_matches_target(
        contract=QueryContract(clean_query="DPO 公式", relation="formula_lookup", targets=["DPO"]),
        claim=claim,
        papers=[CandidatePaper(paper_id="paper-1", title="Direct Preference Optimization")],
        evidence=[_evidence("The DPO objective formula is shown as a loss function.")],
    )


def test_targets_supported_and_paper_identity_helpers_do_not_need_agent_mixin() -> None:
    paper = CandidatePaper(
        paper_id="paper-1",
        title="Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        metadata={"aliases": "DPO"},
    )

    assert targets_supported(
        targets=["Direct Preference Optimization"],
        papers=[paper],
        evidence=[_evidence("Direct Preference Optimization is a preference optimization method.")],
    )
    assert not targets_supported(
        targets=["DPO", "GRPO"],
        papers=[paper],
        evidence=[_evidence("Direct Preference Optimization compares against PPO baselines.")],
    )
    assert paper_identity_matches_targets(
        paper=paper,
        targets=["DPO"],
        canonicalize_target=lambda text: text,
        normalize_entity_text=_normalize,
    )
