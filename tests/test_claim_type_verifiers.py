from __future__ import annotations

from types import SimpleNamespace

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract
from app.services.claims.type_verifiers import (
    origin_claim_has_intro_support,
    verify_followup_research_claims,
    verify_formula_lookup_claims,
    verify_origin_lookup_claims,
)


def _evidence(snippet: str, *, doc_id: str = "ev-1", paper_id: str = "paper-1") -> EvidenceBlock:
    return EvidenceBlock(
        doc_id=doc_id,
        paper_id=paper_id,
        title="DPO Paper",
        file_path="/tmp/paper.pdf",
        page=1,
        block_type="page_text",
        snippet=snippet,
    )


def test_origin_verifier_uses_intro_support_without_agent_mixin() -> None:
    claim = Claim(
        claim_type="origin",
        entity="DPO",
        value="Direct Preference Optimization",
        evidence_ids=["ev-1"],
        paper_ids=["paper-1"],
    )
    contract = QueryContract(clean_query="DPO 是哪篇论文提出的？", relation="origin_lookup", targets=["DPO"])

    assert origin_claim_has_intro_support(
        contract=contract,
        claim=claim,
        papers=[CandidatePaper(paper_id="paper-1", title="Direct Preference Optimization")],
        evidence=[_evidence("This paper introduces Direct Preference Optimization (DPO).")],
        paper_lookup=lambda paper_id: None,
        paper_doc_lookup=lambda paper_id: SimpleNamespace(page_content=""),
    )
    assert verify_origin_lookup_claims(
        contract=contract,
        claims=[claim],
        papers=[],
        evidence=[_evidence("This paper introduces Direct Preference Optimization (DPO).")],
        origin_supports_claim=lambda contract, claim, papers, evidence: True,
    ) is None


def test_followup_verifier_rejects_seed_recommendation() -> None:
    report = verify_followup_research_claims(
        claims=[
            Claim(
                claim_type="followup_research",
                structured_data={
                    "seed_papers": [{"paper_id": "seed"}],
                    "followup_titles": [{"paper_id": "seed", "title": "Seed Paper"}],
                },
            )
        ]
    )

    assert report is not None
    assert report.recommended_action == "exclude_seed_paper"


def test_formula_verifier_retries_invalid_and_misaligned_claims() -> None:
    invalid_report = verify_formula_lookup_claims(
        contract=QueryContract(clean_query="DPO 公式", relation="formula_lookup", targets=["DPO"]),
        claims=[Claim(claim_type="formula", entity="DPO", value="已定位到相关公式")],
        papers=[],
        evidence=[],
        claim_value_looks_like_formula=lambda value: False,
        verify_formula_claims_with_llm=lambda contract, claims, papers, evidence: None,
        formula_claim_matches_target=lambda contract, claim, papers, evidence: True,
    )

    assert invalid_report is not None
    assert invalid_report.recommended_action == "retry_formula"

    misaligned_report = verify_formula_lookup_claims(
        contract=QueryContract(clean_query="DPO 公式", relation="formula_lookup", targets=["DPO"]),
        claims=[Claim(claim_type="formula", entity="PPO", value="L = x")],
        papers=[],
        evidence=[],
        claim_value_looks_like_formula=lambda value: True,
        verify_formula_claims_with_llm=lambda contract, claims, papers, evidence: None,
        formula_claim_matches_target=lambda contract, claim, papers, evidence: False,
    )

    assert misaligned_report is not None
    assert misaligned_report.recommended_action == "retry_formula_target_alignment"
