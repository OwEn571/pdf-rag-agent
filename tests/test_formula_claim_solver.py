from __future__ import annotations

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services.claims.formula_solver import solve_formula_claims


class NoChatClients:
    chat = None

    def invoke_json(self, **_: object) -> object:
        raise AssertionError("formula fallback should not call the model")


def test_formula_claim_solver_builds_fallback_claim_without_model() -> None:
    paper = CandidatePaper(
        paper_id="DPO",
        title="Direct Preference Optimization",
        metadata={"paper_card_text": "This paper introduces DPO."},
    )
    evidence = [
        EvidenceBlock(
            doc_id="dpo-formula",
            paper_id="DPO",
            title=paper.title,
            file_path="/tmp/dpo.pdf",
            page=3,
            block_type="page_text",
            snippet="The DPO objective is L_DPO(pi_theta; pi_ref) = - log sigma(beta log pi_theta / pi_ref).",
            score=5.0,
        )
    ]

    claims = solve_formula_claims(
        clients=NoChatClients(),
        contract=QueryContract(clean_query="DPO 公式是什么？", relation="formula_lookup", targets=["DPO"]),
        papers=[paper],
        evidence=evidence,
        retrieval_formula_token_weights={"objective": 2.0, "loss": 1.5, "log sigma": 2.0},
    )

    assert len(claims) == 1
    assert claims[0].claim_type == "formula"
    assert claims[0].paper_ids == ["DPO"]
    assert claims[0].evidence_ids == ["dpo-formula"]
    assert claims[0].structured_data["source"] == "formula_window_extractor"
