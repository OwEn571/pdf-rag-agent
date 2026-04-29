from __future__ import annotations

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract, ResearchPlan
from app.services.schema_claim_helpers import claims_from_schema_payload, should_use_schema_claim_solver


def _evidence(doc_id: str, *, paper_id: str = "paper-1") -> EvidenceBlock:
    return EvidenceBlock(
        doc_id=doc_id,
        paper_id=paper_id,
        title="Paper One",
        file_path="/tmp/paper.pdf",
        page=1,
        block_type="page_text",
        snippet="supporting text",
    )


def test_claims_from_schema_payload_filters_unsupported_evidence_ids() -> None:
    claims = claims_from_schema_payload(
        {
            "claims": [
                {"claim_type": "summary", "value": "ok", "evidence_ids": ["missing"]},
                {"claim_type": "summary", "value": "kept", "evidence_ids": ["ev-1"]},
            ]
        },
        contract=QueryContract(clean_query="summary", targets=["AlignX"]),
        papers=[CandidatePaper(paper_id="paper-1", title="Paper One")],
        evidence=[_evidence("ev-1")],
    )

    assert len(claims) == 1
    assert claims[0].value == "kept"
    assert claims[0].evidence_ids == ["ev-1"]
    assert claims[0].paper_ids == ["paper-1"]
    assert claims[0].structured_data["source"] == "schema_claim_solver"


def test_claims_from_schema_payload_falls_back_to_contract_target_and_evidence_paper() -> None:
    claims = claims_from_schema_payload(
        {
            "claims": [
                {
                    "value": "definition",
                    "evidence_ids": "ev-1",
                    "paper_ids": ["unknown"],
                    "confidence": "medium",
                    "required": False,
                }
            ]
        },
        contract=QueryContract(clean_query="what is AlignX", targets=["AlignX"]),
        papers=[],
        evidence=[_evidence("ev-1", paper_id="paper-from-evidence")],
    )

    assert claims[0].claim_type == "general_answer"
    assert claims[0].entity == "AlignX"
    assert claims[0].paper_ids == ["paper-from-evidence"]
    assert claims[0].confidence == 0.72
    assert claims[0].required is False


def test_should_use_schema_claim_solver_blocks_high_precision_goals() -> None:
    general_contract = QueryContract(clean_query="总结一下", requested_fields=["summary"])
    formula_contract = QueryContract(clean_query="公式是什么", requested_fields=["formula"])

    assert should_use_schema_claim_solver(contract=general_contract, plan=ResearchPlan(required_claims=["summary"]))
    assert not should_use_schema_claim_solver(contract=formula_contract, plan=ResearchPlan(required_claims=["formula"]))
    assert not should_use_schema_claim_solver(
        contract=QueryContract(clean_query="哪种 topology 最好", requested_fields=["best_topology"]),
        plan=ResearchPlan(required_claims=["best_topology", "langgraph_recommendation"]),
    )
