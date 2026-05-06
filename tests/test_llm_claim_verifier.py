from __future__ import annotations

from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan
from app.services.claims.llm_verifier import (
    coerce_verifier_string_list,
    verify_claims_with_schema_llm,
    verify_formula_claims_with_llm,
)


def _evidence() -> EvidenceBlock:
    return EvidenceBlock(
        doc_id="ev-1",
        paper_id="paper-1",
        title="DPO Paper",
        file_path="/tmp/dpo.pdf",
        page=4,
        block_type="page_text",
        snippet="DPO objective evidence.",
    )


class _JsonClients:
    chat = object()

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, str]] = []

    def invoke_json(self, *, system_prompt: str, human_prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"system_prompt": system_prompt, "human_prompt": human_prompt})
        return self.payload


def test_schema_llm_verifier_returns_report_for_schema_claims() -> None:
    clients = _JsonClients(
        {
            "status": "retry",
            "missing_fields": "evidence",
            "recommended_action": "expand_recall",
            "contradictions": ["unsupported metric"],
        }
    )
    report = verify_claims_with_schema_llm(
        clients=clients,
        contract=QueryContract(clean_query="总结论文"),
        plan=ResearchPlan(required_claims=["answer"]),
        claims=[
            Claim(
                claim_type="summary",
                value="Summary",
                structured_data={"source": "schema_claim_solver"},
            )
        ],
        papers=[CandidatePaper(paper_id="paper-1", title="Paper")],
        evidence=[_evidence()],
    )

    assert report is not None
    assert report.status == "retry"
    assert report.missing_fields == ["evidence"]
    assert report.contradictory_claims == ["unsupported metric"]
    assert clients.calls


def test_formula_llm_verifier_maps_retry_payload_and_pass_to_none() -> None:
    retry_report = verify_formula_claims_with_llm(
        clients=_JsonClients(
            {
                "status": "unsupported",
                "missing_fields": [],
                "unsupported_claims": ["DPO formula"],
                "recommended_action": "retry_formula",
            }
        ),
        contract=QueryContract(clean_query="DPO 公式", relation="formula_lookup", targets=["DPO"]),
        claims=[Claim(claim_type="formula", entity="DPO", value="L = x", evidence_ids=["ev-1"], paper_ids=["paper-1"])],
        papers=[CandidatePaper(paper_id="paper-1", title="DPO Paper")],
        evidence=[_evidence()],
    )

    assert retry_report is not None
    assert retry_report.status == "retry"
    assert retry_report.missing_fields == ["formula_evidence"]
    assert retry_report.unsupported_claims == ["DPO formula"]

    pass_report = verify_formula_claims_with_llm(
        clients=_JsonClients({"status": "pass"}),
        contract=QueryContract(clean_query="DPO 公式", relation="formula_lookup", targets=["DPO"]),
        claims=[Claim(claim_type="formula", entity="DPO", value="L = x")],
        papers=[],
        evidence=[],
    )

    assert pass_report is None


def test_coerce_verifier_string_list() -> None:
    assert coerce_verifier_string_list("missing") == ["missing"]
    assert coerce_verifier_string_list(["a", "", 3]) == ["a", "3"]
    assert coerce_verifier_string_list(None) == []
