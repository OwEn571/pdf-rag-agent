from __future__ import annotations

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan
from app.services.claims.generic_solver import claim_solver_shadow_summary, solve_claims_with_generic_schema


class _Clients:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.chat = object()
        self.calls: list[dict[str, object]] = []

    def invoke_json(self, **kwargs: object) -> dict:
        self.calls.append(kwargs)
        return self.payload


def _evidence() -> EvidenceBlock:
    return EvidenceBlock(
        doc_id="ev-1",
        paper_id="paper-1",
        title="Paper One",
        file_path="/tmp/paper.pdf",
        page=1,
        block_type="page_text",
        snippet="Paper One reports a strong result.",
    )


def test_generic_claim_solver_returns_empty_without_chat_or_evidence() -> None:
    contract = QueryContract(clean_query="summary")
    plan = ResearchPlan(required_claims=["summary"])

    assert solve_claims_with_generic_schema(
        clients=None,
        contract=contract,
        plan=plan,
        papers=[],
        evidence=[_evidence()],
        conversation_context={},
    ) == []
    assert solve_claims_with_generic_schema(
        clients=object(),
        contract=contract,
        plan=plan,
        papers=[],
        evidence=[_evidence()],
        conversation_context={},
    ) == []
    assert solve_claims_with_generic_schema(
        clients=_Clients({"claims": []}),
        contract=contract,
        plan=plan,
        papers=[],
        evidence=[],
        conversation_context={},
    ) == []


def test_generic_claim_solver_builds_schema_claims_from_chat_payload() -> None:
    clients = _Clients(
        {
            "claims": [
                {
                    "claim_type": "summary",
                    "entity": "Paper One",
                    "value": "strong result",
                    "evidence_ids": ["ev-1"],
                    "paper_ids": ["paper-1"],
                }
            ]
        }
    )

    claims = solve_claims_with_generic_schema(
        clients=clients,
        contract=QueryContract(clean_query="总结 Paper One", targets=["Paper One"]),
        plan=ResearchPlan(required_claims=["summary"]),
        papers=[CandidatePaper(paper_id="paper-1", title="Paper One")],
        evidence=[_evidence()],
        conversation_context={"turns": 1},
    )

    assert len(claims) == 1
    assert claims[0].structured_data["source"] == "schema_claim_solver"
    assert claims[0].evidence_ids == ["ev-1"]
    assert clients.calls
    assert "conversation_context" in str(clients.calls[0]["human_prompt"])


def test_claim_solver_shadow_summary_is_stable_for_trace_diff() -> None:
    schema_claims = [
        Claim(
            claim_type="summary",
            value="schema",
            paper_ids=["paper-1"],
            evidence_ids=["ev-1"],
            structured_data={"source": "schema_claim_solver"},
        )
    ]
    deterministic_claims = [
        Claim(
            claim_type="summary",
            value="deterministic",
            paper_ids=["paper-1"],
            evidence_ids=["ev-2"],
        )
    ]

    summary = claim_solver_shadow_summary(
        selected="deterministic",
        schema_claims=schema_claims,
        deterministic_claims=deterministic_claims,
    )

    assert summary["mode"] == "generic_claim_solver_shadow"
    assert summary["selected"] == "deterministic"
    assert summary["schema"]["sources"] == {"schema_claim_solver": 1}
    assert summary["deterministic"]["sources"] == {"deterministic_solver": 1}
    assert summary["schema"]["paper_ids"] == ["paper-1"]
