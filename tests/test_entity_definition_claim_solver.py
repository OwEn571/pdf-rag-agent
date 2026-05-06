from __future__ import annotations

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services.claims.entity_definition_solver import solve_entity_definition_claims


def _evidence(doc_id: str, snippet: str) -> EvidenceBlock:
    return EvidenceBlock(
        doc_id=doc_id,
        paper_id="p1",
        title="Entity Paper",
        file_path="/tmp/entity.pdf",
        page=1,
        block_type="page_text",
        snippet=snippet,
    )


def test_solve_entity_definition_claims_builds_claim_from_selected_paper() -> None:
    paper = CandidatePaper(paper_id="p1", title="Entity Paper", year="2025", doc_ids=["doc-fallback"])
    evidence = [_evidence("ev-definition", "GRPO is an RL optimization algorithm.")]
    contract = QueryContract(clean_query="GRPO 是什么", relation="entity_definition", targets=["GRPO"])

    claims = solve_entity_definition_claims(
        contract=contract,
        papers=[paper],
        evidence=evidence,
        select_supporting_paper=lambda _contract, _papers, _evidence: (paper, evidence),
        infer_entity_type=lambda _contract, _papers, _evidence: "强化学习优化算法",
        entity_supporting_lines=lambda _evidence, kind: [f"{kind} line"] if kind == "definition" else [],
    )

    assert len(claims) == 1
    claim = claims[0]
    assert claim.claim_type == "entity_definition"
    assert claim.entity == "GRPO"
    assert claim.value == "强化学习优化算法"
    assert claim.evidence_ids == ["ev-definition"]
    assert claim.paper_ids == ["p1"]
    assert claim.structured_data["definition_lines"] == ["definition line"]


def test_solve_entity_definition_claims_returns_empty_without_selected_paper() -> None:
    contract = QueryContract(clean_query="未知是什么", relation="entity_definition", targets=["Unknown"])

    claims = solve_entity_definition_claims(
        contract=contract,
        papers=[],
        evidence=[],
        select_supporting_paper=lambda _contract, _papers, _evidence: (None, []),
        infer_entity_type=lambda _contract, _papers, _evidence: "unknown",
        entity_supporting_lines=lambda _evidence, kind: [],
    )

    assert claims == []
