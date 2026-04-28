from __future__ import annotations

from app.domain.models import Claim, EvidenceBlock, QueryContract
from app.services.web_evidence import (
    build_web_research_claim,
    merge_evidence,
    should_add_web_claim,
    web_include_domains,
    web_query_text,
    web_search_topic,
)


def test_web_evidence_builds_query_and_research_domains() -> None:
    contract = QueryContract(
        clean_query="最新的多模态 RAG 论文有哪些？",
        relation="paper_recommendation",
        targets=["RAG"],
        requested_fields=["recommended_papers"],
    )

    assert web_query_text(contract) == "最新的多模态 RAG 论文有哪些？ paper arXiv publication"
    assert web_search_topic("today news about RAG") == "news"
    assert "arxiv.org" in web_include_domains(contract)


def test_web_evidence_merges_by_doc_id() -> None:
    local = [
        EvidenceBlock(
            doc_id="doc-a",
            paper_id="paper-a",
            title="A",
            file_path="/a.pdf",
            page=1,
            block_type="page_text",
            snippet="local",
        )
    ]
    web = [
        EvidenceBlock(
            doc_id="doc-a",
            paper_id="paper-a",
            title="A",
            file_path="/a.pdf",
            page=1,
            block_type="page_text",
            snippet="duplicate",
        ),
        EvidenceBlock(
            doc_id="web-b",
            paper_id="web-b",
            title="B",
            file_path="https://b.example",
            page=0,
            block_type="web",
            snippet="web",
        ),
    ]

    assert [item.doc_id for item in merge_evidence(local, web)] == ["doc-a", "web-b"]


def test_web_evidence_claim_decision_and_payload() -> None:
    contract = QueryContract(
        clean_query="最新 RAG 论文",
        relation="paper_recommendation",
        targets=["RAG"],
        requested_fields=["recommended_papers"],
        allow_web_search=True,
    )
    evidence = [
        EvidenceBlock(
            doc_id="web-a",
            paper_id="web-a",
            title="Recent RAG Paper",
            file_path="https://example.com/rag",
            page=0,
            block_type="web",
            snippet="A recent RAG paper.",
            score=0.9,
        )
    ]

    assert should_add_web_claim(contract=contract, claims=[], explicit_web=False)
    assert should_add_web_claim(
        contract=contract,
        claims=[Claim(claim_type="summary", entity="RAG", value="existing")],
        explicit_web=True,
    )
    claim = build_web_research_claim(contract=contract, web_evidence=evidence)
    assert claim.claim_type == "web_research"
    assert claim.evidence_ids == ["web-a"]
