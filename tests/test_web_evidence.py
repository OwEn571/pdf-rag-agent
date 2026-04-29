from __future__ import annotations

from app.domain.models import Claim, EvidenceBlock, QueryContract
from app.services.web_evidence import (
    build_web_research_claim,
    claims_with_web_research_claim,
    collect_web_evidence,
    merge_evidence,
    search_agent_web_evidence,
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


def test_web_evidence_collects_with_query_override_and_gate() -> None:
    class FakeWebSearch:
        is_configured = True

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def search(self, **kwargs: object) -> list[EvidenceBlock]:
            self.calls.append(kwargs)
            return [
                EvidenceBlock(
                    doc_id="web-a",
                    paper_id="web-a",
                    title="Recent RAG Paper",
                    file_path="https://example.com/rag",
                    page=0,
                    block_type="web",
                    snippet="A recent RAG paper.",
                )
            ]

    contract = QueryContract(
        clean_query="最新 RAG 论文",
        relation="paper_recommendation",
        targets=["RAG"],
        requested_fields=["recommended_papers"],
    )
    web_search = FakeWebSearch()

    evidence = collect_web_evidence(
        web_search=web_search,
        contract=contract,
        use_web_search=True,
        max_web_results=3,
        query_override="today news about RAG",
    )

    assert [item.doc_id for item in evidence] == ["web-a"]
    assert web_search.calls == [
        {
            "query": "today news about RAG",
            "max_results": 3,
            "topic": "news",
            "include_domains": web_include_domains(contract),
        }
    ]
    assert collect_web_evidence(
        web_search=web_search,
        contract=contract,
        use_web_search=False,
        max_web_results=3,
    ) == []


def test_web_evidence_agent_search_normalizes_query_limit_and_merges() -> None:
    contract = QueryContract(
        clean_query="最新 RAG 论文",
        relation="paper_recommendation",
        targets=["RAG"],
        requested_fields=["recommended_papers"],
    )
    local = [
        EvidenceBlock(doc_id="local-a", paper_id="paper-a", title="A", file_path="", page=1, block_type="page_text", snippet="local")
    ]
    web = [
        EvidenceBlock(doc_id="web-a", paper_id="web-a", title="Web", file_path="https://example.com", page=0, block_type="web", snippet="web")
    ]
    calls: list[tuple[QueryContract, bool, int, str]] = []

    result = search_agent_web_evidence(
        contract=contract,
        existing_evidence=local,
        tool_input={"query": "custom web query", "max_results": 99},
        web_enabled=True,
        max_web_results=3,
        collect=lambda item_contract, enabled, limit, query: calls.append((item_contract, enabled, limit, query)) or web,
    )

    assert result.query == "custom web query"
    assert result.max_results == 20
    assert result.web_evidence == web
    assert [item.doc_id for item in result.merged_evidence] == ["local-a", "web-a"]
    assert calls == [(contract, True, 20, "custom web query")]


def test_web_evidence_agent_search_keeps_existing_evidence_when_web_empty() -> None:
    contract = QueryContract(clean_query="RAG")
    local = [
        EvidenceBlock(doc_id="local-a", paper_id="paper-a", title="A", file_path="", page=1, block_type="page_text", snippet="local")
    ]

    result = search_agent_web_evidence(
        contract=contract,
        existing_evidence=local,
        tool_input={},
        web_enabled=False,
        max_web_results=3,
        collect=lambda *_: [],
    )

    assert result.query == web_query_text(contract)
    assert result.web_evidence == []
    assert result.merged_evidence == local


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


def test_web_evidence_claims_with_web_research_claim_appends_when_needed() -> None:
    contract = QueryContract(clean_query="最新 RAG 论文", allow_web_search=True)
    web = [
        EvidenceBlock(
            doc_id="web-a",
            paper_id="web-a",
            title="Web",
            file_path="https://example.com",
            page=0,
            block_type="web",
            snippet="web",
        )
    ]

    claims = claims_with_web_research_claim(
        contract=contract,
        claims=[],
        web_evidence=web,
        explicit_web=False,
        build_claim=lambda item_contract, evidence: Claim(
            claim_type="web_research",
            entity=item_contract.clean_query,
            value=str(len(evidence)),
        ),
    )

    assert [item.claim_type for item in claims] == ["web_research"]
    assert claims_with_web_research_claim(
        contract=contract,
        claims=claims,
        web_evidence=[],
        explicit_web=True,
        build_claim=lambda *_: Claim(claim_type="unused", entity="", value=""),
    ) == claims
