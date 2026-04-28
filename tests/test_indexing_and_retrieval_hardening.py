from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from app.core.config import Settings
from app.domain.models import EvidenceBlock, QueryContract
from app.services.indexing import V4IngestionService
from app.services.retrieval import DualIndexRetriever
from app.services.zotero_sqlite import PaperRecord


def _paper_record() -> PaperRecord:
    return PaperRecord(
        parent_item_id=1,
        attachment_item_id=2,
        attachment_key="ATTACH",
        item_type="journalArticle",
        title="Demo Paper",
        authors=["Ada"],
        year="2026",
        tags=[],
        abstract_note="",
        source_url="",
        website_title="",
        file_path="/tmp/demo.pdf",
        file_exists=True,
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="",
        data_dir=tmp_path / "data",
        paper_store_path=tmp_path / "data" / "papers.jsonl",
        block_store_path=tmp_path / "data" / "blocks.jsonl",
        ingestion_state_path=tmp_path / "data" / "state.json",
        session_store_path=tmp_path / "data" / "sessions.sqlite3",
        eval_cases_path=tmp_path / "evals" / "cases.yaml",
    )


def test_block_doc_id_is_not_coupled_to_text_content() -> None:
    record = _paper_record()

    first = V4IngestionService._block_doc_id(record, 1, "page_text", 1, "old text")
    second = V4IngestionService._block_doc_id(record, 1, "page_text", 1, "new text")
    next_chunk = V4IngestionService._block_doc_id(record, 1, "page_text", 2, "new text")

    assert first == second
    assert first != next_chunk


def test_persist_jsonl_and_state_use_complete_replacement(tmp_path: Path) -> None:
    docs = [
        Document(
            page_content="fresh content",
            metadata={"doc_id": "b1", "paper_id": "P1"},
        )
    ]
    jsonl_path = tmp_path / "blocks.jsonl"
    state_path = tmp_path / "state.json"
    jsonl_path.write_text("stale partial write", encoding="utf-8")

    V4IngestionService._persist_jsonl(jsonl_path, docs)
    V4IngestionService._persist_json(state_path, {"papers": {"P1": {"title": "Demo"}}})

    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert rows == [{"page_content": "fresh content", "metadata": {"doc_id": "b1", "paper_id": "P1"}}]
    assert state["papers"]["P1"]["title"] == "Demo"
    assert not list(tmp_path.glob(".*.tmp"))


def test_retriever_lookup_indexes_filter_evidence_by_paper_id(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    paper_docs = [
        Document(page_content="Paper one", metadata={"doc_id": "paper::P1", "paper_id": "P1", "title": "Paper One"}),
        Document(page_content="Paper two target", metadata={"doc_id": "paper::P2", "paper_id": "P2", "title": "Paper Two"}),
    ]
    block_docs = [
        Document(page_content="target evidence from P1", metadata={"doc_id": "b1", "paper_id": "P1", "title": "Paper One", "block_type": "page_text", "page": 1}),
        Document(page_content="target evidence from P2", metadata={"doc_id": "b2", "paper_id": "P2", "title": "Paper Two", "block_type": "page_text", "page": 1}),
    ]
    V4IngestionService._persist_jsonl(settings.paper_store_path, paper_docs)
    V4IngestionService._persist_jsonl(settings.block_store_path, block_docs)
    retriever = DualIndexRetriever(settings)
    contract = QueryContract(clean_query="target", relation="general_question", targets=["target"])

    evidence = retriever.expand_evidence(
        paper_ids=["P2"],
        query="target",
        contract=contract,
        limit=5,
    )

    paper = retriever.paper_doc_by_id("P2")
    assert paper is not None
    assert paper.metadata["paper_id"] == "P2"
    assert retriever.block_doc_by_id("b2") is not None
    assert [doc.metadata["doc_id"] for doc in retriever.block_documents_for_paper("P1", limit=10)] == ["b1"]
    assert {item.paper_id for item in evidence} == {"P2"}


def test_retriever_exposes_atomic_bm25_and_hybrid_search_tools(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    paper_docs = [
        Document(page_content="title: Direct Preference Optimization\nDPO optimizes a preference objective.", metadata={"doc_id": "paper::DPO", "paper_id": "DPO", "title": "Direct Preference Optimization"}),
        Document(page_content="title: Proximal Policy Optimization\nPPO clips policy ratios.", metadata={"doc_id": "paper::PPO", "paper_id": "PPO", "title": "Proximal Policy Optimization"}),
    ]
    block_docs = [
        Document(page_content="The DPO loss directly optimizes preferences without a reward model.", metadata={"doc_id": "dpo-block", "paper_id": "DPO", "title": "Direct Preference Optimization", "block_type": "page_text", "page": 2}),
        Document(page_content="The PPO clipped surrogate objective limits policy updates.", metadata={"doc_id": "ppo-block", "paper_id": "PPO", "title": "Proximal Policy Optimization", "block_type": "page_text", "page": 3}),
    ]
    V4IngestionService._persist_jsonl(settings.paper_store_path, paper_docs)
    V4IngestionService._persist_jsonl(settings.block_store_path, block_docs)
    retriever = DualIndexRetriever(settings)
    contract = QueryContract(clean_query="DPO loss", targets=["DPO"], requested_fields=["formula"])

    bm25 = retriever.bm25_search(query="DPO loss", contract=contract, scope="blocks", limit=3)
    hybrid = retriever.hybrid_search(query="DPO loss", contract=contract, scope="auto", limit=3)

    assert bm25
    assert bm25[0].paper_id == "DPO"
    assert bm25[0].metadata["search_source"] == "block_bm25"
    assert hybrid
    assert hybrid[0].paper_id == "DPO"


def test_retriever_reranks_existing_evidence_by_focus_terms(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    retriever = DualIndexRetriever(settings)
    evidence = [
        EvidenceBlock(doc_id="a", paper_id="P1", title="A", file_path="", page=1, block_type="page_text", snippet="generic policy text", score=1.0),
        EvidenceBlock(doc_id="b", paper_id="P2", title="B", file_path="", page=1, block_type="page_text", snippet="DPO loss optimizes preferences", score=0.1),
    ]

    reranked = retriever.rerank_evidence(query="DPO loss", evidence=evidence, top_k=2, focus=["DPO"])

    assert [item.doc_id for item in reranked] == ["b", "a"]
    assert reranked[0].metadata["rerank_score"] > reranked[1].metadata["rerank_score"]


def test_retriever_reads_pdf_page_blocks_and_greps_corpus(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    paper_docs = [
        Document(
            page_content="title: Direct Preference Optimization",
            metadata={"doc_id": "paper::DPO", "paper_id": "DPO", "title": "Direct Preference Optimization"},
        )
    ]
    block_docs = [
        Document(page_content="Page one introduction", metadata={"doc_id": "dpo-p1", "paper_id": "DPO", "title": "Direct Preference Optimization", "block_type": "page_text", "page": 1}),
        Document(page_content="Page two contains L_DPO and preference loss details.", metadata={"doc_id": "dpo-p2", "paper_id": "DPO", "title": "Direct Preference Optimization", "block_type": "page_text", "page": 2}),
        Document(page_content="Page three appendix", metadata={"doc_id": "dpo-p3", "paper_id": "DPO", "title": "Direct Preference Optimization", "block_type": "page_text", "page": 3}),
    ]
    V4IngestionService._persist_jsonl(settings.paper_store_path, paper_docs)
    V4IngestionService._persist_jsonl(settings.block_store_path, block_docs)
    retriever = DualIndexRetriever(settings)

    page_blocks = retriever.read_pdf_pages(paper_id="DPO", page_from=2, page_to=2, max_chars=2000)
    grep_hits = retriever.grep_corpus(pattern=r"L_DPO", scope="blocks", paper_ids=["DPO"], max_hits=5)

    assert [item.doc_id for item in page_blocks] == ["dpo-p2"]
    assert "preference loss" in page_blocks[0].snippet
    assert [item.doc_id for item in grep_hits] == ["dpo-p2"]
    assert grep_hits[0].metadata["grep_pattern"] == r"L_DPO"


def test_ingestion_extracts_body_acronym_aliases_from_definitions_and_formulae() -> None:
    aliases = V4IngestionService._extract_acronym_aliases(
        "Preference-Bridged Alignment (PBA) defines the L_{PBA} objective for persona-conditioned generation."
    )

    assert "PBA" in aliases
    assert "Preference-Bridged Alignment" in aliases
    assert "L_PBA" in aliases


def test_retriever_title_anchor_uses_body_acronyms(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    paper_docs = [
        Document(
            page_content="title: Personalized Preference Alignment\nabstract_or_summary: no acronym in card",
            metadata={
                "doc_id": "paper::PBA",
                "paper_id": "PBA",
                "title": "Personalized Preference Alignment",
                "body_acronyms": "PBA||Preference-Bridged Alignment||L_PBA",
            },
        ),
        Document(
            page_content="title: Proximal Optimization Notes",
            metadata={"doc_id": "paper::PPO", "paper_id": "PPO", "title": "Proximal Optimization Notes"},
        ),
    ]
    V4IngestionService._persist_jsonl(settings.paper_store_path, paper_docs)
    V4IngestionService._persist_jsonl(settings.block_store_path, [])
    retriever = DualIndexRetriever(settings)

    anchored = retriever.title_anchor(["PBA"])

    assert [doc.metadata["paper_id"] for doc in anchored] == ["PBA"]


def test_search_papers_uses_target_alias_notes_for_body_acronym_anchor(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    paper_docs = [
        Document(
            page_content="title: Personalized Preference Alignment\nabstract_or_summary: no acronym in card",
            metadata={
                "doc_id": "paper::PBA",
                "paper_id": "PBA",
                "title": "Personalized Preference Alignment",
                "body_acronyms": "PBA||Preference-Bridged Alignment",
            },
        ),
        Document(
            page_content="title: L_PBA Notation Survey\nabstract_or_summary: unrelated notation overview",
            metadata={"doc_id": "paper::NOISE", "paper_id": "NOISE", "title": "L_PBA Notation Survey"},
        ),
    ]
    V4IngestionService._persist_jsonl(settings.paper_store_path, paper_docs)
    V4IngestionService._persist_jsonl(settings.block_store_path, [])
    retriever = DualIndexRetriever(settings)
    contract = QueryContract(
        clean_query="L_PBA 是什么意思？",
        targets=["L_PBA"],
        notes=["target_alias=PBA"],
    )

    papers = retriever.search_papers(query=contract.clean_query, contract=contract, limit=2)

    assert papers
    assert papers[0].paper_id == "PBA"


def test_retriever_filters_historical_book_documents_at_runtime(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    paper_docs = [
        Document(
            page_content="title: 大模型RAG实战_RAG原理、应用与系统构建\n本书介绍 Transformer 和 RAG 的系统构建。",
            metadata={
                "doc_id": "paper::BOOK",
                "paper_id": "BOOK",
                "title": "大模型RAG实战_RAG原理、应用与系统构建",
                "authors": "",
                "year": "",
                "tags": "",
            },
        ),
        Document(
            page_content="title: Attention Is All You Need\nWe propose Transformer, a new simple network architecture.",
            metadata={
                "doc_id": "paper::AIAYN",
                "paper_id": "AIAYN",
                "title": "Attention Is All You Need",
                "authors": "Ashish Vaswani",
                "year": "2017",
                "tags": "",
            },
        ),
        Document(
            page_content="title: Seedream 4.0\nWe develop a highly efficient diffusion transformer for image generation.",
            metadata={
                "doc_id": "paper::SEEDREAM",
                "paper_id": "SEEDREAM",
                "title": "Seedream 4.0",
                "authors": "Team Seedream",
                "year": "2025",
                "tags": "",
            },
        ),
    ]
    block_docs = [
        Document(
            page_content="本书第1章介绍 Transformer 背景。",
            metadata={
                "doc_id": "book-block",
                "paper_id": "BOOK",
                "title": "大模型RAG实战_RAG原理、应用与系统构建",
                "authors": "",
                "year": "",
                "block_type": "page_text",
                "page": 1,
            },
        ),
        Document(
            page_content="We propose Transformer, a model architecture based solely on attention mechanisms.",
            metadata={
                "doc_id": "aiayn-block",
                "paper_id": "AIAYN",
                "title": "Attention Is All You Need",
                "authors": "Ashish Vaswani",
                "year": "2017",
                "block_type": "page_text",
                "page": 1,
            },
        ),
    ]
    V4IngestionService._persist_jsonl(settings.paper_store_path, paper_docs)
    V4IngestionService._persist_jsonl(settings.block_store_path, block_docs)
    retriever = DualIndexRetriever(settings)
    contract = QueryContract(
        clean_query="Transformer架构最先由哪篇论文提出？",
        relation="origin_lookup",
        targets=["Transformer"],
        answer_slots=["origin"],
    )

    papers = retriever.search_papers(query=contract.clean_query, contract=contract, limit=5)

    assert retriever.paper_doc_by_id("BOOK") is None
    assert retriever.block_documents_for_paper("BOOK") == []
    assert [paper.paper_id for paper in papers if paper.paper_id == "BOOK"] == []
    assert papers[0].paper_id == "AIAYN"
