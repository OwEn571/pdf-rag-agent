from __future__ import annotations

from types import SimpleNamespace

from app.services.claims.paper_summary import paper_summary_text


def test_paper_summary_text_prefers_generated_summary() -> None:
    doc = SimpleNamespace(
        metadata={"generated_summary": "generated", "abstract_note": "abstract"},
        page_content="content",
    )

    assert paper_summary_text("p1", paper_doc_lookup=lambda _: doc) == "generated"


def test_paper_summary_text_falls_back_to_abstract_then_content() -> None:
    abstract_doc = SimpleNamespace(metadata={"abstract_note": "abstract"}, page_content="content")
    content_doc = SimpleNamespace(metadata={}, page_content="content" * 200)

    assert paper_summary_text("p1", paper_doc_lookup=lambda _: abstract_doc) == "abstract"
    assert paper_summary_text("p1", paper_doc_lookup=lambda _: content_doc, content_limit=12) == "contentconte"


def test_paper_summary_text_handles_missing_doc() -> None:
    assert paper_summary_text("missing", paper_doc_lookup=lambda _: None) == ""
