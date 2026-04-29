from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.services.intent_marker_matching import query_matches_any
from app.services.zotero_sqlite import (
    PAPERLIKE_WEBPAGE_TITLE_MARKERS,
    PaperRecord,
    ZoteroSQLiteReader,
)


def _settings() -> Settings:
    return Settings(_env_file=None, openai_api_key="", data_dir=Path("/tmp/zprag-v4-test-data"))


def _record(
    *,
    item_type: str,
    tags: list[str] | None = None,
    title: str = "Attention Is All You Need",
    abstract_note: str = "Transformer is introduced in this paper.",
    source_url: str = "https://arxiv.org/abs/1706.03762",
    website_title: str = "arXiv.org",
) -> PaperRecord:
    return PaperRecord(
        parent_item_id=1,
        attachment_item_id=2,
        attachment_key="DEMO",
        item_type=item_type,
        title=title,
        authors=["Ashish Vaswani"],
        year="2017",
        tags=tags or [],
        abstract_note=abstract_note,
        source_url=source_url,
        website_title=website_title,
        file_path="/tmp/demo.pdf",
        file_exists=True,
    )


def test_reader_excludes_book_tag_even_if_item_looks_like_paper() -> None:
    reader = ZoteroSQLiteReader(_settings())

    allowed = reader.should_include_record(_record(item_type="preprint", tags=[]))
    blocked = reader.should_include_record(_record(item_type="preprint", tags=["书籍"]))

    assert allowed is True
    assert blocked is False


def test_reader_keeps_arxiv_webpage_papers() -> None:
    reader = ZoteroSQLiteReader(_settings())

    record = _record(
        item_type="webpage",
        source_url="https://arxiv.org/abs/1706.03762v7",
        website_title="arXiv.org",
    )

    assert reader.should_include_record(record) is True


def test_reader_excludes_non_academic_webpage_pdf() -> None:
    reader = ZoteroSQLiteReader(_settings())

    record = _record(
        item_type="webpage",
        title="Meeting Notes for Product Planning",
        abstract_note="",
        source_url="https://example.com/internal-notes.pdf",
        website_title="Example Docs",
    )

    assert reader.should_include_record(record) is False


def test_reader_uses_paperlike_webpage_title_markers() -> None:
    reader = ZoteroSQLiteReader(_settings())

    record = _record(
        item_type="webpage",
        title="A Survey on Alignment Reasoning",
        source_url="https://example.com/paper.pdf",
        website_title="Example",
    )

    assert query_matches_any("alignment survey", "", PAPERLIKE_WEBPAGE_TITLE_MARKERS)
    assert reader.should_include_record(record) is True
