from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.library.metadata_sql import (
    execute_library_metadata_sql,
    fallback_library_metadata_sql_answer,
    library_metadata_rows,
    validate_library_metadata_sql,
)


def _row(paper_id: str, title: str, year: int) -> dict[str, object]:
    return {
        "paper_id": paper_id,
        "title": title,
        "authors": "Alice, Bob",
        "year": str(year),
        "year_int": year,
        "tags": "rag||survey",
        "categories": "AI/RAG",
        "aliases": "",
        "abstract": "",
        "summary": "A useful summary.",
        "has_pdf": 1,
        "file_path": "/tmp/paper.pdf",
        "searchable_text": title,
        "_author_list": ["Alice", "Bob"],
        "_tag_list": ["rag", "survey"],
        "_category_list": ["AI/RAG"],
    }


def test_validate_library_metadata_sql_rejects_non_select() -> None:
    assert validate_library_metadata_sql("SELECT title FROM papers;") == "SELECT title FROM papers"
    with pytest.raises(ValueError, match="only_select"):
        validate_library_metadata_sql("DELETE FROM papers")
    with pytest.raises(ValueError, match="unknown_table"):
        validate_library_metadata_sql("SELECT * FROM other_table")


def test_library_metadata_rows_deduplicates_and_expands_lists() -> None:
    rows = library_metadata_rows(
        paper_documents=[
            SimpleNamespace(
                metadata={
                    "paper_id": "p1",
                    "title": " Paper One ",
                    "authors": "Alice, Bob and Carol",
                    "year": "2025",
                    "tags": "rag||survey",
                    "aliases": "P1",
                    "file_path": "/tmp/p1.pdf",
                },
                page_content="paper content",
            ),
            SimpleNamespace(metadata={"paper_id": "p1", "title": "Duplicate"}, page_content=""),
            SimpleNamespace(
                metadata={
                    "paper_id": "p2",
                    "title": "No Collection Paper",
                    "authors": "Dana",
                    "year": "unknown",
                    "tags": "retrieval||agent",
                    "file_path": "/tmp/p2.txt",
                },
                page_content="fallback searchable body",
            ),
        ],
        collection_paths={"p1": ["AI/RAG"]},
    )

    assert len(rows) == 2
    assert rows[0]["title"] == "Paper One"
    assert rows[0]["year_int"] == 2025
    assert rows[0]["categories"] == "AI/RAG"
    assert rows[0]["_author_list"] == ["Alice", "Bob", "Carol"]
    assert rows[0]["has_pdf"] == 1
    assert "paper content" in str(rows[0]["searchable_text"])
    assert rows[1]["year_int"] is None
    assert rows[1]["categories"] == "retrieval||agent"
    assert rows[1]["_category_list"] == ["retrieval", "agent"]
    assert rows[1]["has_pdf"] == 0
    assert "fallback searchable body" in str(rows[1]["searchable_text"])


def test_execute_library_metadata_sql_returns_rows_and_truncation() -> None:
    result = execute_library_metadata_sql(
        sql="SELECT title, year_int FROM papers ORDER BY year_int DESC",
        paper_rows=[_row("p1", "Old Paper", 2020), _row("p2", "New Paper", 2025)],
        max_rows=1,
    )

    assert result["columns"] == ["title", "year_int"]
    assert result["rows"] == [{"title": "New Paper", "year_int": 2025}]
    assert result["truncated"] is True


def test_fallback_library_metadata_sql_answer_formats_rows_and_aggregates() -> None:
    aggregate = fallback_library_metadata_sql_answer(
        query="多少篇",
        result={"columns": ["count"], "rows": [{"count": 2}]},
    )
    assert "count=2" in aggregate

    listing = fallback_library_metadata_sql_answer(
        query="有哪些",
        result={"columns": ["title", "year"], "rows": [{"title": "New Paper", "year": "2025", "paper_id": "p2"}]},
    )
    assert "《New Paper》" in listing
    assert "paper_id：p2" in listing
