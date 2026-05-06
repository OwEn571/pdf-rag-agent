from __future__ import annotations

import re
import sqlite3
from typing import Any

from app.services.answers.library_recommendations import split_library_authors


def library_metadata_rows(
    *,
    paper_documents: list[Any],
    collection_paths: dict[str, list[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_paper_ids: set[str] = set()
    for doc in paper_documents:
        meta = dict(getattr(doc, "metadata", None) or {})
        paper_id = str(meta.get("paper_id", "")).strip()
        if not paper_id or paper_id in seen_paper_ids:
            continue
        seen_paper_ids.add(paper_id)
        title = " ".join(str(meta.get("title", "") or "").split())
        authors = " ".join(str(meta.get("authors", "") or "").split())
        year = str(meta.get("year", "") or "").strip()
        year_int = int(year) if year.isdigit() else None
        tags = [tag.strip() for tag in str(meta.get("tags", "") or "").split("||") if tag.strip()]
        categories = [str(item or "未分类").strip() for item in (collection_paths.get(paper_id) or tags[:3] or ["未分类"])]
        aliases = [alias.strip() for alias in str(meta.get("aliases", "") or "").split("||") if alias.strip()]
        abstract = " ".join(str(meta.get("abstract_note", "") or "").split())
        summary = " ".join(str(meta.get("generated_summary", "") or "").split())
        file_path = str(meta.get("file_path", "") or "").strip()
        page_content = str(getattr(doc, "page_content", "") or "")
        searchable_text = " ".join(
            item
            for item in [
                title,
                authors,
                year,
                " ".join(tags),
                " ".join(categories),
                " ".join(aliases),
                abstract,
                summary,
                " ".join(page_content.split())[:1200],
            ]
            if item
        )
        rows.append(
            {
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "year": year,
                "year_int": year_int,
                "tags": "||".join(tags),
                "categories": "||".join(categories),
                "aliases": "||".join(aliases),
                "abstract": abstract[:3000],
                "summary": summary[:3000],
                "has_pdf": 1 if file_path.lower().endswith(".pdf") else 0,
                "file_path": file_path,
                "searchable_text": searchable_text[:6000],
                "_author_list": split_library_authors(authors),
                "_tag_list": tags,
                "_category_list": categories,
            }
        )
    return rows


def library_metadata_sql_schema_description() -> dict[str, Any]:
    return {
        "dialect": "SQLite",
        "tables": {
            "papers": {
                "description": "One row per indexed local paper.",
                "columns": {
                    "paper_id": "Stable local paper id.",
                    "title": "Paper title.",
                    "authors": "Comma-separated author names.",
                    "year": "Original year string from metadata.",
                    "year_int": "Integer publication year when parseable, otherwise NULL.",
                    "tags": "Tags joined by ||.",
                    "categories": "Zotero collection/category paths joined by ||.",
                    "aliases": "Known aliases joined by ||.",
                    "abstract": "Abstract text when available.",
                    "summary": "Generated summary when available.",
                    "has_pdf": "1 if the indexed file path is a PDF, else 0.",
                    "file_path": "Local PDF path when available.",
                    "searchable_text": "Concatenated title/authors/year/tags/categories/aliases/abstract/summary for broad LIKE matching.",
                },
            },
            "paper_authors": {
                "description": "One row per parsed author name.",
                "columns": {
                    "paper_id": "Stable local paper id.",
                    "title": "Paper title copied from papers.",
                    "author": "Single author name.",
                    "year_int": "Integer publication year when parseable.",
                },
            },
            "paper_tags": {
                "description": "One row per tag.",
                "columns": {
                    "paper_id": "Stable local paper id.",
                    "title": "Paper title copied from papers.",
                    "tag": "Single tag.",
                    "year_int": "Integer publication year when parseable.",
                },
            },
            "paper_categories": {
                "description": "One row per Zotero collection/category path.",
                "columns": {
                    "paper_id": "Stable local paper id.",
                    "title": "Paper title copied from papers.",
                    "category": "Single collection/category path.",
                    "year_int": "Integer publication year when parseable.",
                },
            },
        },
    }


def validate_library_metadata_sql(sql: str) -> str:
    normalized = " ".join(str(sql or "").strip().split())
    if normalized.endswith(";"):
        normalized = normalized[:-1].strip()
    if not normalized:
        raise ValueError("empty_sql")
    lowered = normalized.lower()
    if ";" in normalized:
        raise ValueError("multiple_sql_statements_are_not_allowed")
    if any(token in lowered for token in ["--", "/*", "*/"]):
        raise ValueError("sql_comments_are_not_allowed")
    if not (lowered.startswith("select ") or lowered.startswith("with ")):
        raise ValueError("only_select_sql_is_allowed")
    forbidden = {
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "replace",
        "attach",
        "detach",
        "pragma",
        "vacuum",
        "reindex",
        "analyze",
        "load_extension",
    }
    tokens = set(re.findall(r"[a-z_]+", lowered))
    blocked = sorted(tokens & forbidden)
    if blocked:
        raise ValueError(f"forbidden_sql_keyword={blocked[0]}")
    if re.search(r"\b(sqlite_master|sqlite_schema|sqlite_temp_master)\b", lowered):
        raise ValueError("sqlite_internal_tables_are_not_allowed")
    allowed_tables = {"papers", "paper_authors", "paper_tags", "paper_categories"}
    table_refs = re.findall(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered)
    disallowed_tables = [table for table in table_refs if table not in allowed_tables]
    if disallowed_tables:
        raise ValueError(f"unknown_table={disallowed_tables[0]}")
    return normalized


def execute_library_metadata_sql(
    *,
    sql: str,
    paper_rows: list[dict[str, Any]],
    max_rows: int,
) -> dict[str, Any]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(
            """
            CREATE TABLE papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year TEXT,
                year_int INTEGER,
                tags TEXT,
                categories TEXT,
                aliases TEXT,
                abstract TEXT,
                summary TEXT,
                has_pdf INTEGER,
                file_path TEXT,
                searchable_text TEXT
            );
            CREATE TABLE paper_authors (
                paper_id TEXT,
                title TEXT,
                author TEXT,
                year_int INTEGER
            );
            CREATE TABLE paper_tags (
                paper_id TEXT,
                title TEXT,
                tag TEXT,
                year_int INTEGER
            );
            CREATE TABLE paper_categories (
                paper_id TEXT,
                title TEXT,
                category TEXT,
                year_int INTEGER
            );
            CREATE INDEX idx_papers_year ON papers(year_int);
            CREATE INDEX idx_papers_title ON papers(title);
            CREATE INDEX idx_paper_authors_author ON paper_authors(author);
            CREATE INDEX idx_paper_tags_tag ON paper_tags(tag);
            CREATE INDEX idx_paper_categories_category ON paper_categories(category);
            """
        )
        paper_insert_rows = [
            (
                row.get("paper_id"),
                row.get("title"),
                row.get("authors"),
                row.get("year"),
                row.get("year_int"),
                row.get("tags"),
                row.get("categories"),
                row.get("aliases"),
                row.get("abstract"),
                row.get("summary"),
                row.get("has_pdf"),
                row.get("file_path"),
                row.get("searchable_text"),
            )
            for row in paper_rows
        ]
        conn.executemany(
            """
            INSERT INTO papers (
                paper_id, title, authors, year, year_int, tags, categories, aliases,
                abstract, summary, has_pdf, file_path, searchable_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            paper_insert_rows,
        )
        author_rows = [
            (row["paper_id"], row["title"], author, row["year_int"])
            for row in paper_rows
            for author in list(row.get("_author_list", []) or [])
        ]
        tag_rows = [
            (row["paper_id"], row["title"], tag, row["year_int"])
            for row in paper_rows
            for tag in list(row.get("_tag_list", []) or [])
        ]
        category_rows = [
            (row["paper_id"], row["title"], category, row["year_int"])
            for row in paper_rows
            for category in list(row.get("_category_list", []) or [])
        ]
        conn.executemany("INSERT INTO paper_authors (paper_id, title, author, year_int) VALUES (?, ?, ?, ?)", author_rows)
        conn.executemany("INSERT INTO paper_tags (paper_id, title, tag, year_int) VALUES (?, ?, ?, ?)", tag_rows)
        conn.executemany("INSERT INTO paper_categories (paper_id, title, category, year_int) VALUES (?, ?, ?, ?)", category_rows)
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.execute(sql)
        columns = [str(item[0]) for item in (cursor.description or [])]
        fetched = cursor.fetchmany(max_rows + 1)
        truncated = len(fetched) > max_rows
        result_rows = [sqlite_row_to_payload(row) for row in fetched[:max_rows]]
        return {
            "sql": sql,
            "columns": columns,
            "rows": result_rows,
            "row_count": len(result_rows),
            "truncated": truncated,
            "error": "",
        }
    finally:
        conn.close()


def sqlite_row_to_payload(row: sqlite3.Row) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in row.keys():
        value = row[key]
        if value is None or isinstance(value, (str, int, float)):
            payload[str(key)] = value
        else:
            payload[str(key)] = str(value)
    return payload


def fallback_library_metadata_sql_answer(*, query: str, result: dict[str, Any]) -> str:
    _ = query
    rows = list(result.get("rows", []) or [])
    columns = [str(item) for item in list(result.get("columns", []) or [])]
    if not rows:
        return "当前本地 paper index 的元信息查询没有返回匹配记录。"
    if len(rows) == 1 and "title" not in {column.lower() for column in columns}:
        values = "，".join(f"{key}={value}" for key, value in rows[0].items())
        return f"我查的是当前本地 paper index 元信息，结果是：{values}。"
    lines = [
        "我查的是当前本地 paper index 元信息，SQL 查询返回这些记录：",
        "",
    ]
    for row in rows[:12]:
        title = str(row.get("title", "") or row.get("paper_title", "") or "").strip()
        year = str(row.get("year", "") or row.get("year_int", "") or "").strip()
        authors = str(row.get("authors", "") or row.get("author", "") or "").strip()
        paper_id = str(row.get("paper_id", "") or "").strip()
        parts = [f"《{title}》" if title else paper_id or "未命名论文"]
        if year:
            parts.append(f"年份：{year}")
        if authors:
            parts.append(f"作者：{authors}")
        if paper_id and title:
            parts.append(f"paper_id：{paper_id}")
        lines.append("- " + "；".join(parts))
    if bool(result.get("truncated")):
        lines.append("")
        lines.append("结果较多，以上只展示前几条。")
    return "\n".join(lines)
