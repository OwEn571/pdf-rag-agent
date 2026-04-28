from __future__ import annotations

import logging
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.parse import unquote

from app.core.config import Settings

logger = logging.getLogger(__name__)

ATTACHMENT_SQL = """
SELECT
    ia.parentItemID AS parent_item_id,
    ia.itemID AS attachment_item_id,
    ia.path AS attachment_path,
    ia.contentType AS content_type,
    parent.key AS parent_key,
    attachment.key AS attachment_key,
    parent_type.typeName AS parent_item_type
FROM itemAttachments ia
JOIN items parent ON parent.itemID = ia.parentItemID
JOIN items attachment ON attachment.itemID = ia.itemID
JOIN itemTypes parent_type ON parent_type.itemTypeID = parent.itemTypeID
WHERE ia.parentItemID IS NOT NULL
  AND ia.path IS NOT NULL
  AND lower(ia.contentType) = 'application/pdf'
ORDER BY ia.parentItemID
"""

ITEM_FIELDS_SQL = """
SELECT f.fieldName AS field_name, v.value AS value
FROM itemData d
JOIN fields f ON f.fieldID = d.fieldID
JOIN itemDataValues v ON v.valueID = d.valueID
WHERE d.itemID = ?
"""

AUTHORS_SQL_CREATOR_DATA = """
SELECT cd.firstName AS first_name, cd.lastName AS last_name
FROM itemCreators ic
JOIN creators c ON c.creatorID = ic.creatorID
JOIN creatorData cd ON cd.creatorDataID = c.creatorDataID
WHERE ic.itemID = ?
ORDER BY ic.orderIndex
"""

AUTHORS_SQL_DIRECT = """
SELECT c.firstName AS first_name, c.lastName AS last_name
FROM itemCreators ic
JOIN creators c ON c.creatorID = ic.creatorID
WHERE ic.itemID = ?
ORDER BY ic.orderIndex
"""

TAGS_SQL = """
SELECT t.name AS tag_name
FROM itemTags it
JOIN tags t ON t.tagID = it.tagID
WHERE it.itemID = ?
ORDER BY t.name
"""

COLLECTION_ITEMS_SQL = """
SELECT
    c.collectionID AS collection_id,
    c.collectionName AS collection_name,
    c.parentCollectionID AS parent_collection_id,
    ci.itemID AS item_id
FROM collectionItems ci
JOIN collections c ON c.collectionID = ci.collectionID
ORDER BY c.collectionName
"""

YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


@dataclass(slots=True)
class PaperRecord:
    parent_item_id: int
    attachment_item_id: int
    attachment_key: str
    item_type: str
    title: str
    authors: list[str]
    year: str
    tags: list[str]
    abstract_note: str
    source_url: str
    website_title: str
    file_path: str
    file_exists: bool


class ZoteroSQLiteReader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._authors_sql = AUTHORS_SQL_DIRECT

    def read_records(self, max_papers: int | None = None) -> list[PaperRecord]:
        db_path = self.settings.resolved_zotero_sqlite_path
        if not db_path.exists():
            raise FileNotFoundError(f"zotero sqlite not found: {db_path}")
        try:
            return self._read_records_from_db(db_path=db_path, max_papers=max_papers, immutable=True)
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            logger.warning("zotero sqlite locked, retry with snapshot copy: %s", exc)
            snapshot_path = self._copy_sqlite_snapshot(db_path)
            try:
                return self._read_records_from_db(snapshot_path, max_papers=max_papers, immutable=True)
            finally:
                self._cleanup_sqlite_snapshot(snapshot_path)

    def read_attachment_collection_paths(self) -> dict[str, list[str]]:
        db_path = self.settings.resolved_zotero_sqlite_path
        if not db_path.exists():
            return {}
        try:
            return self._read_attachment_collection_paths_from_db(db_path=db_path, immutable=True)
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                logger.warning("zotero collection read failed: %s", exc)
                return {}
            logger.warning("zotero sqlite locked, read collections from snapshot: %s", exc)
            snapshot_path = self._copy_sqlite_snapshot(db_path)
            try:
                return self._read_attachment_collection_paths_from_db(db_path=snapshot_path, immutable=True)
            finally:
                self._cleanup_sqlite_snapshot(snapshot_path)

    def _read_records_from_db(
        self,
        db_path: Path,
        max_papers: int | None,
        immutable: bool,
    ) -> list[PaperRecord]:
        uri = f"file:{db_path}?mode=ro"
        if immutable:
            uri += "&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        try:
            self._authors_sql = self._resolve_authors_sql(connection)
            rows = connection.execute(ATTACHMENT_SQL).fetchall()
            records: list[PaperRecord] = []
            excluded = 0
            for row in rows:
                parent_item_id = int(row["parent_item_id"])
                attachment_path = str(row["attachment_path"])
                attachment_key = str(row["attachment_key"])
                resolved_file = self._resolve_pdf_path(attachment_path, attachment_key)
                item_fields = self._read_item_fields(connection, parent_item_id)
                item_type = str(row["parent_item_type"] or "").strip()
                title = item_fields.get("title") or "未命名论文"
                year = self._extract_year(item_fields.get("date") or item_fields.get("year") or "")
                authors = self._read_authors(connection, parent_item_id)
                tags = self._read_tags(connection, parent_item_id)
                abstract_note = item_fields.get("abstractNote", "").strip()
                source_url = str(item_fields.get("url", "")).strip()
                website_title = str(item_fields.get("websiteTitle", "")).strip()
                record = PaperRecord(
                    parent_item_id=parent_item_id,
                    attachment_item_id=int(row["attachment_item_id"]),
                    attachment_key=attachment_key,
                    item_type=item_type,
                    title=title,
                    authors=authors,
                    year=year,
                    tags=tags,
                    abstract_note=abstract_note,
                    source_url=source_url,
                    website_title=website_title,
                    file_path=str(resolved_file),
                    file_exists=resolved_file.exists(),
                )
                if not self.should_include_record(record):
                    excluded += 1
                    continue
                records.append(record)
                if max_papers and len(records) >= max_papers:
                    break
            logger.info("loaded zotero records: %s (excluded=%s)", len(records), excluded)
            return records
        finally:
            connection.close()

    def _read_attachment_collection_paths_from_db(self, *, db_path: Path, immutable: bool) -> dict[str, list[str]]:
        uri = f"file:{db_path}?mode=ro"
        if immutable:
            uri += "&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        try:
            if not self._table_exists(connection, "collections") or not self._table_exists(connection, "collectionItems"):
                return {}
            collection_rows = connection.execute(COLLECTION_ITEMS_SQL).fetchall()
            collection_names: dict[int, str] = {}
            collection_parents: dict[int, int | None] = {}
            item_collection_ids: dict[int, list[int]] = {}
            for row in collection_rows:
                collection_id = int(row["collection_id"])
                collection_names[collection_id] = str(row["collection_name"] or "").strip()
                parent_id = row["parent_collection_id"]
                collection_parents[collection_id] = int(parent_id) if parent_id is not None else None
                item_id = int(row["item_id"])
                item_collection_ids.setdefault(item_id, []).append(collection_id)

            def collection_path(collection_id: int) -> str:
                names: list[str] = []
                seen: set[int] = set()
                current: int | None = collection_id
                while current is not None and current not in seen:
                    seen.add(current)
                    name = collection_names.get(current, "").strip()
                    if name:
                        names.append(name)
                    current = collection_parents.get(current)
                names.reverse()
                return " / ".join(names)

            attachment_paths: dict[str, list[str]] = {}
            attachment_rows = connection.execute(ATTACHMENT_SQL).fetchall()
            for row in attachment_rows:
                attachment_key = str(row["attachment_key"] or "").strip()
                if not attachment_key:
                    continue
                item_ids = [int(row["parent_item_id"]), int(row["attachment_item_id"])]
                paths: list[str] = []
                for item_id in item_ids:
                    for collection_id in item_collection_ids.get(item_id, []):
                        path = collection_path(collection_id)
                        if path and path not in paths:
                            paths.append(path)
                if paths:
                    attachment_paths[attachment_key] = paths
            return attachment_paths
        finally:
            connection.close()

    @staticmethod
    def _copy_sqlite_snapshot(db_path: Path) -> Path:
        with tempfile.NamedTemporaryFile(prefix="zotero_snapshot_", suffix=".sqlite", delete=False) as tmp:
            snapshot_path = Path(tmp.name)
        shutil.copy2(db_path, snapshot_path)
        for suffix in ("-wal", "-shm"):
            source = Path(f"{db_path}{suffix}")
            if source.exists():
                shutil.copy2(source, Path(f"{snapshot_path}{suffix}"))
        return snapshot_path

    @staticmethod
    def _cleanup_sqlite_snapshot(snapshot_path: Path) -> None:
        for suffix in ("", "-wal", "-shm"):
            Path(f"{snapshot_path}{suffix}").unlink(missing_ok=True)

    def _read_item_fields(self, conn: sqlite3.Connection, item_id: int) -> dict[str, str]:
        rows = conn.execute(ITEM_FIELDS_SQL, (item_id,)).fetchall()
        return {str(row["field_name"]): str(row["value"]) for row in rows if row["value"] is not None}

    def _read_authors(self, conn: sqlite3.Connection, item_id: int) -> list[str]:
        rows = conn.execute(self._authors_sql, (item_id,)).fetchall()
        authors: list[str] = []
        for row in rows:
            first_name = (row["first_name"] or "").strip()
            last_name = (row["last_name"] or "").strip()
            name = " ".join(part for part in (first_name, last_name) if part).strip()
            if name:
                authors.append(name)
        return authors

    def _resolve_authors_sql(self, conn: sqlite3.Connection) -> str:
        creators_columns = self._table_columns(conn, "creators")
        has_creator_data_table = self._table_exists(conn, "creatorData")
        has_creator_data_column = "creatorDataID" in creators_columns
        if has_creator_data_table and has_creator_data_column:
            return AUTHORS_SQL_CREATOR_DATA
        return AUTHORS_SQL_DIRECT

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table_name,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        columns = set()
        for row in rows:
            if isinstance(row, sqlite3.Row):
                columns.add(str(row["name"]))
            else:
                columns.add(str(row[1]))
        return columns

    def _read_tags(self, conn: sqlite3.Connection, item_id: int) -> list[str]:
        rows = conn.execute(TAGS_SQL, (item_id,)).fetchall()
        return [str(row["tag_name"]) for row in rows if row["tag_name"]]

    def should_include_record(self, record: PaperRecord) -> bool:
        if self._has_excluded_tag(record.tags):
            return False
        item_type = record.item_type.strip()
        if item_type in self.settings.ingestion_allowed_item_types:
            return True
        if item_type == "webpage":
            return self._is_paper_like_webpage(record)
        return False

    def _has_excluded_tag(self, tags: list[str]) -> bool:
        excluded = {tag.lower().strip() for tag in self.settings.ingestion_excluded_tags if tag.strip()}
        for tag in tags:
            normalized = str(tag).lower().strip()
            if normalized in excluded:
                return True
        return False

    def _is_paper_like_webpage(self, record: PaperRecord) -> bool:
        host = self._normalize_host(record.source_url)
        if host and any(host == candidate or host.endswith(f".{candidate}") for candidate in self.settings.ingestion_academic_web_hosts):
            return True
        website_title = record.website_title.lower().strip()
        if website_title in {"arxiv.org", "openreview", "acl anthology"}:
            return True
        title = record.title.lower().strip()
        if any(marker in title for marker in ("survey", "benchmark", "transformer", "alignment", "reasoning")) and record.abstract_note:
            return True
        return False

    @staticmethod
    def _normalize_host(url: str) -> str:
        if not url.strip():
            return ""
        parsed = urlparse(url)
        return (parsed.netloc or "").lower().strip()

    def _extract_year(self, raw: str) -> str:
        if not raw:
            return ""
        matched = YEAR_PATTERN.search(raw)
        return matched.group(0) if matched else ""

    def _resolve_pdf_path(self, attachment_path: str, attachment_key: str) -> Path:
        normalized = attachment_path.strip()
        if normalized.startswith("storage:"):
            filename = normalized.split("storage:", maxsplit=1)[1].strip()
            return (self.settings.resolved_zotero_storage_dir / attachment_key / filename).expanduser()
        if normalized.startswith("file://"):
            raw = normalized.removeprefix("file://")
            if raw.startswith("localhost/"):
                raw = raw.removeprefix("localhost/")
            decoded = unquote(raw)
            return Path(decoded).expanduser()
        normalized = normalized.replace("\\", "/")
        candidate = Path(normalized).expanduser()
        if candidate.is_absolute():
            return candidate
        return (self.settings.zotero_root / candidate).expanduser()
