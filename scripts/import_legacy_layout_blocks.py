from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_BLOCKS = PROJECT_ROOT.parent / "pdf-rag-agent" / "data" / "bm25_documents.jsonl"
DEFAULT_V4_PAPERS = PROJECT_ROOT / "data" / "v4_papers.jsonl"
DEFAULT_V4_BLOCKS = PROJECT_ROOT / "data" / "v4_blocks.jsonl"
FORMULA_HINT_RE = re.compile(r"(π|β|sigma|sigmoid|log\s*σ|loss|objective|公式|目标函数|reward)", re.IGNORECASE)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _paper_ids_from_v4_papers(path: Path) -> set[str]:
    paper_ids: set[str] = set()
    for row in _read_jsonl(path):
        meta = dict(row.get("metadata") or {})
        paper_id = str(meta.get("paper_id", "") or "").strip()
        if paper_id:
            paper_ids.add(paper_id)
    return paper_ids


def _legacy_paper_id(meta: dict[str, Any]) -> str:
    return str(meta.get("attachment_key") or meta.get("paper_id") or meta.get("paper_key") or "").strip()


def _fallback_doc_id(*, paper_id: str, page: int, block_type: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    raw = f"{paper_id}|{page}|{block_type}|{digest}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _normalize_legacy_block(row: dict[str, Any]) -> dict[str, Any] | None:
    text = str(row.get("page_content") or "").strip()
    if not text:
        return None
    legacy_meta = dict(row.get("metadata") or {})
    paper_id = _legacy_paper_id(legacy_meta)
    if not paper_id:
        return None
    block_type = str(legacy_meta.get("block_type") or "page_text").strip() or "page_text"
    try:
        page = int(legacy_meta.get("page", 0) or 0)
    except (TypeError, ValueError):
        page = 0
    doc_id = str(legacy_meta.get("doc_id") or "").strip()
    if not doc_id:
        doc_id = _fallback_doc_id(paper_id=paper_id, page=page, block_type=block_type, text=text)
    metadata = dict(legacy_meta)
    metadata.update(
        {
            "doc_id": doc_id,
            "paper_id": paper_id,
            "page": page,
            "block_type": block_type,
            "caption": str(legacy_meta.get("caption") or ""),
            "bbox": str(legacy_meta.get("bbox") or ""),
            "attachment_key": str(legacy_meta.get("attachment_key") or paper_id),
            "formula_hint": int(bool(FORMULA_HINT_RE.search(text))),
        }
    )
    return {"page_content": text, "metadata": metadata}


def build_import_rows(
    *,
    legacy_blocks_path: Path,
    v4_papers_path: Path,
    existing_v4_blocks_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    v4_paper_ids = _paper_ids_from_v4_papers(v4_papers_path)
    rows: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    matched_paper_ids: set[str] = set()
    block_type_counts: dict[str, int] = {}

    for legacy_row in _read_jsonl(legacy_blocks_path):
        normalized = _normalize_legacy_block(legacy_row)
        if normalized is None:
            continue
        meta = dict(normalized["metadata"])
        paper_id = str(meta.get("paper_id") or "")
        if paper_id not in v4_paper_ids:
            continue
        doc_id = str(meta.get("doc_id") or "")
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        matched_paper_ids.add(paper_id)
        block_type = str(meta.get("block_type") or "")
        block_type_counts[block_type] = block_type_counts.get(block_type, 0) + 1
        rows.append(normalized)

    missing_paper_ids = v4_paper_ids - matched_paper_ids
    if missing_paper_ids:
        for existing_row in _read_jsonl(existing_v4_blocks_path):
            meta = dict(existing_row.get("metadata") or {})
            paper_id = str(meta.get("paper_id") or "")
            if paper_id not in missing_paper_ids:
                continue
            doc_id = str(meta.get("doc_id") or "")
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            block_type = str(meta.get("block_type") or "")
            block_type_counts[block_type] = block_type_counts.get(block_type, 0) + 1
            rows.append(existing_row)

    stats = {
        "v4_papers": len(v4_paper_ids),
        "matched_legacy_papers": len(matched_paper_ids),
        "missing_papers_kept_from_v4": len(missing_paper_ids),
        "rows": len(rows),
    }
    for key, value in sorted(block_type_counts.items()):
        stats[f"block_type_{key or 'unknown'}"] = value
    return rows, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import legacy layout-aware block corpus into V4 block store.")
    parser.add_argument("--legacy-blocks", type=Path, default=DEFAULT_LEGACY_BLOCKS)
    parser.add_argument("--v4-papers", type=Path, default=DEFAULT_V4_PAPERS)
    parser.add_argument("--v4-blocks", type=Path, default=DEFAULT_V4_BLOCKS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, stats = build_import_rows(
        legacy_blocks_path=args.legacy_blocks,
        v4_papers_path=args.v4_papers,
        existing_v4_blocks_path=args.v4_blocks,
    )
    if not args.dry_run:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = args.v4_blocks.with_suffix(args.v4_blocks.suffix + f".bak.{timestamp}")
        if args.v4_blocks.exists():
            shutil.copy2(args.v4_blocks, backup_path)
            stats["backup_created"] = 1
        _write_jsonl(args.v4_blocks, rows)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
