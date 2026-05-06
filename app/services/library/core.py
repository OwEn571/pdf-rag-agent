from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.services.retrieval import DualIndexRetriever
from app.services.library.zotero_sqlite import ZoteroSQLiteReader


class LibraryBrowserService:
    def __init__(self, *, settings: Settings, retriever: DualIndexRetriever) -> None:
        self.settings = settings
        self.retriever = retriever

    def list_library(self) -> dict[str, Any]:
        collection_paths = ZoteroSQLiteReader(self.settings).read_attachment_collection_paths()
        papers = [self._paper_payload(doc.metadata or {}, doc.page_content or "", collection_paths) for doc in self.retriever.paper_documents()]
        papers = [paper for paper in papers if paper["paper_id"]]
        papers.sort(key=lambda item: (str(item.get("categories", ["未分类"])[0]), str(item.get("title", "")).lower()))
        grouped: dict[str, list[dict[str, Any]]] = {}
        for paper in papers:
            categories = paper.get("categories") or ["未分类"]
            for category in categories:
                grouped.setdefault(str(category or "未分类"), []).append(paper)
        categories_payload = [
            {"name": name, "count": len(items), "papers": items}
            for name, items in sorted(grouped.items(), key=lambda item: (item[0] == "未分类", item[0].lower()))
        ]
        return {"categories": categories_payload, "total_papers": len(papers)}

    def paper_preview(self, paper_id: str) -> dict[str, Any] | None:
        doc = self.retriever.paper_doc_by_id(paper_id)
        if doc is None:
            return None
        collection_paths = ZoteroSQLiteReader(self.settings).read_attachment_collection_paths()
        paper = self._paper_payload(doc.metadata or {}, doc.page_content or "", collection_paths)
        snippets = []
        for block in self.retriever.block_documents_for_paper(paper_id, limit=10):
            meta = dict(block.metadata or {})
            snippets.append(
                {
                    "doc_id": str(meta.get("doc_id", "")),
                    "paper_id": str(meta.get("paper_id", paper_id)),
                    "page": int(meta.get("page", 0) or 0),
                    "block_type": str(meta.get("block_type", "")),
                    "caption": str(meta.get("caption", "")),
                    "snippet": " ".join(str(block.page_content or "").split())[:500],
                }
            )
        return {"paper": paper, "snippets": snippets}

    def citation_preview(self, *, doc_id: str = "", paper_id: str = "") -> dict[str, Any] | None:
        doc = self.retriever.block_doc_by_id(doc_id) if doc_id else None
        if doc is None and paper_id:
            doc = self.retriever.paper_doc_by_id(paper_id)
        if doc is None:
            return None
        meta = dict(doc.metadata or {})
        resolved_paper_id = str(meta.get("paper_id", paper_id))
        return {
            "paper_id": resolved_paper_id,
            "doc_id": str(meta.get("doc_id", doc_id)),
            "title": str(meta.get("title", "")),
            "authors": str(meta.get("authors", "")),
            "year": str(meta.get("year", "")),
            "file_path": str(meta.get("file_path", "")),
            "page": int(meta.get("page", 0) or 0),
            "block_type": str(meta.get("block_type", "paper_card")),
            "caption": str(meta.get("caption", "")),
            "snippet": " ".join(str(doc.page_content or "").split())[:900],
        }

    def pdf_path(self, paper_id: str) -> Path | None:
        doc = self.retriever.paper_doc_by_id(paper_id)
        if doc is None:
            return None
        raw_path = str((doc.metadata or {}).get("file_path", "")).strip()
        if not raw_path:
            return None
        path = Path(raw_path).expanduser()
        if not path.exists() or not path.is_file() or path.suffix.lower() != ".pdf":
            return None
        try:
            resolved_path = path.resolve()
            allowed_roots = [
                self.settings.resolved_zotero_storage_dir.expanduser().resolve(),
                self.settings.zotero_root.expanduser().resolve(),
            ]
        except OSError:
            return None
        if not any(resolved_path.is_relative_to(root) for root in allowed_roots):
            return None
        return resolved_path

    @staticmethod
    def _paper_payload(meta: dict[str, Any], content: str, collection_paths: dict[str, list[str]]) -> dict[str, Any]:
        paper_id = str(meta.get("paper_id", "")).strip()
        tags = [tag for tag in str(meta.get("tags", "")).split("||") if tag]
        categories = collection_paths.get(paper_id) or tags[:3] or ["未分类"]
        preview = str(meta.get("abstract_note", "") or meta.get("generated_summary", "") or content or "")
        preview = " ".join(preview.split())
        return {
            "paper_id": paper_id,
            "title": str(meta.get("title", "")),
            "authors": str(meta.get("authors", "")),
            "year": str(meta.get("year", "")),
            "tags": tags,
            "categories": categories,
            "file_path": str(meta.get("file_path", "")),
            "preview": preview[:520],
        }
