from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings
from app.services.infra.model_clients import ModelClients
from app.services.retrieval.pdf_extractor import ExtractedBlock, ExtractedPage, PDFExtractor
from app.services.retrieval.vector_index import CollectionVectorIndex
from app.services.library.zotero_sqlite import PaperRecord, ZoteroSQLiteReader

logger = logging.getLogger(__name__)
FORMULA_HINT_RE = re.compile(r"(π|β|sigma|sigmoid|log\s*σ|loss|objective|公式|目标函数|reward)", re.IGNORECASE)


@dataclass(slots=True)
class IngestionStats:
    paper_records: int = 0
    papers_indexed: int = 0
    papers_missing_pdf: int = 0
    block_docs: int = 0
    paper_docs: int = 0
    vectors_upserted: int = 0
    papers_with_generated_summary: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class V4IngestionService:
    def __init__(self, settings: Settings, clients: ModelClients | None = None) -> None:
        self.settings = settings
        self.clients = clients or ModelClients(settings)
        self.reader = ZoteroSQLiteReader(settings)
        self.extractor = PDFExtractor(settings=settings, prefer_unstructured=True)
        # P2-5: Dual splitter for Chinese/English mixed content.
        # Chinese content (split by 。) gets 600 chars; English (split by .) gets 800.
        # This avoids chunks that are too coarse for mixed-language PDFs.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", "。", ". ", " ", ""],
        )
        self.splitter_coarse = RecursiveCharacterTextSplitter(
            chunk_size=1600,
            chunk_overlap=240,
            separators=["\n\n", "\n", "。", ". ", " ", ""],
        )

    def rebuild(self, *, max_papers: int | None = None, force_rebuild: bool = True) -> IngestionStats:
        records = self.reader.read_records(max_papers=max_papers)
        stats = IngestionStats(paper_records=len(records))
        paper_docs: list[Document] = []
        block_docs: list[Document] = []
        paper_ids: list[str] = []
        block_ids: list[str] = []
        state: dict[str, Any] = {"papers": {}}

        for record in records:
            paper_id = record.attachment_key
            if not record.file_exists:
                stats.papers_missing_pdf += 1
                continue
            pages = self.extractor.extract_pages(Path(record.file_path))
            if not pages:
                logger.warning("skip empty extraction: %s", record.file_path)
                continue
            paper_doc, generated = self._build_paper_card(record=record, pages=pages)
            paper_doc_id = str(paper_doc.metadata["doc_id"])
            paper_docs.append(paper_doc)
            paper_ids.append(paper_doc_id)
            stats.paper_docs += 1
            if generated:
                stats.papers_with_generated_summary += 1

            one_block_docs = self._build_block_documents(record=record, pages=pages)
            for doc in one_block_docs:
                block_docs.append(doc)
                block_ids.append(str(doc.metadata["doc_id"]))
            stats.block_docs += len(one_block_docs)
            stats.papers_indexed += 1
            state["papers"][paper_id] = {
                "title": record.title,
                "year": record.year,
                "file_path": record.file_path,
                "paper_doc_id": paper_doc_id,
                "block_doc_ids": [str(doc.metadata["doc_id"]) for doc in one_block_docs],
            }

        self._persist_jsonl(self.settings.paper_store_path, paper_docs)
        self._persist_jsonl(self.settings.block_store_path, block_docs)
        self._persist_json(
            self.settings.ingestion_state_path,
            state,
        )

        vectors_upserted = 0
        if self.settings.openai_api_key:
            paper_index = self._build_vector_index(self.settings.milvus_paper_collection)
            block_index = self._build_vector_index(self.settings.milvus_block_collection)
            try:
                vectors_upserted += paper_index.upsert_documents(
                    paper_docs,
                    force_rebuild=force_rebuild,
                    batch_size=self.settings.upsert_batch_size,
                    doc_ids=paper_ids,
                )
                vectors_upserted += block_index.upsert_documents(
                    block_docs,
                    force_rebuild=force_rebuild,
                    batch_size=self.settings.upsert_batch_size,
                    doc_ids=block_ids,
                )
            except Exception as exc:  # noqa: BLE001
                if self.settings.embedding_fallback_model != self.settings.embedding_model:
                    logger.warning("primary embedding failed, retry fallback model: %s", exc)
                    paper_index = self._build_vector_index(
                        self.settings.milvus_paper_collection,
                        embedding_model=self.settings.embedding_fallback_model,
                    )
                    block_index = self._build_vector_index(
                        self.settings.milvus_block_collection,
                        embedding_model=self.settings.embedding_fallback_model,
                    )
                    vectors_upserted += paper_index.upsert_documents(
                        paper_docs,
                        force_rebuild=force_rebuild,
                        batch_size=self.settings.upsert_batch_size,
                        doc_ids=paper_ids,
                    )
                    vectors_upserted += block_index.upsert_documents(
                        block_docs,
                        force_rebuild=force_rebuild,
                        batch_size=self.settings.upsert_batch_size,
                        doc_ids=block_ids,
                    )
                else:
                    raise
        stats.vectors_upserted = vectors_upserted
        return stats

    def _build_vector_index(self, collection_name: str, embedding_model: str | None = None) -> CollectionVectorIndex:
        return CollectionVectorIndex(
            self.settings,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )

    def _build_paper_card(self, *, record: PaperRecord, pages: list[ExtractedPage]) -> tuple[Document, bool]:
        aliases = self._build_aliases(record.title)
        body_acronyms = self._body_acronyms(pages)
        abstract_text = record.abstract_note.strip()
        generated = False
        if abstract_text:
            summary = abstract_text
        else:
            summary = self._generate_paper_summary(record=record, pages=pages)
            generated = bool(summary)
        summary = summary.strip() or self._fallback_summary_from_pages(pages)
        hints = self._build_evidence_hints(pages)
        card_text = (
            f"title: {record.title}\n"
            f"aliases: {' | '.join(aliases)}\n"
            f"body_acronyms: {' | '.join(body_acronyms)}\n"
            f"authors: {', '.join(record.authors)}\n"
            f"year: {record.year}\n"
            f"tags: {' | '.join(record.tags)}\n"
            f"abstract_or_summary:\n{summary}\n\n"
            f"top_evidence_hints:\n{hints}"
        ).strip()
        doc_id = f"paper::{record.attachment_key}"
        metadata = {
            "doc_id": doc_id,
            "paper_id": record.attachment_key,
            "title": record.title,
            "authors": ", ".join(record.authors),
            "year": record.year,
            "tags": "||".join(record.tags),
            "item_type": record.item_type,
            "source_url": record.source_url,
            "website_title": record.website_title,
            "file_path": record.file_path,
            "attachment_key": record.attachment_key,
            "aliases": "||".join(aliases),
            "body_acronyms": "||".join(body_acronyms),
            "abstract_note": record.abstract_note,
            "generated_summary": summary if generated else "",
            "block_type": "paper_card",
            "caption": "",
            "bbox": "",
        }
        return Document(page_content=card_text, metadata=metadata), generated

    def _build_block_documents(self, *, record: PaperRecord, pages: list[ExtractedPage]) -> list[Document]:
        docs: list[Document] = []
        for page in pages:
            page_text = page.text.strip()
            if page_text:
                page_doc = Document(
                    page_content=page_text,
                    metadata=self._block_metadata(
                        record=record,
                        page=page.page,
                        block_type="page_text",
                        caption="",
                        bbox="",
                    ),
                )
                for index, chunk_doc in enumerate(self.splitter.split_documents([page_doc]), start=1):
                    chunk_doc.metadata["doc_id"] = self._block_doc_id(
                        record,
                        page.page,
                        "page_text",
                        index,
                        chunk_doc.page_content,
                    )
                    chunk_doc.metadata["formula_hint"] = int(bool(FORMULA_HINT_RE.search(chunk_doc.page_content)))
                    chunk_doc.metadata["mentioned_acronyms"] = "||".join(
                        self._extract_acronym_aliases(chunk_doc.page_content)
                    )
                    docs.append(chunk_doc)
            for block_index, block in enumerate(page.blocks, start=1):
                block_doc = self._structured_block_document(
                    record=record,
                    page=page.page,
                    block=block,
                    block_index=block_index,
                )
                if block_doc is not None:
                    docs.append(block_doc)
        return docs

    def _structured_block_document(
        self,
        *,
        record: PaperRecord,
        page: int,
        block: ExtractedBlock,
        block_index: int,
    ) -> Document | None:
        content = self._structured_block_content(block)
        if not content:
            return None
        metadata = self._block_metadata(
            record=record,
            page=page,
            block_type=block.block_type,
            caption=block.caption,
            bbox=self._serialize_bbox(block.bbox),
        )
        metadata["formula_hint"] = int(bool(FORMULA_HINT_RE.search(content)))
        metadata["mentioned_acronyms"] = "||".join(self._extract_acronym_aliases(content))
        metadata["doc_id"] = self._block_doc_id(record, page, block.block_type, block_index, content)
        return Document(page_content=content, metadata=metadata)

    @staticmethod
    def _structured_block_content(block: ExtractedBlock) -> str:
        parts: list[str] = []
        if block.block_type == "caption":
            parts.append(block.text.strip())
        elif block.block_type == "table":
            if block.caption:
                parts.append(block.caption.strip())
            parts.append(block.text.strip())
        elif block.block_type == "figure":
            if block.caption:
                parts.append(block.caption.strip())
            if block.text.strip():
                parts.append(block.text.strip())
        else:
            parts.append(block.text.strip())
        return "\n".join(part for part in parts if part).strip()

    def _generate_paper_summary(self, *, record: PaperRecord, pages: list[ExtractedPage]) -> str:
        seed_text = self._fallback_summary_from_pages(pages, limit=2800)
        if not seed_text:
            return ""
        summary = self.clients.invoke_text(
            system_prompt=(
                "你是论文 ingestion 摘要器。请基于论文标题、作者、标签和正文前几页，"
                "生成 120-200 字的中文摘要，重点覆盖：研究对象、方法、主要结论或指标。"
                "不要编造论文中没有出现的结果。"
            ),
            human_prompt=(
                f"title: {record.title}\n"
                f"authors: {', '.join(record.authors)}\n"
                f"year: {record.year}\n"
                f"tags: {' | '.join(record.tags)}\n"
                f"seed_text:\n{seed_text}"
            ),
            fallback=seed_text[:600],
        )
        return summary.strip()

    def _build_evidence_hints(self, pages: list[ExtractedPage]) -> str:
        hints: list[str] = []
        for page in pages[:6]:
            for block in page.blocks:
                if block.block_type == "caption" and block.text.strip():
                    hints.append(block.text.strip())
                elif block.block_type == "table" and block.caption.strip():
                    hints.append(block.caption.strip())
                elif block.block_type == "figure" and block.caption.strip():
                    hints.append(block.caption.strip())
            if page.text.strip():
                hints.append(page.text.strip()[:220])
            if len(hints) >= 6:
                break
        return "\n".join(f"- {item}" for item in hints[:6])

    @staticmethod
    def _fallback_summary_from_pages(pages: list[ExtractedPage], limit: int = 1800) -> str:
        parts: list[str] = []
        total = 0
        for page in pages[:3]:
            text = page.text.strip()
            if not text:
                continue
            remaining = max(0, limit - total)
            if remaining <= 0:
                break
            clipped = text[:remaining]
            parts.append(clipped)
            total += len(clipped)
        return "\n".join(parts).strip()

    @classmethod
    def _body_acronyms(cls, pages: list[ExtractedPage]) -> list[str]:
        aliases: list[str] = []
        for page in pages:
            if page.text.strip():
                aliases.extend(cls._extract_acronym_aliases(page.text))
            for block in page.blocks:
                text = "\n".join(
                    part
                    for part in [block.caption.strip(), block.text.strip()]
                    if part
                )
                if text:
                    aliases.extend(cls._extract_acronym_aliases(text))
        return cls._dedupe_aliases(aliases, max_items=80)

    @classmethod
    def _extract_acronym_aliases(cls, text: str) -> list[str]:
        raw = " ".join(str(text or "").split())
        if not raw:
            return []
        aliases: list[str] = []
        stopwords = {"and", "or", "the", "a", "an", "to", "for", "with", "via", "on", "in", "of", "from"}
        definition_pattern = re.compile(
            r"(?P<expansion>[A-Z][A-Za-z0-9][A-Za-z0-9\-/ ]{2,100}?)\s*[\(（]\s*(?P<acro>[A-Z][A-Z0-9\-]{1,9})\s*[\)）]"
        )
        for match in definition_pattern.finditer(raw):
            expansion = " ".join(str(match.group("expansion") or "").strip(" ,.;:").split())
            acronym = str(match.group("acro") or "").strip()
            if not acronym:
                continue
            if expansion and cls._acronym_matches_expansion(acronym=acronym, expansion=expansion, stopwords=stopwords):
                aliases.extend([acronym, expansion])
            else:
                aliases.append(acronym)
        cue_pattern = re.compile(
            r"\b(?P<acro>[A-Z][A-Z0-9\-]{1,9})\b\s+(?:loss|objective|algorithm|method|framework|benchmark|dataset|introduces|models|uses|maps|denotes|refers|aligns)\b",
            flags=re.IGNORECASE,
        )
        for match in cue_pattern.finditer(raw):
            aliases.append(str(match.group("acro") or "").upper())
        formula_subscript_pattern = re.compile(
            r"\bL\s*(?:_\s*\{?|\{\s*)(?:\\mathrm\{?)?\s*(?P<acro>[A-Z][A-Z0-9\-]{1,9})",
            flags=re.IGNORECASE,
        )
        compact_loss_pattern = re.compile(r"\bL(?P<acro>[A-Z][A-Z0-9\-]{1,9})\b")
        for match in [*formula_subscript_pattern.finditer(raw), *compact_loss_pattern.finditer(raw)]:
            acronym = str(match.group("acro") or "").strip("{}")
            if acronym:
                aliases.extend([acronym.upper(), f"L_{acronym.upper()}"])
        return cls._dedupe_aliases(aliases, max_items=24)

    @staticmethod
    def _acronym_matches_expansion(*, acronym: str, expansion: str, stopwords: set[str]) -> bool:
        expansion = re.sub(r"[-/]+", " ", expansion)
        words = re.findall(r"[A-Za-z][A-Za-z0-9]*", expansion)
        initials = "".join(word[0].upper() for word in words if word.lower() not in stopwords)
        compact_acronym = re.sub(r"[^A-Z0-9]", "", acronym.upper())
        return bool(compact_acronym and initials.endswith(compact_acronym))

    @staticmethod
    def _dedupe_aliases(items: list[str], *, max_items: int) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            alias = " ".join(str(item or "").strip().split())
            key = alias.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(alias)
            if len(deduped) >= max_items:
                break
        return deduped

    @staticmethod
    def _build_aliases(title: str) -> list[str]:
        clean = " ".join(str(title or "").strip().split())
        if not clean:
            return []
        aliases = [clean]
        english_words = re.findall(r"[A-Za-z][A-Za-z0-9\-']*", clean)
        if english_words:
            aliases.append("".join(word[0] for word in english_words).upper())
        short_title = re.sub(r"[:\-–].*$", "", clean).strip()
        if short_title and short_title not in aliases:
            aliases.append(short_title)
        dedup: list[str] = []
        seen: set[str] = set()
        for item in aliases:
            normalized = item.lower().strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                dedup.append(item)
        return dedup

    @staticmethod
    def _serialize_bbox(bbox: tuple[float, float, float, float] | None) -> str:
        if bbox is None:
            return ""
        return ",".join(f"{value:.2f}" for value in bbox)

    @staticmethod
    def _block_metadata(
        *,
        record: PaperRecord,
        page: int,
        block_type: str,
        caption: str,
        bbox: str,
    ) -> dict[str, Any]:
        return {
            "paper_id": record.attachment_key,
            "title": record.title,
            "authors": ", ".join(record.authors),
            "year": record.year,
            "tags": "||".join(record.tags),
            "item_type": record.item_type,
            "source_url": record.source_url,
            "website_title": record.website_title,
            "file_path": record.file_path,
            "page": int(page),
            "block_type": block_type,
            "caption": caption,
            "bbox": bbox,
            "attachment_key": record.attachment_key,
        }

    @staticmethod
    def _block_doc_id(record: PaperRecord, page: int, block_type: str, chunk_index: int, _text: str) -> str:
        raw = f"{record.attachment_key}|{page}|{block_type}|{chunk_index}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _persist_jsonl(path: Path, docs: list[Document]) -> None:
        lines = [
            json.dumps({"page_content": doc.page_content, "metadata": doc.metadata}, ensure_ascii=False)
            for doc in docs
        ]
        V4IngestionService._atomic_write_text(
            path,
            "\n".join(lines) + ("\n" if lines else ""),
        )

    @staticmethod
    def _persist_json(path: Path, payload: dict[str, Any]) -> None:
        V4IngestionService._atomic_write_text(
            path,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                tmp_name = f.name
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            Path(tmp_name).replace(path)
        except Exception:
            if tmp_name:
                try:
                    Path(tmp_name).unlink(missing_ok=True)
                except OSError:
                    pass
            raise
