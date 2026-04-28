from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.core.config import Settings
from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services.vector_index import CollectionVectorIndex

logger = logging.getLogger(__name__)


BOOK_ITEM_TYPES = {
    "book",
    "bookSection",
    "dictionaryEntry",
    "encyclopediaArticle",
    "magazineArticle",
    "newspaperArticle",
}
BOOKISH_TITLE_MARKERS = (
    "实战",
    "教程",
    "指南",
    "手册",
    "原理、应用",
    "原理与应用",
    "系统构建",
    "从入门到精通",
)
BOOKISH_CONTENT_MARKERS = ("本书", "本章", "第1章", "第 1 章", "章节", "读者")


class DualIndexRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._paper_docs: list[Document] = []
        self._block_docs: list[Document] = []
        self._paper_docs_by_id: dict[str, Document] = {}
        self._block_docs_by_id: dict[str, Document] = {}
        self._block_docs_by_paper_id: dict[str, list[Document]] = {}
        self._load_library_docs()
        self._paper_bm25: BM25Retriever | None = self._build_bm25(self._paper_docs, settings.paper_bm25_top_k)
        self._block_bm25: BM25Retriever | None = self._build_bm25(self._block_docs, settings.block_bm25_top_k)
        self._paper_dense = CollectionVectorIndex(settings, collection_name=settings.milvus_paper_collection)
        self._block_dense = CollectionVectorIndex(settings, collection_name=settings.milvus_block_collection)

    def refresh(self) -> None:
        self._load_library_docs()
        self._paper_bm25 = self._build_bm25(self._paper_docs, self.settings.paper_bm25_top_k)
        self._block_bm25 = self._build_bm25(self._block_docs, self.settings.block_bm25_top_k)

    def close(self) -> None:
        self._paper_dense.close()
        self._block_dense.close()

    def search_papers(self, *, query: str, contract: QueryContract, limit: int | None = None) -> list[CandidatePaper]:
        limit = limit or self.settings.paper_limit_default
        search_text = query.strip()
        target_terms = self._contract_target_terms(contract)
        target_text = " ".join(target_terms).strip()
        if target_text and target_text.lower() not in search_text.lower():
            search_text = f"{target_text} {search_text}".strip()
        weighted_docs: list[tuple[float, list[Document]]] = []
        anchors = self.title_anchor(target_terms)
        if anchors:
            weighted_docs.append((1.6, anchors))
        relation_anchors = self.relation_anchor_docs(contract)
        if relation_anchors:
            weighted_docs.append((1.3, relation_anchors))
        if self._paper_bm25 is not None:
            weighted_docs.append((0.9, self._paper_bm25.invoke(search_text)))
        dense_docs = [
            doc
            for doc in self._paper_dense.search_documents(search_text, limit=self.settings.paper_dense_top_k)
            if self._is_allowed_library_doc(doc)
        ]
        if dense_docs:
            weighted_docs.append((0.8, dense_docs))
        fused = self._rrf_fuse(weighted_docs)
        candidates: list[CandidatePaper] = []
        for rank, doc in enumerate(fused[: max(limit * 2, 8)], start=1):
            if not self._is_allowed_library_doc(doc):
                continue
            meta = dict(doc.metadata or {})
            meta.setdefault("paper_card_text", str(doc.page_content or ""))
            paper_id = str(meta.get("paper_id", "")).strip()
            if not paper_id:
                continue
            score = (1.0 / rank) + self._paper_match_boost(doc, contract)
            candidates.append(
                CandidatePaper(
                    paper_id=paper_id,
                    title=str(meta.get("title", "")),
                    year=str(meta.get("year", "")),
                    score=score,
                    match_reason="hybrid_rrf",
                    anchor_terms=target_terms,
                    doc_ids=[str(meta.get("doc_id", ""))] if meta.get("doc_id") else [],
                    metadata=meta,
                )
            )
        deduped = self._deduplicate_candidates(candidates)
        return self.screen_papers(deduped, contract=contract, limit=limit)

    def bm25_search(
        self,
        *,
        query: str,
        contract: QueryContract,
        scope: str = "auto",
        paper_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[EvidenceBlock]:
        query = str(query or "").strip()
        if not query:
            return []
        limit = max(1, min(int(limit or self.settings.evidence_limit_default), 50))
        docs: list[tuple[str, list[Document]]] = []
        normalized_scope = scope if scope in {"auto", "papers", "blocks"} else "auto"
        if normalized_scope in {"auto", "papers"} and self._paper_bm25 is not None:
            docs.append(("paper_bm25", self._paper_bm25.invoke(query)))
        if normalized_scope in {"auto", "blocks"} and self._block_bm25 is not None:
            docs.append(("block_bm25", self._filter_docs_by_paper_ids(self._block_bm25.invoke(query), paper_ids or [])))
        return self._rank_search_tool_documents(
            docs=docs,
            query=query,
            contract=contract,
            limit=limit,
        )

    def vector_search(
        self,
        *,
        query: str,
        contract: QueryContract,
        scope: str = "auto",
        paper_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[EvidenceBlock]:
        query = str(query or "").strip()
        if not query:
            return []
        limit = max(1, min(int(limit or self.settings.evidence_limit_default), 50))
        normalized_scope = scope if scope in {"auto", "papers", "blocks"} else "auto"
        docs: list[tuple[str, list[Document]]] = []
        if normalized_scope in {"auto", "papers"}:
            paper_docs = [
                doc
                for doc in self._paper_dense.search_documents(query, limit=min(limit, self.settings.paper_dense_top_k))
                if self._is_allowed_library_doc(doc)
            ]
            docs.append(("paper_dense", paper_docs))
        if normalized_scope in {"auto", "blocks"}:
            block_docs = self._block_dense.search_documents(query, limit=min(limit, self.settings.block_dense_top_k))
            docs.append(("block_dense", self._filter_docs_by_paper_ids(block_docs, paper_ids or [])))
        return self._rank_search_tool_documents(
            docs=docs,
            query=query,
            contract=contract,
            limit=limit,
        )

    def hybrid_search(
        self,
        *,
        query: str,
        contract: QueryContract,
        scope: str = "auto",
        paper_ids: list[str] | None = None,
        limit: int | None = None,
        alpha: float = 0.5,
    ) -> list[EvidenceBlock]:
        query = str(query or "").strip()
        if not query:
            return []
        limit = max(1, min(int(limit or self.settings.evidence_limit_default), 50))
        try:
            dense_weight = max(0.0, min(1.0, float(alpha)))
        except (TypeError, ValueError):
            dense_weight = 0.5
        sparse_weight = 1.0 - dense_weight
        bm25 = self.bm25_search(
            query=query,
            contract=contract,
            scope=scope,
            paper_ids=paper_ids,
            limit=limit,
        )
        dense = self.vector_search(
            query=query,
            contract=contract,
            scope=scope,
            paper_ids=paper_ids,
            limit=limit,
        )
        weighted_docs = [
            (sparse_weight or 0.01, [self._evidence_to_document(item) for item in bm25]),
            (dense_weight or 0.01, [self._evidence_to_document(item) for item in dense]),
        ]
        fused_docs = self._rrf_fuse(weighted_docs)
        return self._rank_search_tool_documents(
            docs=[("hybrid_search", fused_docs)],
            query=query,
            contract=contract,
            limit=limit,
        )

    def rerank_evidence(
        self,
        *,
        query: str,
        evidence: list[EvidenceBlock],
        top_k: int | None = None,
        focus: list[str] | None = None,
    ) -> list[EvidenceBlock]:
        if not evidence:
            return []
        limit = max(1, min(int(top_k or len(evidence)), 50))
        tokens = self._query_tokens(query, extra=list(focus or []))
        reranked: list[EvidenceBlock] = []
        for item in evidence:
            haystack = f"{item.title}\n{item.caption}\n{item.snippet}"
            lexical = self._lexical_score(haystack, tokens)
            focus_bonus = 0.0
            for target in list(focus or []):
                if target and self._matches_target(haystack, target):
                    focus_bonus += 1.5
            score = float(item.score or 0.0) + lexical + focus_bonus
            metadata = dict(item.metadata or {})
            metadata["rerank_score"] = score
            reranked.append(item.model_copy(update={"score": score, "metadata": metadata}))
        reranked.sort(key=lambda item: (-item.score, item.page, item.doc_id))
        return reranked[:limit]

    def read_pdf_pages(
        self,
        *,
        paper_id: str,
        page_from: int,
        page_to: int | None = None,
        max_chars: int = 4000,
    ) -> list[EvidenceBlock]:
        paper_id = str(paper_id or "").strip()
        if not paper_id:
            return []
        start = max(1, int(page_from or 1))
        end = max(start, int(page_to or start))
        max_chars = max(200, min(int(max_chars or 4000), 20000))
        used_chars = 0
        blocks: list[EvidenceBlock] = []
        block_order = {"page_text": 0, "table": 1, "caption": 2, "figure": 3}
        docs = sorted(
            self._block_docs_by_paper_id.get(paper_id, []),
            key=lambda doc: (
                int((doc.metadata or {}).get("page", 0) or 0),
                block_order.get(str((doc.metadata or {}).get("block_type", "")), 9),
                str((doc.metadata or {}).get("doc_id", "")),
            ),
        )
        for doc in docs:
            meta = dict(doc.metadata or {})
            page = int(meta.get("page", 0) or 0)
            if page < start or page > end:
                continue
            remaining = max_chars - used_chars
            if remaining <= 0:
                break
            text = str(doc.page_content or "").strip()
            if not text:
                continue
            snippet = text[:remaining]
            used_chars += len(snippet)
            meta["read_source"] = "read_pdf_page"
            blocks.append(
                EvidenceBlock(
                    doc_id=str(meta.get("doc_id", "")),
                    paper_id=str(meta.get("paper_id", "")),
                    title=str(meta.get("title", "")),
                    file_path=str(meta.get("file_path", "")),
                    page=page,
                    block_type=str(meta.get("block_type", "")),
                    caption=str(meta.get("caption", "")),
                    bbox=str(meta.get("bbox", "")),
                    snippet=snippet,
                    score=1.0,
                    metadata=meta,
                )
            )
        return blocks

    def grep_corpus(
        self,
        *,
        pattern: str,
        scope: str = "auto",
        paper_ids: list[str] | None = None,
        max_hits: int = 20,
    ) -> list[EvidenceBlock]:
        pattern = str(pattern or "").strip()
        if not pattern or len(pattern) > 240:
            return []
        try:
            regex = re.compile(pattern, flags=re.IGNORECASE)
        except re.error:
            return []
        max_hits = max(1, min(int(max_hits or 20), 100))
        normalized_scope = scope if scope in {"auto", "papers", "blocks"} else "auto"
        docs: list[Document] = []
        if normalized_scope in {"auto", "papers"}:
            docs.extend(self._paper_docs)
        if normalized_scope in {"auto", "blocks"}:
            docs.extend(self._filter_docs_by_paper_ids(self._block_docs, paper_ids or []))
        hits: list[EvidenceBlock] = []
        for doc in docs:
            if len(hits) >= max_hits:
                break
            meta = dict(doc.metadata or {})
            text = str(doc.page_content or "")
            match = regex.search(text)
            if match is None:
                continue
            start = max(0, match.start() - 220)
            end = min(len(text), match.end() + 420)
            snippet = text[start:end].strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            meta["grep_pattern"] = pattern
            meta["search_source"] = "grep_corpus"
            hits.append(
                EvidenceBlock(
                    doc_id=str(meta.get("doc_id", "")),
                    paper_id=str(meta.get("paper_id", "")),
                    title=str(meta.get("title", "")),
                    file_path=str(meta.get("file_path", "")),
                    page=int(meta.get("page", 0) or 0),
                    block_type=str(meta.get("block_type", "") or "paper_card"),
                    caption=str(meta.get("caption", "")),
                    bbox=str(meta.get("bbox", "")),
                    snippet=snippet,
                    score=1.0,
                    metadata=meta,
                )
            )
        return hits

    def search_concept_evidence(
        self,
        *,
        query: str,
        contract: QueryContract,
        paper_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[EvidenceBlock]:
        limit = limit or self.settings.evidence_limit_default
        target_terms = self._contract_target_terms(contract)
        tokens = self._query_tokens(query, extra=target_terms)
        candidates: list[EvidenceBlock] = []
        has_definition_like = False
        block_docs = (
            self._block_documents_for_paper_ids(paper_ids or [])
            if paper_ids
            else self._block_docs
        )
        for doc in block_docs:
            meta = dict(doc.metadata or {})
            text = str(doc.page_content or "")
            title = str(meta.get("title", ""))
            snippet = self._focused_snippet(text=text, targets=target_terms, query=query)
            definition_score = self._definition_like_score(text=text, targets=target_terms)
            score = self._concept_score(text=text, title=title, tokens=tokens, target_terms=target_terms)
            if score <= 0:
                continue
            if definition_score > 0:
                has_definition_like = True
                score += definition_score * 2.0
            meta["definition_score"] = definition_score
            candidates.append(
                EvidenceBlock(
                    doc_id=str(meta.get("doc_id", "")),
                    paper_id=str(meta.get("paper_id", "")),
                    title=title,
                    file_path=str(meta.get("file_path", "")),
                    page=int(meta.get("page", 0) or 0),
                    block_type=str(meta.get("block_type", "")),
                    caption=str(meta.get("caption", "")),
                    bbox=str(meta.get("bbox", "")),
                    snippet=snippet,
                    score=score,
                    metadata=meta,
                )
            )
        allowed_paper_ids = set(paper_ids or [])
        paper_docs = (
            [
                doc
                for doc in self._paper_docs
                if str((doc.metadata or {}).get("paper_id", "")) in allowed_paper_ids
            ]
            if allowed_paper_ids
            else self._paper_docs
        )
        for doc in paper_docs:
            meta = dict(doc.metadata or {})
            text = str(doc.page_content or "")
            title = str(meta.get("title", ""))
            snippet = self._focused_snippet(text=text, targets=target_terms, query=query)
            definition_score = self._definition_like_score(text=text, targets=target_terms)
            score = self._concept_score(text=text, title=title, tokens=tokens, target_terms=target_terms) + 0.2
            if score <= 0:
                continue
            if definition_score > 0:
                has_definition_like = True
                score += definition_score * 2.0
            meta["definition_score"] = definition_score
            candidates.append(
                EvidenceBlock(
                    doc_id=str(meta.get("doc_id", "")),
                    paper_id=str(meta.get("paper_id", "")),
                    title=title,
                    file_path=str(meta.get("file_path", "")),
                    page=0,
                    block_type="paper_card",
                    caption="",
                    bbox="",
                    snippet=snippet,
                    score=score,
                    metadata=meta,
                )
            )
        block_order = {"page_text": 0, "table": 1, "caption": 1, "figure": 2, "paper_card": 3}
        candidates.sort(key=lambda item: (-item.score, block_order.get(item.block_type, 4), item.page, item.doc_id))
        deduped: list[EvidenceBlock] = []
        seen: set[str] = set()
        for item in candidates:
            if item.doc_id in seen:
                continue
            seen.add(item.doc_id)
            deduped.append(item)
        if has_definition_like:
            definition_first = [item for item in deduped if float(item.metadata.get("definition_score", 0) or 0) > 0]
            if definition_first:
                deduped = definition_first
        return deduped[: max(1, limit)]

    def search_entity_evidence(
        self,
        *,
        query: str,
        contract: QueryContract,
        limit: int | None = None,
    ) -> list[EvidenceBlock]:
        limit = limit or self.settings.evidence_limit_default
        target_terms = self._contract_target_terms(contract)
        tokens = self._query_tokens(query, extra=target_terms)
        requested_fields = {self._normalize_text(item) for item in contract.requested_fields if item}
        formula_requested = bool(requested_fields & {"formula", "objective", "variable_explanation"})
        detail_requested = bool(
            requested_fields
            & {
                "mechanism",
                "workflow",
                "objective",
                "reward_signal",
                "training_signal",
                "formula",
                "variable_explanation",
            }
        )
        candidates: list[EvidenceBlock] = []
        has_definition_like = False
        has_mechanism_like = False
        for docs, paper_card_bonus in ((self._block_docs, 0.0), (self._paper_docs, 0.2)):
            for doc in docs:
                meta = dict(doc.metadata or {})
                text = str(doc.page_content or "")
                title = str(meta.get("title", ""))
                snippet = self._focused_snippet(text=text, targets=target_terms, query=query)
                if (
                    not formula_requested
                    and self.settings.retrieval_filter_formula_heavy_non_formula
                    and self._looks_formula_heavy(snippet)
                ):
                    continue
                entity_score = self._entity_score(text=text, title=title, tokens=tokens, target_terms=target_terms)
                if entity_score <= 0:
                    continue
                definition_score = self._definition_like_score(text=text, targets=target_terms)
                mechanism_score = self._mechanism_like_score(
                    text=text,
                    title=title,
                    targets=target_terms,
                    requested_fields=requested_fields,
                )
                application_score = self._application_like_score(text=text, targets=target_terms)
                score = entity_score + paper_card_bonus
                if definition_score > 0:
                    has_definition_like = True
                    score += definition_score * (0.8 if detail_requested else 1.9)
                if mechanism_score > 0:
                    has_mechanism_like = True
                    score += mechanism_score * (2.0 if detail_requested else 1.0)
                if application_score > 0:
                    score += application_score * (0.7 if detail_requested else 1.0)
                if detail_requested and int(meta.get("formula_hint", 0) or 0):
                    score += 0.8
                meta["definition_score"] = definition_score
                meta["mechanism_score"] = mechanism_score
                meta["application_score"] = application_score
                candidates.append(
                    EvidenceBlock(
                        doc_id=str(meta.get("doc_id", "")),
                        paper_id=str(meta.get("paper_id", "")),
                        title=title,
                        file_path=str(meta.get("file_path", "")),
                        page=int(meta.get("page", 0) or 0),
                        block_type=str(meta.get("block_type", "paper_card" if docs is self._paper_docs else "")),
                        caption=str(meta.get("caption", "")),
                        bbox=str(meta.get("bbox", "")),
                        snippet=snippet,
                        score=score,
                        metadata=meta,
                    )
                )
        block_order = {"page_text": 0, "table": 1, "caption": 1, "figure": 2, "paper_card": 3}
        candidates.sort(key=lambda item: (-item.score, block_order.get(item.block_type, 4), item.page, item.doc_id))
        deduped: list[EvidenceBlock] = []
        seen: set[str] = set()
        for item in candidates:
            if item.doc_id in seen:
                continue
            seen.add(item.doc_id)
            deduped.append(item)
        if detail_requested and has_mechanism_like:
            detail_first = [item for item in deduped if float(item.metadata.get("mechanism_score", 0) or 0) > 0]
            remainder = [item for item in deduped if item not in detail_first]
            if detail_first:
                deduped = [*detail_first, *remainder]
        elif has_definition_like:
            definition_first = [item for item in deduped if float(item.metadata.get("definition_score", 0) or 0) > 0]
            remainder = [item for item in deduped if item not in definition_first]
            if definition_first:
                deduped = [*definition_first, *remainder]
        return deduped[: max(1, limit)]

    def title_anchor(self, targets: list[str]) -> list[Document]:
        normalized_targets = [str(item).strip() for item in targets if item]
        if not normalized_targets:
            return []
        anchored: list[Document] = []
        for doc in self._paper_docs:
            meta = dict(doc.metadata or {})
            title = str(meta.get("title", ""))
            aliases = [alias for alias in str(meta.get("aliases", "")).split("||") if alias]
            body_acronyms = [alias for alias in str(meta.get("body_acronyms", "")).split("||") if alias]
            haystack = [title, *aliases, *body_acronyms]
            if any(target and any(self._matches_target(candidate, target) for candidate in haystack if candidate) for target in normalized_targets):
                anchored.append(doc)
        return anchored

    @staticmethod
    def _focused_snippet(text: str, *, targets: list[str], query: str, max_chars: int = 420) -> str:
        raw = " ".join(str(text or "").split())
        if not raw:
            return ""
        lowered = raw.lower()
        anchor_terms = [str(item).strip() for item in targets if str(item).strip()]
        if not anchor_terms:
            anchor_terms = [str(token).strip() for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", query)[:3] if token.strip()]
        anchors: list[tuple[int, int]] = []
        for term in anchor_terms:
            index = lowered.find(term.lower())
            if index >= 0:
                anchors.append((0, index))
        cue_terms = [
            "definition",
            "algorithm",
            "method",
            "objective",
            "reward",
            "advantage",
            "critic",
            "workflow",
            "benchmark",
            "dataset",
            "定义",
            "方法",
            "算法",
            "目标",
            "奖励",
            "流程",
        ]
        for term in cue_terms:
            index = lowered.find(term)
            if index >= 0:
                anchors.append((1, index))
        if not anchors:
            return raw[:max_chars]
        _, index = min(anchors, key=lambda item: (item[0], item[1]))
        start = max(0, index - 80)
        end = min(len(raw), start + max_chars)
        snippet = raw[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(raw):
            snippet = snippet.rstrip() + "..."
        return snippet

    @staticmethod
    def _looks_formula_heavy(text: str) -> bool:
        compact = " ".join(str(text or "").split())
        if not compact:
            return False
        weird_math_chars = "∑𝜋𝜃𝑜𝑡𝑞𝐴ˆβϵμ"
        if sum(1 for ch in compact if ch in weird_math_chars) >= 2:
            return True
        if compact.count("|") >= 2:
            return True
        if re.search(r"\([0-9]{1,2}\)\s*$", compact):
            return True
        letters = sum(1 for ch in compact if ch.isalpha())
        digits = sum(1 for ch in compact if ch.isdigit())
        symbols = sum(1 for ch in compact if not ch.isalnum() and ch not in " .,;:!?()[]{}-_/")
        return letters < 24 and (symbols > max(4, letters) or digits > letters)

    def relation_anchor_docs(self, contract: QueryContract) -> list[Document]:
        return []

    def screen_papers(
        self,
        candidates: list[CandidatePaper],
        *,
        contract: QueryContract,
        limit: int,
    ) -> list[CandidatePaper]:
        screened: list[CandidatePaper] = []
        normalized_targets = [self._normalize_text(item) for item in self._contract_target_terms(contract) if item]
        goals = self._contract_goals(contract, query=contract.clean_query)
        for candidate in candidates:
            score = candidate.score
            title = self._normalize_text(candidate.title)
            content = self._normalize_text(
                str(
                    candidate.metadata.get("paper_card_text")
                    or candidate.metadata.get("generated_summary")
                    or candidate.metadata.get("abstract_note")
                    or ""
                )
            )
            if normalized_targets and any(self._matches_target(candidate.title, term) for term in normalized_targets):
                score += 0.8
            if normalized_targets and any(self._matches_target(content, term) for term in normalized_targets):
                score += 0.4
            year = self._safe_year(candidate.year)
            if goals & {"origin", "paper_title", "year"} and year:
                score += max(0.0, (2100 - year) / 1000.0)
                score += self._origin_signal_score(candidate, targets=normalized_targets)
            if goals & {"followup_papers", "candidate_relationship", "strict_followup"} and year:
                score += min(0.8, year / 10000.0)
            screened.append(candidate.model_copy(update={"score": score}))
        screened.sort(key=lambda item: (-item.score, self._safe_year(item.year), item.title))
        return screened[: max(1, limit)]

    def expand_evidence(
        self,
        *,
        paper_ids: list[str],
        query: str,
        contract: QueryContract,
        limit: int | None = None,
    ) -> list[EvidenceBlock]:
        if not paper_ids:
            return []
        limit = limit or self.settings.evidence_limit_default
        candidates: list[EvidenceBlock] = []
        target_terms = self._contract_target_terms(contract)
        tokens = self._query_tokens(query, extra=target_terms)
        goals = self._contract_goals(contract, query=query)
        for doc in self._block_documents_for_paper_ids(paper_ids):
            meta = dict(doc.metadata or {})
            text = str(doc.page_content or "")
            score = self._lexical_score(text, tokens)
            block_type = str(meta.get("block_type", ""))
            if block_type in contract.required_modalities:
                score += 1.2
            if "formula" in goals and int(meta.get("formula_hint", 0) or 0):
                score += 1.6
            if "formula" in goals:
                score += self._formula_snippet_score(text, targets=contract.targets)
            mentioned = str(meta.get("mentioned_acronyms", "") or "")
            if target_terms and mentioned:
                for target in target_terms:
                    if self._matches_target(mentioned, target):
                        score += 1.8
                        break
            if "figure_conclusion" in goals and block_type in {"figure", "caption"}:
                score += 1.5
            if "figure_conclusion" in goals and block_type == "page_text":
                lowered = text.lower()
                if "figure " in lowered or "fig." in lowered or "图" in text:
                    score += 1.6
                if any(token in lowered for token in ["figure 1", "fig. 1", "fig 1"]) or "图1" in text:
                    score += 2.2
            if goals & {"metric_value", "setting", "summary", "results", "key_findings"} and block_type in {"table", "caption"}:
                score += 1.4 + self._metric_signal_score(text)
            if goals & {
                "origin",
                "paper_title",
                "year",
                "entity_type",
                "definition",
                "role_in_context",
                "followup_papers",
                "candidate_relationship",
            } and block_type == "page_text":
                score += 0.3
            candidates.append(
                EvidenceBlock(
                    doc_id=str(meta.get("doc_id", "")),
                    paper_id=str(meta.get("paper_id", "")),
                    title=str(meta.get("title", "")),
                    file_path=str(meta.get("file_path", "")),
                    page=int(meta.get("page", 0) or 0),
                    block_type=block_type,
                    caption=str(meta.get("caption", "")),
                    bbox=str(meta.get("bbox", "")),
                    snippet=text[:900],
                    score=score,
                    metadata=meta,
                )
            )
        if self.settings.openai_api_key and len(candidates) < limit:
            dense_docs = self._block_dense.search_documents(query, limit=self.settings.block_dense_top_k)
            dense_ids = {str(doc.metadata.get("doc_id", "")) for doc in dense_docs}
            for doc in self._block_documents_for_paper_ids(paper_ids):
                meta = dict(doc.metadata or {})
                if str(meta.get("doc_id", "")) not in dense_ids:
                    continue
                if any(item.doc_id == str(meta.get("doc_id", "")) for item in candidates):
                    continue
                candidates.append(
                    EvidenceBlock(
                        doc_id=str(meta.get("doc_id", "")),
                        paper_id=str(meta.get("paper_id", "")),
                        title=str(meta.get("title", "")),
                        file_path=str(meta.get("file_path", "")),
                        page=int(meta.get("page", 0) or 0),
                        block_type=str(meta.get("block_type", "")),
                        caption=str(meta.get("caption", "")),
                        bbox=str(meta.get("bbox", "")),
                        snippet=str(doc.page_content or "")[:900],
                        score=0.6,
                        metadata=meta,
                    )
                )
        candidates.sort(key=lambda item: (-item.score, item.page, item.doc_id))
        deduped: list[EvidenceBlock] = []
        seen: set[str] = set()
        for item in candidates:
            if item.doc_id in seen:
                continue
            seen.add(item.doc_id)
            deduped.append(item)
        return deduped[: max(1, limit)]

    @staticmethod
    def _contract_target_terms(contract: QueryContract) -> list[str]:
        terms: list[str] = []
        for item in list(contract.targets or []):
            text = str(item or "").strip()
            if text:
                terms.append(text)
        terms.extend(DualIndexRetriever._contract_target_alias_terms(contract))
        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            key = term.lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(term)
        return deduped

    @staticmethod
    def _contract_target_alias_terms(contract: QueryContract) -> list[str]:
        terms: list[str] = []
        for note in list(contract.notes or []):
            text = str(note or "")
            if not text.startswith("target_alias="):
                continue
            alias = text.split("=", 1)[1].strip()
            if alias:
                terms.append(alias)
        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            key = term.lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(term)
        return deduped

    @staticmethod
    def _contract_goals(contract: QueryContract, *, query: str = "") -> set[str]:
        goals: set[str] = set()
        for value in [
            *list(getattr(contract, "answer_slots", []) or []),
            *list(contract.requested_fields or []),
            *[
                str(note).split("=", 1)[1]
                for note in contract.notes
                if str(note).startswith("answer_slot=") and "=" in str(note)
            ],
        ]:
            goals.update(DualIndexRetriever._normalize_contract_goal(value))
        lowered = " ".join(str(query or contract.clean_query or "").lower().split())
        if not goals or goals <= {"answer", "general_answer"}:
            if any(token in lowered for token in ["最早", "最先", "谁提出", "提出的", "origin", "first proposed"]):
                goals.update({"origin", "paper_title", "year"})
            if any(token in lowered for token in ["公式", "损失函数", "objective", "loss", "gradient", "梯度"]):
                goals.add("formula")
            if any(token in lowered for token in ["后续", "followup", "follow-up", "successor"]):
                goals.add("followup_papers")
            if any(token in lowered for token in ["figure", "fig.", "图", "caption"]):
                goals.add("figure_conclusion")
            if any(token in lowered for token in ["结果", "实验", "核心结论", "贡献", "summary", "result"]):
                goals.update({"summary", "results"})
            if any(token in lowered for token in ["多少", "数值", "准确率", "得分", "score", "accuracy", "metric", "win rate"]):
                goals.add("metric_value")
            if any(token in lowered for token in ["推荐", "值得", "入门", "recommend"]):
                goals.add("recommended_papers")
            if contract.targets and (
                any(token in str(query or contract.clean_query or "") for token in ["是什么", "什么意思", "定义"])
                or any(token in lowered for token in ["what is", "what are"])
            ):
                goals.update({"definition", "entity_type", "mechanism"})
        if "figure" in contract.required_modalities:
            goals.add("figure_conclusion")
        if "table" in contract.required_modalities and any(
            token in lowered for token in ["多少", "数值", "准确率", "得分", "score", "accuracy", "metric", "win rate"]
        ):
            goals.add("metric_value")
        return goals or {"answer"}

    @staticmethod
    def _normalize_contract_goal(value: str) -> set[str]:
        key = "_".join(str(value or "").strip().lower().replace("-", "_").split())
        aliases = {
            "general_answer": {"answer"},
            "origin": {"origin", "paper_title", "year"},
            "formula": {"formula"},
            "paper_summary": {"summary", "results"},
            "metric_value": {"metric_value", "setting"},
            "figure": {"figure_conclusion", "caption"},
            "paper_recommendation": {"recommended_papers"},
            "followup_research": {"followup_papers", "candidate_relationship"},
            "entity_definition": {"entity_type", "definition", "mechanism", "role_in_context"},
            "concept_definition": {"definition", "mechanism", "examples"},
            "topology_discovery": {"relevant_papers", "topology_types"},
            "topology_recommendation": {"best_topology", "langgraph_recommendation"},
            "training_component": {"mechanism", "reward_model_requirement"},
        }
        return set(aliases.get(key, {key} if key else set()))

    def paper_doc_by_id(self, paper_id: str) -> Document | None:
        return self._paper_docs_by_id.get(paper_id)

    def paper_documents(self) -> list[Document]:
        return list(self._paper_docs)

    def block_documents_for_paper(self, paper_id: str, *, limit: int = 8) -> list[Document]:
        docs = self._block_docs_by_paper_id.get(paper_id, [])
        return list(docs[: max(0, limit)])

    def canonicalize_target(self, target: str) -> str:
        raw = str(target or "").strip()
        normalized_target = self._normalize_entity_text(raw)
        if not normalized_target:
            return raw
        best_text = raw
        best_score = 0
        for doc in self._paper_docs:
            meta = dict(doc.metadata or {})
            for candidate in self._paper_candidate_names(meta):
                normalized_candidate = self._normalize_entity_text(candidate)
                if not normalized_candidate:
                    continue
                score = 0
                if normalized_candidate == normalized_target:
                    score = 100
                elif normalized_candidate.startswith(normalized_target) and len(normalized_target) >= 4:
                    score = 80
                elif normalized_target.startswith(normalized_candidate) and len(normalized_candidate) >= 4:
                    score = 70
                elif normalized_target in normalized_candidate and len(normalized_target) >= 6:
                    score = 60
                if score <= 0:
                    continue
                if score > best_score or (score == best_score and len(candidate) < len(best_text)):
                    best_text = candidate
                    best_score = score
        return best_text if best_score >= 70 else raw

    def canonicalize_targets(self, targets: list[str]) -> list[str]:
        normalized: list[str] = []
        for target in targets:
            canonical = self.canonicalize_target(target)
            if canonical and canonical not in normalized:
                normalized.append(canonical)
        return normalized

    def block_doc_by_id(self, doc_id: str) -> Document | None:
        return self._block_docs_by_id.get(doc_id)

    def _block_documents_for_paper_ids(self, paper_ids: list[str]) -> list[Document]:
        ordered_ids = list(dict.fromkeys(str(item) for item in paper_ids if item))
        docs: list[Document] = []
        for paper_id in ordered_ids:
            docs.extend(self._block_docs_by_paper_id.get(paper_id, []))
        return docs

    @staticmethod
    def _filter_docs_by_paper_ids(docs: list[Document], paper_ids: list[str]) -> list[Document]:
        allowed = {str(item).strip() for item in paper_ids if str(item).strip()}
        if not allowed:
            return docs
        return [doc for doc in docs if str((doc.metadata or {}).get("paper_id", "")).strip() in allowed]

    def _rank_search_tool_documents(
        self,
        *,
        docs: list[tuple[str, list[Document]]],
        query: str,
        contract: QueryContract,
        limit: int,
    ) -> list[EvidenceBlock]:
        target_terms = self._contract_target_terms(contract)
        tokens = self._query_tokens(query, extra=target_terms)
        candidates: list[EvidenceBlock] = []
        for source, source_docs in docs:
            for rank, doc in enumerate(source_docs, start=1):
                if not self._is_allowed_library_doc(doc):
                    continue
                evidence = self._evidence_from_document(
                    doc=doc,
                    query=query,
                    target_terms=target_terms,
                    tokens=tokens,
                    score_seed=1.0 / rank,
                    source=source,
                )
                if evidence is not None:
                    candidates.append(evidence)
        candidates.sort(key=lambda item: (-item.score, item.page, item.doc_id))
        deduped: list[EvidenceBlock] = []
        seen: set[str] = set()
        for item in candidates:
            key = item.doc_id or f"{item.paper_id}:{item.page}:{item.snippet[:80]}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[: max(1, limit)]

    def _evidence_from_document(
        self,
        *,
        doc: Document,
        query: str,
        target_terms: list[str],
        tokens: list[str],
        score_seed: float,
        source: str,
    ) -> EvidenceBlock | None:
        meta = dict(doc.metadata or {})
        text = str(doc.page_content or "")
        if not text:
            return None
        title = str(meta.get("title", ""))
        block_type = str(meta.get("block_type", "") or "paper_card")
        snippet = self._focused_snippet(text=text, targets=target_terms, query=query, max_chars=700)
        score = score_seed + self._lexical_score(self._normalize_text(f"{title}\n{text}"), tokens) * 0.2
        if target_terms and any(target and self._matches_target(f"{title}\n{text}", target) for target in target_terms):
            score += 1.0
        if "dense_score" in meta:
            try:
                score += max(0.0, float(meta.get("dense_score", 0.0) or 0.0)) * 0.05
            except (TypeError, ValueError):
                pass
        meta["search_source"] = source
        return EvidenceBlock(
            doc_id=str(meta.get("doc_id", "")) or f"{source}:{self._doc_key(doc)}",
            paper_id=str(meta.get("paper_id", "")),
            title=title,
            file_path=str(meta.get("file_path", "")),
            page=int(meta.get("page", 0) or 0),
            block_type=block_type,
            caption=str(meta.get("caption", "")),
            bbox=str(meta.get("bbox", "")),
            snippet=snippet,
            score=score,
            metadata=meta,
        )

    @staticmethod
    def _evidence_to_document(item: EvidenceBlock) -> Document:
        metadata = dict(item.metadata or {})
        metadata.setdefault("doc_id", item.doc_id)
        metadata.setdefault("paper_id", item.paper_id)
        metadata.setdefault("title", item.title)
        metadata.setdefault("page", item.page)
        metadata.setdefault("block_type", item.block_type)
        metadata.setdefault("file_path", item.file_path)
        metadata.setdefault("caption", item.caption)
        metadata.setdefault("bbox", item.bbox)
        return Document(page_content=item.snippet, metadata=metadata)

    def _rebuild_lookup_indexes(self) -> None:
        self._paper_docs_by_id = {}
        self._block_docs_by_id = {}
        self._block_docs_by_paper_id = {}
        for doc in self._paper_docs:
            paper_id = str((doc.metadata or {}).get("paper_id", "")).strip()
            if paper_id:
                self._paper_docs_by_id[paper_id] = doc
        for doc in self._block_docs:
            meta = doc.metadata or {}
            doc_id = str(meta.get("doc_id", "")).strip()
            paper_id = str(meta.get("paper_id", "")).strip()
            if doc_id:
                self._block_docs_by_id[doc_id] = doc
            if paper_id:
                self._block_docs_by_paper_id.setdefault(paper_id, []).append(doc)

    def _load_library_docs(self) -> None:
        raw_paper_docs = self._load_docs(self.settings.paper_store_path)
        raw_block_docs = self._load_docs(self.settings.block_store_path)
        self._paper_docs = [doc for doc in raw_paper_docs if self._is_allowed_library_doc(doc)]
        allowed_paper_ids = {
            str((doc.metadata or {}).get("paper_id", "")).strip()
            for doc in self._paper_docs
            if str((doc.metadata or {}).get("paper_id", "")).strip()
        }
        self._block_docs = [
            doc
            for doc in raw_block_docs
            if self._is_allowed_library_doc(doc)
            and (
                not allowed_paper_ids
                or str((doc.metadata or {}).get("paper_id", "")).strip() in allowed_paper_ids
            )
        ]
        self._rebuild_lookup_indexes()

    def _is_allowed_library_doc(self, doc: Document) -> bool:
        meta = doc.metadata or {}
        tags = {
            item.strip().lower()
            for item in str(meta.get("tags", "") or "").replace("/", "||").split("||")
            if item.strip()
        }
        excluded_tags = {item.strip().lower() for item in self.settings.ingestion_excluded_tags if item.strip()}
        if tags & excluded_tags:
            return False

        item_type = str(meta.get("item_type", "") or "").strip()
        if item_type in BOOK_ITEM_TYPES:
            return False
        if item_type and item_type not in set(self.settings.ingestion_allowed_item_types):
            source_url = str(meta.get("source_url", "") or "")
            if not any(host and host in source_url for host in self.settings.ingestion_academic_web_hosts):
                return False

        title = str(meta.get("title", "") or "")
        authors = str(meta.get("authors", "") or "").strip()
        year = str(meta.get("year", "") or "").strip()
        content = str(doc.page_content or "")
        sparse_bibliography = not authors and not year
        if sparse_bibliography and self._looks_like_book_doc(title=title, content=content):
            logger.info("Filtered non-paper library document from retrieval: %s", title)
            return False
        return True

    @staticmethod
    def _looks_like_book_doc(*, title: str, content: str) -> bool:
        title_text = str(title or "")
        content_text = str(content or "")
        if any(marker in title_text for marker in BOOKISH_TITLE_MARKERS):
            return True
        marker_hits = sum(1 for marker in BOOKISH_CONTENT_MARKERS if marker in content_text[:2500])
        return marker_hits >= 2

    @staticmethod
    def _load_docs(path: Any) -> list[Document]:
        doc_path = path if hasattr(path, "exists") else None
        if doc_path is None or not doc_path.exists():
            return []
        docs: list[Document] = []
        with doc_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                payload = json.loads(line)
                docs.append(Document(page_content=payload["page_content"], metadata=payload["metadata"]))
        return docs

    @staticmethod
    def _build_bm25(docs: list[Document], k: int) -> BM25Retriever | None:
        if not docs:
            return None
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = max(1, k)
        return retriever

    @staticmethod
    def _rrf_fuse(weighted_docs: list[tuple[float, list[Document]]], rrf_k: int = 60) -> list[Document]:
        merged: dict[str, tuple[Document, float]] = {}
        for weight, docs in weighted_docs:
            for rank, doc in enumerate(docs, start=1):
                key = DualIndexRetriever._doc_key(doc)
                prev_doc, prev_score = merged.get(key, (doc, 0.0))
                merged[key] = (prev_doc, prev_score + (weight / (rrf_k + rank)))
        ranked = sorted(merged.values(), key=lambda item: item[1], reverse=True)
        return [item[0] for item in ranked]

    @staticmethod
    def _doc_key(doc: Document) -> str:
        meta = doc.metadata or {}
        return f"{meta.get('doc_id','')}|{meta.get('paper_id','')}"

    @staticmethod
    def _deduplicate_candidates(candidates: list[CandidatePaper]) -> list[CandidatePaper]:
        deduped: dict[str, CandidatePaper] = {}
        for item in candidates:
            prev = deduped.get(item.paper_id)
            if prev is None or item.score > prev.score:
                deduped[item.paper_id] = item
        ranked = sorted(deduped.values(), key=lambda item: (-item.score, item.title))
        return ranked

    def _paper_match_boost(self, doc: Document, contract: QueryContract) -> float:
        meta = dict(doc.metadata or {})
        title = self._normalize_text(str(meta.get("title", "")))
        content = self._normalize_text(str(doc.page_content or ""))
        body_acronyms = str(meta.get("body_acronyms", "") or "")
        score = 0.0
        for target in self._contract_target_terms(contract):
            normalized = str(target).strip()
            if normalized and self._matches_target(title, normalized):
                score += 0.6
            if normalized and self._matches_target(content, normalized):
                score += 0.2
            if normalized and self._matches_target(body_acronyms, normalized):
                score += 0.9
            score += self._configured_paper_boost(target=normalized, title=title, content=content)
        for alias in self._contract_target_alias_terms(contract):
            normalized_alias = str(alias).strip()
            if normalized_alias and self._matches_target(body_acronyms, normalized_alias):
                score += 1.5
        return score

    def _origin_signal_score(self, candidate: CandidatePaper, *, targets: list[str]) -> float:
        if not targets:
            return 0.0
        content = self._normalize_text(
            str(
                candidate.metadata.get("paper_card_text")
                or candidate.metadata.get("generated_summary")
                or candidate.metadata.get("abstract_note")
                or ""
            )
        )
        if not content:
            return 0.0
        score = 0.0
        for target in targets:
            target_text = str(target or "").strip()
            if not target_text:
                continue
            target_pattern = re.escape(target_text.lower())
            proposal_pattern = (
                rf"\b(?:we|this paper|our paper|the paper)?\s*"
                rf"(?:propose|proposes|proposed|present|presents|presented|introduce|introduces|introduced)\b"
                rf"[^.。;；]{{0,180}}\b{target_pattern}\b"
            )
            contribution_pattern = rf"\b(?:main contribution|contribution)\b[^.。;；]{{0,160}}\b{target_pattern}\b"
            if re.search(proposal_pattern, content) or re.search(contribution_pattern, content):
                score += 1.4
                continue
            if re.search(rf"\bthe\s+{target_pattern}\b", content):
                score += 0.25
        return score

    def _configured_paper_boost(self, *, target: str, title: str, content: str) -> float:
        normalized_target = str(target or "").strip().lower()
        if not normalized_target:
            return 0.0
        score = 0.0
        for rule in self.settings.retrieval_paper_match_boosts:
            rule_target = str(rule.get("target", "") or "").strip().lower()
            if rule_target and rule_target != normalized_target:
                continue
            title_contains = str(rule.get("title_contains", "") or "").strip().lower()
            content_contains = str(rule.get("content_contains", "") or "").strip().lower()
            try:
                weight = float(rule.get("weight", 0.0) or 0.0)
            except (TypeError, ValueError):
                weight = 0.0
            if title_contains and title_contains in title:
                score += weight
            if content_contains and content_contains in content:
                score += weight
        return score

    @staticmethod
    def _normalize_entity_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())

    @staticmethod
    def _paper_candidate_names(meta: dict[str, Any]) -> list[str]:
        title = str(meta.get("title", "")).strip()
        aliases = [alias.strip() for alias in str(meta.get("aliases", "")).split("||") if alias.strip()]
        body_acronyms = [alias.strip() for alias in str(meta.get("body_acronyms", "")).split("||") if alias.strip()]
        candidates: list[str] = []
        for item in [title, *aliases, *body_acronyms]:
            if item and item not in candidates:
                candidates.append(item)
        if title:
            for separator in [":", " - ", " — ", " – "]:
                if separator in title:
                    head = title.split(separator, 1)[0].strip()
                    if head and head not in candidates:
                        candidates.append(head)
        return candidates

    @staticmethod
    def _concept_score(text: str, title: str, *, tokens: list[str], target_terms: list[str]) -> float:
        haystack = DualIndexRetriever._normalize_text(f"{title}\n{text}")
        if not haystack:
            return 0.0
        score = DualIndexRetriever._lexical_score(haystack, tokens)
        if target_terms:
            target_hits = 0
            for target in target_terms:
                if target and DualIndexRetriever._matches_target(f"{title}\n{text}", target):
                    target_hits += 1
                    score += 2.2
            if target_hits == 0:
                return 0.0
        if any(
            token in haystack
            for token in [
                "refers to",
                "stands for",
                "is a",
                "is an",
                "on-policy",
                "reinforcement learning",
                "policy optimization",
                "reward model",
                "算法",
                "方法",
            ]
        ):
            score += 0.8
        return score

    @staticmethod
    def _entity_score(text: str, title: str, *, tokens: list[str], target_terms: list[str]) -> float:
        haystack = f"{title}\n{text}"
        normalized = DualIndexRetriever._normalize_text(haystack)
        if not normalized:
            return 0.0
        score = DualIndexRetriever._lexical_score(normalized, tokens) * 0.8
        target_hits = 0
        for target in target_terms:
            if target and DualIndexRetriever._matches_target(haystack, target):
                target_hits += 1
                score += 2.4
        if target_terms and target_hits == 0:
            return 0.0
        return score

    @staticmethod
    def _definition_like_score(text: str, *, targets: list[str]) -> float:
        raw = " ".join(str(text or "").split())
        lowered = raw.lower()
        score = 0.0
        for target in targets:
            target_text = str(target or "").strip()
            if not target_text:
                continue
            if DualIndexRetriever._expansion_before_target(raw, target_text):
                score = max(score, 3.2)
            normalized_target = target_text.lower()
            if normalized_target and re.search(rf"\b{re.escape(normalized_target)}\b\s+(is|refers to|means|denotes|stands for)\b", lowered):
                score = max(score, 2.4)
            if normalized_target and re.search(
                rf"\b{re.escape(normalized_target)}\b.{0,40}\b(variant|algorithm|method|framework|system|dataset|benchmark)\b",
                lowered,
            ):
                score = max(score, 2.0)
            if normalized_target and re.search(
                rf"\b(propose|introduce|present)\b.{0,60}\b{re.escape(normalized_target)}\b",
                lowered,
            ):
                score = max(score, 1.8)
        return score

    @staticmethod
    def _mechanism_like_score(text: str, title: str, *, targets: list[str], requested_fields: set[str]) -> float:
        raw = " ".join(str(text or "").split())
        lowered = raw.lower()
        haystack = f"{title}\n{raw}"
        score = 0.0
        if targets and not any(target and DualIndexRetriever._matches_target(haystack, target) for target in targets):
            return 0.0
        strong_cues = [
            "objective",
            "advantage",
            "relative rewards",
            "group-based reward",
            "value critic",
            "value estimates",
            "rule-based rewards",
            "rubric-based rewards",
            "policy ratio",
            "clip",
            "kl penalty",
            "workflow",
            "work flow",
            "机制",
            "流程",
            "目标函数",
            "奖励",
            "优势函数",
        ]
        medium_cues = [
            "reinforcement learning",
            "policy optimization",
            "algorithm",
            "operates on groups",
            "compute advantages",
            "sampled from the old policy",
            "eliminating the need",
            "optimiz",
            "训练",
            "优化",
            "算法",
        ]
        if any(token in lowered for token in strong_cues):
            score += 1.4
        if any(token in lowered for token in medium_cues):
            score += 0.8
        if requested_fields & {"formula", "objective", "variable_explanation"} and any(
            token in lowered for token in ["objective", "advantage", "clip", "kl", "reward", "公式", "目标函数"]
        ):
            score += 0.8
        if requested_fields & {"mechanism", "workflow", "reward_signal", "training_signal"} and any(
            token in lowered for token in ["workflow", "operates", "uses", "guiding signals", "reward", "流程", "机制"]
        ):
            score += 0.6
        return score

    @staticmethod
    def _application_like_score(text: str, *, targets: list[str]) -> float:
        raw = " ".join(str(text or "").split())
        lowered = raw.lower()
        if targets and not any(target and DualIndexRetriever._matches_target(raw, target) for target in targets):
            return 0.0
        if any(
            token in lowered
            for token in [
                "we employ",
                "we use",
                "used in",
                "is used",
                "applied to",
                "guiding signals",
                "用于",
                "用来",
                "应用于",
            ]
        ):
            return 0.8
        return 0.0

    @staticmethod
    def _expansion_before_target(text: str, target: str) -> str:
        stopwords = {"and", "or", "the", "a", "an", "to", "for", "with", "via", "on", "in", "of", "from"}
        initials_target = "".join(ch for ch in target.upper() if ch.isalnum())
        if len(initials_target) < 2:
            return ""
        for needle in [f"({target})", f"（{target}）"]:
            index = text.find(needle)
            if index < 0:
                continue
            prefix = text[max(0, index - 100) : index]
            words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", prefix)
            if len(words) < 2:
                continue
            for size in range(2, min(8, len(words)) + 1):
                phrase_words = words[-size:]
                initials = "".join(word[0].upper() for word in phrase_words if word.lower() not in stopwords)
                if initials != initials_target:
                    continue
                phrase = " ".join(phrase_words).strip(" .,:;")
                if phrase:
                    return phrase.title() if phrase.islower() else phrase
        return ""

    @staticmethod
    def _matches_target(text: str, target: str) -> bool:
        raw_text = str(text or "")
        raw_target = str(target or "").strip()
        if not raw_text or not raw_target:
            return False
        if " " not in raw_target and len(raw_target) <= 24 and re.fullmatch(r"[A-Za-z0-9\-]{2,}", raw_target):
            lowered_text = raw_text.lower()
            target_key = raw_target.lower()
            pattern = re.compile(rf"(?<![A-Za-z0-9\-]){re.escape(target_key)}(?![A-Za-z0-9\-])")
            if pattern.search(lowered_text) is not None:
                return True
            if raw_target.upper() == raw_target and len(raw_target) <= 10:
                loss_patterns = [
                    rf"\bl[_\{{\s]*(?:\\mathrm\{{?)?\s*{re.escape(target_key)}\b",
                    rf"\bl{re.escape(target_key)}\b",
                ]
                return any(re.search(loss_pattern, lowered_text) for loss_pattern in loss_patterns)
            return False
        return DualIndexRetriever._normalize_text(raw_target) in DualIndexRetriever._normalize_text(raw_text)

    @staticmethod
    def _query_tokens(query: str, *, extra: list[str]) -> list[str]:
        english = re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,}", query)
        chinese = re.findall(r"[\u4e00-\u9fff]{2,}", query)
        tokens = [*english, *chinese, *extra]
        normalized: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            key = DualIndexRetriever._normalize_text(token)
            if key and key not in seen:
                seen.add(key)
                normalized.append(key)
        return normalized

    @staticmethod
    def _lexical_score(text: str, tokens: list[str]) -> float:
        haystack = DualIndexRetriever._normalize_text(text)
        if not haystack or not tokens:
            return 0.0
        score = 0.0
        for token in tokens:
            if token in haystack:
                score += 1.0
        return score

    def _formula_snippet_score(self, text: str, targets: list[str] | None = None) -> float:
        raw = str(text or "")
        haystack = raw.lower()
        score = 0.0
        for token, weight in self.settings.retrieval_formula_token_weights.items():
            if str(token).lower() in haystack:
                score += float(weight)
        target_set = {str(item).strip().upper() for item in (targets or [])}
        for target in target_set:
            for token, weight in self.settings.retrieval_target_formula_token_weights.get(target, {}).items():
                if str(token).lower() in haystack:
                    score += float(weight)
        if "reward model" in haystack and any(token in haystack for token in ["preferred", "dispreferred", "yw", "yl"]):
            score -= 3.0
        if "figure " in haystack or haystack.startswith("figure"):
            score -= 2.0
        if "table " in haystack or haystack.startswith("table"):
            score -= 1.0
        if any(token in haystack for token in ["a.4", "appendix", "plackett-luce", "rankings"]):
            score -= 4.0
        return max(0.0, score)

    @staticmethod
    def _metric_signal_score(text: str) -> float:
        haystack = str(text or "").lower()
        weighted_tokens = {
            "win rate": 1.4,
            "winrate": 1.4,
            "accuracy": 1.1,
            "acc.": 1.1,
            "acc ": 0.8,
            "elix": 1.0,
            "review": 0.8,
            "roleplay": 0.8,
            "benchmark": 0.6,
            "performance comparison": 0.8,
            "ablation": 0.6,
        }
        return sum(weight for token, weight in weighted_tokens.items() if token in haystack)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text or "").lower().strip().split())

    @staticmethod
    def _safe_year(value: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 9999
