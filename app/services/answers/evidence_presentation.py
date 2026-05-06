from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.domain.models import AssistantCitation, Claim, EvidenceBlock
from app.services.intents.figure import extract_figure_benchmarks, figure_signal_score, has_explicit_figure_reference


DocumentLookup = Callable[[str], Any | None]


def paper_recommendation_reason(summary_text: str) -> str:
    compact = " ".join(str(summary_text or "").split())
    if not compact:
        return "与当前主题直接相关。"
    if len(compact) > 120:
        compact = compact[:117].rstrip() + "..."
    return compact


def extract_topology_terms(evidence: list[EvidenceBlock]) -> list[str]:
    joined = "\n".join(item.snippet.lower() for item in evidence[:8])
    ordered_terms = [
        ("dag", ["dag"]),
        ("irregular/random", ["irregular", "random"]),
        ("chain", ["chain"]),
        ("tree", ["tree"]),
        ("mesh", ["mesh"]),
        ("star", ["star"]),
    ]
    found: list[str] = []
    for label, tokens in ordered_terms:
        if any(token in joined for token in tokens):
            found.append(label)
    if not found:
        found = ["dag", "chain", "tree", "mesh", "irregular/random"]
    return found


def formula_terms(text: str) -> list[str]:
    joined = text.lower()
    mapping = {
        "pi_theta": ["pi_theta", "πθ", "\\pi_\\theta", "\\pi_{\\theta}", "policy"],
        "pi_phi": ["pi_phi", "πϕ", "πφ", "\\pi_\\phi", "\\pi_{\\phi}", "\\pi_\\varphi"],
        "pi_ref": ["pi_ref", "πref", "\\pi_{\\mathrm{ref}}", "\\pi_ref", "reference"],
        "p_tilde": ["p̃", "\\tilde{p}", "preference direction vector"],
        "beta": ["β", "beta"],
        "preferred": ["preferred", "y_w", "yw"],
        "rejected": ["rejected", "y_l", "yl"],
        "log_sigma": ["log σ", "log sigma", "\\log \\sigma", "sigmoid"],
        "ratio": ["r_t", "ratio", "probability ratio"],
        "advantage": ["advantage", "\\hat{a}", "â", "a_t"],
        "epsilon": ["epsilon", "\\epsilon", "ϵ"],
        "clip": ["clip", "clipped"],
    }
    found: list[str] = []
    for label, tokens in mapping.items():
        if any(token in joined for token in tokens):
            found.append(label)
    return found


def top_evidence_ids(evidence: list[EvidenceBlock], *, limit: int) -> list[str]:
    return [item.doc_id for item in evidence[:limit]]


def evidence_ids_for_paper(evidence: list[EvidenceBlock], paper_id: str, *, limit: int) -> list[str]:
    return [item.doc_id for item in evidence if item.paper_id == paper_id][:limit]


def claim_evidence_ids(claims: list[Claim]) -> list[str]:
    doc_ids: list[str] = []
    for claim in claims:
        for doc_id in claim.evidence_ids:
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
    return doc_ids


def citations_from_doc_ids(
    doc_ids: list[str],
    evidence: list[EvidenceBlock],
    *,
    block_doc_lookup: DocumentLookup | None = None,
    paper_doc_lookup: DocumentLookup | None = None,
    screened_paper_ids: set[str] | None = None,
) -> list[AssistantCitation]:
    by_id = {item.doc_id: item for item in evidence}
    citations: list[AssistantCitation] = []
    seen: set[str] = set()
    for doc_id in doc_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        item = by_id.get(doc_id)
        if item is not None:
            citations.append(citation_from_evidence(item))
            continue
        # P0-2: Only fallback to paper lookup for IDs within screened_papers scope
        if screened_paper_ids is not None:
            lookup_id = doc_id.split("paper::", maxsplit=1)[1] if doc_id.startswith("paper::") else doc_id
            if lookup_id not in screened_paper_ids:
                continue
        lookup_id = doc_id.split("paper::", maxsplit=1)[1] if doc_id.startswith("paper::") else doc_id
        doc = block_doc_lookup(doc_id) if block_doc_lookup is not None else None
        if doc is None and paper_doc_lookup is not None:
            doc = paper_doc_lookup(lookup_id)
        if doc is None:
            continue
        citations.append(citation_from_document(doc=doc, doc_id=doc_id, fallback_paper_id=lookup_id))
    return citations


def dedupe_citations(citations: list[AssistantCitation]) -> list[AssistantCitation]:
    seen: set[tuple[str, str, int, str]] = set()
    deduped: list[AssistantCitation] = []
    for citation in citations:
        key = (citation.title, citation.file_path, citation.page, citation.block_type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def citation_from_evidence(item: EvidenceBlock) -> AssistantCitation:
    tags_raw = str(item.metadata.get("tags", ""))
    return AssistantCitation(
        doc_id=item.doc_id,
        paper_id=item.paper_id,
        title=item.title,
        authors=str(item.metadata.get("authors", "")),
        year=str(item.metadata.get("year", "")),
        tags=[tag for tag in tags_raw.split("||") if tag],
        file_path=item.file_path,
        page=item.page,
        block_type=item.block_type,
        caption=item.caption,
        snippet=item.snippet[:220],
    )


def citation_from_document(*, doc: Any, doc_id: str, fallback_paper_id: str) -> AssistantCitation:
    meta = dict(getattr(doc, "metadata", {}) or {})
    tags_raw = str(meta.get("tags", ""))
    snippet = " ".join(str(getattr(doc, "page_content", "") or "").split())[:220]
    return AssistantCitation(
        doc_id=str(meta.get("doc_id", doc_id)),
        paper_id=str(meta.get("paper_id", fallback_paper_id)),
        title=str(meta.get("title", "")),
        authors=str(meta.get("authors", "")),
        year=str(meta.get("year", "")),
        tags=[tag for tag in tags_raw.split("||") if tag],
        file_path=str(meta.get("file_path", "")),
        page=int(meta.get("page", 0) or 0),
        block_type=str(meta.get("block_type", "paper_card")),
        caption=str(meta.get("caption", "")),
        snippet=snippet,
    )


def build_figure_contexts(evidence: list[EvidenceBlock], limit: int = 2) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], dict[str, Any]] = {}
    for item in evidence:
        if item.block_type not in {"figure", "caption", "page_text"}:
            continue
        key = (item.file_path, item.page)
        group = groups.setdefault(
            key,
            {
                "paper_id": item.paper_id,
                "file_path": item.file_path,
                "page": item.page,
                "title": item.title,
                "doc_ids": [],
                "caption_parts": [],
                "figure_parts": [],
                "page_parts": [],
                "score": 0.0,
                "page_text_score": 0.0,
            },
        )
        if item.doc_id not in group["doc_ids"]:
            group["doc_ids"].append(item.doc_id)
        if item.block_type == "caption" and item.snippet:
            group["caption_parts"].append(item.snippet)
            group["score"] += 2.0 + (figure_signal_score(item.snippet) * 1.5)
        elif item.block_type == "figure" and item.snippet:
            group["figure_parts"].append(item.snippet)
            group["score"] += 3.0 + (figure_signal_score(item.snippet) * 1.5)
        elif item.block_type == "page_text" and item.snippet:
            group["page_parts"].append(item.snippet)
            page_score = 0.8 + (figure_signal_score(item.snippet) * 1.8)
            if has_explicit_figure_reference(item.snippet):
                page_score += 6.0
            if page_score > float(group.get("page_text_score", 0.0)):
                group["score"] += page_score - float(group.get("page_text_score", 0.0))
                group["page_text_score"] = page_score
    ranked = sorted(groups.values(), key=lambda item: (-item["score"], item["page"]))
    contexts: list[dict[str, Any]] = []
    for item in ranked[:limit]:
        contexts.append(
            {
                "paper_id": item["paper_id"],
                "file_path": item["file_path"],
                "page": item["page"],
                "title": item["title"],
                "doc_ids": item["doc_ids"][:6],
                "caption": join_unique_text(item["caption_parts"], limit=500),
                "figure_text": join_unique_text(item["figure_parts"], limit=500),
                "page_text": join_unique_text(item["page_parts"], limit=700),
            }
        )
    return contexts


def figure_fallback_summary(contexts: list[dict[str, Any]]) -> str:
    item = contexts[0]
    caption = str(item.get("caption", "")).strip()
    page_text = str(item.get("page_text", "")).strip()
    parts = [part for part in [caption, page_text] if part]
    if not parts:
        return "当前没有足够的图像证据。"
    combined = " ".join(parts)
    if "figure 1" not in combined.lower() and "图1" not in combined:
        combined = f"Figure 1 / 图1 相关内容：{combined}"
    benchmarks = extract_figure_benchmarks(combined)
    benchmark_suffix = ""
    if len(benchmarks) >= 3:
        benchmark_suffix = " 图中提到的 benchmark 包括：" + "、".join(benchmarks) + "。"
    if benchmark_suffix:
        remaining = max(80, 700 - len(benchmark_suffix))
        combined = combined[:remaining].rstrip() + benchmark_suffix
        return combined[:700]
    return combined[:700]


def join_unique_text(items: list[str], *, limit: int) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    total = 0
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        remaining = max(0, limit - total)
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[: max(0, remaining - 3)] + "..."
        parts.append(text)
        total += len(text) + 1
    return "\n".join(parts).strip()


def safe_year(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 9999


def chunk_text(text: str, *, size: int) -> list[str]:
    if not text:
        return []
    return [text[index : index + size] for index in range(0, len(text), size)]
