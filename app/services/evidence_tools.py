from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.domain.models import EvidenceBlock


@dataclass(frozen=True, slots=True)
class ClaimCheck:
    status: str
    confidence: float
    supporting_evidence_ids: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)
    reason: str = ""


def summarize_text(*, text: str, target_words: int = 120, focus: list[str] | None = None) -> str:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return ""
    target_chars = max(120, min(int(target_words or 120), 1000) * 6)
    if len(normalized) <= target_chars:
        return normalized
    sentences = _split_sentences(normalized)
    if not sentences:
        return normalized[:target_chars].rstrip()
    focus_terms = [_normalize(term) for term in list(focus or []) if _normalize(term)]
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        lowered = _normalize(sentence)
        score = 1.0 / (index + 1)
        for term in focus_terms:
            if term in lowered:
                score += 2.0
        scored.append((score, index, sentence))
    selected = sorted(sorted(scored, key=lambda item: item[0], reverse=True)[:6], key=lambda item: item[1])
    summary = " ".join(sentence for _, _, sentence in selected)
    return summary[:target_chars].rstrip()


def summarize_evidence(
    *,
    evidence: list[EvidenceBlock],
    target_words: int = 120,
    focus: list[str] | None = None,
) -> str:
    text = "\n".join(
        f"[{item.doc_id}] {item.title} p.{item.page}: {item.snippet}"
        for item in evidence
        if str(item.snippet or "").strip()
    )
    return summarize_text(text=text, target_words=target_words, focus=focus)


def verify_claim_against_evidence(
    *,
    claim: str,
    evidence: list[EvidenceBlock],
    min_overlap: int = 2,
) -> ClaimCheck:
    claim_text = str(claim or "").strip()
    if not claim_text:
        return ClaimCheck(status="clarify", confidence=0.0, reason="empty_claim")
    claim_terms = _claim_terms(claim_text)
    if not claim_terms:
        return ClaimCheck(status="clarify", confidence=0.0, reason="no_checkable_terms")
    support: list[str] = []
    covered: set[str] = set()
    for item in evidence:
        haystack = _normalize(f"{item.title} {item.caption} {item.snippet}")
        item_hits = {term for term in claim_terms if term in haystack}
        if not item_hits:
            continue
        covered.update(item_hits)
        support.append(item.doc_id or item.paper_id or item.title)
    coverage = len(covered) / max(1, len(claim_terms))
    matched = [term for term in claim_terms if term in covered]
    missing = [term for term in claim_terms if term not in covered]
    required_overlap = min(len(claim_terms), max(1, int(min_overlap or 2)))
    if len(covered) >= required_overlap and coverage >= 0.5 and support:
        return ClaimCheck(
            status="pass",
            confidence=min(0.95, 0.35 + coverage * 0.6),
            supporting_evidence_ids=list(dict.fromkeys(support)),
            matched_terms=matched,
            missing_terms=missing,
            reason="claim_terms_covered_by_evidence",
        )
    return ClaimCheck(
        status="retry",
        confidence=max(0.05, coverage * 0.55),
        supporting_evidence_ids=list(dict.fromkeys(support)),
        matched_terms=matched,
        missing_terms=missing,
        reason="claim_terms_missing_from_evidence",
    )


def evidence_from_payload(items: object) -> list[EvidenceBlock]:
    if not isinstance(items, list):
        return []
    evidence: list[EvidenceBlock] = []
    for item in items:
        if isinstance(item, EvidenceBlock):
            evidence.append(item)
            continue
        if isinstance(item, str):
            snippet = " ".join(item.split())
            if snippet:
                evidence.append(
                    EvidenceBlock(
                        doc_id=f"inline::{len(evidence) + 1}",
                        paper_id="",
                        title="inline evidence",
                        file_path="",
                        page=0,
                        block_type="inline",
                        caption="",
                        bbox="",
                        snippet=snippet,
                        score=0.0,
                        metadata={},
                    )
                )
            continue
        if not isinstance(item, dict):
            continue
        snippet = str(item.get("snippet", "") or item.get("text", "") or item.get("content", "") or "").strip()
        if not snippet:
            continue
        doc_id = str(item.get("doc_id", "") or item.get("id", "") or item.get("url", "") or f"payload::{len(evidence) + 1}")
        try:
            evidence.append(
                EvidenceBlock(
                    doc_id=doc_id,
                    paper_id=str(item.get("paper_id", "") or ""),
                    title=str(item.get("title", "") or item.get("url", "") or ""),
                    file_path=str(item.get("file_path", "") or ""),
                    page=int(item.get("page", 0) or 0),
                    block_type=str(item.get("block_type", "") or "text"),
                    caption=str(item.get("caption", "") or ""),
                    bbox=str(item.get("bbox", "") or ""),
                    snippet=snippet,
                    score=float(item.get("score", 0.0) or 0.0),
                    metadata=dict(item.get("metadata", {}) or {}),
                )
            )
        except (TypeError, ValueError):
            continue
    return evidence


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[。！？.!?])\s+", text) if part.strip()]


def _claim_terms(text: str) -> list[str]:
    tokens: list[str] = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text)
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        if len(chunk) <= 4:
            tokens.append(chunk)
            continue
        tokens.extend(chunk[index : index + 2] for index in range(0, len(chunk) - 1))
    stop = {"the", "and", "with", "from", "that", "this", "claim", "paper", "论文", "方法", "结果"}
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = _normalize(token)
        if not key or key in stop or key in seen:
            continue
        seen.add(key)
        terms.append(key)
    return terms[:20]


def _normalize(text: str) -> str:
    return " ".join(str(text or "").lower().split())
