from __future__ import annotations

import re
from typing import Any

from app.services.contracts.normalization import normalize_lookup_text

# 匹配中文书名号引用：《论文标题》
_CJK_BOOKMARK_RE = re.compile(r"《([^》]{2,220})》")
# 匹配英文斜体 *Paper Title*
_EN_ITALIC_RE = re.compile(r"\*([A-Z][^*\n]{3,200})\*")
# 匹配方括号引用 [12]
_BRACKET_REF_RE = re.compile(r"\[(\d{1,3})\]")
# 匹配 "..." 引号内的长标题（至少8字符且含大写）
_EN_QUOTED_TITLE_RE = re.compile(r'"([^"]{8,200})"')


def _extract_cited_titles(answer: str) -> list[str]:
    titles: list[str] = []
    for m in _CJK_BOOKMARK_RE.finditer(answer):
        titles.append(m.group(1))
    for m in _EN_ITALIC_RE.finditer(answer):
        titles.append(m.group(1))
    for m in _EN_QUOTED_TITLE_RE.finditer(answer):
        titles.append(m.group(1))
    return titles


def _extract_cited_bracket_ids(answer: str) -> list[int]:
    return [int(m.group(1)) for m in _BRACKET_REF_RE.finditer(answer)]


def build_answer_whitelist(
    *,
    evidence: list[Any],
    citations: list[Any],
    screened_papers: list[Any],
) -> set[str]:
    allowed: set[str] = set()
    for item in evidence:
        title = getattr(item, "title", "") or ""
        if title:
            allowed.add(normalize_lookup_text(title))
        paper_id = getattr(item, "paper_id", "") or getattr(item, "doc_id", "") or ""
        if paper_id:
            allowed.add(normalize_lookup_text(paper_id))
    for item in citations:
        title = getattr(item, "title", "") or ""
        if title:
            allowed.add(normalize_lookup_text(title))
        paper_id = getattr(item, "paper_id", "") or ""
        if paper_id:
            allowed.add(normalize_lookup_text(paper_id))
    for item in screened_papers:
        title = getattr(item, "title", "") or ""
        if title:
            allowed.add(normalize_lookup_text(title))
        paper_id = getattr(item, "paper_id", "") or ""
        if paper_id:
            allowed.add(normalize_lookup_text(paper_id))
    allowed.discard("")
    return allowed


def audit_answer_citations(
    *,
    answer: str,
    allowed_titles: set[str],
    max_citation_index: int = 0,
) -> list[str]:
    """Returns a list of violations (human-readable descriptions)."""
    violations: list[str] = []
    for title in _extract_cited_titles(answer):
        if normalize_lookup_text(title) not in allowed_titles:
            violations.append(f"引用标题不在白名单中：《{title[:80]}》")
    if max_citation_index > 0:
        for idx in _extract_cited_bracket_ids(answer):
            if idx < 1 or idx > max_citation_index:
                violations.append(f"引用编号超出范围：[{idx}]（有效范围 1-{max_citation_index}）")
    return violations
