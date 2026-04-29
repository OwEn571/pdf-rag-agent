from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Callable

import httpx

from app.domain.models import EvidenceBlock, SessionContext
from app.services.contract_normalization import normalize_lookup_text
from app.services.library_intents import library_query_prefers_previous_candidates


CITATION_COUNT_PATTERNS = [
    r"citationCount[\"'\s:=]+([0-9][0-9,]*)",
    r"cited by\s+([0-9][0-9,]*)",
    r"citations?\s*[:：]?\s+([0-9][0-9,]*)",
    r"([0-9][0-9,]*)\s+citations?",
    r"被引\s*[:：]?\s*([0-9][0-9,]*)",
]

RankLibraryPapersFn = Callable[..., list[dict[str, Any]]]
HttpGetFn = Callable[..., Any]
logger = logging.getLogger(__name__)


def parse_citation_count(value: str) -> int | None:
    digits = re.sub(r"[^0-9]", "", str(value or ""))
    if not digits:
        return None
    try:
        count = int(digits)
    except ValueError:
        return None
    return count if count >= 0 else None


def title_token_overlap(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", str(left or "").lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", str(right or "").lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def extract_citation_count_from_evidence(*, title: str, evidence: list[EvidenceBlock]) -> dict[str, Any]:
    title_tokens = set(re.findall(r"[a-z0-9]+", title.lower()))
    best: dict[str, Any] = {}
    best_score = -1.0
    for item in evidence:
        text = " ".join([item.title, item.snippet, item.caption, item.file_path])
        lowered = text.lower()
        source = str(item.metadata.get("source", "") or "").lower()
        item_title_overlap = title_token_overlap(title, item.title)
        if source != "semantic_scholar" and item_title_overlap < 0.55:
            continue
        text_tokens = set(re.findall(r"[a-z0-9]+", lowered))
        overlap = len(title_tokens & text_tokens) / max(1, min(len(title_tokens), len(text_tokens)))
        source_bonus = 0.0
        if "semanticscholar.org" in lowered:
            source_bonus += 1.0
        if "openalex.org" in lowered:
            source_bonus += 0.8
        for pattern in CITATION_COUNT_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.I):
                count = parse_citation_count(match.group(1))
                if count is None:
                    continue
                score = overlap + source_bonus + min(count, 100000) / 1000000.0
                if score > best_score:
                    best_score = score
                    best = {
                        "citation_count": count,
                        "source_title": item.title,
                        "source_url": item.file_path,
                        "doc_id": item.doc_id,
                        "source_snippet": item.snippet[:260],
                    }
    return best


def select_citation_ranking_candidates(
    *,
    paper_documents: list[Any],
    session: SessionContext,
    query: str,
    limit: int,
    rank_library_papers_for_recommendation: RankLibraryPapersFn,
) -> list[dict[str, str]]:
    docs: list[dict[str, object]] = []
    seen_paper_ids: set[str] = set()
    by_title: dict[str, dict[str, object]] = {}
    for doc in paper_documents:
        meta = dict(doc.metadata or {})
        paper_id = str(meta.get("paper_id", "")).strip()
        title = str(meta.get("title", "") or "").strip()
        if not paper_id or paper_id in seen_paper_ids or not title:
            continue
        seen_paper_ids.add(paper_id)
        docs.append(meta)
        by_title[normalize_lookup_text(title)] = meta

    selected: list[dict[str, str]] = []
    selected_keys: set[str] = set()

    def add_candidate(*, title: str, year: str = "", reason: str = "") -> None:
        clean_title = " ".join(str(title or "").split()).strip()
        if not clean_title:
            return
        key = normalize_lookup_text(clean_title)
        if not key or key in selected_keys:
            return
        meta = by_title.get(key)
        selected_keys.add(key)
        selected.append(
            {
                "title": str(meta.get("title", clean_title) if meta else clean_title),
                "year": str(meta.get("year", year) if meta else year),
                "paper_id": str(meta.get("paper_id", "") if meta else ""),
                "reason": reason or str(meta.get("generated_summary", "") if meta else ""),
            }
        )

    if library_query_prefers_previous_candidates(query):
        for turn in reversed(session.turns[-4:]):
            if turn.relation not in {"library_recommendation", "compound_query", "library_citation_ranking"}:
                continue
            for title, year in re.findall(r"《([^》]{2,220})》(?:（(\d{4})）)?", turn.answer):
                add_candidate(title=title, year=year)
                if len(selected) >= limit:
                    break
            if selected:
                break

    if not selected:
        for item in rank_library_papers_for_recommendation(docs=docs, query=query, limit=limit):
            add_candidate(title=item["title"], year=item.get("year", ""), reason=item.get("reason", ""))
            if len(selected) >= limit:
                break
    return selected[:limit]


def semantic_scholar_citation_evidence(
    *,
    title: str,
    web_search: Any,
    timeout_seconds: float,
    http_get: HttpGetFn = httpx.get,
) -> EvidenceBlock | None:
    if type(web_search).__name__ != "TavilyWebSearchClient":
        return None
    try:
        response = http_get(
            "https://api.semanticscholar.org/graph/v1/paper/search/match",
            params={
                "query": title,
                "fields": "title,year,citationCount,url",
            },
            timeout=min(max(float(timeout_seconds), 2.0), 5.0),
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.info("semantic scholar citation lookup failed for %s: %s", title, exc)
        return None
    records = payload.get("data", [])
    if not isinstance(records, list):
        return None
    best_record: dict[str, Any] | None = None
    best_overlap = 0.0
    for record in records:
        if not isinstance(record, dict):
            continue
        record_title = str(record.get("title", "") or "").strip()
        overlap = title_token_overlap(title, record_title)
        if overlap > best_overlap:
            best_overlap = overlap
            best_record = record
    if best_record is None or best_overlap < 0.55:
        return None
    count = parse_citation_count(str(best_record.get("citationCount", "")))
    if count is None:
        return None
    record_title = str(best_record.get("title", "") or title).strip()
    url = str(best_record.get("url", "") or "").strip() or "https://www.semanticscholar.org/search"
    year = str(best_record.get("year", "") or "").strip()
    doc_id = "web::semantic-scholar::" + hashlib.sha1(f"{record_title}\n{url}".encode("utf-8")).hexdigest()[:16]
    snippet = (
        f"Semantic Scholar citationCount: {count:,}. "
        f"Matched paper title: {record_title}."
    )
    return EvidenceBlock(
        doc_id=doc_id,
        paper_id=doc_id,
        title=f"{record_title} | Semantic Scholar",
        file_path=url,
        page=0,
        block_type="web",
        caption=url,
        snippet=snippet,
        score=best_overlap,
        metadata={
            "source": "semantic_scholar",
            "query": title,
            "year": year,
            "citation_count": count,
            "title_overlap": best_overlap,
        },
    )


def format_citation_ranking_answer(
    *,
    candidates: list[dict[str, str]],
    citation_results: list[dict[str, Any]],
    web_enabled: bool,
) -> str:
    if not candidates:
        return "## 按引用数重排\n\n当前库里没有可用于推荐的候选论文。"
    if not web_enabled:
        titles = "、".join(f"《{item['title']}》" for item in candidates[:5])
        return (
            "## 按引用数重排\n\n"
            "引用数是外部动态指标，不能只靠本地 PDF 摘要推断。当前 Web/Tavily 检索没有可用配置，"
            "所以我不会把上一轮默认推荐硬改成“按引用数”的答案。\n\n"
            f"已识别的候选是：{titles}。配置 Web 检索后，我会逐篇查 citation count 再排序。"
        )

    counted = [item for item in citation_results if item.get("citation_count") is not None]
    missing = [item for item in citation_results if item.get("citation_count") is None]
    if not counted:
        titles = "、".join(f"《{item['title']}》" for item in candidates[:5])
        return (
            "## 按引用数重排\n\n"
            "我已经对候选做了外部 citation count 检索，但返回摘要里没有稳定抽出引用数。"
            "因此不能诚实地按引用数排序。\n\n"
            f"这次候选是：{titles}。\n\n"
            "边界说明：没抽到 citation count 不等于低引用，只是当前 Web 摘要不足。"
        )

    counted.sort(key=lambda item: (-int(item["citation_count"]), item["title"].lower()))
    lines = [
        "## 按引用数重排",
        "",
        "我没有复用上一轮“默认推荐”的本地启发式；下面只按外部检索中能抽取到的 citation count 排序。",
        "",
        "| 排名 | 论文 | 引用数 | 来源 |",
        "|---:|---|---:|---|",
    ]
    for index, item in enumerate(counted, start=1):
        year = f"（{item['year']}）" if item.get("year") else ""
        source = str(item.get("source_url") or item.get("source_title") or "web evidence")
        if item.get("source_url"):
            source = f"[来源]({item['source_url']})"
        lines.append(f"| {index} | 《{item['title']}》{year} | {int(item['citation_count']):,} | {source} |")
    if missing:
        missing_titles = "、".join(f"《{item['title']}》" for item in missing[:4])
        lines.extend(
            [
                "",
                f"未排序候选：{missing_titles}。这些只是没有从当前检索摘要里抽到引用数，不能视为引用数更低。",
            ]
        )
    lines.extend(
        [
            "",
            "边界说明：引用数会随平台和时间变化；这里是一次外部检索的可验证快照，适合用于粗排，不适合当作精确 bibliometrics 报告。",
        ]
    )
    return "\n".join(lines)
