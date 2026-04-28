from __future__ import annotations

import re
from typing import Any

from app.domain.models import EvidenceBlock


CITATION_COUNT_PATTERNS = [
    r"citationCount[\"'\s:=]+([0-9][0-9,]*)",
    r"cited by\s+([0-9][0-9,]*)",
    r"citations?\s*[:：]?\s+([0-9][0-9,]*)",
    r"([0-9][0-9,]*)\s+citations?",
    r"被引\s*[:：]?\s*([0-9][0-9,]*)",
]


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
