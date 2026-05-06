from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any

from app.domain.models import SessionContext
from app.services.contracts.normalization import normalize_lookup_text
from app.services.intents.marker_matching import MarkerProfile, query_matches_any
from app.services.contracts.session_context import agent_session_conversation_context


LIBRARY_COMPOSER_MARKERS: dict[str, MarkerProfile] = {
    "status_recommendation": (
        "最值得",
        "值得一读",
        "值得读",
        "值得一看",
        "值得看",
        "推荐",
        "哪篇",
        "哪几篇",
        "must read",
        "worth reading",
        "recommend",
    ),
    "status_listing": ("有哪些", "有哪些文章", "有哪些论文", "文章列表", "论文列表", "列出", "list"),
    "same_best": ("最值得", "best", "top", "first"),
    "recent": ("最新", "最近", "newest", "latest", "recent"),
    "survey": ("综述", "survey", "review", "入门", "overview"),
}


def split_library_authors(authors: str) -> list[str]:
    normalized = str(authors or "").replace(" and ", ",")
    names = [" ".join(item.split()) for item in normalized.split(",")]
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            deduped.append(name)
    return deduped


def library_status_query_wants_recommendation(query: str) -> bool:
    compact = " ".join(str(query or "").lower().split())
    return query_matches_any(compact, "", LIBRARY_COMPOSER_MARKERS["status_recommendation"])


def library_status_query_wants_listing(query: str) -> bool:
    compact = " ".join(str(query or "").lower().split())
    return query_matches_any(compact, "", LIBRARY_COMPOSER_MARKERS["status_listing"])


def library_paper_preview_lines(
    *,
    docs: list[dict[str, object]],
    collection_paths: dict[str, list[str]],
    limit: int,
) -> list[str]:
    def year_value(meta: dict[str, object]) -> int:
        year = str(meta.get("year", "") or "").strip()
        return int(year) if year.isdigit() else 0

    ranked = sorted(
        docs,
        key=lambda meta: (-year_value(meta), str(meta.get("title", "") or "").lower()),
    )
    lines: list[str] = []
    for meta in ranked[:limit]:
        title = str(meta.get("title", "") or "").strip()
        if not title:
            continue
        year = str(meta.get("year", "") or "").strip()
        paper_id = str(meta.get("paper_id", "") or "").strip()
        tags = [tag for tag in str(meta.get("tags", "") or "").split("||") if tag]
        categories = collection_paths.get(paper_id) or tags[:1]
        suffix_parts = []
        if year:
            suffix_parts.append(year)
        if categories:
            suffix_parts.append(str(categories[0]))
        suffix = f"（{' · '.join(suffix_parts)}）" if suffix_parts else ""
        lines.append(f"- 《{title}》{suffix}")
    return lines


def library_unique_paper_metadata(*, paper_documents: list[Any]) -> list[dict[str, object]]:
    docs: list[dict[str, object]] = []
    seen_paper_ids: set[str] = set()
    for doc in paper_documents:
        meta = dict(getattr(doc, "metadata", None) or {})
        paper_id = str(meta.get("paper_id", "")).strip()
        if not paper_id or paper_id in seen_paper_ids:
            continue
        seen_paper_ids.add(paper_id)
        docs.append(meta)
    return docs


def compose_library_status_markdown(
    *,
    query: str,
    docs: list[dict[str, object]],
    collection_paths: dict[str, list[str]],
) -> str:
    category_counter: Counter[str] = Counter()
    year_values: list[int] = []
    pdf_paths = 0
    for meta in docs:
        paper_id = str(meta.get("paper_id", "")).strip()
        tags = [tag for tag in str(meta.get("tags", "")).split("||") if tag]
        categories = collection_paths.get(paper_id) or tags[:3] or ["未分类"]
        for category in dict.fromkeys(str(item or "未分类") for item in categories):
            category_counter[category] += 1
        year = str(meta.get("year", "")).strip()
        if year.isdigit():
            year_values.append(int(year))
        if str(meta.get("file_path", "")).lower().endswith(".pdf"):
            pdf_paths += 1

    total = len(docs)
    lines = [
        "## 当前论文库",
        "",
        f"我当前索引的本地 Zotero/PDF 论文库共有 **{total} 篇论文**。",
    ]
    if pdf_paths:
        lines.append(f"其中有 PDF 路径的记录是 **{pdf_paths} 篇**。")
    if year_values:
        lines.append(f"年份范围大约是 **{min(year_values)}–{max(year_values)}**。")
    if category_counter:
        top_categories = "、".join(f"{name}（{count}）" for name, count in category_counter.most_common(6))
        lines.extend(["", "## 分类概览", "", f"主要分类：{top_categories}。"])
    if library_status_query_wants_listing(query):
        preview_limit = 18 if total <= 24 else 12
        preview_lines = library_paper_preview_lines(
            docs=docs,
            collection_paths=collection_paths,
            limit=preview_limit,
        )
        if preview_lines:
            lines.extend(
                [
                    "",
                    "## 文章预览",
                    "",
                    f"你问“有哪些文章”时，我先按年份较新、元数据较完整列出 **{len(preview_lines)} 篇预览**；完整列表可以在左侧 Zotero 分类栏继续浏览。",
                    "",
                    *preview_lines,
                ]
            )
    if library_status_query_wants_recommendation(query):
        recommendations = rank_library_papers_for_recommendation(docs=docs, query=query, limit=3)
        if recommendations:
            primary = recommendations[0]
            primary_year = f"（{primary['year']}）" if primary.get("year") else ""
            lines.extend(
                [
                    "",
                    "## 默认推荐",
                    "",
                    "如果你没有指定“值得一读”的标准，我默认按基础性、覆盖面和本地摘要可支撑性来选。",
                    f"我会先读《{primary['title']}》{primary_year}：{primary['reason']}",
                ]
            )
            if len(recommendations) > 1:
                runners_up_items = []
                for item in recommendations[1:]:
                    year_suffix = f"（{item['year']}）" if item.get("year") else ""
                    runners_up_items.append(f"《{item['title']}》{year_suffix}")
                runners_up = "；".join(runners_up_items)
                lines.append(f"备选：{runners_up}。")
    lines.extend(
        [
            "",
            "这个数字来自当前索引的 paper store，不是一次检索召回的 top-k 候选数；候选论文数量不能当成总论文数。",
        ]
    )
    return "\n".join(lines)


def clean_library_recommendation_criteria_note(note: str, *, has_recent_recommendations: bool) -> str:
    fallback = (
        "这次我会避开刚刚已经推荐过的论文，换几个不同入口。"
        if has_recent_recommendations
        else "我会按当前问题、主题覆盖、论文类型和摘要证据强度综合挑选。"
    )
    compact = " ".join(str(note or "").split())
    if not compact:
        return fallback
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", compact))
    ascii_letters = len(re.findall(r"[A-Za-z]", compact))
    if ascii_letters > max(12, chinese_chars * 2):
        return fallback
    return compact[:120]


def diversify_library_recommendations(
    *,
    candidates: list[dict[str, str]],
    recent_titles: list[str],
    query: str,
    limit: int,
) -> list[dict[str, str]]:
    recent_keys = {normalize_lookup_text(title) for title in recent_titles}
    wants_same_best = query_matches_any(str(query).lower(), "", LIBRARY_COMPOSER_MARKERS["same_best"])
    fresh: list[dict[str, str]] = []
    repeated: list[dict[str, str]] = []
    for item in candidates:
        key = normalize_lookup_text(item.get("title", ""))
        if key in recent_keys and not wants_same_best:
            repeated.append(item)
        else:
            fresh.append(item)
    return [*fresh, *repeated][:limit]


def recent_library_recommendation_titles(session: SessionContext | None) -> list[str]:
    if session is None:
        return []
    titles: list[str] = []
    memory = dict(session.working_memory or {})
    for item in reversed([entry for entry in list(memory.get("tool_results", []) or []) if isinstance(entry, dict)]):
        if item.get("tool") != "get_library_recommendation":
            continue
        titles.extend(re.findall(r"《([^》]{2,220})》", str(item.get("answer_preview", "") or "")))
        if titles:
            break
    if not titles:
        for turn in reversed(session.turns[-4:]):
            if turn.relation not in {"library_recommendation", "compound_query"}:
                continue
            titles.extend(re.findall(r"《([^》]{2,220})》", turn.answer))
            if titles:
                break
    deduped: list[str] = []
    seen: set[str] = set()
    for title in titles:
        key = normalize_lookup_text(title)
        if key and key not in seen:
            seen.add(key)
            deduped.append(title)
    return deduped[:8]


def rank_library_papers_for_recommendation(
    *,
    docs: list[dict[str, object]],
    query: str,
    limit: int = 3,
) -> list[dict[str, str]]:
    lowered_query = str(query or "").lower()
    wants_recent = query_matches_any(lowered_query, "", LIBRARY_COMPOSER_MARKERS["recent"])
    wants_survey = query_matches_any(lowered_query, "", LIBRARY_COMPOSER_MARKERS["survey"])
    scored: list[tuple[float, dict[str, str]]] = []
    for meta in docs:
        title = str(meta.get("title", "") or "").strip()
        if not title:
            continue
        year_text = str(meta.get("year", "") or "").strip()
        tags = [tag for tag in str(meta.get("tags", "") or "").split("||") if tag]
        summary = str(meta.get("generated_summary") or meta.get("abstract_note") or "").strip()
        text = " ".join([title, summary, " ".join(tags)]).lower()
        score = 0.0
        if wants_recent and year_text.isdigit():
            score += max(0, int(year_text) - 2000) * 0.2
        if wants_survey and query_matches_any(text, "", LIBRARY_COMPOSER_MARKERS["survey"]):
            score += 8.0
        signal_weights = {
            "survey": 2.2,
            "benchmark": 2.0,
            "foundational": 3.0,
            "foundation": 2.4,
            "seminal": 3.0,
            "introduce": 2.0,
            "introduces": 2.0,
            "introduced": 1.8,
            "propose": 1.2,
            "all you need": 3.5,
            "comprehensive": 1.5,
            "framework": 1.0,
        }
        for token, weight in signal_weights.items():
            if token in text:
                score += weight
        score += min(len(tags), 5) * 0.15
        if year_text.isdigit() and not wants_recent:
            year = int(year_text)
            if 2016 <= year <= 2024:
                score += 0.8
            elif year > 2024:
                score += 0.35
        reason = library_recommendation_reason(title=title, year=year_text, summary=summary, tags=tags)
        scored.append(
            (
                score,
                {
                    "title": title,
                    "year": year_text,
                    "paper_id": str(meta.get("paper_id", "") or "").strip(),
                    "tags": "、".join(tags[:5]),
                    "summary": summary[:360],
                    "reason": reason,
                },
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1]["title"].lower()))
    return [item for _score, item in scored[:limit]]


def select_library_recommendations(
    *,
    query: str,
    candidates: list[dict[str, str]],
    session: SessionContext | None,
    limit: int,
    clients: Any,
    settings: Any,
) -> tuple[list[dict[str, str]], str]:
    recent_titles = recent_library_recommendation_titles(session)
    llm_selected, llm_note = llm_select_library_recommendations(
        query=query,
        candidates=candidates,
        session=session,
        recent_titles=recent_titles,
        limit=limit,
        clients=clients,
        settings=settings,
    )
    if llm_selected:
        return llm_selected[:limit], llm_note
    diversified = diversify_library_recommendations(
        candidates=candidates,
        recent_titles=recent_titles,
        query=query,
        limit=limit,
    )
    note = "这次我会避开刚刚已经推荐过的论文，换几个不同入口。" if recent_titles else "我会按主题覆盖、论文类型和摘要证据强度挑几个不同入口。"
    return diversified, note


def llm_select_library_recommendations(
    *,
    query: str,
    candidates: list[dict[str, str]],
    session: SessionContext | None,
    recent_titles: list[str],
    limit: int,
    clients: Any,
    settings: Any,
) -> tuple[list[dict[str, str]], str]:
    if getattr(clients, "chat", None) is None or not candidates:
        return [], ""
    try:
        context = (
            agent_session_conversation_context(
                session,
                settings=settings,
            )
            if session is not None
            else {}
        )
    except Exception:  # noqa: BLE001
        context = {}
    payload = clients.invoke_json(
        system_prompt=(
            "你是库内论文推荐重排器。"
            "你的任务不是直接回答用户，而是从 candidate_papers 中选择 3-5 篇最适合当前问题的论文。"
            "必须结合 current_query、conversation_context 和 recently_recommended_titles。"
            "不要固定偏好某一篇；如果最近已经推荐过某篇，除非用户明确追问它，否则优先换不同方向。"
            "推荐理由只能基于候选的 title/year/tags/summary，不要编造引用数或未提供事实。"
            "criteria_note 和 reason 必须使用中文，criteria_note 不超过 60 个汉字，直接说明本轮选择标准。"
            "只输出 JSON：criteria_note, recommendations，其中 recommendations 每项包含 title, reason。"
        ),
        human_prompt=json.dumps(
            {
                "current_query": query,
                "recently_recommended_titles": recent_titles,
                "candidate_papers": candidates[:18],
                "conversation_context": context,
            },
            ensure_ascii=False,
        ),
        fallback={},
    )
    if not isinstance(payload, dict):
        return [], ""
    by_title = {normalize_lookup_text(item.get("title", "")): item for item in candidates}
    selected: list[dict[str, str]] = []
    raw_recommendations = payload.get("recommendations", [])
    if isinstance(raw_recommendations, list):
        for raw in raw_recommendations:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title", "") or "").strip()
            candidate = by_title.get(normalize_lookup_text(title))
            if candidate is None:
                continue
            reason = str(raw.get("reason", "") or "").strip() or candidate.get("reason", "")
            selected.append({**candidate, "reason": reason})
            if len(selected) >= limit:
                break
    note = str(payload.get("criteria_note", "") or "").strip()
    return selected, note


def library_recommendation_reason(*, title: str, year: str, summary: str, tags: list[str]) -> str:
    _ = title
    compact = " ".join(summary.split())
    if compact:
        if len(compact) > 140:
            compact = compact[:137].rstrip() + "..."
        return compact
    if tags:
        return f"它覆盖 {', '.join(tags[:3])} 等主题，适合作为进入当前库主题的切入口。"
    suffix = f"（{year}）" if year else ""
    return f"这是库中元数据完整的一篇论文{suffix}，适合作为默认起点。"
