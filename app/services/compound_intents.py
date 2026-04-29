from __future__ import annotations

from typing import Any

from app.services.contract_normalization import normalize_lookup_text
from app.services.intent_marker_matching import MarkerProfile, query_matches_any
from app.services.query_shaping import extract_targets


COMPOUND_INTENT_MARKERS: dict[str, MarkerProfile] = {
    "library_scope": ("论文库", "知识库"),
    "library_count": ("多少", "几篇", "有哪些", "列表"),
    "library_recommendation": ("推荐", "值得", "哪篇"),
    "comparison": ("两者", "区别", "比较", "对比", "不同"),
    "compound": (
        "分别",
        "各自",
        "同时",
        "顺便",
        "比较",
        "对比",
        "区别",
        "不同",
        "又",
        "以及",
        "和",
        " vs ",
        " versus ",
    ),
    "compound_task": ("公式", "结果", "实验", "指标", "是什么", "核心结论", "figure", "图"),
}


def _matches_text(text: str, markers: MarkerProfile) -> bool:
    return query_matches_any(text, text, markers)


def should_try_compound_decomposition_heuristic(
    clean_query: str,
    *,
    normalized_query: str,
    target_count: int,
    has_memory_context: bool,
) -> bool:
    if len(normalized_query) < 8:
        return False
    if (
        (
            _matches_text(clean_query, COMPOUND_INTENT_MARKERS["library_scope"])
            or "zotero" in normalized_query
        )
        and _matches_text(clean_query, COMPOUND_INTENT_MARKERS["library_count"])
        and _matches_text(clean_query, COMPOUND_INTENT_MARKERS["library_recommendation"])
    ):
        return True
    if (
        query_matches_any(normalized_query, clean_query, COMPOUND_INTENT_MARKERS["comparison"])
        and has_memory_context
    ):
        return True
    if target_count < 2:
        return False
    if query_matches_any(normalized_query, clean_query, COMPOUND_INTENT_MARKERS["compound"]):
        return True
    return (
        sum(1 for cue in COMPOUND_INTENT_MARKERS["compound_task"] if cue in normalized_query or cue in clean_query)
        >= 2
    )


def should_try_compound_decomposition(
    clean_query: str,
    *,
    session: Any | None = None,
) -> bool:
    normalized = normalize_lookup_text(clean_query)
    memory = dict((getattr(session, "working_memory", None) if session is not None else {}) or {})
    bindings = dict(memory.get("target_bindings", {}) or {})
    active_targets = []
    if session is not None:
        active = session.effective_active_research()
        active_targets = list(getattr(active, "targets", []) or [])
    has_memory_context = bool(bindings or active_targets)
    target_count = len({target.lower() for target in extract_targets(clean_query)})
    return should_try_compound_decomposition_heuristic(
        clean_query,
        normalized_query=normalized,
        target_count=target_count,
        has_memory_context=has_memory_context,
    )
