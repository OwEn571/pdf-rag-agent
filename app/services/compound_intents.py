from __future__ import annotations

from typing import Any

from app.services.contract_normalization import normalize_lookup_text
from app.services.query_shaping import extract_targets


LIBRARY_COMPOUND_SCOPE_MARKERS = ["论文库", "知识库"]
LIBRARY_COMPOUND_COUNT_MARKERS = ["多少", "几篇", "有哪些", "列表"]
LIBRARY_COMPOUND_RECOMMEND_MARKERS = ["推荐", "值得", "哪篇"]

COMPARISON_CUES = ["两者", "区别", "比较", "对比", "不同"]
COMPOUND_CUES = ["分别", "各自", "同时", "顺便", "比较", "对比", "区别", "不同", "又", "以及", "和", " vs ", " versus "]
COMPOUND_TASK_CUES = ["公式", "结果", "实验", "指标", "是什么", "核心结论", "figure", "图"]


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
            any(marker in clean_query for marker in LIBRARY_COMPOUND_SCOPE_MARKERS)
            or "zotero" in normalized_query
        )
        and any(marker in clean_query for marker in LIBRARY_COMPOUND_COUNT_MARKERS)
        and any(marker in clean_query for marker in LIBRARY_COMPOUND_RECOMMEND_MARKERS)
    ):
        return True
    if any(cue in normalized_query or cue in clean_query for cue in COMPARISON_CUES) and has_memory_context:
        return True
    if target_count < 2:
        return False
    if any(cue in normalized_query for cue in COMPOUND_CUES) or any(cue in clean_query for cue in COMPOUND_CUES):
        return True
    return sum(1 for cue in COMPOUND_TASK_CUES if cue in normalized_query or cue in clean_query) >= 2


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
