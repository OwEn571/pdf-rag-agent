from __future__ import annotations


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
