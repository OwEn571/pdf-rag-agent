from __future__ import annotations

import re
from collections.abc import Callable

from app.services.intent_marker_matching import (
    MarkerProfile,
    normalized_query_text,
    query_matches_any,
)


FOLLOWUP_INTENT_MARKERS: dict[str, MarkerProfile] = {
    "memory_synthesis": ("区别", "比较", "对比", "两者", "二者", "它们", "difference", "compare"),
    "formula_interpretation": (
        "怎么理解",
        "如何理解",
        "怎样理解",
        "怎么读",
        "如何读",
        "什么意思",
        "解释一下",
        "讲一下",
        "直觉",
        "intuition",
        "interpret",
        "understand",
    ),
    "formula_reference": ("这个公式", "该公式", "这个式子", "这条公式", "公式"),
    "language_preference": ("中文", "全中文", "不要英文", "中英文混杂", "用中文", "说中文", "chinese"),
    "language_research": ("公式", "是什么", "多少", "结果", "实验", "对比", "比较", "区别", "figure", "table"),
    "active_paper_reference": (
        "这篇论文",
        "这篇文章",
        "这篇",
        "该论文",
        "该文",
        "文中",
        "论文中",
        "论文里",
        "里面",
        "其中",
        "this paper",
        "the paper",
        "in this paper",
    ),
    "formula": ("公式", "式子", "目标函数", "损失", "objective", "loss", "formula"),
    "formula_correction": (
        "不是这个",
        "不是这个公式",
        "不应该是这个",
        "我觉得不是",
        "不对",
        "错了",
        "好像不是",
        "应该不是",
        "not this",
        "wrong formula",
    ),
    "paper_scope_correction": (
        "我问的是",
        "我问的就是",
        "问的是",
        "问的就是",
        "限定在",
        "不是这篇",
        "不是这个",
        "i mean",
        "i meant",
    ),
    "paper_scope": ("论文中", "论文里", "文中", "这篇", "该论文", "paper"),
    "contextual_metric": (
        "具体效果",
        "效果如何",
        "表现如何",
        "结果分别",
        "分别如何",
        "准确率",
        "得分",
        "指标",
        "win rate",
        "accuracy",
        "score",
        "performance",
    ),
    "metric_reference": (
        "准确度",
        "准确率",
        "指标",
        "分数",
        "得分",
        "结果",
        "accuracy",
        "metric",
        "score",
        "win rate",
    ),
    "metric_definition": (
        "怎么定义",
        "如何定义",
        "怎么计算",
        "如何计算",
        "怎么算",
        "计算方式",
        "统计口径",
        "评价口径",
        "定义",
        "defined",
        "definition",
        "calculated",
    ),
    "formula_active_context": (
        "那",
        "这篇",
        "这篇论文",
        "该论文",
        "文中",
        "其中",
        "里面",
        "这里",
        "上面",
        "刚才",
        "它",
        "this paper",
        "in this paper",
        "there",
    ),
    "formula_location": (
        "就在",
        "论文里",
        "论文中",
        "那篇",
        "这篇",
        "中啊",
        "里啊",
        "in the paper",
        "in ",
    ),
    "negative_correction": (
        "不是这个",
        "不是这篇",
        "不是它",
        "另一个",
        "不对",
        "错了",
        "不一样",
        "not this",
        "another",
        "different one",
    ),
}


def _normalized_query_text(query: str) -> tuple[str, str, str]:
    text = str(query or "")
    lowered, compact = normalized_query_text(text)
    return text, lowered, compact


def _matches_query_text(text: str, lowered: str, compact: str, markers: MarkerProfile) -> bool:
    return query_matches_any(lowered, compact, markers) or any(marker in text for marker in markers)


def is_memory_synthesis_query(query: str) -> bool:
    _, lowered, _ = _normalized_query_text(query)
    return query_matches_any(lowered, "", FOLLOWUP_INTENT_MARKERS["memory_synthesis"])


def is_formula_interpretation_followup_query(query: str, *, had_formula_context: bool) -> bool:
    if not had_formula_context:
        return False
    _, lowered, compact = _normalized_query_text(query)
    formula_reference = query_matches_any("", compact, FOLLOWUP_INTENT_MARKERS["formula_reference"])
    return formula_reference and query_matches_any(
        lowered,
        compact,
        FOLLOWUP_INTENT_MARKERS["formula_interpretation"],
    )


def is_language_preference_followup(query: str, *, has_turns: bool) -> bool:
    if not has_turns:
        return False
    _, lowered, compact = _normalized_query_text(query)
    if not query_matches_any(lowered, compact, FOLLOWUP_INTENT_MARKERS["language_preference"]):
        return False
    return not query_matches_any(lowered, compact, FOLLOWUP_INTENT_MARKERS["language_research"])


def looks_like_active_paper_reference(query: str) -> bool:
    text, lowered, compact = _normalized_query_text(query)
    return _matches_query_text(text, lowered, compact, FOLLOWUP_INTENT_MARKERS["active_paper_reference"])


def looks_like_formula_answer_correction(query: str) -> bool:
    text, lowered, compact = _normalized_query_text(query)
    return (
        _matches_query_text(text, lowered, compact, FOLLOWUP_INTENT_MARKERS["formula_correction"])
        and _matches_query_text(text, lowered, compact, FOLLOWUP_INTENT_MARKERS["formula"])
    )


def looks_like_paper_scope_correction(query: str) -> bool:
    text, lowered, compact = _normalized_query_text(query)
    return (
        _matches_query_text(text, lowered, compact, FOLLOWUP_INTENT_MARKERS["paper_scope_correction"])
        and _matches_query_text(text, lowered, compact, FOLLOWUP_INTENT_MARKERS["paper_scope"])
    )


def looks_like_contextual_metric_query(
    query: str,
    *,
    targets: list[str],
    is_short_acronym: Callable[[str], bool],
) -> bool:
    if not targets:
        return False
    text, lowered, compact = _normalized_query_text(query)
    if not _matches_query_text(text, lowered, compact, FOLLOWUP_INTENT_MARKERS["contextual_metric"]):
        return False
    acronym_targets = [target for target in targets if is_short_acronym(target)]
    return len(targets) >= 2 or bool(acronym_targets)


def is_metric_definition_followup_query(query: str, *, has_metric_context: bool) -> bool:
    if not has_metric_context:
        return False
    text, lowered, compact = _normalized_query_text(query)
    has_metric_reference = _matches_query_text(
        text,
        lowered,
        compact,
        FOLLOWUP_INTENT_MARKERS["metric_reference"],
    )
    has_definition_request = _matches_query_text(
        text,
        lowered,
        compact,
        FOLLOWUP_INTENT_MARKERS["metric_definition"],
    )
    return has_metric_reference and has_definition_request


def formula_query_allows_active_paper_context(
    query: str,
    *,
    active_names: list[str],
    normalize_entity_key: Callable[[str], str],
) -> bool:
    text, lowered, compact = _normalized_query_text(query)
    if _matches_query_text(text, lowered, compact, FOLLOWUP_INTENT_MARKERS["formula_active_context"]):
        return True
    query_key = normalize_entity_key(text)
    for name in active_names:
        name_key = normalize_entity_key(name)
        if len(name_key) >= 4 and name_key in query_key:
            return True
    return False


def looks_like_formula_location_correction(query: str) -> bool:
    text = " ".join(str(query or "").strip().split())
    lowered = text.lower()
    if not text:
        return False
    if _matches_query_text(text, lowered, "", FOLLOWUP_INTENT_MARKERS["formula_location"]):
        return True
    return bool(
        re.search(r"在\s*[A-Za-z0-9][^。？！?]{8,}\s*(?:中|里|里面)", text)
        or re.search(r"\bFrom\s+1[,0-9]*\s+Users\b", text, flags=re.IGNORECASE)
    )


def is_negative_correction_query(query: str) -> bool:
    lowered = str(query or "").lower()
    return query_matches_any(lowered, "", FOLLOWUP_INTENT_MARKERS["negative_correction"])
