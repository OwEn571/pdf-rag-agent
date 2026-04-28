from __future__ import annotations

import re
from collections.abc import Callable


def _normalized_query_text(query: str) -> tuple[str, str, str]:
    text = str(query or "")
    lowered = " ".join(text.lower().split())
    compact = re.sub(r"\s+", "", text.lower())
    return text, lowered, compact


def is_memory_synthesis_query(query: str) -> bool:
    _, lowered, _ = _normalized_query_text(query)
    return any(token in lowered for token in ["区别", "比较", "对比", "两者", "二者", "它们", "difference", "compare"])


def is_formula_interpretation_followup_query(query: str, *, had_formula_context: bool) -> bool:
    if not had_formula_context:
        return False
    _, lowered, compact = _normalized_query_text(query)
    interpretation_cues = [
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
    ]
    formula_reference = any(marker in compact for marker in ["这个公式", "该公式", "这个式子", "这条公式", "公式"])
    return formula_reference and any(marker in lowered or marker in compact for marker in interpretation_cues)


def is_language_preference_followup(query: str, *, has_turns: bool) -> bool:
    if not has_turns:
        return False
    _, lowered, compact = _normalized_query_text(query)
    language_cues = ["中文", "全中文", "不要英文", "中英文混杂", "用中文", "说中文", "chinese"]
    if not any(cue in lowered or cue in compact for cue in language_cues):
        return False
    research_cues = ["公式", "是什么", "多少", "结果", "实验", "对比", "比较", "区别", "figure", "table"]
    return not any(cue in lowered or cue in compact for cue in research_cues)


def looks_like_active_paper_reference(query: str) -> bool:
    text, lowered, compact = _normalized_query_text(query)
    markers = [
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
    ]
    return any(marker in lowered or marker in compact or marker in text for marker in markers)


def looks_like_formula_answer_correction(query: str) -> bool:
    text, lowered, compact = _normalized_query_text(query)
    formula_markers = ["公式", "式子", "目标函数", "损失", "objective", "loss", "formula"]
    correction_markers = [
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
    ]
    return (
        any(marker in lowered or marker in compact or marker in text for marker in correction_markers)
        and any(marker in lowered or marker in compact or marker in text for marker in formula_markers)
    )


def looks_like_paper_scope_correction(query: str) -> bool:
    text, lowered, compact = _normalized_query_text(query)
    correction_markers = [
        "我问的是",
        "我问的就是",
        "问的是",
        "问的就是",
        "限定在",
        "不是这篇",
        "不是这个",
        "i mean",
        "i meant",
    ]
    scope_markers = ["论文中", "论文里", "文中", "这篇", "该论文", "paper"]
    return (
        any(marker in lowered or marker in compact or marker in text for marker in correction_markers)
        and any(marker in lowered or marker in compact or marker in text for marker in scope_markers)
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
    markers = [
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
    ]
    if not any(marker in lowered or marker in compact or marker in text for marker in markers):
        return False
    acronym_targets = [target for target in targets if is_short_acronym(target)]
    return len(targets) >= 2 or bool(acronym_targets)


def looks_like_formula_location_correction(query: str) -> bool:
    text = " ".join(str(query or "").strip().split())
    lowered = text.lower()
    if not text:
        return False
    markers = [
        "就在",
        "论文里",
        "论文中",
        "那篇",
        "这篇",
        "中啊",
        "里啊",
        "in the paper",
        "in ",
    ]
    if any(marker in lowered or marker in text for marker in markers):
        return True
    return bool(
        re.search(r"在\s*[A-Za-z0-9][^。？！?]{8,}\s*(?:中|里|里面)", text)
        or re.search(r"\bFrom\s+1[,0-9]*\s+Users\b", text, flags=re.IGNORECASE)
    )


def is_negative_correction_query(query: str) -> bool:
    lowered = str(query or "").lower()
    markers = [
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
    ]
    return any(marker in lowered for marker in markers)
