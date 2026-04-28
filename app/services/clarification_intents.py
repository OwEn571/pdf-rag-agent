from __future__ import annotations

import re


CLARIFICATION_CHOICE_MARKERS = [
    "我说",
    "选",
    "选择",
    "就是",
    "应该是",
    "指的是",
    "the one",
    "choose",
    "select",
]

CLARIFICATION_ORDINAL_PATTERNS = [
    (
        0,
        ["第一个", "第一项", "第1个", "第 1 个", "第1项", "第 1 项", "first", "the first"],
    ),
    (
        1,
        ["第二个", "第二项", "第2个", "第 2 个", "第2项", "第 2 项", "second", "the second"],
    ),
    (
        2,
        ["第三个", "第三项", "第3个", "第 3 个", "第3项", "第 3 项", "third", "the third"],
    ),
    (
        3,
        ["第四个", "第四项", "第4个", "第 4 个", "第4项", "第 4 项", "fourth", "the fourth"],
    ),
]


def looks_like_clarification_choice_text(normalized_query: str) -> bool:
    return any(marker in normalized_query for marker in CLARIFICATION_CHOICE_MARKERS)


def pending_clarification_selection_index(query: str) -> int | None:
    compact = " ".join(str(query or "").strip().lower().split())
    if not compact:
        return None
    digit_match = re.search(r"(?<!\d)([1-9])(?!\d)", compact)
    if digit_match:
        return int(digit_match.group(1)) - 1
    for index, markers in CLARIFICATION_ORDINAL_PATTERNS:
        if any(marker in compact for marker in markers):
            return index
    return None
