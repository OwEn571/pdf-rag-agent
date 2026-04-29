from __future__ import annotations

import hashlib
import json
import re
from typing import Any


CLARIFICATION_OPTION_SCHEMA_VERSION = "clarification_option.v1"

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


def clarification_option_public_payload(option: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": option.get("schema_version", CLARIFICATION_OPTION_SCHEMA_VERSION),
        "option_id": option.get("option_id", ""),
        "kind": option.get("kind", ""),
        "target": option.get("target", ""),
        "label": option.get("label", ""),
        "description": option.get("description", ""),
        "paper_id": option.get("paper_id", ""),
        "title": option.get("title", ""),
        "year": option.get("year", ""),
        "meaning": option.get("meaning", ""),
        "snippet": option.get("snippet", ""),
        "source": option.get("source", ""),
        "source_relation": option.get("source_relation", ""),
        "source_requested_fields": option.get("source_requested_fields", []),
        "source_answer_slots": option.get("source_answer_slots", []),
    }
    for key in [
        "display_title",
        "display_label",
        "display_reason",
        "judge_recommended",
        "disambiguation_confidence",
        "source_required_modalities",
    ]:
        if key in option:
            payload[key] = option.get(key)
    return payload


def clarification_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def clarification_option_description(option: dict[str, Any], *, title: str, year: str) -> str:
    meta = " · ".join(item for item in [title, year] if item)
    context = str(option.get("context_text", "") or option.get("snippet", "") or "").strip()
    context = " ".join(context.split())
    return context or meta


def clarification_option_id(
    *,
    kind: str,
    target: str,
    label: str,
    paper_id: str,
    title: str,
    index: int,
) -> str:
    seed = json.dumps(
        {
            "kind": kind,
            "target": target,
            "label": label,
            "paper_id": paper_id,
            "title": title,
            "index": index,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    prefix = re.sub(r"[^a-z0-9]+", "-", f"{kind}-{target}".lower()).strip("-") or "clarification"
    return f"{prefix}-{digest}"


def ambiguity_options_from_notes(notes: list[str]) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for note in notes:
        raw = str(note or "")
        if not raw.startswith("ambiguity_option="):
            continue
        try:
            payload = json.loads(raw.split("=", 1)[1])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("title"):
            options.append(payload)
    return options
