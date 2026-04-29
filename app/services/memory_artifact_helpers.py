from __future__ import annotations

import re
from typing import Any

from app.domain.models import SessionContext
from app.services.session_context_helpers import truncate_context_text


def conversation_tool_result_artifact(*, tool: str, result: dict[str, Any]) -> dict[str, Any]:
    if tool != "query_library_metadata" or not isinstance(result, dict):
        return {}
    rows = [dict(item) for item in list(result.get("rows", []) or []) if isinstance(item, dict)]
    items: list[dict[str, Any]] = []
    for index, row in enumerate(rows[:80], start=1):
        compact_row: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or isinstance(value, (int, float)):
                compact_row[str(key)] = value
                continue
            compact_row[str(key)] = truncate_context_text(str(value), limit=900)
        item = {
            "ordinal": index,
            "row": compact_row,
        }
        for key in ["paper_id", "title", "year", "year_int", "authors", "author"]:
            if key in compact_row:
                item[key] = compact_row[key]
        items.append(item)
    return {
        "type": "tabular_sql_result",
        "tool": tool,
        "sql": truncate_context_text(str(result.get("sql", "") or ""), limit=1200),
        "columns": [str(item) for item in list(result.get("columns", []) or [])],
        "row_count": int(result.get("row_count", len(rows)) or 0),
        "truncated": bool(result.get("truncated", False)),
        "items": items,
    }


def answer_from_recent_tool_artifact_reference(*, query: str, session: SessionContext) -> str:
    item_index = referenced_list_item_index(query)
    if item_index is None:
        return ""
    artifact = latest_list_tool_artifact(session)
    if not artifact:
        return ""
    items = [item for item in list(artifact.get("items", []) or []) if isinstance(item, dict)]
    if item_index < 0:
        return ""
    if item_index >= len(items):
        source_query = str(artifact.get("query", "") or "上一轮工具结果").strip()
        return f"上一轮“{source_query}”只保留了 {len(items)} 条结构化结果，找不到第 {item_index + 1} 条。"
    item = items[item_index]
    row = dict(item.get("row", {}) or {})
    ordinal = int(item.get("ordinal", item_index + 1) or item_index + 1)
    source_query = str(artifact.get("query", "") or "").strip()
    source_tool = str(artifact.get("tool", "") or "tool").strip()
    lines = []
    if source_query:
        lines.append(f"按上一轮“{source_query}”的 `{source_tool}` 结果，第 {ordinal} 条是：")
    else:
        lines.append(f"按上一轮 `{source_tool}` 结果，第 {ordinal} 条是：")
    lines.append("")
    title = str(row.get("title", "") or row.get("paper_title", "") or "").strip()
    if title:
        lines.append(f"- 标题：{title}")
    for key, label in [
        ("year", "年份"),
        ("year_int", "年份"),
        ("authors", "作者"),
        ("author", "作者"),
        ("paper_id", "paper_id"),
        ("categories", "分类"),
        ("tags", "标签"),
    ]:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text or (key == "year_int" and str(row.get("year", "") or "").strip()):
            continue
        lines.append(f"- {label}：{text}")
    if len(lines) <= 2:
        for key, value in row.items():
            text = "" if value is None else str(value).strip()
            if text:
                lines.append(f"- {key}：{text}")
    return "\n".join(lines).strip()


def latest_list_tool_artifact(session: SessionContext) -> dict[str, Any]:
    memory = dict(session.working_memory or {})
    direct = memory.get("last_displayed_list")
    if isinstance(direct, dict) and isinstance(direct.get("items"), list):
        return dict(direct)
    for result in reversed([item for item in list(memory.get("tool_results", []) or []) if isinstance(item, dict)]):
        artifact = result.get("artifact")
        if isinstance(artifact, dict) and isinstance(artifact.get("items"), list):
            merged = dict(artifact)
            merged.setdefault("query", result.get("query", ""))
            merged.setdefault("tool", result.get("tool", ""))
            return merged
    return {}


def referenced_list_item_index(query: str) -> int | None:
    compact = re.sub(r"\s+", "", str(query or "").strip().lower())
    if not compact:
        return None
    digit_match = re.search(r"第(\d+)(篇|个|项|条|篇论文|篇文章)?", compact)
    if digit_match:
        return max(0, int(digit_match.group(1)) - 1)
    chinese_match = re.search(r"第([一二三四五六七八九十两]+)(篇|个|项|条|篇论文|篇文章)?", compact)
    if chinese_match:
        value = chinese_ordinal_value(chinese_match.group(1))
        if value is not None:
            return max(0, value - 1)
    english_ordinals = {
        "first": 1,
        "1st": 1,
        "second": 2,
        "2nd": 2,
        "third": 3,
        "3rd": 3,
        "fourth": 4,
        "4th": 4,
        "fifth": 5,
        "5th": 5,
        "sixth": 6,
        "6th": 6,
        "seventh": 7,
        "7th": 7,
        "eighth": 8,
        "8th": 8,
        "ninth": 9,
        "9th": 9,
        "tenth": 10,
        "10th": 10,
    }
    for token, value in english_ordinals.items():
        if token in compact:
            return value - 1
    return None


def chinese_ordinal_value(text: str) -> int | None:
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw == "十":
        return 10
    if "十" in raw:
        left, _, right = raw.partition("十")
        tens = digits.get(left, 1 if left == "" else 0)
        ones = digits.get(right, 0) if right else 0
        value = tens * 10 + ones
        return value if value > 0 else None
    return digits.get(raw)
