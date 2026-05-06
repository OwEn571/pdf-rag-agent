from __future__ import annotations

import re

from app.domain.models import SessionContext
from app.services.intents.marker_matching import (
    MarkerProfile,
    normalized_query_text,
    query_matches_any,
)


MEMORY_INTENT_MARKERS: dict[str, MarkerProfile] = {
    "pdf_agent_explicit": ("pdf-agent", "pdf agent", "pdfagent"),
    "pdf_agent_terms": ("agent", "智能体"),
    "multi_agent": ("multi-agent", "multiagent", "多智能体", "智能体", "agents", "agent"),
    "design_signal": (
        "拓扑",
        "topology",
        "组织",
        "设计",
        "通信",
        "交流",
        "交互式问答",
        "问答",
        "解析",
        "框架",
        "系统",
        "应该用",
        "最应该",
    ),
    "comparison": ("区别", "比较", "对比", "两者", "二者", "它们", "difference", "compare"),
    "memory_reference": (
        "上一轮",
        "上一条",
        "刚才",
        "上面",
        "列表",
        "它",
        "他",
        "这个",
        "这篇",
        "那篇",
        "这些",
    ),
    "short_followup": ("呢", "这个", "具体", "变量", "公式", "结果", "图", "来源", "是啥", "是什么"),
    "recent_tool_result": ("上面", "列表", "刚才", "上一条", "上一轮"),
    "ordinal": ("first", "1st", "second", "2nd", "third", "3rd", "fourth", "4th", "fifth", "5th"),
}


def is_pdf_agent_topology_design_query(*, lowered: str, compact: str) -> bool:
    has_pdf_agent = (
        query_matches_any(lowered, compact, MEMORY_INTENT_MARKERS["pdf_agent_explicit"])
        or ("pdf" in lowered and query_matches_any(lowered, compact, MEMORY_INTENT_MARKERS["pdf_agent_terms"]))
    )
    if not has_pdf_agent:
        return False
    has_multi_agent = query_matches_any(lowered, compact, MEMORY_INTENT_MARKERS["multi_agent"])
    has_design_signal = query_matches_any(lowered, compact, MEMORY_INTENT_MARKERS["design_signal"])
    return has_multi_agent and has_design_signal


def is_memory_comparison_query(lowered: str) -> bool:
    return query_matches_any(lowered, "", MEMORY_INTENT_MARKERS["comparison"])


def looks_like_memory_reference(query: str) -> bool:
    lowered, _ = normalized_query_text(query)
    return contains_ordinal_reference(query) or query_matches_any(
        lowered,
        "",
        MEMORY_INTENT_MARKERS["memory_reference"],
    )


def is_short_followup(query: str) -> bool:
    compact = re.sub(r"\s+", "", str(query or ""))
    return 0 < len(compact) <= 18 and (
        contains_ordinal_reference(query)
        or query_matches_any("", compact, MEMORY_INTENT_MARKERS["short_followup"])
    )


def looks_like_recent_tool_result_reference(query: str, *, session: SessionContext | None) -> bool:
    if session is None or not session.turns:
        return False
    memory = dict(session.working_memory or {})
    has_recent_list = isinstance(memory.get("last_displayed_list"), dict)
    if not has_recent_list:
        tool_results = [entry for entry in list(memory.get("tool_results", []) or []) if isinstance(entry, dict)]
        for item in reversed(tool_results):
            artifact = item.get("artifact")
            if isinstance(artifact, dict) and isinstance(artifact.get("items"), list):
                has_recent_list = True
                break
    if not has_recent_list:
        return False
    lowered, compact = normalized_query_text(query)
    return contains_ordinal_reference(query) or (
        len(compact) <= 24 and query_matches_any(lowered, compact, MEMORY_INTENT_MARKERS["recent_tool_result"])
    )


def contains_ordinal_reference(query: str) -> bool:
    compact = re.sub(r"\s+", "", str(query or "").strip().lower())
    if not compact:
        return False
    if re.search(r"第(\d+|[一二三四五六七八九十两]+)(篇论文|篇文章|篇|个|项|条)?", compact):
        return True
    return query_matches_any("", compact, MEMORY_INTENT_MARKERS["ordinal"])
