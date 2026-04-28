from __future__ import annotations

import re

from app.domain.models import SessionContext


def is_pdf_agent_topology_design_query(*, lowered: str, compact: str) -> bool:
    has_pdf_agent = (
        "pdf-agent" in lowered
        or "pdf agent" in lowered
        or "pdfagent" in compact
        or ("pdf" in lowered and ("agent" in lowered or "智能体" in lowered or "智能体" in compact))
    )
    if not has_pdf_agent:
        return False
    has_multi_agent = any(
        marker in lowered or marker in compact
        for marker in ["multi-agent", "multiagent", "多智能体", "智能体", "agents", "agent"]
    )
    has_design_signal = any(
        marker in lowered or marker in compact
        for marker in [
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
        ]
    )
    return has_multi_agent and has_design_signal


def is_memory_comparison_query(lowered: str) -> bool:
    return any(token in lowered for token in ["区别", "比较", "对比", "两者", "二者", "它们", "difference", "compare"])


def looks_like_memory_reference(query: str) -> bool:
    lowered = " ".join(str(query or "").lower().split())
    return contains_ordinal_reference(query) or any(
        token in lowered for token in ["上一轮", "上一条", "刚才", "上面", "列表", "它", "他", "这个", "这篇", "那篇", "这些"]
    )


def is_short_followup(query: str) -> bool:
    compact = re.sub(r"\s+", "", str(query or ""))
    return 0 < len(compact) <= 18 and (
        contains_ordinal_reference(query)
        or any(token in compact for token in ["呢", "这个", "具体", "变量", "公式", "结果", "图", "来源", "是啥", "是什么"])
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
    lowered = " ".join(str(query or "").lower().split())
    compact = re.sub(r"\s+", "", str(query or ""))
    return contains_ordinal_reference(query) or (
        len(compact) <= 24 and any(token in lowered or token in compact for token in ["上面", "列表", "刚才", "上一条", "上一轮"])
    )


def contains_ordinal_reference(query: str) -> bool:
    compact = re.sub(r"\s+", "", str(query or "").strip().lower())
    if not compact:
        return False
    if re.search(r"第(\d+|[一二三四五六七八九十两]+)(篇论文|篇文章|篇|个|项|条)?", compact):
        return True
    return any(
        token in compact
        for token in ["first", "1st", "second", "2nd", "third", "3rd", "fourth", "4th", "fifth", "5th"]
    )
