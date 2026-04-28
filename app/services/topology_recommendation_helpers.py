from __future__ import annotations


def fallback_topology_recommendation(topology_terms: list[str]) -> dict[str, str]:
    terms_text = " / ".join(topology_terms) if topology_terms else "chain / tree / mesh / DAG"
    return {
        "overall_best": "",
        "engineering_best": "DAG",
        "rationale": f"当前证据主要覆盖这些 topology：{terms_text}；工程选择仍要看任务依赖、并行验证、可追溯性和节点调度成本。",
        "summary": f"当前证据讨论了 {terms_text} 等 topology，但没有给出脱离任务的绝对最优。",
    }


def is_unusable_topology_recommendation_text(text: str) -> bool:
    lowered = " ".join(str(text or "").lower().split())
    if not lowered:
        return True
    negative_markers = [
        "does not address",
        "does not contain",
        "impossible to determine",
        "no direct analysis",
        "not provide specific",
        "cannot determine",
        "无法确定",
        "不能确定",
        "没有覆盖",
        "不包含",
    ]
    return any(marker in lowered for marker in negative_markers)
