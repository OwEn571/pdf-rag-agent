from __future__ import annotations


def _normalized_query_text(query: str) -> tuple[str, str]:
    normalized = " ".join(str(query or "").lower().split())
    return normalized, normalized.replace(" ", "")


def looks_like_origin_lookup_query(query: str) -> bool:
    lowered, compact = _normalized_query_text(query)
    markers = [
        "最先",
        "最早",
        "最初",
        "首次",
        "第一个提出",
        "第一篇提出",
        "第一篇论文",
        "第一个引入",
        "第一篇引入",
        "最初的论文",
        "最初论文",
        "哪篇论文提出",
        "哪篇提出",
        "谁提出",
        "提出的第一篇",
        "first proposed",
        "first introduced",
        "origin",
    ]
    return any(marker in lowered or marker in compact for marker in markers)


def looks_like_metric_value_query(query: str) -> bool:
    lowered, compact = _normalized_query_text(query)
    markers = [
        "具体效果",
        "效果如何",
        "表现如何",
        "结果分别",
        "准确率",
        "得分",
        "数值",
        "多少",
        "score",
        "accuracy",
        "metric",
        "win rate",
    ]
    return any(marker in lowered or marker in compact for marker in markers)


def looks_like_summary_results_query(query: str) -> bool:
    lowered, compact = _normalized_query_text(query)
    markers = [
        "主要结论",
        "核心结论",
        "一句话结论",
        "什么结论",
        "数据支持",
        "用什么数据支持",
        "实验结果",
        "贡献",
        "summary",
        "result",
        "results",
        "contribution",
    ]
    return any(marker in lowered or marker in compact for marker in markers)
