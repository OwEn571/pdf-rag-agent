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


def research_answer_slots(
    *,
    clean_query: str,
    lowered: str,
    compact: str,
    active_relation: str = "",
) -> list[str]:
    if any(marker in lowered or marker in compact for marker in ["后续", "followup", "follow-up", "扩展工作", "继承工作"]):
        return ["followup_research"]
    if looks_like_origin_lookup_query(clean_query):
        return ["origin"]
    if any(marker in lowered or marker in compact for marker in ["公式", "损失函数", "objective", "loss", "gradient", "梯度"]):
        return ["formula"]
    if any(marker in lowered or marker in compact for marker in ["figure", "fig.", "图", "caption", "可视化"]):
        return ["figure"]
    if any(marker in lowered or marker in compact for marker in ["结果", "实验", "核心结论", "贡献", "消融", "ablation", "performance"]):
        return ["paper_summary"]
    if any(marker in lowered or marker in compact for marker in ["多少", "数值", "准确率", "得分", "score", "accuracy", "metric"]):
        return ["metric_value"]
    if any(marker in lowered or marker in compact for marker in ["推荐", "哪些论文", "值得一看", "值得看", "入门", "papers to read"]):
        return ["paper_recommendation"]
    if any(marker in lowered or marker in compact for marker in ["拓扑", "topology", "langgraph"]):
        if any(
            marker in lowered or marker in compact
            for marker in [
                "哪种最好",
                "比较好",
                "推荐",
                "最应该",
                "应该用",
                "应该使用",
                "怎么组织",
                "如何组织",
                "怎样组织",
                "怎么设计",
                "如何设计",
                "适合",
                "选择",
            ]
        ):
            return ["topology_recommendation"]
        return ["topology_discovery"]
    if any(marker in lowered or marker in compact for marker in ["reward model", "奖励模型", "critic", "value model", "价值模型"]):
        return ["training_component"]
    if any(marker in clean_query for marker in ["是什么", "什么是", "什么意思", "定义"]) or any(
        marker in lowered for marker in ["what is", "what are", "definition"]
    ):
        return ["definition"]
    if active_relation == "formula_lookup" and any(marker in compact for marker in ["变量", "解释", "呢"]):
        return ["formula"]
    return ["general_answer"]


EXTERNAL_SEARCH_MARKERS = [
    "最新",
    "最近",
    "今天",
    "昨天",
    "新闻",
    "刚发布",
    "新论文",
    "近期论文",
    "arxiv",
    "latest",
    "recent",
    "today",
    "yesterday",
    "news",
    "new paper",
    "new papers",
    "newly released",
    "current",
    "引用数",
    "引用量",
    "被引",
    "citation count",
    "citations",
    "cited by",
    "most cited",
]

ROUTER_WEB_EXTRA_MARKERS = ["当前", "现在", "citation"]


def normalized_query_needs_external_search(
    lowered: str,
    compact: str,
    *,
    include_router_extras: bool = False,
) -> bool:
    markers = EXTERNAL_SEARCH_MARKERS
    if include_router_extras:
        markers = [*markers, *ROUTER_WEB_EXTRA_MARKERS]
    return any(marker in lowered or marker in compact for marker in markers)


def query_needs_external_search(query: str) -> bool:
    lowered, compact = _normalized_query_text(query)
    return normalized_query_needs_external_search(lowered, compact)
