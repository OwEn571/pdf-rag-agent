from __future__ import annotations

from app.services.research_intents import (
    looks_like_metric_value_query,
    looks_like_origin_lookup_query,
    looks_like_summary_results_query,
    normalized_query_needs_external_search,
    query_needs_external_search,
)


def test_research_intent_classifier_detects_origin_metric_and_summary() -> None:
    assert looks_like_origin_lookup_query("DPO 最早是哪篇论文提出的？")
    assert looks_like_metric_value_query("PPO 的 win rate 具体多少？")
    assert looks_like_summary_results_query("这篇论文的核心结论和实验结果是什么？")
    assert not looks_like_origin_lookup_query("帮我总结一下这篇论文")


def test_research_intent_classifier_detects_external_search_need() -> None:
    assert query_needs_external_search("最新的多模态 RAG 论文有哪些？")
    assert query_needs_external_search("DPO citation count 是多少？")
    assert not query_needs_external_search("citation graph 的定义是什么？")
    assert normalized_query_needs_external_search(
        "citation graph",
        "citationgraph",
        include_router_extras=True,
    )
    assert not query_needs_external_search("这篇论文的主要结论是什么？")
