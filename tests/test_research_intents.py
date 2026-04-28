from __future__ import annotations

from app.services.research_intents import (
    looks_like_metric_value_query,
    looks_like_origin_lookup_query,
    looks_like_summary_results_query,
)


def test_research_intent_classifier_detects_origin_metric_and_summary() -> None:
    assert looks_like_origin_lookup_query("DPO 最早是哪篇论文提出的？")
    assert looks_like_metric_value_query("PPO 的 win rate 具体多少？")
    assert looks_like_summary_results_query("这篇论文的核心结论和实验结果是什么？")
    assert not looks_like_origin_lookup_query("帮我总结一下这篇论文")
