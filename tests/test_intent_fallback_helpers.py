from __future__ import annotations

from app.services.intent_fallback_helpers import non_research_fallback_intent, research_fallback_intent


def test_non_research_fallback_detects_library_status_and_citation_ranking() -> None:
    status = non_research_fallback_intent(
        clean_query="我的论文库里有多少篇论文？",
        lowered="我的论文库里有多少篇论文？",
        session_has_turns=False,
        active_targets=[],
        extracted_targets=[],
    )
    citation = non_research_fallback_intent(
        clean_query="按引用数排序",
        lowered="按引用数排序",
        session_has_turns=True,
        active_targets=["DPO"],
        extracted_targets=[],
    )

    assert status is not None
    assert status.intent_kind == "meta_library"
    assert status.answer_slots == ["library_status"]
    assert citation is not None
    assert citation.needs_web
    assert citation.refers_previous_turn
    assert citation.answer_slots == ["citation_ranking"]


def test_non_research_fallback_detects_memory_comparison_and_previous_rationale() -> None:
    comparison = non_research_fallback_intent(
        clean_query="它们有什么区别？",
        lowered="它们有什么区别？",
        session_has_turns=True,
        active_targets=["DPO", "PPO"],
        extracted_targets=[],
    )
    rationale = non_research_fallback_intent(
        clean_query="为什么推荐这个？",
        lowered="为什么推荐这个？",
        session_has_turns=True,
        active_targets=["DPO"],
        extracted_targets=["PBA"],
    )

    assert comparison is not None
    assert comparison.answer_slots == ["comparison"]
    assert comparison.target_entities == ["DPO", "PPO"]
    assert rationale is not None
    assert rationale.answer_slots == ["previous_rationale"]
    assert rationale.target_entities == ["PBA"]


def test_research_fallback_builds_research_and_memory_payloads() -> None:
    fresh = research_fallback_intent(
        clean_query="PBA 准确率多少",
        lowered="pba 准确率多少",
        compact="pba准确率多少",
        active_relation="",
        active_targets=[],
        extracted_targets=["PBA"],
        refers_previous=False,
    )
    followup = research_fallback_intent(
        clean_query="变量呢",
        lowered="变量呢",
        compact="变量呢",
        active_relation="formula_lookup",
        active_targets=["DPO"],
        extracted_targets=[],
        refers_previous=True,
    )

    assert fresh.intent_kind == "research"
    assert fresh.answer_slots == ["metric_value"]
    assert fresh.target_aliases == ["L_PBA", "LPBA", "L_{PBA}", "L_{\\mathrm{PBA}}"]
    assert followup.intent_kind == "memory_op"
    assert followup.topic_state == "continue"
    assert followup.answer_slots == ["formula"]
    assert followup.target_entities == ["DPO"]
