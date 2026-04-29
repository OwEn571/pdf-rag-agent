from __future__ import annotations

from app.services.followup_intents import (
    FOLLOWUP_INTENT_MARKERS,
    formula_query_allows_active_paper_context,
    is_formula_interpretation_followup_query,
    is_language_preference_followup,
    is_memory_synthesis_query,
    is_negative_correction_query,
    looks_like_active_paper_reference,
    looks_like_contextual_metric_query,
    looks_like_formula_answer_correction,
    looks_like_formula_location_correction,
    looks_like_paper_scope_correction,
)
from app.services.intent_marker_matching import query_matches_any


def test_followup_intent_markers_use_centralized_profiles() -> None:
    assert query_matches_any("不对", "不对", FOLLOWUP_INTENT_MARKERS["negative_correction"])
    assert query_matches_any(
        "怎么理解",
        "怎么理解",
        FOLLOWUP_INTENT_MARKERS["formula_interpretation"],
    )
    assert not query_matches_any("随便聊聊", "随便聊聊", FOLLOWUP_INTENT_MARKERS["language_research"])


def test_followup_intents_detect_corrections_and_context() -> None:
    assert looks_like_formula_answer_correction("我觉得不是这个公式哦")
    assert looks_like_paper_scope_correction("我问的是这篇论文中 PBA 的结果")
    assert looks_like_active_paper_reference("这篇论文中 PBA 和 ICA 的具体效果如何")
    assert looks_like_formula_location_correction("公式就在这篇论文里啊")
    assert is_negative_correction_query("我是说另一个PBA，不是这个")


def test_followup_intents_detect_memory_and_language_followups() -> None:
    assert is_memory_synthesis_query("你觉得两者的区别是什么")
    assert is_formula_interpretation_followup_query("怎么理解这个公式？", had_formula_context=True)
    assert not is_formula_interpretation_followup_query("怎么理解这个公式？", had_formula_context=False)
    assert is_language_preference_followup("你怎么回答还中英文混杂，我要中文", has_turns=True)
    assert not is_language_preference_followup("中文解释这个公式", has_turns=True)


def test_contextual_metric_query_requires_metric_signal_and_targets() -> None:
    assert looks_like_contextual_metric_query(
        "这篇论文中PBA和ICA的具体效果如何呢",
        targets=["PBA", "ICA"],
        is_short_acronym=lambda target: len(target) <= 4 and target.isupper(),
    )
    assert not looks_like_contextual_metric_query(
        "这篇论文主要说了什么",
        targets=["PBA"],
        is_short_acronym=lambda target: len(target) <= 4 and target.isupper(),
    )


def test_formula_query_allows_active_paper_context_from_cues_and_names() -> None:
    def normalize(text: str) -> str:
        return "".join(str(text or "").lower().split())

    assert formula_query_allows_active_paper_context(
        "那PBA的公式是什么",
        active_names=[],
        normalize_entity_key=normalize,
    )
    assert formula_query_allows_active_paper_context(
        "AlignX 里的 PBA 公式是什么",
        active_names=["AlignX"],
        normalize_entity_key=normalize,
    )
    assert not formula_query_allows_active_paper_context(
        "PBA 的公式是什么",
        active_names=["PersonaDual"],
        normalize_entity_key=normalize,
    )
