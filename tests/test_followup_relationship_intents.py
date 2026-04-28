from __future__ import annotations

from app.services.followup_relationship_intents import (
    followup_relationship_recheck_requested,
    followup_relevance_score,
    has_followup_domain_signal,
    has_followup_seed_intro_signal,
    has_followup_soft_relation_signal,
    has_followup_support_relation_signal,
    target_relation_cue_near_text,
)


def test_followup_relationship_recheck_requested_detects_strict_followups() -> None:
    assert followup_relationship_recheck_requested("确认一下是不是严格后续工作", "确认一下是不是严格后续工作")
    assert followup_relationship_recheck_requested(
        "is it really a strict follow-up?",
        "is it really a strict follow-up?",
    )
    assert not followup_relationship_recheck_requested("有哪些后续论文？", "有哪些后续论文？")


def test_followup_relation_signals_detect_soft_support_and_target_windows() -> None:
    text = "The new benchmark builds on AlignX and evaluates personalized preference inference."

    assert has_followup_soft_relation_signal("A subsequent personalization benchmark")
    assert has_followup_support_relation_signal(text)
    assert target_relation_cue_near_text(text=text, target="AlignX")
    assert not target_relation_cue_near_text(text="AlignX is mentioned without relation words.", target="AlignX")


def test_followup_domain_relevance_and_intro_signals() -> None:
    text = "We introduce a user-level alignment benchmark for personalized preference inference."

    assert has_followup_domain_signal(text)
    assert followup_relevance_score(text) > 2.0
    assert has_followup_seed_intro_signal(text)
    assert not has_followup_domain_signal("A generic retrieval paper")
