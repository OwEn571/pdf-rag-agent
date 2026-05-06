from __future__ import annotations

from app.services.intents.followup import FOLLOWUP_INTENT_MARKERS, looks_like_formula_answer_correction
from app.services.intents.marker_matching import marker_profile, marker_profile_map
from app.services.intents.library import is_citation_ranking_query, is_library_recommendation_query
from app.services.intents.research import looks_like_origin_lookup_query, research_answer_slots


def test_intent_marker_profiles_load_research_library_and_followup_sections() -> None:
    assert "source paper" in marker_profile("research", "origin_lookup")
    assert "recommend" in marker_profile_map("library")["recommendation"]
    assert "wrong formula" in marker_profile_map("followup")["formula_correction"]


def test_research_intents_use_profile_backed_origin_and_slots() -> None:
    assert looks_like_origin_lookup_query("DeepSeek R1 的 source paper 是哪篇")
    lowered = "alignx 核心结论和实验结果"
    assert research_answer_slots(clean_query=lowered, lowered=lowered, compact=lowered) == ["paper_summary"]


def test_library_and_followup_intents_use_profile_backed_markers() -> None:
    assert is_library_recommendation_query("论文库里最值得一读的是哪篇")
    assert is_citation_ranking_query("按 citation count 排序")
    assert looks_like_formula_answer_correction("不是这个公式，不对")
    assert "negative_correction" in FOLLOWUP_INTENT_MARKERS
