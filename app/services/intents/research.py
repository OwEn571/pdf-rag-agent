from __future__ import annotations

from app.services.intents.marker_matching import (
    MarkerProfile,
    marker_profile,
    marker_profile_map,
    normalized_query_text,
    query_matches_any,
)

ORIGIN_LOOKUP_MARKERS: MarkerProfile = marker_profile("research", "origin_lookup")
METRIC_VALUE_MARKERS: MarkerProfile = marker_profile("research", "metric_value")
SUMMARY_RESULTS_MARKERS: MarkerProfile = marker_profile("research", "summary_results")
RESEARCH_SLOT_MARKERS: dict[str, MarkerProfile] = marker_profile_map("research_slots")
TOPOLOGY_RECOMMENDATION_MARKERS: MarkerProfile = marker_profile("research", "topology_recommendation")
DEFINITION_QUERY_MARKERS: MarkerProfile = marker_profile("research", "definition_query")
DEFINITION_LOWERED_MARKERS: MarkerProfile = marker_profile("research", "definition_lowered")
FORMULA_FOLLOWUP_MARKERS: MarkerProfile = marker_profile("research", "formula_followup")
EXTERNAL_SEARCH_MARKERS: MarkerProfile = marker_profile("research", "external_search")
ROUTER_WEB_EXTRA_MARKERS: MarkerProfile = marker_profile("research", "router_web_extra")


def _normalized_query_text(query: str) -> tuple[str, str]:
    return normalized_query_text(query)


def looks_like_origin_lookup_query(query: str) -> bool:
    lowered, compact = _normalized_query_text(query)
    return query_matches_any(lowered, compact, ORIGIN_LOOKUP_MARKERS)


def looks_like_metric_value_query(query: str) -> bool:
    lowered, compact = _normalized_query_text(query)
    return query_matches_any(lowered, compact, METRIC_VALUE_MARKERS)


def looks_like_summary_results_query(query: str) -> bool:
    lowered, compact = _normalized_query_text(query)
    return query_matches_any(lowered, compact, SUMMARY_RESULTS_MARKERS)


def research_answer_slots(
    *,
    clean_query: str,
    lowered: str,
    compact: str,
    active_relation: str = "",
) -> list[str]:
    if query_matches_any(lowered, compact, RESEARCH_SLOT_MARKERS["followup_research"]):
        return ["followup_research"]
    if looks_like_origin_lookup_query(clean_query):
        return ["origin"]
    if looks_like_summary_results_query(clean_query):
        return ["paper_summary"]
    if query_matches_any(lowered, compact, RESEARCH_SLOT_MARKERS["formula"]):
        return ["formula"]
    if query_matches_any(lowered, compact, RESEARCH_SLOT_MARKERS["figure"]):
        return ["figure"]
    if query_matches_any(lowered, compact, RESEARCH_SLOT_MARKERS["paper_summary"]):
        return ["paper_summary"]
    if query_matches_any(lowered, compact, RESEARCH_SLOT_MARKERS["metric_value"]):
        return ["metric_value"]
    if query_matches_any(lowered, compact, RESEARCH_SLOT_MARKERS["paper_recommendation"]):
        return ["paper_recommendation"]
    if query_matches_any(lowered, compact, RESEARCH_SLOT_MARKERS["comparison"]):
        return ["comparison"]
    if query_matches_any(lowered, compact, RESEARCH_SLOT_MARKERS["topology"]):
        if query_matches_any(lowered, compact, TOPOLOGY_RECOMMENDATION_MARKERS):
            return ["topology_recommendation"]
        return ["topology_discovery"]
    if query_matches_any(lowered, compact, RESEARCH_SLOT_MARKERS["training_component"]):
        return ["training_component"]
    if any(marker in clean_query for marker in DEFINITION_QUERY_MARKERS) or query_matches_any(
        lowered,
        "",
        DEFINITION_LOWERED_MARKERS,
    ):
        return ["definition"]
    if active_relation == "formula_lookup" and query_matches_any("", compact, FORMULA_FOLLOWUP_MARKERS):
        return ["formula"]
    return ["general_answer"]


def normalized_query_needs_external_search(
    lowered: str,
    compact: str,
    *,
    include_router_extras: bool = False,
) -> bool:
    markers = EXTERNAL_SEARCH_MARKERS
    if include_router_extras:
        markers = (*markers, *ROUTER_WEB_EXTRA_MARKERS)
    return query_matches_any(lowered, compact, markers)


def query_needs_external_search(query: str) -> bool:
    lowered, compact = _normalized_query_text(query)
    return normalized_query_needs_external_search(lowered, compact)
