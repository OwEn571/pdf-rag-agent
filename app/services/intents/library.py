from __future__ import annotations

from app.domain.models import QueryContract, SessionContext
from app.services.intents.marker_matching import (
    MarkerProfile,
    marker_profile_map,
    normalized_query_text as normalize_query_text,
    query_matches_any,
)


LIBRARY_INTENT_MARKERS: dict[str, MarkerProfile] = marker_profile_map("library")


def normalized_query_text(query: str) -> tuple[str, str]:
    return normalize_query_text(query)


def has_library_scope(query: str) -> bool:
    normalized, compact = normalized_query_text(query)
    return query_matches_any(normalized, compact, LIBRARY_INTENT_MARKERS["scope"])


def is_library_status_query(query: str) -> bool:
    normalized, compact = normalized_query_text(query)
    has_scope = query_matches_any(normalized, compact, LIBRARY_INTENT_MARKERS["status_scope"])
    asks_list = query_matches_any(normalized, compact, LIBRARY_INTENT_MARKERS["list"])
    return has_scope and (is_library_count_query(query) or asks_list)


def is_library_count_query(query: str) -> bool:
    normalized, compact = normalized_query_text(query)
    return query_matches_any(normalized, compact, LIBRARY_INTENT_MARKERS["count"])


def is_library_recommendation_query(query: str) -> bool:
    normalized, compact = normalized_query_text(query)
    return query_matches_any(normalized, compact, LIBRARY_INTENT_MARKERS["recommendation"])


def is_scoped_library_recommendation_query(query: str) -> bool:
    return is_library_recommendation_query(query) and has_library_scope(query)


def library_status_contract(clean_query: str) -> QueryContract:
    return QueryContract(
        clean_query=clean_query,
        interaction_mode="conversation",
        relation="library_status",
        targets=[],
        requested_fields=[],
        required_modalities=[],
        answer_shape="bullets",
        precision_requirement="exact",
        continuation_mode="fresh",
        notes=["self_knowledge", "dynamic_library_stats"],
    )


def library_recommendation_contract(clean_query: str) -> QueryContract:
    return QueryContract(
        clean_query=clean_query,
        interaction_mode="conversation",
        relation="library_recommendation",
        targets=[],
        requested_fields=[],
        required_modalities=[],
        answer_shape="bullets",
        precision_requirement="normal",
        continuation_mode="fresh",
        notes=["self_knowledge", "dynamic_library_recommendation"],
    )


def library_query_prefers_previous_candidates(query: str) -> bool:
    normalized, compact = normalized_query_text(query)
    return not query_matches_any(normalized, compact, LIBRARY_INTENT_MARKERS["fresh_library_scope"])


def is_citation_query(query: str) -> bool:
    normalized, compact = normalized_query_text(query)
    return query_matches_any(normalized, compact, LIBRARY_INTENT_MARKERS["citation"])


def is_citation_ranking_query(query: str) -> bool:
    normalized, compact = normalized_query_text(query)
    has_citation = query_matches_any(normalized, compact, LIBRARY_INTENT_MARKERS["citation_ranking"])
    has_rank = query_matches_any(normalized, compact, LIBRARY_INTENT_MARKERS["ranking"])
    return has_citation and (has_rank or len(compact) <= 20)


def citation_ranking_has_library_context(*, clean_query: str, session: SessionContext) -> bool:
    if has_library_scope(clean_query):
        return True
    if is_library_recommendation_query(clean_query) or is_library_status_query(clean_query):
        return True
    for turn in reversed(session.turns[-4:]):
        if turn.relation == "library_recommendation":
            return True
        if turn.relation == "compound_query" and (
            "library_recommendation" in turn.requested_fields
            or "默认推荐" in turn.answer
            or "最值得" in turn.query
            or "推荐" in turn.answer
        ):
            return True
        if turn.relation == "library_citation_ranking":
            return True
    return False
