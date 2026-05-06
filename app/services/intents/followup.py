from __future__ import annotations

import re
from collections.abc import Callable

from app.services.intents.marker_matching import (
    MarkerProfile,
    marker_profile_map,
    normalized_query_text,
    query_matches_any,
)


FOLLOWUP_INTENT_MARKERS: dict[str, MarkerProfile] = marker_profile_map("followup")


def _normalized_query_text(query: str) -> tuple[str, str, str]:
    text = str(query or "")
    lowered, compact = normalized_query_text(text)
    return text, lowered, compact


def _matches_query_text(text: str, lowered: str, compact: str, markers: MarkerProfile) -> bool:
    return query_matches_any(lowered, compact, markers) or any(marker in text for marker in markers)


def is_memory_synthesis_query(query: str) -> bool:
    _, lowered, _ = _normalized_query_text(query)
    return query_matches_any(lowered, "", FOLLOWUP_INTENT_MARKERS["memory_synthesis"])


def is_formula_interpretation_followup_query(query: str, *, had_formula_context: bool) -> bool:
    if not had_formula_context:
        return False
    _, lowered, compact = _normalized_query_text(query)
    formula_reference = query_matches_any("", compact, FOLLOWUP_INTENT_MARKERS["formula_reference"])
    return formula_reference and query_matches_any(
        lowered,
        compact,
        FOLLOWUP_INTENT_MARKERS["formula_interpretation"],
    )


def is_language_preference_followup(query: str, *, has_turns: bool) -> bool:
    if not has_turns:
        return False
    _, lowered, compact = _normalized_query_text(query)
    if not query_matches_any(lowered, compact, FOLLOWUP_INTENT_MARKERS["language_preference"]):
        return False
    return not query_matches_any(lowered, compact, FOLLOWUP_INTENT_MARKERS["language_research"])


def looks_like_active_paper_reference(query: str) -> bool:
    text, lowered, compact = _normalized_query_text(query)
    return _matches_query_text(text, lowered, compact, FOLLOWUP_INTENT_MARKERS["active_paper_reference"])


def looks_like_formula_answer_correction(query: str) -> bool:
    text, lowered, compact = _normalized_query_text(query)
    return (
        _matches_query_text(text, lowered, compact, FOLLOWUP_INTENT_MARKERS["formula_correction"])
        and _matches_query_text(text, lowered, compact, FOLLOWUP_INTENT_MARKERS["formula"])
    )


def looks_like_paper_scope_correction(query: str) -> bool:
    text, lowered, compact = _normalized_query_text(query)
    return (
        _matches_query_text(text, lowered, compact, FOLLOWUP_INTENT_MARKERS["paper_scope_correction"])
        and _matches_query_text(text, lowered, compact, FOLLOWUP_INTENT_MARKERS["paper_scope"])
    )


def looks_like_contextual_metric_query(
    query: str,
    *,
    targets: list[str],
    is_short_acronym: Callable[[str], bool],
) -> bool:
    if not targets:
        return False
    text, lowered, compact = _normalized_query_text(query)
    if not _matches_query_text(text, lowered, compact, FOLLOWUP_INTENT_MARKERS["contextual_metric"]):
        return False
    acronym_targets = [target for target in targets if is_short_acronym(target)]
    return len(targets) >= 2 or bool(acronym_targets)


def is_metric_definition_followup_query(query: str, *, has_metric_context: bool) -> bool:
    if not has_metric_context:
        return False
    text, lowered, compact = _normalized_query_text(query)
    has_metric_reference = _matches_query_text(
        text,
        lowered,
        compact,
        FOLLOWUP_INTENT_MARKERS["metric_reference"],
    )
    has_definition_request = _matches_query_text(
        text,
        lowered,
        compact,
        FOLLOWUP_INTENT_MARKERS["metric_definition"],
    )
    return has_metric_reference and has_definition_request


def formula_query_allows_active_paper_context(
    query: str,
    *,
    active_names: list[str],
    normalize_entity_key: Callable[[str], str],
) -> bool:
    text, lowered, compact = _normalized_query_text(query)
    if _matches_query_text(text, lowered, compact, FOLLOWUP_INTENT_MARKERS["formula_active_context"]):
        return True
    query_key = normalize_entity_key(text)
    for name in active_names:
        name_key = normalize_entity_key(name)
        if len(name_key) >= 4 and name_key in query_key:
            return True
    return False


def looks_like_formula_location_correction(query: str) -> bool:
    text = " ".join(str(query or "").strip().split())
    lowered = text.lower()
    if not text:
        return False
    if _matches_query_text(text, lowered, "", FOLLOWUP_INTENT_MARKERS["formula_location"]):
        return True
    return bool(
        re.search(r"在\s*[A-Za-z0-9][^。？！?]{8,}\s*(?:中|里|里面)", text)
        or re.search(r"\bFrom\s+1[,0-9]*\s+Users\b", text, flags=re.IGNORECASE)
    )


def is_negative_correction_query(query: str) -> bool:
    lowered = str(query or "").lower()
    return query_matches_any(lowered, "", FOLLOWUP_INTENT_MARKERS["negative_correction"])
