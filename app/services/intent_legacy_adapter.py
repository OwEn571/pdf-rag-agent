from __future__ import annotations

from typing import Any

from app.services.intent_contract_adapter import answer_slots_from_relation
from app.services.intent_fallback_helpers import FallbackIntentPayload
from app.services.intent_marker_matching import MarkerProfile, query_matches_any
from app.services.query_shaping import fallback_query_targets
from app.services.research_intents import looks_like_origin_lookup_query


LEGACY_CONTRACT_ADAPTER_MARKERS: dict[str, MarkerProfile] = {
    "contextual_origin_refine": ("最早", "起源", "最初", "首次", "出处", "来源", "不对", "不是", "确定"),
    "strong_origin_refine": ("最早", "起源"),
}


def legacy_contract_payload_to_intent_payload(
    payload: dict[str, Any],
    *,
    clean_query: str = "",
) -> FallbackIntentPayload | None:
    relation = str(payload.get("relation", "") or "").strip()
    interaction_mode = str(payload.get("interaction_mode", "") or "").strip()
    if not relation or interaction_mode not in {"conversation", "research"}:
        return None
    targets = payload.get("targets", [])
    if not isinstance(targets, list):
        targets = []
    notes = payload.get("notes", [])
    if not isinstance(notes, list):
        notes = []
    requested_fields = payload.get("requested_fields", [])
    if not isinstance(requested_fields, list):
        requested_fields = []
    query_for_goal = clean_query or str(payload.get("clean_query", "") or "")
    slots = answer_slots_from_relation(relation)
    needs_local_corpus = interaction_mode == "research"
    continuation_mode = str(payload.get("continuation_mode", "") or "").strip()
    topic_state = "continue" if continuation_mode == "followup" else "switch" if continuation_mode == "context_switch" else "new"
    refers_previous_turn = topic_state == "continue"
    origin_like_goal = looks_like_origin_lookup_query(query_for_goal)
    has_explicit_origin_target = bool(targets) or bool(fallback_query_targets(query_for_goal))
    if origin_like_goal and relation in {
        "memory_followup",
        "clarify_user_intent",
        "correction_without_context",
        "general_question",
    } and has_explicit_origin_target:
        slots = ["origin"]
        needs_local_corpus = True
        refers_previous_turn = False
        topic_state = "new"
        notes = [*notes, "local_origin_lookup_override"]
    if relation == "memory_followup" and (
        any(str(field) in {"paper_content", "method", "experiments", "summary", "key_findings"} for field in requested_fields)
        or "needs_contextual_refine" in {str(item) for item in notes}
    ):
        slots = ["paper_summary"]
        needs_local_corpus = True
        refers_previous_turn = True
    if relation in {"clarify_user_intent", "correction_without_context"} and query_matches_any(
        query_for_goal,
        query_for_goal,
        LEGACY_CONTRACT_ADAPTER_MARKERS["contextual_origin_refine"],
    ):
        has_strong_origin_refine = query_matches_any(
            query_for_goal,
            query_for_goal,
            LEGACY_CONTRACT_ADAPTER_MARKERS["strong_origin_refine"],
        )
        slots = ["origin"] if origin_like_goal or has_strong_origin_refine else ["general_answer"]
        needs_local_corpus = True
        refers_previous_turn = not (origin_like_goal and has_explicit_origin_target)
        notes = [*notes, "needs_contextual_refine"] if refers_previous_turn else notes
    if interaction_mode == "conversation" and relation.startswith("library"):
        intent_kind = "meta_library"
    elif interaction_mode == "conversation" and relation.startswith("memory"):
        intent_kind = "memory_op"
    elif interaction_mode == "conversation" and not needs_local_corpus:
        intent_kind = "smalltalk"
    else:
        intent_kind = "memory_op" if relation in {"clarify_user_intent", "correction_without_context", "memory_followup"} else "research"
    return FallbackIntentPayload(
        intent_kind=intent_kind,  # type: ignore[arg-type]
        topic_state=topic_state,  # type: ignore[arg-type]
        active_topic=query_for_goal,
        needs_local_corpus=needs_local_corpus,
        needs_web=bool(payload.get("allow_web_search")) or "citation_ranking" in slots,
        refers_previous_turn=refers_previous_turn,
        target_entities=[str(item).strip() for item in targets if str(item).strip()],
        target_aliases=[],
        user_goal=query_for_goal,
        answer_slots=slots,
        confidence=0.82,
        notes=["legacy_router_payload", *[str(item) for item in notes]],
    )
