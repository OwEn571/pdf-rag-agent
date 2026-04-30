from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.services.intent_marker_matching import MarkerProfile, query_matches_any
from app.services.library_intents import (
    is_citation_query,
    is_library_status_query,
    is_scoped_library_recommendation_query,
)
from app.services.memory_intents import is_memory_comparison_query


IntentKindValue = Literal["smalltalk", "meta_library", "research", "memory_op"]
TopicStateValue = Literal["continue", "switch", "new"]

FALLBACK_INTENT_MARKERS: dict[str, MarkerProfile] = {
    "previous_rationale": ("为什么选择", "为什么推荐", "推荐理由", "排序依据"),
}


@dataclass(frozen=True)
class FallbackIntentPayload:
    intent_kind: IntentKindValue
    topic_state: TopicStateValue
    needs_local_corpus: bool
    target_entities: list[str]
    user_goal: str
    answer_slots: list[str]
    confidence: float
    notes: list[str]
    needs_web: bool = False
    refers_previous_turn: bool = False
    target_aliases: list[str] = field(default_factory=list)
    active_topic: str = ""

    def as_intent_kwargs(self) -> dict[str, object]:
        return {
            "intent_kind": self.intent_kind,
            "topic_state": self.topic_state,
            "active_topic": self.active_topic,
            "needs_local_corpus": self.needs_local_corpus,
            "needs_web": self.needs_web,
            "refers_previous_turn": self.refers_previous_turn,
            "target_entities": list(self.target_entities),
            "target_aliases": list(self.target_aliases),
            "user_goal": self.user_goal,
            "answer_slots": list(self.answer_slots),
            "confidence": self.confidence,
            "notes": list(self.notes),
        }


def non_research_fallback_intent(
    *,
    clean_query: str,
    lowered: str,
    session_has_turns: bool,
    active_targets: list[str],
    extracted_targets: list[str],
) -> FallbackIntentPayload | None:
    if is_library_status_query(clean_query):
        return FallbackIntentPayload(
            intent_kind="meta_library",
            topic_state="new",
            needs_local_corpus=False,
            target_entities=[],
            user_goal="读取本地论文库状态。",
            answer_slots=["library_status"],
            confidence=0.9,
            notes=["local_intent_library_status"],
        )
    if is_scoped_library_recommendation_query(clean_query):
        return FallbackIntentPayload(
            intent_kind="meta_library",
            topic_state="continue" if session_has_turns else "new",
            needs_local_corpus=False,
            target_entities=[],
            user_goal="基于本地论文库给出阅读推荐。",
            answer_slots=["library_recommendation"],
            confidence=0.86,
            notes=["local_intent_library_recommendation"],
        )
    if is_citation_query(clean_query):
        return FallbackIntentPayload(
            intent_kind="meta_library",
            topic_state="continue" if session_has_turns else "new",
            needs_local_corpus=False,
            needs_web=True,
            refers_previous_turn=session_has_turns,
            target_entities=[],
            user_goal="对库内或上一轮候选按引用数排序。",
            answer_slots=["citation_ranking"],
            confidence=0.82,
            notes=["local_intent_citation_ranking"],
        )
    if is_memory_comparison_query(lowered) and session_has_turns:
        return FallbackIntentPayload(
            intent_kind="memory_op",
            topic_state="continue",
            needs_local_corpus=False,
            refers_previous_turn=True,
            target_entities=list(active_targets),
            user_goal="基于上一轮结果做比较或综合。",
            answer_slots=["comparison"],
            confidence=0.82,
            notes=["local_intent_memory_synthesis"],
        )
    if query_matches_any(lowered, "", FALLBACK_INTENT_MARKERS["previous_rationale"]) and session_has_turns:
        return FallbackIntentPayload(
            intent_kind="memory_op",
            topic_state="continue",
            needs_local_corpus=False,
            refers_previous_turn=True,
            target_entities=list(extracted_targets or active_targets),
            user_goal="解释上一轮工具输出的依据。",
            answer_slots=["previous_rationale"],
            confidence=0.84,
            notes=["local_intent_memory_followup"],
        )
    return None
