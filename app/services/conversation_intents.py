from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.intent_marker_matching import MarkerProfile, query_matches_any


CONVERSATION_INTENT_MARKERS: dict[str, MarkerProfile] = {
    "self_identity": ("你是谁", "who are you", "你的身份"),
    "capability": ("你能做什么", "有什么功能", "capability", "abilities"),
}

GREETING_QUERIES: set[str] = {
    "你好",
    "您好",
    "你好吗",
    "嗨",
    "嗨嗨",
    "哈喽",
    "哈啰",
    "hello",
    "hi",
    "hey",
    "yo",
    "在吗",
    "早上好",
    "下午好",
    "晚上好",
}

SHORT_CLARIFICATION_QUERIES: set[str] = {"何意味", "什么意思", "啥意思"}


@dataclass(frozen=True)
class ProtectedConversationIntent:
    user_goal: str
    answer_slots: list[str]
    confidence: float
    notes: list[str]
    ambiguous_slots: list[str] = field(default_factory=list)


def compact_conversation_query(query: str) -> str:
    return re.sub(r"[\s,.!?。！？~～，、；;：:]+", "", str(query or "").strip().lower())


def protected_conversation_intent(query: str) -> ProtectedConversationIntent | None:
    compact = compact_conversation_query(query)
    lowered = " ".join(str(query or "").strip().lower().split())
    if not compact:
        return ProtectedConversationIntent(
            user_goal="用户没有输入有效内容，需要澄清。",
            answer_slots=["clarify"],
            confidence=0.95,
            notes=["local_protected_empty_query"],
        )
    punctuation_stripped = re.sub(r"[\s,.!?。！？~～]+", "", lowered)
    if punctuation_stripped in GREETING_QUERIES:
        return ProtectedConversationIntent(
            user_goal="回应用户寒暄。",
            answer_slots=["greeting"],
            confidence=0.99,
            notes=["local_protected_greeting"],
        )
    if query_matches_any(lowered, "", CONVERSATION_INTENT_MARKERS["self_identity"]):
        return ProtectedConversationIntent(
            user_goal="介绍助手身份。",
            answer_slots=["self_identity"],
            confidence=0.96,
            notes=["local_protected_self_identity"],
        )
    if query_matches_any(lowered, "", CONVERSATION_INTENT_MARKERS["capability"]):
        return ProtectedConversationIntent(
            user_goal="介绍助手能力范围。",
            answer_slots=["capability"],
            confidence=0.96,
            notes=["local_protected_capability"],
        )
    if compact in SHORT_CLARIFICATION_QUERIES:
        return ProtectedConversationIntent(
            user_goal="用户表达过短，需要补充想问的对象。",
            answer_slots=["clarify"],
            confidence=0.88,
            ambiguous_slots=["missing_target"],
            notes=["local_protected_short_clarification"],
        )
    return None
