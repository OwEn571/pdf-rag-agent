from __future__ import annotations

import re


FOLLOWUP_RECHECK_CUES = [
    "严格后续",
    "严格意义",
    "仔细",
    "确认",
    "确定",
    "真是",
    "是不是",
    "是吗",
    "后续工作",
    "strict",
    "really",
]

FOLLOWUP_SOFT_RELATION_CUES = [
    "extend",
    "extension",
    "transfer",
    "continual",
    "streaming",
    "behavioral",
    "follow-up",
    "subsequent",
]

FOLLOWUP_SUPPORT_RELATION_CUES = [
    "extend",
    "extension",
    "builds on",
    "based on",
    "uses",
    "evaluate",
    "benchmark",
    "transfer",
    "follow-up",
    "subsequent",
]

TARGET_RELATION_CUES = [
    "extend",
    "extends",
    "extension",
    "builds on",
    "based on",
    "uses",
    "using",
    "evaluate",
    "evaluates",
    "evaluation",
    "benchmark",
    "trained on",
    "derived from",
    "follow-up",
    "subsequent",
    "successor",
]

FOLLOWUP_DOMAIN_TOKENS = [
    "personalization",
    "personalized",
    "preference",
    "user-level",
    "persona",
    "alignment",
    "benchmark",
    "dataset",
    "inference",
    "transferable",
    "conditioned generation",
    "profile",
]

FOLLOWUP_RELEVANCE_WEIGHTS = {
    "preference inference": 1.2,
    "personalized preference": 1.1,
    "user-level alignment": 1.0,
    "transferable personalization": 1.0,
    "conditioned generation": 1.0,
    "personalized alignment": 0.9,
    "user preference": 0.8,
    "personalization": 0.7,
    "modular": 0.5,
    "benchmark": 0.3,
    "dataset": 0.3,
}

FOLLOWUP_SEED_INTRO_CUES = [
    "introduce",
    "introduces",
    "we introduce",
    "propose",
    "present",
    "dataset",
    "benchmark",
    "定义",
    "提出",
    "数据集",
]


def followup_relationship_recheck_requested(clean_query: str, normalized_query: str) -> bool:
    return any(cue in normalized_query or cue in clean_query for cue in FOLLOWUP_RECHECK_CUES)


def has_followup_soft_relation_signal(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(cue in lowered for cue in FOLLOWUP_SOFT_RELATION_CUES)


def has_followup_support_relation_signal(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(cue in lowered for cue in FOLLOWUP_SUPPORT_RELATION_CUES)


def target_relation_cue_near_text(*, text: str, target: str) -> bool:
    if not text or not target:
        return False
    lowered = str(text).lower()
    target_lower = target.lower()
    pattern = re.compile(rf"(?<![a-z0-9\-]){re.escape(target_lower)}(?![a-z0-9\-])")
    for match in pattern.finditer(lowered):
        window = lowered[max(0, match.start() - 90) : match.end() + 120]
        if any(cue in window for cue in TARGET_RELATION_CUES):
            return True
    return False


def has_followup_domain_signal(text: str) -> bool:
    lowered = str(text or "").lower()
    return sum(1 for token in FOLLOWUP_DOMAIN_TOKENS if token in lowered) >= 2


def followup_relevance_score(text: str) -> float:
    lowered = str(text or "").lower()
    return sum(weight for token, weight in FOLLOWUP_RELEVANCE_WEIGHTS.items() if token in lowered)


def has_followup_seed_intro_signal(text: str) -> bool:
    return any(token in str(text or "").lower() for token in FOLLOWUP_SEED_INTRO_CUES)
