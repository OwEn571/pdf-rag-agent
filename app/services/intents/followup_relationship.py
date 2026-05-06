from __future__ import annotations

import re

from app.services.intents.marker_matching import MarkerProfile, query_matches_any


FOLLOWUP_RECHECK_CUES: MarkerProfile = (
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
)

FOLLOWUP_SOFT_RELATION_CUES: MarkerProfile = (
    "extend",
    "extension",
    "transfer",
    "continual",
    "streaming",
    "behavioral",
    "follow-up",
    "subsequent",
)

FOLLOWUP_SUPPORT_RELATION_CUES: MarkerProfile = (
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
)

TARGET_RELATION_CUES: MarkerProfile = (
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
)

FOLLOWUP_DOMAIN_TOKENS: MarkerProfile = (
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
)

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

FOLLOWUP_SEED_INTRO_CUES: MarkerProfile = (
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
)


FOLLOWUP_RELATIONSHIP_MARKERS: dict[str, MarkerProfile] = {
    "recheck": FOLLOWUP_RECHECK_CUES,
    "soft_relation": FOLLOWUP_SOFT_RELATION_CUES,
    "support_relation": FOLLOWUP_SUPPORT_RELATION_CUES,
    "target_relation": TARGET_RELATION_CUES,
    "domain": FOLLOWUP_DOMAIN_TOKENS,
    "seed_intro": FOLLOWUP_SEED_INTRO_CUES,
}


def followup_relationship_recheck_requested(clean_query: str, normalized_query: str) -> bool:
    return query_matches_any(normalized_query, clean_query, FOLLOWUP_RELATIONSHIP_MARKERS["recheck"])


def has_followup_soft_relation_signal(text: str) -> bool:
    lowered = str(text or "").lower()
    return query_matches_any(lowered, "", FOLLOWUP_RELATIONSHIP_MARKERS["soft_relation"])


def has_followup_support_relation_signal(text: str) -> bool:
    lowered = str(text or "").lower()
    return query_matches_any(lowered, "", FOLLOWUP_RELATIONSHIP_MARKERS["support_relation"])


def target_relation_cue_near_text(*, text: str, target: str) -> bool:
    if not text or not target:
        return False
    lowered = str(text).lower()
    target_lower = target.lower()
    pattern = re.compile(rf"(?<![a-z0-9\-]){re.escape(target_lower)}(?![a-z0-9\-])")
    for match in pattern.finditer(lowered):
        window = lowered[max(0, match.start() - 90) : match.end() + 120]
        if query_matches_any(window, "", FOLLOWUP_RELATIONSHIP_MARKERS["target_relation"]):
            return True
    return False


def has_followup_domain_signal(text: str) -> bool:
    lowered = str(text or "").lower()
    return sum(1 for token in FOLLOWUP_RELATIONSHIP_MARKERS["domain"] if token in lowered) >= 2


def followup_relevance_score(text: str) -> float:
    lowered = str(text or "").lower()
    return sum(weight for token, weight in FOLLOWUP_RELEVANCE_WEIGHTS.items() if token in lowered)


def has_followup_seed_intro_signal(text: str) -> bool:
    return query_matches_any(
        str(text or "").lower(),
        "",
        FOLLOWUP_RELATIONSHIP_MARKERS["seed_intro"],
    )
