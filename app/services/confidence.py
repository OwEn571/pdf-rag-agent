from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.domain.models import QueryContract


@dataclass(frozen=True, slots=True)
class Confidence:
    score: float
    basis: str
    detail: dict[str, Any] = field(default_factory=dict)


def confidence_from_contract(contract: QueryContract) -> Confidence:
    notes = [str(item) for item in list(contract.notes or [])]
    ambiguous_slots = [
        note.split("=", 1)[1]
        for note in notes
        if note.startswith("ambiguous_slot=") and "=" in note
    ]
    explicit_score = _note_float(notes=notes, prefix="intent_confidence=")
    forced_clarify = bool(ambiguous_slots) or "low_intent_confidence" in notes or "intent_needs_clarification" in notes
    if forced_clarify:
        return Confidence(
            score=0.0,
            basis="contract_clarification_notes",
            detail={"ambiguous_slots": ambiguous_slots, "notes": notes},
        )
    if explicit_score is not None:
        return Confidence(
            score=max(0.0, min(1.0, explicit_score)),
            basis="intent_confidence_note",
            detail={"notes": notes},
        )
    return Confidence(score=1.0, basis="implicit_high_confidence", detail={"notes": notes})


def should_ask_human(confidence: Confidence, settings: Any) -> bool:
    floor = getattr(settings, "confidence_floor", 0.6)
    try:
        parsed_floor = float(floor)
    except (TypeError, ValueError):
        parsed_floor = 0.6
    return confidence.score < max(0.0, min(1.0, parsed_floor))


def confidence_from_self_consistency(samples: list[str], *, min_samples: int = 2) -> Confidence:
    normalized = [" ".join(str(item or "").split()) for item in list(samples or [])]
    normalized = [item for item in normalized if item]
    if len(normalized) < max(2, min_samples):
        return Confidence(
            score=0.5 if normalized else 0.0,
            basis="self_consistency",
            detail={"sample_count": len(normalized), "reason": "insufficient_samples"},
        )
    token_sets = [_meaningful_terms(item) for item in normalized]
    pairwise_scores: list[float] = []
    for index, left in enumerate(token_sets):
        for right in token_sets[index + 1 :]:
            pairwise_scores.append(_jaccard(left, right))
    score = sum(pairwise_scores) / len(pairwise_scores) if pairwise_scores else 0.0
    return Confidence(
        score=max(0.0, min(1.0, score)),
        basis="self_consistency",
        detail={
            "sample_count": len(normalized),
            "pairwise_scores": [round(item, 4) for item in pairwise_scores],
            "min_pairwise": round(min(pairwise_scores), 4) if pairwise_scores else 0.0,
        },
    )


def confidence_from_verification_report(report: Any) -> Confidence:
    payload = report.model_dump() if hasattr(report, "model_dump") else dict(report or {})
    status = str(payload.get("status", "") or "pending").strip()
    missing_fields = [str(item) for item in list(payload.get("missing_fields", []) or [])]
    unsupported_claims = [str(item) for item in list(payload.get("unsupported_claims", []) or [])]
    contradictory_claims = [str(item) for item in list(payload.get("contradictory_claims", []) or [])]
    if status == "pass":
        penalty = min(0.35, 0.08 * (len(missing_fields) + len(unsupported_claims)) + 0.12 * len(contradictory_claims))
        score = 0.9 - penalty
    elif status == "retry":
        score = 0.35
    elif status == "clarify":
        score = 0.0
    else:
        score = 0.5
    return Confidence(
        score=max(0.0, min(1.0, score)),
        basis="verifier",
        detail={
            "status": status,
            "missing_fields": missing_fields,
            "unsupported_claims": unsupported_claims,
            "contradictory_claims": contradictory_claims,
            "recommended_action": str(payload.get("recommended_action", "") or ""),
        },
    )


def confidence_payload(confidence: Confidence) -> dict[str, Any]:
    return {
        "score": confidence.score,
        "basis": confidence.basis,
        "detail": dict(confidence.detail),
    }


def _note_float(*, notes: list[str], prefix: str) -> float | None:
    for note in notes:
        if not note.startswith(prefix):
            continue
        try:
            return float(note.split("=", 1)[1])
        except (IndexError, ValueError):
            return None
    return None


def _meaningful_terms(text: str) -> set[str]:
    terms: list[str] = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text.lower())
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        if len(chunk) <= 4:
            terms.append(chunk)
            continue
        terms.extend(chunk[index : index + 2] for index in range(0, len(chunk) - 1))
    stop = {"the", "and", "with", "from", "that", "this", "论文", "方法", "结果", "这个"}
    return {item for item in terms if item and item not in stop}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
