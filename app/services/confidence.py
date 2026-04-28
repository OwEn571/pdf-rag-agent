from __future__ import annotations

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


def _note_float(*, notes: list[str], prefix: str) -> float | None:
    for note in notes:
        if not note.startswith(prefix):
            continue
        try:
            return float(note.split("=", 1)[1])
        except (IndexError, ValueError):
            return None
    return None
