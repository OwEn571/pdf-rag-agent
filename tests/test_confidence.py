from __future__ import annotations

from types import SimpleNamespace

from app.domain.models import QueryContract
from app.services.confidence import confidence_from_contract, should_ask_human


def test_confidence_from_contract_uses_intent_confidence_note() -> None:
    confidence = confidence_from_contract(QueryContract(clean_query="hello", notes=["intent_confidence=0.72"]))

    assert confidence.score == 0.72
    assert confidence.basis == "intent_confidence_note"
    assert should_ask_human(confidence, SimpleNamespace(confidence_floor=0.7)) is False


def test_confidence_from_contract_forces_clarification_for_ambiguous_slots() -> None:
    confidence = confidence_from_contract(
        QueryContract(clean_query="它的公式", notes=["intent_confidence=0.91", "ambiguous_slot=target"])
    )

    assert confidence.score == 0.0
    assert confidence.basis == "contract_clarification_notes"
    assert confidence.detail["ambiguous_slots"] == ["target"]
    assert should_ask_human(confidence, SimpleNamespace(confidence_floor=0.6)) is True


def test_confidence_defaults_to_high_when_no_uncertainty_signal_exists() -> None:
    confidence = confidence_from_contract(QueryContract(clean_query="你好"))

    assert confidence.score == 1.0
    assert confidence.basis == "implicit_high_confidence"
    assert should_ask_human(confidence, SimpleNamespace(confidence_floor=0.99)) is False
