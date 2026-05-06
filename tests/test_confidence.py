from __future__ import annotations

from types import SimpleNamespace

from app.domain.models import QueryContract, VerificationReport
from app.services.infra.confidence import (
    confidence_from_contract,
    confidence_from_logprobs,
    confidence_from_self_consistency,
    confidence_from_verification_report,
    confidence_payload,
    coerce_claim_confidence,
    coerce_confidence_value,
    should_ask_human,
)


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


def test_confidence_from_self_consistency_scores_similar_samples_high() -> None:
    confidence = confidence_from_self_consistency(
        [
            "DPO optimizes preference likelihood without a separate reward model.",
            "DPO directly optimizes preference likelihood and avoids training a separate reward model.",
            "Direct Preference Optimization optimizes preferences without fitting a separate reward model.",
        ]
    )

    assert confidence.basis == "self_consistency"
    assert confidence.score > 0.45
    assert confidence.detail["sample_count"] == 3


def test_confidence_from_self_consistency_scores_conflicting_samples_low() -> None:
    confidence = confidence_from_self_consistency(
        [
            "DPO removes the reward model and uses preference likelihood.",
            "Transformer introduced self-attention for sequence modeling.",
            "A citation count ranking requires external web evidence.",
        ]
    )

    assert confidence.score < 0.2
    assert should_ask_human(confidence, SimpleNamespace(confidence_floor=0.6)) is True


def test_confidence_from_logprobs_maps_average_token_probability() -> None:
    confidence = confidence_from_logprobs([-0.1, "-0.2", -0.3, "bad"])

    assert confidence.basis == "logprobs"
    assert 0.8 < confidence.score < 0.9
    assert confidence.detail["token_count"] == 3
    assert confidence.detail["avg_logprob"] == -0.2


def test_confidence_from_logprobs_reports_missing_signal() -> None:
    confidence = confidence_from_logprobs([], min_tokens=2)
    sparse = confidence_from_logprobs([-0.2], min_tokens=2)

    assert confidence.score == 0.0
    assert confidence.detail["reason"] == "insufficient_logprobs"
    assert sparse.score == 0.5
    assert sparse.detail["token_count"] == 1


def test_confidence_from_verification_report_maps_status_to_score() -> None:
    passed = confidence_from_verification_report(VerificationReport(status="pass"))
    retry = confidence_from_verification_report(
        VerificationReport(status="retry", missing_fields=["formula"], recommended_action="expand_recall")
    )
    clarify = confidence_from_verification_report(VerificationReport(status="clarify", missing_fields=["target"]))

    assert passed.score > retry.score > clarify.score
    assert confidence_payload(retry)["detail"]["recommended_action"] == "expand_recall"


def test_coerce_confidence_value_supports_labels_numbers_and_callsite_defaults() -> None:
    assert coerce_confidence_value("high") == 0.88
    assert coerce_confidence_value("0.63") == 0.63
    assert coerce_confidence_value(None, default=0.12) == 0.12
    assert coerce_confidence_value("low", label_scores={"low": 0.45}) == 0.45


def test_coerce_claim_confidence_uses_shared_claim_label_mapping() -> None:
    assert coerce_claim_confidence("high") == 0.88
    assert coerce_claim_confidence("medium") == 0.72
    assert coerce_claim_confidence("low") == 0.55
