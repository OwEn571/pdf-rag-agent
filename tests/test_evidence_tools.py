from __future__ import annotations

from app.domain.models import EvidenceBlock
from app.services.evidence_tools import evidence_from_payload, summarize_text, verify_claim_against_evidence


def _block(snippet: str, *, doc_id: str = "doc-1", title: str = "PPO paper") -> EvidenceBlock:
    return EvidenceBlock(
        doc_id=doc_id,
        paper_id="paper-1",
        title=title,
        file_path="",
        page=1,
        block_type="text",
        caption="",
        bbox="",
        snippet=snippet,
        score=0.8,
        metadata={},
    )


def test_summarize_text_prefers_focus_sentences_and_limits_output() -> None:
    text = (
        "This paragraph introduces optimization background. "
        "DPO directly optimizes preference likelihood without fitting a separate reward model. "
        + "Unrelated implementation detail. " * 20
    )

    summary = summarize_text(text=text, target_words=20, focus=["DPO", "preference"])

    assert "DPO directly optimizes preference likelihood" in summary
    assert len(summary) < len(text)


def test_verify_claim_against_evidence_returns_pass_with_matched_terms() -> None:
    check = verify_claim_against_evidence(
        claim="PPO uses a clipped surrogate objective",
        evidence=[_block("The PPO algorithm optimizes a clipped surrogate objective to limit policy updates.")],
    )

    assert check.status == "pass"
    assert check.confidence > 0.5
    assert check.supporting_evidence_ids == ["doc-1"]
    assert "clipped" in check.matched_terms
    assert "surrogate" in check.matched_terms


def test_verify_claim_against_evidence_retries_when_terms_are_missing() -> None:
    check = verify_claim_against_evidence(
        claim="PPO requires a KL penalty objective",
        evidence=[_block("The passage only says PPO limits policy updates with clipping.")],
    )

    assert check.status == "retry"
    assert "penalty" in check.missing_terms


def test_evidence_from_payload_accepts_inline_strings_and_fetch_payloads() -> None:
    evidence = evidence_from_payload(
        [
            "Inline PPO evidence",
            {"url": "https://example.test/paper", "title": "Fetched page", "text": "Fetched evidence text"},
        ]
    )

    assert [item.doc_id for item in evidence] == ["inline::1", "https://example.test/paper"]
    assert evidence[1].title == "Fetched page"
    assert evidence[1].snippet == "Fetched evidence text"
