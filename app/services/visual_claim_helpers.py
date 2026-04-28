from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.domain.models import Claim
from app.services.confidence import coerce_confidence_value


def table_metric_claim_from_vlm_payload(
    payload: Any,
    *,
    entity: str,
    evidence_ids: list[str],
    paper_ids: list[str],
) -> Claim | None:
    raw_claim = _first_raw_claim(payload)
    claim_text = str(
        raw_claim.get("claim", "") or (payload.get("draft_answer", "") if isinstance(payload, dict) else "")
    ).strip()
    if not claim_text:
        return None
    raw_lines = raw_claim.get("metric_lines", [])
    metric_lines = [str(item).strip() for item in raw_lines if str(item).strip()] if isinstance(raw_lines, list) else []
    return Claim(
        claim_type="metric_value",
        entity=entity,
        value=claim_text,
        structured_data={"metric_lines": metric_lines or [claim_text], "mode": "vlm_table"},
        evidence_ids=evidence_ids,
        paper_ids=paper_ids,
        confidence=coerce_confidence_value(raw_claim.get("confidence", 0.84), default=0.82),
    )


def figure_conclusion_claim_from_vlm_payload(
    payload: Any,
    *,
    entity: str,
    evidence_ids: list[str],
    paper_id: str,
    fallback_text: str,
    signal_score: Callable[[str], float],
) -> Claim | None:
    raw_claim = _first_raw_claim(payload)
    if not raw_claim:
        return None
    claim_text = str(raw_claim.get("claim", "") or (payload.get("draft_answer", "") if isinstance(payload, dict) else "")).strip()
    if not claim_text:
        return None
    if signal_score(claim_text) < max(3, signal_score(fallback_text)):
        return None
    return Claim(
        claim_type="figure_conclusion",
        entity=entity,
        value=claim_text,
        structured_data={"mode": "vlm"},
        evidence_ids=evidence_ids,
        paper_ids=[paper_id],
        confidence=coerce_confidence_value(raw_claim.get("confidence", 0.82), default=0.82),
    )


def _first_raw_claim(payload: Any) -> dict[str, Any]:
    raw_claims = payload.get("claims", []) if isinstance(payload, dict) else []
    if isinstance(raw_claims, list) and raw_claims and isinstance(raw_claims[0], dict):
        return raw_claims[0]
    return {}
