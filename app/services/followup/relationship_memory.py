from __future__ import annotations

from typing import Any

from app.domain.models import Claim, QueryContract
from app.services.contracts.normalization import normalize_lookup_text
from app.services.contracts.session_context import truncate_context_text


def followup_relationship_memory(
    *,
    contract: QueryContract,
    claims: list[Claim],
    answer: str,
) -> dict[str, Any]:
    claim = next((item for item in claims if item.claim_type == "followup_research"), None)
    if claim is None:
        return {}
    structured = dict(claim.structured_data or {})
    candidate_title = str(structured.get("selected_candidate_title", "") or "").strip()
    rows = [dict(item or {}) for item in list(structured.get("followup_titles", []) or []) if isinstance(item, dict)]
    selected_row: dict[str, Any] = {}
    if candidate_title:
        selected_key = normalize_lookup_text(candidate_title)
        selected_row = next(
            (
                row
                for row in rows
                if selected_key
                and (
                    selected_key in normalize_lookup_text(str(row.get("title", "")))
                    or normalize_lookup_text(str(row.get("title", ""))) in selected_key
                )
            ),
            rows[0] if rows else {},
        )
    elif rows:
        selected_row = rows[0]
        candidate_title = str(selected_row.get("title", "") or "").strip()
    if not candidate_title:
        return {}
    seed = next((dict(item or {}) for item in list(structured.get("seed_papers", []) or []) if isinstance(item, dict)), {})
    seed_target = contract.targets[0] if contract.targets else str(claim.entity or "")
    return {
        "seed_target": seed_target,
        "seed_title": str(seed.get("title", "") or "").strip(),
        "seed_paper_id": str(seed.get("paper_id", "") or "").strip(),
        "candidate_title": candidate_title,
        "candidate_paper_id": str(selected_row.get("paper_id", "") or "").strip(),
        "relationship_strength": str(selected_row.get("relationship_strength", "") or "").strip(),
        "relation_type": str(selected_row.get("relation_type", "") or "").strip(),
        "strict_followup": bool(selected_row.get("strict_followup", False)),
        "clean_query": contract.clean_query,
        "answer_preview": truncate_context_text(answer, limit=900),
    }
