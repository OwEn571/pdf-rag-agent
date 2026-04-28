from __future__ import annotations

from typing import Any

from app.domain.models import CandidatePaper, Claim


def followup_research_claim(
    *,
    entity: str,
    seed_papers: list[CandidatePaper],
    followups: list[dict[str, Any]],
    selected_candidate_title: str,
    limit: int = 10,
) -> Claim:
    selected_followups = followups[:limit]
    seed_payload = [
        {"title": item.title, "year": item.year, "paper_id": item.paper_id}
        for item in seed_papers[:2]
    ]
    followup_payload = [
        {
            "title": item["paper"].title,
            "year": item["paper"].year,
            "paper_id": item["paper"].paper_id,
            "relation_type": item.get("relation_type", ""),
            "reason": item.get("reason", ""),
            "relationship_strength": item.get("relationship_strength", ""),
            "strict_followup": bool(item.get("strict_followup", False)),
            "classification": item.get("classification", ""),
            "evidence_ids": list(item.get("evidence_ids", []) or []),
        }
        for item in selected_followups
    ]
    evidence_ids: list[str] = []
    for paper in seed_papers[:1]:
        for doc_id in paper.doc_ids[:1]:
            if doc_id not in evidence_ids:
                evidence_ids.append(doc_id)
    for item in selected_followups:
        for doc_id in list(item.get("evidence_ids", []) or []):
            if doc_id not in evidence_ids:
                evidence_ids.append(str(doc_id))
        for doc_id in item["paper"].doc_ids[:1]:
            if doc_id not in evidence_ids:
                evidence_ids.append(doc_id)
    paper_ids = list(
        dict.fromkeys(
            [item.paper_id for item in seed_papers[:1]]
            + [item["paper"].paper_id for item in selected_followups]
        )
    )
    confidence_values = [float(item.get("confidence", 0.8)) for item in selected_followups]
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.8
    return Claim(
        claim_type="followup_research",
        entity=entity,
        value="; ".join(f"{item['paper'].title} ({item['paper'].year})" for item in selected_followups),
        structured_data={
            "seed_papers": seed_payload,
            "followup_titles": followup_payload,
            "selected_candidate_title": selected_candidate_title,
            "mode": "candidate_validation" if selected_candidate_title else "followup_discovery",
            "plan_steps": ["resolve_seed_paper", "broad_recall_followups", "rank_relationships"],
        },
        evidence_ids=evidence_ids,
        paper_ids=paper_ids,
        confidence=confidence,
    )
