from __future__ import annotations

from collections.abc import Callable

from app.domain.models import CandidatePaper, Claim


def paper_summary_claim(
    *,
    entity: str,
    paper: CandidatePaper,
    summary_text: str,
    metric_lines: list[str],
    evidence_ids: list[str],
) -> Claim:
    return Claim(
        claim_type="paper_summary",
        entity=entity,
        value=summary_text or paper.title,
        structured_data={
            "metric_lines": [
                line for line in metric_lines if paper.title.lower() in line.lower()
            ]
            or metric_lines[:4],
            "paper_title": paper.title,
            "paper_year": paper.year,
        },
        evidence_ids=evidence_ids or paper.doc_ids[:1],
        paper_ids=[paper.paper_id],
        confidence=0.82,
    )


def paper_recommendation_claim(
    *,
    entity: str,
    papers: list[CandidatePaper],
    reason_for_paper: Callable[[CandidatePaper], str],
) -> Claim | None:
    recommended = papers[:5]
    if not recommended:
        return None
    recommendations = []
    evidence_ids: list[str] = []
    for item in recommended:
        recommendations.append(
            {
                "title": item.title,
                "year": item.year,
                "paper_id": item.paper_id,
                "reason": reason_for_paper(item),
            }
        )
        evidence_ids.extend(item.doc_ids[:1])
    return Claim(
        claim_type="paper_recommendation",
        entity=entity,
        value="; ".join(f"{row['title']} ({row['year']})" for row in recommendations),
        structured_data={"recommended_papers": recommendations},
        evidence_ids=list(dict.fromkeys(evidence_ids)),
        paper_ids=[item.paper_id for item in recommended],
        confidence=0.84,
    )


def default_text_claim(
    *,
    entity: str,
    paper: CandidatePaper,
    summary: str,
    evidence_ids: list[str],
) -> Claim:
    return Claim(
        claim_type="text_answer",
        entity=entity,
        value=summary or paper.title,
        structured_data={"paper_title": paper.title, "paper_year": paper.year},
        evidence_ids=evidence_ids or paper.doc_ids[:1],
        paper_ids=[paper.paper_id],
        confidence=0.72,
    )
