from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock
from app.services.intent_marker_matching import MarkerProfile, query_matches_any


EvidenceIdsForPaper = Callable[[str], list[str]]

UNUSABLE_TOPOLOGY_RECOMMENDATION_MARKERS: MarkerProfile = (
    "does not address",
    "does not contain",
    "impossible to determine",
    "no direct analysis",
    "not provide specific",
    "cannot determine",
    "无法确定",
    "不能确定",
    "没有覆盖",
    "不包含",
)


def fallback_topology_recommendation(topology_terms: list[str]) -> dict[str, str]:
    terms_text = " / ".join(topology_terms) if topology_terms else "chain / tree / mesh / DAG"
    return {
        "overall_best": "",
        "engineering_best": "DAG",
        "rationale": f"当前证据主要覆盖这些 topology：{terms_text}；工程选择仍要看任务依赖、并行验证、可追溯性和节点调度成本。",
        "summary": f"当前证据讨论了 {terms_text} 等 topology，但没有给出脱离任务的绝对最优。",
    }


def is_unusable_topology_recommendation_text(text: str) -> bool:
    lowered = " ".join(str(text or "").lower().split())
    if not lowered:
        return True
    return query_matches_any(lowered, "", UNUSABLE_TOPOLOGY_RECOMMENDATION_MARKERS)


def topology_recommendation_system_prompt() -> str:
    return (
        "你是 topology 证据分析器。"
        "请只输出 JSON，字段为 overall_best, engineering_best, rationale, summary。"
        "必须严格基于给定证据，不要使用外部知识。"
    )


def topology_recommendation_human_prompt(*, topology_terms: list[str], evidence: list[EvidenceBlock]) -> str:
    return json.dumps(
        {
            "topology_terms": topology_terms,
            "evidence": [item.snippet[:260] for item in evidence[:6]],
        },
        ensure_ascii=False,
    )


def topology_recommendation_from_payload(
    payload: Any,
    *,
    topology_terms: list[str],
) -> dict[str, str]:
    if isinstance(payload, dict):
        summary = str(payload.get("summary", "")).strip()
        if summary and not is_unusable_topology_recommendation_text(summary):
            return {
                "overall_best": str(payload.get("overall_best", "")).strip(),
                "engineering_best": str(payload.get("engineering_best", "")).strip(),
                "rationale": str(payload.get("rationale", "")).strip(),
                "summary": summary,
            }
    return fallback_topology_recommendation(topology_terms)


def topology_discovery_claim(
    *,
    papers: list[CandidatePaper],
    topology_terms: list[str],
    evidence_ids_for_paper: EvidenceIdsForPaper,
) -> Claim | None:
    if not papers:
        return None
    relevant_papers: list[dict[str, str]] = []
    evidence_ids: list[str] = []
    paper_ids: list[str] = []
    for paper in papers[:5]:
        ids = evidence_ids_for_paper(paper.paper_id)
        if not ids and not paper.doc_ids:
            continue
        relevant_papers.append({"title": paper.title, "year": paper.year, "paper_id": paper.paper_id})
        evidence_ids.extend(ids or paper.doc_ids[:1])
        paper_ids.append(paper.paper_id)
    if not relevant_papers:
        relevant_papers = [{"title": paper.title, "year": paper.year, "paper_id": paper.paper_id} for paper in papers[:3]]
        paper_ids = [paper.paper_id for paper in papers[:3]]
    return Claim(
        claim_type="topology_discovery",
        entity="agent topology",
        value="; ".join(item["title"] for item in relevant_papers),
        structured_data={"topology_terms": topology_terms, "relevant_papers": relevant_papers},
        evidence_ids=list(dict.fromkeys(evidence_ids)),
        paper_ids=list(dict.fromkeys(paper_ids)),
        confidence=0.82,
    )


def topology_recommendation_claim(
    *,
    recommendation: dict[str, str],
    topology_terms: list[str],
    evidence: list[EvidenceBlock],
) -> Claim:
    return Claim(
        claim_type="topology_recommendation",
        entity="agent topology",
        value=recommendation.get("summary", ""),
        structured_data={
            "topology_terms": topology_terms,
            "overall_best": recommendation.get("overall_best", ""),
            "engineering_best": recommendation.get("engineering_best", ""),
            "rationale": recommendation.get("rationale", ""),
        },
        evidence_ids=evidence[:3] and [item.doc_id for item in evidence[:3]] or [],
        paper_ids=list(dict.fromkeys(item.paper_id for item in evidence[:3])),
        confidence=0.8,
    )
