from __future__ import annotations

from collections.abc import Mapping

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract


def extract_metric_lines(evidence: list[EvidenceBlock], *, token_weights: Mapping[str, float]) -> list[str]:
    scored_lines: list[tuple[int, str]] = []
    for item in evidence:
        snippet = item.snippet.replace("\n", " ")
        score = metric_line_score(snippet, token_weights=token_weights)
        if score > 0:
            scored_lines.append((score, snippet[:280]))
    dedup: list[str] = []
    seen: set[str] = set()
    for _, line in sorted(scored_lines, key=lambda item: (-item[0], item[1])):
        norm = " ".join(line.lower().split())
        if norm not in seen:
            seen.add(norm)
            dedup.append(line)
    return dedup


def metric_line_score(text: str, *, token_weights: Mapping[str, float]) -> int:
    haystack = text.lower()
    score = 0.0
    for token, weight in token_weights.items():
        if token in haystack:
            score += float(weight)
    return int(score)


def metric_block_score(
    *,
    item: EvidenceBlock,
    contract: QueryContract,
    paper_by_id: dict[str, CandidatePaper],
    token_weights: Mapping[str, float],
    target_paper_match: bool = False,
) -> float:
    text = "\n".join([item.title, item.caption, item.snippet]).lower()
    score = float(metric_line_score(text, token_weights=token_weights)) + float(item.score)
    if item.block_type == "table":
        score += 1.5
    elif item.block_type == "caption":
        score += 0.75
    query = contract.clean_query.lower()
    if "pba" in query and "pba" in text:
        score += 5.0
    if "win rate" in text:
        score += 1.0
    if "accuracy" in text or " acc" in f" {text}":
        score += 1.0
    for target in contract.targets:
        normalized = target.strip().lower()
        if normalized and normalized in text:
            score += 4.0
    if paper_by_id.get(item.paper_id) is not None and contract.targets and target_paper_match:
        score += 6.0
    return score


def metric_context_claim(
    *,
    entity: str,
    selected_paper: CandidatePaper,
    selected_papers: list[CandidatePaper],
    metric_lines: list[str],
    metric_evidence: list[EvidenceBlock],
    fallback_evidence_ids: list[str],
    paper_ids: list[str],
) -> Claim:
    return Claim(
        claim_type="metric_context",
        entity=entity,
        value="table-backed metric answer",
        structured_data={
            "metric_lines": metric_lines,
            "paper_titles": [paper.title for paper in selected_papers],
        },
        evidence_ids=[item.doc_id for item in metric_evidence[:4]] or fallback_evidence_ids,
        paper_ids=paper_ids or [selected_paper.paper_id],
        confidence=0.74,
    )


def text_table_metric_claim(
    *,
    entity: str,
    metric_lines: list[str],
    evidence_ids: list[str],
    paper_ids: list[str],
    selected_paper: CandidatePaper,
) -> Claim:
    return Claim(
        claim_type="metric_value",
        entity=entity,
        value=metric_lines[0] if metric_lines else "已定位到表格指标证据。",
        structured_data={"metric_lines": metric_lines, "mode": "text_table"},
        evidence_ids=evidence_ids,
        paper_ids=paper_ids or [selected_paper.paper_id],
        confidence=0.86,
    )
