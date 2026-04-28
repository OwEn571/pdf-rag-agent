from __future__ import annotations

from collections.abc import Mapping

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract


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
