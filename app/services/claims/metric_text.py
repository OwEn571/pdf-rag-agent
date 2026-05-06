from __future__ import annotations

from collections.abc import Callable, Mapping

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract

PaperTargetMatcher = Callable[[CandidatePaper, list[str]], bool]


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


def ranked_metric_context_evidence(
    *,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    token_weights: Mapping[str, float],
    paper_target_matcher: PaperTargetMatcher,
) -> list[EvidenceBlock]:
    return _rank_metric_evidence(
        contract=contract,
        papers=papers,
        evidence=[
            item
            for item in evidence
            if metric_line_score(item.snippet, token_weights=token_weights) > 0
            or item.block_type in {"table", "caption"}
        ],
        token_weights=token_weights,
        paper_target_matcher=paper_target_matcher,
    )


def ranked_table_metric_blocks(
    *,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    token_weights: Mapping[str, float],
    paper_target_matcher: PaperTargetMatcher,
) -> list[EvidenceBlock]:
    return _rank_metric_evidence(
        contract=contract,
        papers=papers,
        evidence=[
            item
            for item in evidence
            if item.block_type in {"table", "caption"}
            or (
                item.block_type == "page_text"
                and metric_line_score(item.snippet, token_weights=token_weights) >= 3
            )
        ],
        token_weights=token_weights,
        paper_target_matcher=paper_target_matcher,
    )


def metric_paper_selection(
    *, papers: list[CandidatePaper], ranked_evidence: list[EvidenceBlock]
) -> tuple[CandidatePaper | None, list[CandidatePaper], list[str]]:
    paper_by_id = {paper.paper_id: paper for paper in papers}
    paper_ids = list(dict.fromkeys(item.paper_id for item in ranked_evidence[:4] if item.paper_id))
    selected_papers = [paper_by_id[paper_id] for paper_id in paper_ids if paper_id in paper_by_id]
    selected_paper = selected_papers[0] if selected_papers else max(papers, key=lambda item: item.score) if papers else None
    return selected_paper, selected_papers, paper_ids


def _rank_metric_evidence(
    *,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    token_weights: Mapping[str, float],
    paper_target_matcher: PaperTargetMatcher,
) -> list[EvidenceBlock]:
    paper_by_id = {paper.paper_id: paper for paper in papers}
    return sorted(
        evidence,
        key=lambda item: (
            -metric_block_score(
                item=item,
                contract=contract,
                paper_by_id=paper_by_id,
                token_weights=token_weights,
                target_paper_match=bool(
                    paper_by_id.get(item.paper_id) is not None
                    and contract.targets
                    and paper_target_matcher(paper_by_id[item.paper_id], contract.targets)
                ),
            ),
            item.page,
            item.doc_id,
        ),
    )


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
