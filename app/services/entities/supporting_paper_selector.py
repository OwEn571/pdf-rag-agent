from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services.claims import origin_selection as origin_helpers
from app.services.infra.confidence import coerce_confidence_value
from app.services.answers.evidence_presentation import safe_year
from app.services.claims.paper_summary import paper_summary_text
from app.services.planning.query_shaping import matches_target

PaperDocLookupFn = Callable[[str], Any]
PaperLookupFn = Callable[[str], CandidatePaper | None]
PaperIdentityMatchesFn = Callable[[CandidatePaper, list[str]], bool]


def candidate_from_paper_id(
    paper_id: str,
    *,
    paper_doc_lookup: PaperDocLookupFn,
) -> CandidatePaper | None:
    doc = paper_doc_lookup(paper_id)
    if doc is None:
        return None
    meta = dict(getattr(doc, "metadata", None) or {})
    return CandidatePaper(
        paper_id=paper_id,
        title=str(meta.get("title", "")),
        year=str(meta.get("year", "")),
        score=0.0,
        match_reason="paper_doc_lookup",
        doc_ids=[str(meta.get("doc_id", ""))] if meta.get("doc_id") else [],
        metadata=meta,
    )


def ground_entity_papers(
    *,
    candidates: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    limit: int,
    paper_lookup: PaperLookupFn,
) -> list[CandidatePaper]:
    if not evidence:
        return candidates[: max(1, limit)]
    by_id = {item.paper_id: item for item in candidates}
    aggregated: dict[str, dict[str, Any]] = {}
    for item in evidence:
        bucket = aggregated.setdefault(item.paper_id, {"score": 0.0, "doc_ids": []})
        bucket["score"] += float(item.score)
        if item.doc_id not in bucket["doc_ids"]:
            bucket["doc_ids"].append(item.doc_id)
    grounded: list[CandidatePaper] = []
    for paper_id, payload in aggregated.items():
        paper = by_id.get(paper_id) or paper_lookup(paper_id)
        if paper is None:
            continue
        grounded.append(
            paper.model_copy(
                update={
                    "score": paper.score + float(payload["score"]) + (len(payload["doc_ids"]) * 0.25),
                    "match_reason": "entity_evidence_grounded",
                    "doc_ids": list(dict.fromkeys([*paper.doc_ids, *payload["doc_ids"]]))[:6],
                }
            )
        )
    grounded.sort(key=lambda item: (-item.score, safe_year(item.year), item.title))
    return grounded[: max(1, limit)] or candidates[: max(1, limit)]


def select_entity_supporting_paper(
    *,
    clients: Any,
    paper_doc_lookup: PaperDocLookupFn,
    paper_identity_matches_targets: PaperIdentityMatchesFn,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
) -> tuple[CandidatePaper | None, list[EvidenceBlock]]:
    if not papers:
        return None, []
    target = contract.targets[0] if contract.targets else ""
    if not target:
        paper = best_entity_fallback_paper(papers=papers, evidence=evidence)
        return paper, prune_entity_supporting_evidence([item for item in evidence if item.paper_id == paper.paper_id])
    context_targets = [item for item in contract.targets[1:] if str(item).strip()]
    paper_lookup = lambda paper_id: candidate_from_paper_id(paper_id, paper_doc_lookup=paper_doc_lookup)

    matching_evidence = [
        item
        for item in evidence
        if any(
            matches_target(haystack, target)
            for haystack in [item.snippet, item.caption, item.title]
            if haystack
        )
    ]
    if context_targets:
        identity_contextual_evidence = [
            item
            for item in matching_evidence
            if entity_context_identity_matches(
                item=item,
                context_targets=context_targets,
                paper_lookup=paper_lookup,
                paper_identity_matches_targets=paper_identity_matches_targets,
            )
        ]
        contextual_evidence = [
            item
            for item in matching_evidence
            if entity_context_matches(
                item=item,
                context_targets=context_targets,
                paper_lookup=paper_lookup,
            )
        ]
        if identity_contextual_evidence:
            matching_evidence = identity_contextual_evidence
        elif contextual_evidence:
            matching_evidence = contextual_evidence
    paper_rank = {item.paper_id: idx for idx, item in enumerate(papers)}

    if matching_evidence:
        llm_paper, llm_evidence = (
            llm_select_entity_supporting_paper(
                clients=clients,
                paper_doc_lookup=paper_doc_lookup,
                contract=contract,
                papers=papers,
                matching_evidence=matching_evidence,
            )
            if not context_targets
            else (None, [])
        )
        if llm_paper is not None:
            return llm_paper, llm_evidence
        scored: list[tuple[float, CandidatePaper]] = []
        for paper in papers:
            support = [item for item in matching_evidence if item.paper_id == paper.paper_id]
            definition_bonus = sum(float(item.metadata.get("definition_score", 0) or 0) for item in support)
            mechanism_bonus = sum(float(item.metadata.get("mechanism_score", 0) or 0) for item in support)
            application_bonus = sum(float(item.metadata.get("application_score", 0) or 0) for item in support)
            paper_text = "\n".join(
                [
                    paper.title,
                    str(paper.metadata.get("aliases", "")),
                    str(paper.metadata.get("paper_card_text", "")),
                    str(paper.metadata.get("generated_summary", "")),
                    str(paper.metadata.get("abstract_note", "")),
                    "\n".join(item.snippet for item in support[:3]),
                ]
            )
            score = sum(item.score for item in support)
            if support:
                score += 3.0
            score += definition_bonus * 2.5
            score += mechanism_bonus * 1.0
            score += application_bonus * 0.3
            if matches_target(paper_text, target):
                score += 1.2
            if context_targets:
                if paper_identity_matches_targets(paper, context_targets) or paper_introduces_context_target(
                    paper=paper,
                    context_targets=context_targets,
                ):
                    score += 12.0
                elif any(matches_target(paper_text, context_target) for context_target in context_targets):
                    score += 4.0
            if definition_bonus > 0:
                score += 1.6
            elif any(entity_definition_score(item.snippet) > 0 for item in support):
                score += 0.8
            scored.append((score, paper))
        scored.sort(key=lambda item: (-item[0], paper_rank.get(item[1].paper_id, 999), item[1].title))
        best_paper = scored[0][1]
        best_evidence = [item for item in matching_evidence if item.paper_id == best_paper.paper_id]
        best_evidence.sort(
            key=lambda item: (
                -entity_definition_score(item.snippet),
                -item.score,
                item.page,
                item.doc_id,
            )
        )
        return best_paper, prune_entity_supporting_evidence(best_evidence)

    for paper in papers:
        paper_text = "\n".join(
            [
                paper.title,
                str(paper.metadata.get("aliases", "")),
                str(paper.metadata.get("paper_card_text", "")),
                str(paper.metadata.get("generated_summary", "")),
                str(paper.metadata.get("abstract_note", "")),
            ]
        )
        if matches_target(paper_text, target) and (
            not context_targets or any(matches_target(paper_text, context_target) for context_target in context_targets)
        ):
            fallback_evidence = [item for item in evidence if item.paper_id == paper.paper_id]
            fallback_evidence.sort(
                key=lambda item: (
                    -entity_definition_score(item.snippet),
                    -item.score,
                    item.page,
                    item.doc_id,
                )
            )
            return paper, prune_entity_supporting_evidence(fallback_evidence)

    paper = best_entity_fallback_paper(papers=papers, evidence=evidence)
    fallback_evidence = [item for item in evidence if item.paper_id == paper.paper_id]
    fallback_evidence.sort(
        key=lambda item: (
            -entity_definition_score(item.snippet),
            -item.score,
            item.page,
            item.doc_id,
        )
    )
    return paper, prune_entity_supporting_evidence(fallback_evidence)


def best_entity_fallback_paper(*, papers: list[CandidatePaper], evidence: list[EvidenceBlock]) -> CandidatePaper:
    evidence_score_by_paper: dict[str, float] = {}
    for item in evidence:
        evidence_score_by_paper[item.paper_id] = evidence_score_by_paper.get(item.paper_id, 0.0) + float(item.score)
    return max(
        papers,
        key=lambda paper: (
            evidence_score_by_paper.get(paper.paper_id, 0.0),
            float(paper.score),
            -len(paper.title),
        ),
    )


def entity_context_identity_matches(
    *,
    item: EvidenceBlock,
    context_targets: list[str],
    paper_lookup: PaperLookupFn,
    paper_identity_matches_targets: PaperIdentityMatchesFn,
) -> bool:
    paper = paper_lookup(item.paper_id)
    if paper is None:
        return False
    return paper_identity_matches_targets(paper, context_targets) or paper_introduces_context_target(
        paper=paper,
        context_targets=context_targets,
    )


def paper_introduces_context_target(*, paper: CandidatePaper, context_targets: list[str]) -> bool:
    paper_text = "\n".join(
        [
            paper.title,
            str(paper.metadata.get("aliases", "")),
            str(paper.metadata.get("paper_card_text", "")),
            str(paper.metadata.get("generated_summary", "")),
            str(paper.metadata.get("abstract_note", "")),
        ]
    )
    return origin_helpers.origin_target_definition_score(paper_text, [str(item) for item in context_targets]) >= 4.0


def entity_context_matches(
    *,
    item: EvidenceBlock,
    context_targets: list[str],
    paper_lookup: PaperLookupFn,
) -> bool:
    paper = paper_lookup(item.paper_id)
    paper_text = ""
    if paper is not None:
        paper_text = "\n".join(
            [
                paper.title,
                str(paper.metadata.get("aliases", "")),
                str(paper.metadata.get("paper_card_text", "")),
                str(paper.metadata.get("generated_summary", "")),
                str(paper.metadata.get("abstract_note", "")),
            ]
        )
    haystack = "\n".join([item.title, item.caption, item.snippet, paper_text])
    return any(matches_target(haystack, context_target) for context_target in context_targets if str(context_target).strip())


def llm_select_entity_supporting_paper(
    *,
    clients: Any,
    paper_doc_lookup: PaperDocLookupFn,
    contract: QueryContract,
    papers: list[CandidatePaper],
    matching_evidence: list[EvidenceBlock],
) -> tuple[CandidatePaper | None, list[EvidenceBlock]]:
    if getattr(clients, "chat", None) is None or not contract.targets or not matching_evidence:
        return None, []
    target = contract.targets[0]
    grouped: dict[str, list[EvidenceBlock]] = {}
    for item in matching_evidence:
        grouped.setdefault(item.paper_id, []).append(item)
    candidates_payload: list[dict[str, Any]] = []
    by_id = {item.paper_id: item for item in papers}
    for paper in papers[:6]:
        support = grouped.get(paper.paper_id, [])
        if not support:
            continue
        ordered_support = sorted(
            support,
            key=lambda item: (
                -float(item.metadata.get("definition_score", 0) or 0),
                -float(item.metadata.get("mechanism_score", 0) or 0),
                -float(item.score),
                item.page,
                item.doc_id,
            ),
        )[:2]
        candidates_payload.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "summary": paper_summary_text(paper.paper_id, paper_doc_lookup=paper_doc_lookup),
                "evidence": [
                    {
                        "doc_id": item.doc_id,
                        "page": item.page,
                        "snippet": item.snippet[:260],
                    }
                    for item in ordered_support
                ],
            }
        )
    if not candidates_payload:
        return None, []
    payload = clients.invoke_json(
        system_prompt=(
            "你是论文实体 grounding 裁判器。"
            "你的任务是在多个候选论文里，挑出最适合回答“某个实体/术语是什么”的来源论文。"
            "优先级："
            "1. 直接定义或首次引入该实体的论文；"
            "2. 明确解释该实体机制/组成的论文；"
            "3. 只是使用、对比或顺带提到该实体的论文不要优先。"
            "只输出 JSON，字段为 paper_id, evidence_doc_ids, relation_to_target, confidence, reason。"
            "relation_to_target 只能是 [origin, direct_definition, mechanism_explanation, usage_only, incidental_mention]。"
        ),
        human_prompt=json.dumps(
            {
                "query": contract.clean_query,
                "target": target,
                "requested_fields": contract.requested_fields,
                "candidates": candidates_payload,
            },
            ensure_ascii=False,
        ),
        fallback={},
    )
    if not isinstance(payload, dict) or not payload:
        return None, []
    paper_id = str(payload.get("paper_id", "")).strip()
    relation_to_target = str(payload.get("relation_to_target", "")).strip().lower()
    if paper_id not in by_id:
        return None, []
    if relation_to_target not in {
        "origin",
        "direct_definition",
        "mechanism_explanation",
        "usage_only",
        "incidental_mention",
    }:
        return None, []
    if relation_to_target in {"usage_only", "incidental_mention"}:
        return None, []
    confidence = coerce_confidence_value(
        payload.get("confidence", 0),
        default=0.0,
        label_scores={"high": 0.88, "medium": 0.72, "low": 0.45},
    )
    if confidence < 0.55:
        return None, []
    raw_doc_ids = payload.get("evidence_doc_ids", [])
    evidence_doc_ids = [str(item).strip() for item in raw_doc_ids if str(item).strip()] if isinstance(raw_doc_ids, list) else []
    selected_evidence = [item for item in grouped.get(paper_id, []) if item.doc_id in evidence_doc_ids]
    if not selected_evidence:
        selected_evidence = grouped.get(paper_id, [])
    return by_id[paper_id], prune_entity_supporting_evidence(selected_evidence)


def prune_entity_supporting_evidence(evidence: list[EvidenceBlock]) -> list[EvidenceBlock]:
    if not evidence:
        return []
    cleaned = [item for item in evidence if not is_noisy_entity_line(item.snippet)]
    pool = cleaned or evidence
    deduped: list[EvidenceBlock] = []
    seen: set[str] = set()
    for item in pool:
        key = f"{item.paper_id}:{item.page}:{item.block_type}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:4]


def entity_definition_score(text: str) -> int:
    haystack = " ".join(str(text or "").lower().split())
    if not haystack:
        return 0
    score = 0
    if any(
        token in haystack
        for token in [
            "algorithm",
            "framework",
            "method",
            "model",
            "system",
            "dataset",
            "benchmark",
            "objective",
            "loss",
            "reinforcement learning",
            "policy optimization",
            "reward",
            "算法",
            "方法",
            "模型",
            "系统",
            "框架",
            "数据集",
            "基准",
            "目标函数",
            "优化",
        ]
    ):
        score += 1
    if any(
        token in haystack
        for token in [
            " is a ",
            " is an ",
            " refers to ",
            " stands for ",
            " denotes ",
            " employ the ",
            " uses the ",
            " introduce ",
            " propose ",
        ]
    ):
        score += 1
    return score


def is_noisy_entity_line(text: str) -> bool:
    compact = " ".join(str(text or "").split())
    if not compact:
        return True
    weird_math_chars = "∑𝜋𝜃𝑜𝑡𝑞𝐴ˆβϵμ"
    if sum(1 for ch in compact if ch in weird_math_chars) >= 2:
        return True
    if compact.count("|") >= 2:
        return True
    if re.search(r"\([0-9]{1,2}\)\s*$", compact):
        return True
    letters = sum(1 for ch in compact if ch.isalpha())
    digits = sum(1 for ch in compact if ch.isdigit())
    symbols = sum(1 for ch in compact if not ch.isalnum() and ch not in " .,;:!?()[]{}-_/")
    if letters < 24 and symbols > max(4, letters):
        return True
    if digits > letters and letters < 18:
        return True
    return False
