from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services.contract_normalization import normalize_lookup_text
from app.services.followup_relationship_intents import has_followup_domain_signal

PaperText = Callable[[str], str]
PaperAnchor = Callable[[CandidatePaper], str]
ConfidenceCoercer = Callable[[Any], float]


def selected_followup_candidate_title(contract: QueryContract) -> str:
    for note in contract.notes:
        raw = str(note or "")
        if raw.startswith("candidate_title="):
            return raw.split("=", 1)[1].strip()
    return ""


def candidate_title_matches(paper: CandidatePaper, selected_title: str) -> bool:
    selected_key = normalize_lookup_text(selected_title)
    if not selected_key:
        return False
    title_key = normalize_lookup_text(paper.title)
    aliases = normalize_lookup_text(str(paper.metadata.get("aliases", "")))
    return selected_key in title_key or title_key in selected_key or selected_key in aliases


def followup_target_aliases(
    *,
    contract: QueryContract,
    seed_papers: list[CandidatePaper],
    paper_anchor_text: PaperAnchor,
) -> list[str]:
    aliases: list[str] = []
    for target in contract.targets:
        target = str(target or "").strip()
        if target:
            aliases.append(target)
    for paper in seed_papers[:2]:
        raw_aliases = str(paper.metadata.get("aliases", ""))
        for alias in re.split(r"\|\||[,;/]", raw_aliases):
            alias = alias.strip()
            if alias and len(alias) <= 48:
                aliases.append(alias)
        anchor = paper_anchor_text(paper)
        if anchor and len(anchor) <= 48:
            aliases.append(anchor)
    normalized: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        key = alias.lower()
        if key and key not in seen:
            seen.add(key)
            normalized.append(alias)
    return normalized


def filter_followup_candidates(
    *,
    contract: QueryContract,
    candidates: list[CandidatePaper],
    paper_summary_text: PaperText,
) -> list[CandidatePaper]:
    if not candidates:
        return []
    target = contract.targets[0].lower() if contract.targets else ""
    filtered: list[CandidatePaper] = []
    for item in candidates:
        title_lower = item.title.lower()
        card_text = str(item.metadata.get("paper_card_text", "")).lower()
        summary_lower = str(item.metadata.get("generated_summary", "") or paper_summary_text(item.paper_id)).lower()
        if (
            target
            and target not in title_lower
            and target not in summary_lower
            and target not in str(item.metadata.get("abstract_note", "")).lower()
            and target not in card_text
            and not has_followup_domain_signal(title_lower + "\n" + card_text + "\n" + summary_lower)
        ):
            continue
        filtered.append(item)
    return filtered or candidates[: min(8, len(candidates))]


def merge_followup_rankings(
    *,
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in (primary, secondary):
        for item in source:
            paper = item.get("paper")
            paper_id = getattr(paper, "paper_id", "")
            if not paper_id or paper_id in seen:
                continue
            seen.add(paper_id)
            merged.append(item)
    return merged


def relationship_evidence_ids_from_payload(
    *,
    payload: dict[str, Any],
    relationship_evidence: list[EvidenceBlock],
) -> list[str]:
    available = {item.doc_id for item in relationship_evidence}
    raw_ids = payload.get("evidence_ids", [])
    selected: list[str] = []
    if isinstance(raw_ids, list):
        selected = [str(item).strip() for item in raw_ids if str(item).strip() in available]
    if selected:
        return selected[:6]
    return [item.doc_id for item in relationship_evidence[:4]]


def paper_relationship_brief(
    *,
    paper: CandidatePaper,
    paper_summary_text: PaperText,
) -> dict[str, Any]:
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "year": paper.year,
        "authors": str(paper.metadata.get("authors", "")),
        "aliases": str(paper.metadata.get("aliases", "")),
        "summary": paper_summary_text(paper.paper_id),
        "paper_card_text": str(paper.metadata.get("paper_card_text", ""))[:1800],
        "tags": str(paper.metadata.get("tags", "")),
    }


def followup_relationship_validator_system_prompt() -> str:
    return (
        "你是论文关系验证器。"
        "任务是判断候选论文是否是种子论文/数据集/方法的严格后续工作。"
        "严格后续只在证据明确显示候选论文使用、继承、引用、复现、评测或直接扩展种子论文/数据集/方法时成立。"
        "仅主题相似、作者重合、关键词相同，不能判为严格后续，只能判为 related_continuation 或 not_enough_evidence。"
        "只能基于输入的 seed_papers、candidate_paper 和 relationship_evidence 判断，不要使用外部记忆补事实。"
        "relationship_evidence 中 role=candidate 的片段必须出现明确引用/使用/评测/继承/扩展 seed 或其数据集/方法，才可以判 strict_followup 或 direct_use_or_evaluation。"
        "只输出 JSON：classification, strict_followup, relation_type, relationship_strength, reason, confidence, evidence_ids。"
        "classification 只能是 strict_followup, direct_use_or_evaluation, related_continuation, not_enough_evidence, unrelated。"
        "relationship_strength 只能是 direct, strong_related, not_enough_evidence, unrelated。"
    )


def followup_relationship_validator_human_prompt(
    *,
    contract: QueryContract,
    seed_papers: list[CandidatePaper],
    paper: CandidatePaper,
    relationship_evidence: list[EvidenceBlock],
    paper_summary_text: PaperText,
) -> str:
    return json.dumps(
        {
            "query": contract.clean_query,
            "targets": contract.targets,
            "seed_papers": [
                paper_relationship_brief(paper=item, paper_summary_text=paper_summary_text)
                for item in seed_papers[:2]
            ],
            "candidate_paper": paper_relationship_brief(paper=paper, paper_summary_text=paper_summary_text),
            "relationship_evidence": [
                {
                    "doc_id": item.doc_id,
                    "paper_id": item.paper_id,
                    "role": "candidate" if item.paper_id == paper.paper_id else "seed",
                    "title": item.title,
                    "page": item.page,
                    "block_type": item.block_type,
                    "snippet": item.snippet[:900],
                }
                for item in relationship_evidence
            ],
        },
        ensure_ascii=False,
    )


def followup_validator_assessment_from_payload(
    *,
    payload: object,
    relationship_evidence: list[EvidenceBlock],
    coerce_confidence: ConfidenceCoercer,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        return {}
    classification = str(payload.get("classification", "") or "").strip()
    strength = str(payload.get("relationship_strength", "") or "").strip()
    if strength not in {"direct", "strong_related", "not_enough_evidence", "unrelated"}:
        if classification in {"strict_followup", "direct_use_or_evaluation"}:
            strength = "direct"
        elif classification == "related_continuation":
            strength = "strong_related"
        elif classification == "unrelated":
            strength = "unrelated"
        else:
            strength = "not_enough_evidence"
    strict = bool(payload.get("strict_followup", False)) and strength == "direct"
    relation_type = str(payload.get("relation_type", "") or "").strip()
    if not relation_type:
        relation_type = "严格后续/直接使用证据" if strict else ("强相关延续候选" if strength == "strong_related" else "证据不足")
    reason = " ".join(str(payload.get("reason", "") or "").split())
    if not reason:
        reason = "关系验证器没有找到足够明确的严格后续证据。"
    return {
        "relation_type": relation_type,
        "reason": reason,
        "confidence": coerce_confidence(payload.get("confidence", 0.68)),
        "relationship_strength": strength,
        "strict_followup": strict,
        "classification": classification,
        "evidence_ids": relationship_evidence_ids_from_payload(
            payload=payload,
            relationship_evidence=relationship_evidence,
        ),
    }
