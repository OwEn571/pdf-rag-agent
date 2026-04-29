from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services.contract_normalization import normalize_lookup_text
from app.services.evidence_presentation import safe_year
from app.services.followup_relationship_intents import (
    followup_relevance_score,
    has_followup_domain_signal,
    has_followup_seed_intro_signal,
    has_followup_soft_relation_signal,
    has_followup_support_relation_signal,
    target_relation_cue_near_text,
)
from app.services.query_shaping import matches_target

PaperText = Callable[[str], str]
PaperAnchor = Callable[[CandidatePaper], str]
ConfidenceCoercer = Callable[[Any], float]
EvidenceExpander = Callable[[list[str], str, QueryContract, int], list[EvidenceBlock]]


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


def paper_anchor_text(paper: CandidatePaper) -> str:
    title = str(paper.title or "").strip()
    if not title:
        return ""
    for separator in [":", " - ", " — ", " – "]:
        if separator in title:
            return title.split(separator, 1)[0].strip()
    return title


def paper_brief(
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
    }


def followup_seed_selector_system_prompt() -> str:
    return (
        "你是论文关系求解器中的种子论文定位器。"
        "请根据当前问题、目标实体和候选论文，找出用户真正想追踪其后续工作的原始/种子论文。"
        "如果目标是数据集、方法或模型，优先选择“引入/提出/定义该对象”的论文，而不是后续扩展。"
        "只输出 JSON，字段为 seed_paper_ids 和 rationale。"
    )


def followup_seed_selector_human_prompt(
    *,
    contract: QueryContract,
    active_titles: list[str],
    candidates: list[CandidatePaper],
    paper_summary_text: PaperText,
) -> str:
    return json.dumps(
        {
            "query": contract.clean_query,
            "targets": contract.targets,
            "active_titles": active_titles,
            "candidates": [
                paper_brief(paper=item, paper_summary_text=paper_summary_text)
                for item in candidates[:8]
            ],
        },
        ensure_ascii=False,
    )


def followup_candidate_ranker_system_prompt() -> str:
    return (
        "你是论文关系分析器。"
        "请判断哪些候选论文是种子论文的后续研究、扩展工作、迁移工作或直接延续。"
        "后续工作必须与种子论文的对象、问题设定或方法线索直接相关；只在同一大领域但关系松散的论文不要选。"
        "绝对不要把种子论文本身选进去。"
        "只输出 JSON，字段为 followups。followups 中每项包含 paper_id, relation_type, reason, confidence。"
    )


def followup_candidate_ranker_human_prompt(
    *,
    contract: QueryContract,
    seed_papers: list[CandidatePaper],
    candidates: list[CandidatePaper],
    paper_summary_text: PaperText,
) -> str:
    return json.dumps(
        {
            "query": contract.clean_query,
            "targets": contract.targets,
            "seed_papers": [
                paper_brief(paper=item, paper_summary_text=paper_summary_text)
                for item in seed_papers[:2]
            ],
            "candidates": [
                paper_brief(paper=item, paper_summary_text=paper_summary_text)
                for item in candidates[:10]
            ],
        },
        ensure_ascii=False,
    )


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


def extract_followup_keyphrases(text: str) -> list[str]:
    lowered = str(text or "").lower()
    phrase_bank = [
        "user-level alignment",
        "personalized preference",
        "personalized alignment",
        "preference inference",
        "conditioned generation",
        "transferable personalization",
        "modular personalization",
        "user preference",
        "preference summary",
        "personalization",
        "alignment",
        "benchmark",
        "dataset",
    ]
    phrases = [phrase for phrase in phrase_bank if phrase in lowered]
    title_like = re.sub(r"[^a-z0-9\s-]", " ", lowered)
    words = [
        word
        for word in title_like.split()
        if len(word) >= 5 and word not in {"large", "language", "models", "paper", "using", "through", "across"}
    ]
    frequent: list[str] = []
    for word in words:
        if words.count(word) >= 2 and word not in frequent:
            frequent.append(word)
    return [*phrases, *frequent[:8]]


def followup_expansion_terms(
    *,
    paper: CandidatePaper,
    paper_summary_text: PaperText,
) -> str:
    text = f"{paper.title}\n{paper_summary_text(paper.paper_id)}\n{paper.metadata.get('paper_card_text', '')}".lower()
    terms = extract_followup_keyphrases(text)
    if has_followup_domain_signal(text):
        terms.extend(["follow-up", "extension", "downstream", "benchmark", "transfer", "personalization", "preference"])
    return " ".join(dict.fromkeys(item for item in terms if item))[:600]


def followup_reason_fallback(
    *,
    seed_papers: list[CandidatePaper],
    paper: CandidatePaper,
    paper_summary_text: PaperText,
) -> str:
    seed_titles = ", ".join(item.title for item in seed_papers[:1])
    summary = " ".join(paper_summary_text(paper.paper_id).split())
    if len(summary) > 120:
        summary = summary[:117].rstrip() + "..."
    if seed_titles:
        return f"它延续了《{seed_titles}》相关主题，重点在于：{summary or '与该方向直接相关。'}"
    return summary or "与当前研究方向直接相关。"


def infer_followup_relation_type(
    *,
    paper: CandidatePaper,
    paper_summary_text: PaperText,
    strict: bool = False,
) -> str:
    summary = paper_summary_text(paper.paper_id).lower()
    if strict and any(token in summary for token in ["uses", "using", "evaluate", "evaluates", "benchmark", "dataset"]):
        return "直接使用/评测证据"
    if strict:
        return "直接后续/扩展证据"
    if any(token in summary for token in ["dataset", "benchmark", "evaluation"]):
        return "dataset/benchmark continuation"
    if any(token in summary for token in ["transfer", "cross-task", "cross model"]):
        return "transfer extension"
    if any(token in summary for token in ["reasoning", "behavioral", "preference inference"]):
        return "method/model extension"
    return "related continuation"


def paper_keyword_set(
    papers: list[CandidatePaper],
    *,
    paper_summary_text: PaperText,
) -> set[str]:
    keywords: set[str] = set()
    stopwords = {
        "that",
        "with",
        "from",
        "this",
        "their",
        "into",
        "through",
        "using",
        "large",
        "language",
        "models",
        "model",
        "paper",
        "across",
        "approach",
        "approaches",
        "average",
        "different",
        "diverse",
        "demonstrate",
        "demonstrates",
        "method",
        "methods",
        "task",
        "tasks",
        "result",
        "results",
        "performance",
        "application",
        "applications",
    }
    for paper in papers:
        text = f"{paper.title} {paper_summary_text(paper.paper_id)}"
        for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", text.lower()):
            if token.endswith("ies") and len(token) > 5:
                token = token[:-3] + "y"
            elif token.endswith("s") and len(token) > 6:
                token = token[:-1]
            if token not in stopwords:
                keywords.add(token)
    return keywords


def paper_author_tokens(papers: list[CandidatePaper]) -> set[str]:
    tokens: set[str] = set()
    for paper in papers:
        authors = str(paper.metadata.get("authors", ""))
        for token in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", authors.lower()):
            if token not in {"and", "et", "al"}:
                tokens.add(token)
    return tokens


def followup_seed_score(
    *,
    contract: QueryContract,
    paper: CandidatePaper,
    active_titles: list[str],
    paper_summary_text: PaperText,
) -> float:
    score = paper.score
    summary = paper_summary_text(paper.paper_id)
    haystack = f"{paper.title}\n{summary}\n{paper.metadata.get('paper_card_text', '')}".lower()
    if paper.title in active_titles:
        score += 2.5
    for target in contract.targets:
        if target and matches_target(haystack, target.lower()):
            score += 1.1
    if has_followup_seed_intro_signal(haystack):
        score += 1.2
    year = safe_year(paper.year)
    if year < 9999:
        score += max(0.0, (2100 - year) / 1000.0)
    return score


def followup_relationship_assessment(
    *,
    contract: QueryContract,
    seed_papers: list[CandidatePaper],
    paper: CandidatePaper,
    paper_summary_text: PaperText,
) -> dict[str, Any]:
    target_aliases = followup_target_aliases(
        contract=contract,
        seed_papers=seed_papers,
        paper_anchor_text=paper_anchor_text,
    )
    seed_keywords = paper_keyword_set(seed_papers, paper_summary_text=paper_summary_text)
    candidate_keywords = paper_keyword_set([paper], paper_summary_text=paper_summary_text)
    seed_phrases: set[str] = set()
    for seed in seed_papers:
        seed_text = f"{seed.title}\n{paper_summary_text(seed.paper_id)}\n{seed.metadata.get('paper_card_text', '')}"
        seed_phrases.update(extract_followup_keyphrases(seed_text))
    seed_author_tokens = paper_author_tokens(seed_papers)
    summary = paper_summary_text(paper.paper_id)
    haystack = f"{paper.title}\n{summary}\n{paper.metadata.get('paper_card_text', '')}\n{paper.metadata.get('abstract_note', '')}"
    lowered = haystack.lower()
    score = 0.0
    explicit_direct_signals: list[str] = []
    support_signals: list[str] = []
    target_seen = ""
    for alias in target_aliases:
        if alias and matches_target(haystack, alias):
            target_seen = alias
            if target_relation_cue_near_text(text=haystack, target=alias):
                score += 3.2
                explicit_direct_signals.append(f"候选摘要/元数据明确提到并使用、评测或扩展 {alias}")
            else:
                score += 1.2
                support_signals.append(f"候选摘要/元数据提到 {alias}")
            break
    candidate_phrases = set(extract_followup_keyphrases(haystack))
    phrase_overlap = {
        phrase
        for phrase in seed_phrases & candidate_phrases
        if " " in phrase and phrase not in {"large language models"}
    }
    if phrase_overlap:
        score += min(1.2, len(phrase_overlap) * 0.35)
        support_signals.append("共享研究线索：" + "、".join(sorted(phrase_overlap)[:4]))
    overlap = seed_keywords & candidate_keywords
    displayable_topic_terms = {
        "behavioral",
        "signal",
        "signals",
        "persona",
        "decoding",
        "transfer",
        "transferable",
        "conditioned",
        "generation",
        "profile",
        "profiles",
        "hypothesis",
        "preference-inference",
        "user-level",
    }
    topical_overlap = {
        token
        for token in overlap
        if token in displayable_topic_terms
    }
    if topical_overlap and not phrase_overlap:
        score += min(1.0, len(topical_overlap) * 0.16)
        support_signals.append("共享部分主题词：" + "、".join(sorted(topical_overlap)[:4]))
    author_overlap = seed_author_tokens & paper_author_tokens([paper])
    if author_overlap:
        score += min(1.0, len(author_overlap) * 0.35)
        support_signals.append("存在作者重合")
    if has_followup_support_relation_signal(lowered):
        score += 0.45
        support_signals.append("包含扩展、使用、评测或迁移类关系词")
    if has_followup_domain_signal(lowered):
        score += 0.35
        support_signals.append("主题属于 personalized preference / alignment 相邻方向")
    if explicit_direct_signals:
        strength = "direct"
        relation_type = infer_followup_relation_type(
            paper=paper,
            paper_summary_text=paper_summary_text,
            strict=True,
        )
        reason_bits = explicit_direct_signals + support_signals[:2]
        confidence = 0.86
    elif score >= 1.4 and (target_seen or support_signals):
        strength = "strong_related"
        relation_type = "强相关延续候选"
        reason_bits = support_signals
        confidence = 0.66
    else:
        strength = "weak_related"
        relation_type = "同主题待确认候选"
        reason_bits = support_signals
        confidence = 0.48
    reason = "；".join(dict.fromkeys(reason_bits)) if reason_bits else followup_reason_fallback(
        seed_papers=seed_papers,
        paper=paper,
        paper_summary_text=paper_summary_text,
    )
    if strength != "direct" and reason:
        reason = f"{reason}；目前证据不足以确认它严格继承、引用或使用种子论文/数据集。"
    return {
        "score": score,
        "strength": strength,
        "relation_type": relation_type,
        "reason": reason,
        "confidence": confidence,
    }


def rank_followup_candidates_fallback(
    *,
    contract: QueryContract,
    seed_papers: list[CandidatePaper],
    candidates: list[CandidatePaper],
    paper_summary_text: PaperText,
) -> list[dict[str, Any]]:
    seed_keywords = paper_keyword_set(seed_papers, paper_summary_text=paper_summary_text)
    seed_author_tokens = paper_author_tokens(seed_papers)
    target_text = " ".join(contract.targets)
    seed_year = min((safe_year(item.year) for item in seed_papers), default=9999)
    seed_ids = {item.paper_id for item in seed_papers}
    scored: list[tuple[float, CandidatePaper, dict[str, Any]]] = []
    for paper in filter_followup_candidates(
        contract=contract,
        candidates=candidates,
        paper_summary_text=paper_summary_text,
    ):
        if paper.paper_id in seed_ids:
            continue
        score = paper.score
        summary = paper_summary_text(paper.paper_id)
        haystack = f"{paper.title}\n{summary}\n{paper.metadata.get('paper_card_text', '')}"
        if target_text and matches_target(haystack.lower(), target_text.lower()):
            score += 1.2
        if seed_year < 9999:
            year = safe_year(paper.year)
            if year >= seed_year:
                score += 0.4 + min(0.5, max(0, year - seed_year) * 0.1)
        overlap = len(seed_keywords & paper_keyword_set([paper], paper_summary_text=paper_summary_text))
        if overlap:
            score += min(1.2, overlap * 0.18)
        author_overlap = len(seed_author_tokens & paper_author_tokens([paper]))
        if author_overlap:
            score += min(0.8, author_overlap * 0.25)
        if has_followup_soft_relation_signal(haystack):
            score += 0.35
        score += followup_relevance_score(haystack)
        assessment = followup_relationship_assessment(
            contract=contract,
            seed_papers=seed_papers,
            paper=paper,
            paper_summary_text=paper_summary_text,
        )
        if assessment["score"] < 0.3:
            continue
        score += float(assessment["score"])
        scored.append((score, paper, assessment))
    ranked = [
        (paper, assessment)
        for _, paper, assessment in sorted(scored, key=lambda item: (-item[0], safe_year(item[1].year), item[1].title))
    ]
    results: list[dict[str, Any]] = []
    for paper, assessment in ranked[:10]:
        results.append(
            {
                "paper": paper,
                "relation_type": str(assessment["relation_type"]),
                "reason": str(assessment["reason"]),
                "confidence": float(assessment["confidence"]),
                "relationship_strength": str(assessment["strength"]),
            }
        )
    return results


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


def followup_relationship_evidence(
    *,
    contract: QueryContract,
    seed_papers: list[CandidatePaper],
    paper: CandidatePaper,
    evidence: list[EvidenceBlock],
    expand_evidence: EvidenceExpander,
) -> list[EvidenceBlock]:
    seed_ids = [item.paper_id for item in seed_papers[:2]]
    pair_ids = list(dict.fromkeys([*seed_ids, paper.paper_id]))
    selected = [item for item in evidence if item.paper_id in set(pair_ids)]
    if len(selected) < 6:
        query_parts = [
            contract.clean_query,
            " ".join(contract.targets),
            paper.title,
            "uses evaluates benchmark extends cites dataset method follow-up",
        ]
        expanded = expand_evidence(
            pair_ids,
            " ".join(part for part in query_parts if part),
            contract.model_copy(update={"required_modalities": ["page_text", "paper_card"]}),
            12,
        )
        by_id = {item.doc_id: item for item in selected}
        for item in expanded:
            by_id.setdefault(item.doc_id, item)
        selected = list(by_id.values())
    selected.sort(key=lambda item: (0 if item.paper_id == paper.paper_id else 1, -item.score, item.page, item.doc_id))
    return selected[:12]
