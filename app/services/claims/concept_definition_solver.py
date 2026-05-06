from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract
from app.services.infra.confidence import coerce_claim_confidence
from app.services.infra.prompt_safety import DOCUMENT_SAFETY_INSTRUCTION, wrap_untrusted_document_text
from app.services.intents.marker_matching import MarkerProfile, query_matches_any
from app.services.claims.paper_summary import paper_summary_text

PaperDocLookupFn = Callable[[str], Any]


CONCEPT_REASONING_MARKERS: dict[str, MarkerProfile] = {
    "category_rl": ("on-policy", "policy optimization", "reinforcement learning", "rlhf", "reward model"),
    "category_dataset": ("dataset", "benchmark", "数据集"),
    "category_framework": ("framework", "platform", "system", "架构"),
    "category_objective": ("loss", "objective", "偏好优化", "优化目标"),
    "detail_human_feedback": ("rlhf", "human feedback", "人类反馈"),
    "detail_policy_optimization": ("policy optimization", "策略优化"),
    "detail_retrieval_generation": (
        "retrieval- augmented generation",
        "retrieval augmented generation",
        "retrieval-augmented generation",
    ),
}


def solve_concept_definition_claims(
    *,
    clients: Any,
    paper_doc_lookup: PaperDocLookupFn,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
) -> list[Claim]:
    claim = build_concept_definition_claim(
        clients=clients,
        paper_doc_lookup=paper_doc_lookup,
        contract=contract,
        papers=papers,
        evidence=evidence,
    )
    return [claim] if claim is not None else []


def build_concept_definition_claim(
    *,
    clients: Any,
    paper_doc_lookup: PaperDocLookupFn,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
) -> Claim | None:
    if not evidence:
        return None
    supporting = evidence[: min(5, len(evidence))]
    llm_payload = llm_build_concept_definition(
        clients=clients,
        paper_doc_lookup=paper_doc_lookup,
        contract=contract,
        papers=papers,
        evidence=supporting,
    )
    if llm_payload:
        expansion = str(llm_payload.get("expansion", "") or "").strip()
        category = canonicalize_concept_category(str(llm_payload.get("category", "") or "").strip())
        definition = str(llm_payload.get("definition", "") or "").strip()
        if definition:
            raw_doc_ids = llm_payload.get("supporting_doc_ids", [])
            supporting_doc_ids = (
                [str(item).strip() for item in raw_doc_ids if str(item).strip()]
                if isinstance(raw_doc_ids, list)
                else []
            )
            selected_evidence = [item for item in supporting if item.doc_id in supporting_doc_ids] or supporting[:3]
            paper_ids = list(dict.fromkeys(item.paper_id for item in selected_evidence))
            confidence = coerce_claim_confidence(llm_payload.get("confidence", 0.84))
            return Claim(
                claim_type="concept_definition",
                entity=contract.targets[0] if contract.targets else supporting[0].title,
                value=definition,
                structured_data={
                    "expansion": expansion,
                    "category": category,
                    "supporting_lines": [item.snippet[:240] for item in selected_evidence[:3]],
                    "relevance_score": selected_evidence[0].score if selected_evidence else supporting[0].score,
                },
                evidence_ids=[item.doc_id for item in selected_evidence[:3]],
                paper_ids=paper_ids,
                confidence=confidence,
            )
    target = contract.targets[0] if contract.targets else ""
    expansion = extract_acronym_expansion(target=target, evidence=supporting)
    category = infer_concept_category(target=target, evidence=supporting, expansion=expansion)
    answer = compose_concept_definition_text(
        target=target or supporting[0].title,
        expansion=expansion,
        category=category,
        evidence=supporting,
    )
    return Claim(
        claim_type="concept_definition",
        entity=target or supporting[0].title,
        value=answer,
        structured_data={
            "expansion": expansion,
            "category": category,
            "supporting_lines": [item.snippet[:240] for item in supporting[:3]],
            "relevance_score": supporting[0].score,
        },
        evidence_ids=[item.doc_id for item in supporting[:3]],
        paper_ids=list(dict.fromkeys(item.paper_id for item in supporting[:3])),
        confidence=0.85 if expansion else 0.76,
    )


def llm_build_concept_definition(
    *,
    clients: Any,
    paper_doc_lookup: PaperDocLookupFn,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
) -> dict[str, object]:
    if getattr(clients, "chat", None) is None:
        return {}
    payload = clients.invoke_json(
        system_prompt=(
            "你是论文概念解释器。"
            "请基于目标术语、当前问题、候选论文和局部证据，生成一个严格基于证据的概念解释。"
            "只输出 JSON，字段为 expansion, category, definition, supporting_doc_ids, confidence。"
            "category 尽量归一到这些类型之一："
            "[强化学习算法, 数据集/benchmark, 框架/系统, 训练目标/优化方法, 概念/方法]。"
            "definition 需要直接回答用户的问题，优先解释这个概念本身是什么、常见用于什么，不要只摘抄一句原文。"
            "如果证据不足，就返回空对象。"
            f"{DOCUMENT_SAFETY_INSTRUCTION}"
        ),
        human_prompt=json.dumps(
            {
                "query": contract.clean_query,
                "target": contract.targets[0] if contract.targets else "",
                "requested_fields": contract.requested_fields,
                "papers": [
                    {
                        "paper_id": item.paper_id,
                        "title": item.title,
                        "summary": paper_summary_text(item.paper_id, paper_doc_lookup=paper_doc_lookup),
                    }
                    for item in papers[:3]
                ],
                "evidence": [
                    {
                        "doc_id": item.doc_id,
                        "paper_id": item.paper_id,
                        "title": item.title,
                        "page": item.page,
                        "snippet": wrap_untrusted_document_text(item.snippet[:280], doc_id=item.doc_id, title=item.title),
                    }
                    for item in evidence[:5]
                ],
            },
            ensure_ascii=False,
        ),
        fallback={},
    )
    return payload if isinstance(payload, dict) else {}


def canonicalize_concept_category(label: str) -> str:
    normalized = " ".join(str(label or "").lower().split())
    if not normalized:
        return ""
    alias_map = {
        "强化学习算法": [
            "强化学习算法",
            "优化算法",
            "algorithm",
            "rl algorithm",
            "policy optimization algorithm",
        ],
        "数据集/benchmark": [
            "数据集/benchmark",
            "数据集",
            "benchmark",
            "dataset",
        ],
        "框架/系统": [
            "框架/系统",
            "框架",
            "系统",
            "framework",
            "system",
            "platform",
        ],
        "训练目标/优化方法": [
            "训练目标/优化方法",
            "训练目标",
            "优化方法",
            "training objective",
            "optimization method",
        ],
        "概念/方法": [
            "概念/方法",
            "概念",
            "方法",
            "concept",
            "method",
        ],
    }
    for canonical, aliases in alias_map.items():
        normalized_aliases = {" ".join(item.lower().split()) for item in aliases}
        if normalized in normalized_aliases:
            return canonical
    return ""


def extract_acronym_expansion(*, target: str, evidence: list[EvidenceBlock]) -> str:
    target_text = str(target or "").strip()
    if not target_text:
        return ""
    stopwords = {"and", "or", "the", "a", "an", "to", "for", "with", "via", "on", "in", "of", "from"}
    initials_target = "".join(ch for ch in target_text.upper() if ch.isalnum())
    for item in evidence[:5]:
        text = " ".join(item.snippet.split())
        match = re.search(rf"{re.escape(target_text)}\s*\(\s*([A-Za-z][A-Za-z0-9 /\-]{{3,80}}?)\s*\)", text, re.IGNORECASE)
        if match is not None:
            phrase = match.group(1).strip(" .,:;")
            if len(phrase) > len(target_text):
                return phrase
        for needle in [f"({target_text})", f"（{target_text}）"]:
            index = text.find(needle)
            if index < 0:
                continue
            prefix = text[max(0, index - 100) : index]
            words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", prefix)
            if len(words) < 2:
                continue
            for size in range(2, min(8, len(words)) + 1):
                phrase_words = words[-size:]
                initials = "".join(word[0].upper() for word in phrase_words if word.lower() not in stopwords)
                if initials != initials_target:
                    continue
                phrase = " ".join(phrase_words).strip(" .,:;")
                if phrase and len(phrase) > len(target_text):
                    return phrase.title() if phrase.islower() else phrase
    return ""


def infer_concept_category(*, target: str, evidence: list[EvidenceBlock], expansion: str) -> str:
    _ = target
    joined = "\n".join(item.snippet for item in evidence[:4]).lower()
    if query_matches_any(joined, "", CONCEPT_REASONING_MARKERS["category_rl"]):
        return "强化学习算法"
    if query_matches_any(joined, "", CONCEPT_REASONING_MARKERS["category_dataset"]):
        return "数据集/benchmark"
    if query_matches_any(joined, "", CONCEPT_REASONING_MARKERS["category_framework"]):
        return "框架/系统"
    if query_matches_any(joined, "", CONCEPT_REASONING_MARKERS["category_objective"]):
        return "训练目标/优化方法"
    if expansion and "optimization" in expansion.lower():
        return "训练目标/优化方法"
    if expansion and "generation" in expansion.lower():
        return "概念/方法"
    return "概念/方法"


def compose_concept_definition_text(
    *,
    target: str,
    expansion: str,
    category: str,
    evidence: list[EvidenceBlock],
) -> str:
    joined = "\n".join(item.snippet for item in evidence[:4]).lower()
    lead = f"{target} 通常指 {expansion}，是一种{category}。" if expansion else f"根据当前语料，{target} 是一种{category}。"
    details: list[str] = []
    if category == "强化学习算法":
        if "on-policy" in joined:
            details.append("它属于 on-policy 方法")
        if query_matches_any(joined, "", CONCEPT_REASONING_MARKERS["detail_human_feedback"]):
            details.append("常用于 RLHF / LLM 对齐场景")
        if "reward model" in joined:
            details.append("通常围绕奖励模型信号更新策略")
        if query_matches_any(joined, "", CONCEPT_REASONING_MARKERS["detail_policy_optimization"]):
            details.append("核心目标是稳定地优化策略")
    elif category in {"框架/系统", "概念/方法"}:
        if query_matches_any(joined, "", CONCEPT_REASONING_MARKERS["detail_retrieval_generation"]):
            details.append("核心思路是先检索相关信息，再结合生成模型组织回答")
        elif "retrieval" in joined and "generation" in joined:
            details.append("它把检索和生成结合在同一个回答链路里")
        if "agent" in joined and "retrieval" in joined:
            details.append("在 agent 场景里常被用来先找资料再执行或回答")
    elif category == "数据集/benchmark":
        details.append("主要用于评测或构造标准化实验任务")
    elif category == "训练目标/优化方法":
        details.append("通常用来定义优化方向、损失形式或训练目标")
    if not details and evidence:
        snippet = " ".join(evidence[0].snippet.split())[:120]
        details.append(f"当前最相关的证据提到：{snippet}")
    return lead + (" " + "，".join(details) + "。" if details else "")
