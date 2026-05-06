from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services.entities.definition_profiles import ENTITY_DEFINITION_MARKERS
from app.services.intents.marker_matching import query_matches_any
from app.services.claims.paper_summary import paper_summary_text

PaperDocLookupFn = Callable[[str], Any]


def infer_entity_type(
    *,
    clients: Any,
    paper_doc_lookup: PaperDocLookupFn,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
) -> str:
    llm_label = llm_infer_entity_type(
        clients=clients,
        paper_doc_lookup=paper_doc_lookup,
        contract=contract,
        papers=papers,
        evidence=evidence,
    )
    if llm_label:
        return llm_label
    text_parts = [item.snippet for item in evidence[:6]]
    for paper in papers[:2]:
        text_parts.extend(
            [
                paper.title,
                str(paper.metadata.get("paper_card_text", "")),
                str(paper.metadata.get("generated_summary", "")),
                str(paper.metadata.get("abstract_note", "")),
            ]
        )
    text = "\n".join(part for part in text_parts if part).lower()
    algorithm_score = 0
    dataset_score = 0
    framework_score = 0
    model_score = 0
    if query_matches_any(text, "", ENTITY_DEFINITION_MARKERS["algorithm_type"]):
        algorithm_score += 3
    if query_matches_any(text, "", ENTITY_DEFINITION_MARKERS["dataset_type"]):
        dataset_score += 2
    if query_matches_any(text, "", ENTITY_DEFINITION_MARKERS["framework_type"]):
        framework_score += 2
    if query_matches_any(text, "", ENTITY_DEFINITION_MARKERS["model_type"]):
        model_score += 2
    scores = {
        "优化算法/训练方法": algorithm_score,
        "数据集/benchmark": dataset_score,
        "框架/系统": framework_score,
        "模型/方法": model_score,
    }
    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score > 0:
        return best_label
    return "方法/框架"


def llm_infer_entity_type(
    *,
    clients: Any,
    paper_doc_lookup: PaperDocLookupFn,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
) -> str:
    if getattr(clients, "chat", None) is None:
        return ""
    payload = clients.invoke_json(
        system_prompt=(
            "你是论文实体类型判别器。"
            "请基于目标术语、问题和局部证据，判断这个目标最像什么类型。"
            "优先围绕目标术语本身判断，不要被整篇论文的大主题带偏。"
            "只输出 JSON，字段为 entity_type, confidence, rationale。"
            "entity_type 尽量归一到这些类型之一："
            "[优化算法/训练方法, 数据集/benchmark, 框架/系统, 模型/方法, 评测任务/设置]。"
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
                    for item in papers[:2]
                ],
                "evidence": [item.snippet[:260] for item in evidence[:5]],
            },
            ensure_ascii=False,
        ),
        fallback={},
    )
    if not isinstance(payload, dict) or not payload:
        return ""
    return canonicalize_entity_type_label(str(payload.get("entity_type", "")).strip())


def canonicalize_entity_type_label(label: str) -> str:
    normalized = " ".join(str(label or "").lower().split())
    if not normalized:
        return ""
    alias_map = {
        "优化算法/训练方法": [
            "优化算法/训练方法",
            "强化学习算法",
            "优化算法",
            "训练方法",
            "algorithm",
            "optimization method",
            "training method",
        ],
        "数据集/benchmark": [
            "数据集/benchmark",
            "数据集",
            "benchmark",
            "dataset",
            "偏好数据集",
            "评测基准",
        ],
        "框架/系统": [
            "框架/系统",
            "框架",
            "系统",
            "framework",
            "system",
            "platform",
        ],
        "模型/方法": [
            "模型/方法",
            "模型",
            "方法",
            "model",
            "method",
        ],
        "评测任务/设置": [
            "评测任务/设置",
            "任务设置",
            "评测任务",
            "task",
            "evaluation setting",
        ],
    }
    for canonical, aliases in alias_map.items():
        normalized_aliases = {" ".join(item.lower().split()) for item in aliases}
        if normalized in normalized_aliases:
            return canonical
    return ""
