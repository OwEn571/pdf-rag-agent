from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock
from app.services.answers.evidence_presentation import extract_topology_terms
from app.services.infra.prompt_safety import DOCUMENT_SAFETY_INSTRUCTION, wrap_untrusted_document_text
from app.services.intents.marker_matching import MarkerProfile, query_matches_any


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


def clean_topology_public_text(text: str) -> str:
    compact = " ".join(str(text or "").split())
    if not compact:
        return ""
    lowered = compact.lower()
    if query_matches_any(lowered, "", UNUSABLE_TOPOLOGY_RECOMMENDATION_MARKERS):
        return ""
    ascii_letters = sum(1 for char in compact if ("a" <= char.lower() <= "z"))
    chinese_chars = sum(1 for char in compact if "\u4e00" <= char <= "\u9fff")
    if ascii_letters > max(80, chinese_chars * 3):
        return ""
    if compact and compact[-1] not in "。.!?！？":
        compact += "。"
    return compact


def compose_topology_recommendation_answer(
    *,
    claims: list[Claim],
    evidence: list[EvidenceBlock],
) -> str:
    if not claims:
        return ""
    claim = claims[0]
    structured = dict(claim.structured_data or {})
    terms = [str(item) for item in structured.get("topology_terms", []) if str(item).strip()]
    if not terms:
        terms = extract_topology_terms(evidence)
    terms_text = "、".join(terms or ["DAG", "irregular/random", "chain", "tree", "mesh"])
    summary = clean_topology_public_text(str(claim.value or structured.get("summary", "") or "").strip())
    rationale = clean_topology_public_text(str(structured.get("rationale", "") or "").strip())
    return (
        "## 结论\n\n"
        "我会用 **DAG 作为主干拓扑** 来组织 PDF-Agent 的 multi-agent 系统；"
        "局部可以嵌入 chain 或 tree，但不建议把整套系统做成全 mesh。\n\n"
        "## 证据边界\n\n"
        f"- 当前论文证据主要覆盖 {terms_text} 等 multi-agent topology。"
        f"{summary if summary else '这些证据没有证明存在一个脱离任务的绝对最优拓扑。'}\n"
        "- 因此下面是“论文证据 + PDF-RAG 工程约束”的设计建议，不是某篇论文直接给出的定理。\n\n"
        "## 组织建议\n\n"
        "- **DAG 主干**：把意图识别、论文召回、证据扩展、公式/表格/图像解析、claim verification、answer compose 做成有依赖的节点；可并行的检索和解析分支并行跑，最后汇总。\n"
        "- **局部 chain**：单篇 PDF 精读、OCR 清洗、公式抽取这类严格顺序步骤可以保持 chain，便于调试和重试。\n"
        "- **局部 tree**：多篇论文比较、多个候选含义消歧、多个公式定义聚合时，用 tree 做分解-汇总会更自然。\n"
        "- **少用 mesh / irregular**：它们适合探索式协作，但对 PDF-Agent 来说更容易带来状态漂移、重复检索和不可控成本。\n"
        f"- 选择时优先看任务约束：{rationale if rationale else '证据质量、节点成本、是否需要并行验证，以及最终回答是否要可追溯。'}"
    )


def topology_recommendation_system_prompt() -> str:
    return (
        "你是 topology 证据分析器。"
        "请只输出 JSON，字段为 overall_best, engineering_best, rationale, summary。"
        "必须严格基于给定证据，不要使用外部知识。"
        f"{DOCUMENT_SAFETY_INSTRUCTION}"
    )


def topology_recommendation_human_prompt(*, topology_terms: list[str], evidence: list[EvidenceBlock]) -> str:
    return json.dumps(
        {
            "topology_terms": topology_terms,
            "evidence": [
                wrap_untrusted_document_text(item.snippet[:260], doc_id=item.doc_id, title=item.title)
                for item in evidence[:6]
            ],
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
