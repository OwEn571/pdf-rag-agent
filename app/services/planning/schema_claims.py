from __future__ import annotations

import json
from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan
from app.services.infra.confidence import coerce_confidence_value
from app.services.contracts.context import contract_notes
from app.services.infra.prompt_safety import DOCUMENT_SAFETY_INSTRUCTION, wrap_untrusted_document_text
from app.services.planning.solver_goals import claim_goal_context_from_contract_plan, claim_goals_for_context


_DEFAULT_BLOCKED_GOALS: frozenset[str] = frozenset({
    "formula",
    "origin",
    "paper_title",
    "year",
    "variable_explanation",
    "followup_papers",
    "candidate_relationship",
    "strict_followup",
    "best_topology",
    "langgraph_recommendation",
})


def _parse_blocked_goals_override(value: str) -> frozenset[str] | None:
    """Parse a comma-separated blocked-goals string. Empty string = use default."""
    text = str(value or "").strip()
    if not text:
        return None
    return frozenset({item.strip() for item in text.split(",") if item.strip()})


def blocked_goals(agent_settings: Any) -> frozenset[str]:
    """Return the effective blocked-goals set, respecting any settings override."""
    raw = getattr(agent_settings, "generic_claim_solver_blocked_goals", "") if agent_settings is not None else ""
    override = _parse_blocked_goals_override(raw)
    return override if override is not None else _DEFAULT_BLOCKED_GOALS


def should_use_schema_claim_solver(*, contract: QueryContract, plan: ResearchPlan, agent_settings: Any = None) -> bool:
    goals = claim_goals_for_context(claim_goal_context_from_contract_plan(contract=contract, plan=plan))
    return not bool(goals & blocked_goals(agent_settings))


def schema_claim_system_prompt() -> str:
    return (
        "你是论文研究助手的通用证据 Claim 抽取器。"
        "不要根据 relation 分支套模板。"
        "只基于输入 evidence 和 papers，输出 JSON："
        "{claims:[{claim_type, entity, value, structured_data, evidence_ids, paper_ids, confidence, required}]}。"
        "claim_type 应来自用户目标和 requested_fields，例如 definition/formula/metric_value/"
        "paper_summary/followup_research/recommendation/general_answer。"
        "每条 claim 必须引用能支撑它的 evidence_ids；证据不足就返回空 claims。"
        "不要编造 evidence 中不存在的论文、指标、公式或结论。"
        f"{DOCUMENT_SAFETY_INSTRUCTION}"
    )


def schema_claim_human_prompt(
    *,
    contract: QueryContract,
    plan: ResearchPlan,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    conversation_context: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "query": contract.clean_query,
            "intent_adapter": {
                "relation": contract.relation,
                "targets": contract.targets,
                "requested_fields": contract.requested_fields,
                "answer_shape": contract.answer_shape,
                "precision_requirement": contract.precision_requirement,
                "notes": contract_notes(contract),
            },
            "required_claims": plan.required_claims,
            "papers": [
                {
                    "paper_id": item.paper_id,
                    "title": item.title,
                    "year": item.year,
                }
                for item in papers[:8]
            ],
            "evidence": [
                {
                    "doc_id": item.doc_id,
                    "paper_id": item.paper_id,
                    "title": item.title,
                    "page": item.page,
                    "block_type": item.block_type,
                    "caption": item.caption,
                    "snippet": wrap_untrusted_document_text(
                        item.snippet,
                        doc_id=item.doc_id,
                        title=item.title,
                        source=item.block_type or "pdf",
                        max_chars=1200,
                    ),
                }
                for item in evidence[:40]
            ],
            "conversation_context": conversation_context,
        },
        ensure_ascii=False,
    )


def claims_from_schema_payload(
    payload: Any,
    *,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    max_claims: int = 12,
) -> list[Claim]:
    if not isinstance(payload, dict):
        return []
    raw_claims = payload.get("claims", [])
    if not isinstance(raw_claims, list):
        return []
    evidence_ids = {item.doc_id for item in evidence}
    paper_ids = {item.paper_id for item in papers} | {item.paper_id for item in evidence}
    claims: list[Claim] = []
    for item in raw_claims[:max_claims]:
        if not isinstance(item, dict):
            continue
        selected_evidence_ids = _selected_ids(item.get("evidence_ids", []), allowed=evidence_ids)
        if not selected_evidence_ids:
            continue
        selected_paper_ids = _selected_ids(item.get("paper_ids", []), allowed=paper_ids) or list(
            dict.fromkeys(block.paper_id for block in evidence if block.doc_id in selected_evidence_ids)
        )
        structured_data = item.get("structured_data", {})
        if not isinstance(structured_data, dict):
            structured_data = {}
        structured_data = dict(structured_data)
        structured_data["source"] = "schema_claim_solver"
        claims.append(
            Claim(
                claim_type=str(item.get("claim_type", "") or "general_answer"),
                entity=str(item.get("entity", "") or (contract.targets[0] if contract.targets else "")),
                value=str(item.get("value", "") or ""),
                structured_data=structured_data,
                evidence_ids=selected_evidence_ids,
                paper_ids=list(dict.fromkeys(selected_paper_ids)),
                confidence=coerce_confidence_value(item.get("confidence", 0.72), default=0.82),
                required=bool(item.get("required", True)),
            )
        )
    return claims


def _selected_ids(value: Any, *, allowed: set[str]) -> list[str]:
    raw_ids = [value] if isinstance(value, str) else value
    if not isinstance(raw_ids, list):
        return []
    return [str(item).strip() for item in raw_ids if str(item).strip() in allowed]
