from __future__ import annotations

import json
from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan, VerificationReport
from app.services.infra.prompt_safety import DOCUMENT_SAFETY_INSTRUCTION, wrap_untrusted_document_text


def verify_claims_with_schema_llm(
    *,
    clients: Any,
    contract: QueryContract,
    plan: ResearchPlan,
    claims: list[Claim],
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
) -> VerificationReport | None:
    if getattr(clients, "chat", None) is None:
        return None
    if not any(dict(claim.structured_data or {}).get("source") == "schema_claim_solver" for claim in claims):
        return None
    payload = clients.invoke_json(
        system_prompt=(
            "你是论文研究助手的通用证据覆盖验证器。"
            "不要按 relation 写分支规则。"
            "给定 query_contract、required_claims、claims 和 evidence，"
            "判断 claims 是否被 evidence 覆盖。"
            f"{DOCUMENT_SAFETY_INSTRUCTION}"
            "只输出 JSON：status(pass|retry|clarify), missing_fields, recommended_action, contradictions。"
            "retry 表示可以通过更多检索补足；clarify 表示必须让用户消歧或补槽。"
        ),
        human_prompt=json.dumps(
            {
                "query": contract.clean_query,
                "contract": contract.model_dump(),
                "required_claims": plan.required_claims,
                "claims": [item.model_dump() for item in claims],
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
                        "snippet": wrap_untrusted_document_text(
                            item.snippet,
                            doc_id=item.doc_id,
                            title=item.title,
                            source=item.block_type or "pdf",
                            max_chars=1000,
                        ),
                    }
                    for item in evidence[:40]
                ],
            },
            ensure_ascii=False,
        ),
        fallback={},
    )
    if not isinstance(payload, dict) or not payload:
        return None
    status = str(payload.get("status", "") or "").strip().lower()
    if status not in {"pass", "retry", "clarify"}:
        return None
    missing = payload.get("missing_fields", [])
    if isinstance(missing, str):
        missing_fields = [missing]
    elif isinstance(missing, list):
        missing_fields = [str(item).strip() for item in missing if str(item).strip()]
    else:
        missing_fields = []
    contradictions = payload.get("contradictions", [])
    contradictory_claims: list[str] = []
    if isinstance(contradictions, list):
        contradictory_claims = [str(item).strip() for item in contradictions if str(item).strip()]
    return VerificationReport(
        status=status,  # type: ignore[arg-type]
        missing_fields=missing_fields,
        recommended_action=str(payload.get("recommended_action", "") or f"schema_verifier_{status}"),
        contradictory_claims=contradictory_claims,
    )


def verify_formula_claims_with_llm(
    *,
    clients: Any,
    contract: QueryContract,
    claims: list[Claim],
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
) -> VerificationReport | None:
    if getattr(clients, "chat", None) is None or not claims:
        return None
    claim_evidence_ids = {doc_id for claim in claims for doc_id in claim.evidence_ids}
    claim_paper_ids = {paper_id for claim in claims for paper_id in claim.paper_ids}
    relevant_evidence = [
        item
        for item in evidence
        if item.doc_id in claim_evidence_ids or item.paper_id in claim_paper_ids
    ]
    if not relevant_evidence:
        relevant_evidence = evidence[:20]
    payload = clients.invoke_json(
        system_prompt=(
            "你是论文公式 claim verifier。给定用户目标、formula claims 和 evidence，"
            "判断每条公式是否被 evidence 支撑，且是否对应用户 targets。"
            "不要因为 evidence 同时讨论 PPO/DPO/其他算法就否决；只看 claim.value 的公式、"
            "claim.entity/paper_title 和 evidence 中的明确指代是否一致。"
            "如果公式是从常识模板补出来、目标不一致或证据不足，返回 retry。"
            "只输出 JSON：status(pass|retry|clarify), missing_fields, unsupported_claims, "
            "contradictory_claims, recommended_action。"
        ),
        human_prompt=json.dumps(
            {
                "query": contract.clean_query,
                "targets": contract.targets,
                "answer_slots": list(getattr(contract, "answer_slots", []) or []),
                "claims": [claim.model_dump() for claim in claims],
                "papers": [
                    {
                        "paper_id": item.paper_id,
                        "title": item.title,
                        "year": item.year,
                    }
                    for item in papers[:12]
                ],
                "evidence": [
                    {
                        "doc_id": item.doc_id,
                        "paper_id": item.paper_id,
                        "title": item.title,
                        "page": item.page,
                        "block_type": item.block_type,
                        "caption": item.caption,
                        "snippet": item.snippet[:1400],
                    }
                    for item in relevant_evidence[:24]
                ],
            },
            ensure_ascii=False,
        ),
        fallback={},
    )
    if not isinstance(payload, dict) or not payload:
        return None
    raw_status = str(payload.get("status", "") or "").strip().lower()
    status_map = {
        "pass": "pass",
        "supported": "pass",
        "retry": "retry",
        "insufficient": "retry",
        "unsupported": "retry",
        "contradicted": "retry",
        "contradiction": "retry",
        "clarify": "clarify",
        "ambiguous": "clarify",
    }
    status = status_map.get(raw_status)
    if status is None:
        return None
    if status == "pass":
        return None
    missing_fields = coerce_verifier_string_list(payload.get("missing_fields"))
    unsupported_claims = coerce_verifier_string_list(payload.get("unsupported_claims"))
    contradictory_claims = coerce_verifier_string_list(
        payload.get("contradictory_claims") or payload.get("contradictions")
    )
    if not missing_fields:
        missing_fields = ["formula_evidence"]
    return VerificationReport(
        status=status,  # type: ignore[arg-type]
        missing_fields=missing_fields,
        unsupported_claims=unsupported_claims,
        contradictory_claims=contradictory_claims,
        recommended_action=str(payload.get("recommended_action") or "retry_formula_evidence"),
    )


def coerce_verifier_string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []
