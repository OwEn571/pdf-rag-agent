from __future__ import annotations

import re
from typing import Any

from app.domain.models import QueryContract, VerificationReport


def compound_task_label(contract: QueryContract) -> str:
    if contract.relation == "library_status":
        return "查看论文库概览和文章预览"
    if contract.relation == "library_recommendation":
        return "从库内给出默认推荐"
    if contract.relation == "formula_lookup":
        target = contract.targets[0] if contract.targets else "目标对象"
        return f"查询 {target} 公式"
    if contract.relation == "comparison_synthesis":
        target_text = " 和 ".join(contract.targets) if contract.targets else "前面结果"
        return f"比较 {target_text}"
    return contract.clean_query


def compound_section_heading(*, contract: QueryContract, index: int) -> str:
    return f"## {index}. {compound_task_label(contract)}"


def compound_research_progress_markdown(*, contract: QueryContract, index: int) -> str:
    heading = compound_section_heading(contract=contract, index=index)
    if contract.relation == "formula_lookup":
        target = contract.targets[0] if contract.targets else "目标对象"
        return f"{heading}\n\n好的，我现在去查询 **{target}** 的公式。"
    return heading


def demote_markdown_headings(answer: str) -> str:
    return re.sub(r"^(#{1,5})\s+", lambda match: "#" + match.group(1) + " ", str(answer or "").strip(), flags=re.M)


def format_compound_section(*, contract: QueryContract, answer: str, index: int) -> str:
    normalized = demote_markdown_headings(str(answer or "").strip())
    return f"{compound_section_heading(contract=contract, index=index)}\n\n{normalized}".strip()


def compound_task_result_from_task_payload(
    task_result: dict[str, Any],
    *,
    fallback_contract: QueryContract,
) -> dict[str, Any]:
    contract = task_result.get("contract_obj")
    if not isinstance(contract, QueryContract):
        raw_contract = task_result.get("contract")
        if isinstance(raw_contract, dict):
            try:
                contract = QueryContract.model_validate(raw_contract)
            except Exception:  # noqa: BLE001
                contract = fallback_contract
        else:
            contract = fallback_contract
    verification = task_result.get("verification_obj")
    if not isinstance(verification, VerificationReport):
        raw_verification = task_result.get("verification")
        if isinstance(raw_verification, dict):
            try:
                verification = VerificationReport.model_validate(raw_verification)
            except Exception:  # noqa: BLE001
                verification = VerificationReport(status="pass", recommended_action="task_subagent")
        else:
            verification = VerificationReport(status="pass", recommended_action="task_subagent")
    return {
        "contract": contract,
        "answer": str(task_result.get("answer", "") or ""),
        "citations": list(task_result.get("citations", []) or []),
        "claims": list(task_result.get("claims", []) or []),
        "evidence": list(task_result.get("evidence", []) or []),
        "verification": verification,
    }
