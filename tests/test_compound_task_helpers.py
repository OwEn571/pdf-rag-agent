from __future__ import annotations

from app.domain.models import QueryContract, VerificationReport
from app.services.compound_task_helpers import (
    compound_research_progress_markdown,
    compound_section_heading,
    compound_task_label,
    compound_task_result_from_task_payload,
    demote_markdown_headings,
    format_compound_section,
)


def test_compound_task_helpers_format_labels_and_sections() -> None:
    formula = QueryContract(clean_query="DPO 公式", relation="formula_lookup", targets=["DPO"])
    comparison = QueryContract(clean_query="比较", relation="comparison_synthesis", targets=["DPO", "PPO"])

    assert compound_task_label(formula) == "查询 DPO 公式"
    assert compound_task_label(comparison) == "比较 DPO 和 PPO"
    assert compound_section_heading(contract=formula, index=2) == "## 2. 查询 DPO 公式"
    assert "好的，我现在去查询 **DPO** 的公式" in compound_research_progress_markdown(contract=formula, index=1)
    assert demote_markdown_headings("# Title\n## Child") == "## Title\n### Child"
    assert format_compound_section(contract=formula, answer="# Answer", index=1).startswith("## 1. 查询 DPO 公式\n\n## Answer")


def test_compound_task_result_from_payload_validates_contract_and_verification() -> None:
    fallback = QueryContract(clean_query="fallback", relation="paper_summary_results")
    payload = {
        "contract": {"clean_query": "DPO", "relation": "formula_lookup", "targets": ["DPO"]},
        "verification": {"status": "clarify", "recommended_action": "ask_human"},
        "answer": "answer",
        "citations": ["c"],
    }

    result = compound_task_result_from_task_payload(payload, fallback_contract=fallback)

    assert isinstance(result["contract"], QueryContract)
    assert result["contract"].relation == "formula_lookup"
    assert isinstance(result["verification"], VerificationReport)
    assert result["verification"].status == "clarify"
    assert result["answer"] == "answer"
    assert result["citations"] == ["c"]


def test_compound_task_result_from_payload_uses_fallbacks_for_bad_values() -> None:
    fallback = QueryContract(clean_query="fallback", relation="paper_summary_results")

    result = compound_task_result_from_task_payload({"contract": "bad", "verification": "bad"}, fallback_contract=fallback)

    assert result["contract"] == fallback
    assert result["verification"].status == "pass"
    assert result["verification"].recommended_action == "task_subagent"
