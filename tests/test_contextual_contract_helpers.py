from __future__ import annotations

from app.domain.models import ActiveResearch, CandidatePaper, QueryContract
from app.services.contextual_contract_helpers import (
    active_paper_reference_notes,
    formula_answer_correction_contract,
    formula_contextual_paper_contract,
    formula_location_followup_contract,
    promote_contextual_metric_contract,
)


def test_active_paper_reference_notes_adds_selected_paper_context_once() -> None:
    paper = CandidatePaper(paper_id="p1", title="Paper Title")

    notes = active_paper_reference_notes(
        notes=["existing", "selected_paper_id=p1"],
        paper=paper,
        marker="active_paper_reference",
    )

    assert notes == [
        "existing",
        "selected_paper_id=p1",
        "active_paper_reference",
        "resolved_from_conversation_memory",
        "memory_title=Paper Title",
    ]


def test_promote_contextual_metric_contract_adds_metric_requirements() -> None:
    contract = QueryContract(
        clean_query="PBA 和 ICA 的具体效果如何？",
        relation="general_question",
        targets=["PBA", "ICA"],
        requested_fields=["answer"],
        required_modalities=["page_text"],
        answer_shape="narrative",
    )

    promoted = promote_contextual_metric_contract(contract)

    assert promoted.relation == "metric_value_lookup"
    assert promoted.answer_slots == ["metric_value"]
    assert promoted.requested_fields == ["answer", "metric_value", "setting", "evidence"]
    assert promoted.required_modalities == ["page_text", "table", "caption"]
    assert "contextual_metric_query" in promoted.notes
    assert "answer_slot=metric_value" in promoted.notes


def test_promote_contextual_metric_contract_keeps_unrelated_contract() -> None:
    contract = QueryContract(clean_query="这篇论文主要讲什么？", relation="paper_summary_results", targets=["DPO"])
    assert promote_contextual_metric_contract(contract) is contract


def test_formula_answer_correction_contract_prefers_active_target_and_paper() -> None:
    contract = QueryContract(clean_query="不是这个公式", relation="formula_lookup", targets=["old"], allow_web_search=True)
    active = ActiveResearch(
        relation="formula_lookup",
        targets=["DPO"],
        titles=["DPO Paper"],
        requested_fields=["formula"],
    )
    paper = CandidatePaper(paper_id="paper-1", title="DPO Paper")

    updated = formula_answer_correction_contract(contract=contract, active=active, paper=paper)

    assert updated.relation == "formula_lookup"
    assert updated.targets == ["DPO"]
    assert updated.allow_web_search is True
    assert "selected_paper_id=paper-1" in updated.notes
    assert "prefer_scalar_objective" in updated.notes
    assert "限定在论文《DPO Paper》" in updated.clean_query


def test_formula_location_followup_contract_binds_named_paper() -> None:
    contract = QueryContract(clean_query="是这篇论文里的", relation="formula_lookup", notes=["existing"])
    paper = CandidatePaper(paper_id="paper-1", title="DPO Paper")

    updated = formula_location_followup_contract(contract=contract, paper=paper, target="DPO")

    assert updated.targets == ["DPO"]
    assert updated.requested_fields == ["formula", "variable_explanation", "source"]
    assert "formula_location_followup" in updated.notes
    assert "resolved_from_user_paper_hint" in updated.notes
    assert "selected_paper_id=paper-1" in updated.notes


def test_formula_contextual_paper_contract_rewrites_existing_contract() -> None:
    contract = QueryContract(clean_query="公式是什么？", relation="general_question", targets=["PBA"], notes=["context"])
    paper = CandidatePaper(paper_id="paper-2", title="AlignX Paper")

    updated = formula_contextual_paper_contract(contract=contract, paper=paper, target="PBA")

    assert updated.relation == "formula_lookup"
    assert updated.targets == ["PBA"]
    assert updated.required_modalities == ["page_text", "table"]
    assert "formula_contextual_paper_binding" in updated.notes
    assert "memory_title=AlignX Paper" in updated.notes
