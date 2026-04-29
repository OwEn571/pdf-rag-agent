from __future__ import annotations

from app.domain.models import CandidatePaper, QueryContract
from app.services.contextual_contract_helpers import active_paper_reference_notes, promote_contextual_metric_contract


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
