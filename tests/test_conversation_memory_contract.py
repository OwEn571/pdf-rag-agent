from __future__ import annotations

from app.domain.models import QueryContract, SessionContext
from app.services.conversation_memory_contract import (
    apply_conversation_memory_to_contract,
    target_binding_from_memory,
)


def _session_with_pba_binding() -> SessionContext:
    return SessionContext(
        session_id="memory-contract",
        working_memory={
            "target_bindings": {
                "pba": {
                    "target": "PBA",
                    "paper_id": "ALIGNX",
                    "title": "From 1,000,000 Users to Every User",
                }
            }
        },
    )


def test_target_binding_from_memory_normalizes_target_key() -> None:
    binding = target_binding_from_memory(session=_session_with_pba_binding(), target=" PBA ")

    assert binding is not None
    assert binding["paper_id"] == "ALIGNX"


def test_apply_conversation_memory_to_contract_adds_selected_paper_notes() -> None:
    contract = QueryContract(
        clean_query="PBA 和 ICA 的具体效果如何？",
        interaction_mode="research",
        relation="metric_value_lookup",
        targets=["PBA"],
        requested_fields=["metric_value"],
        required_modalities=["table"],
        answer_shape="table",
        precision_requirement="exact",
        continuation_mode="fresh",
    )

    updated = apply_conversation_memory_to_contract(contract=contract, session=_session_with_pba_binding())

    assert updated.continuation_mode == "followup"
    assert "resolved_from_conversation_memory" in updated.notes
    assert "selected_paper_id=ALIGNX" in updated.notes
    assert "memory_title=From 1,000,000 Users to Every User" in updated.notes


def test_apply_conversation_memory_to_contract_keeps_fresh_formula_unbound() -> None:
    contract = QueryContract(
        clean_query="PBA 的公式是什么？",
        interaction_mode="research",
        relation="formula_lookup",
        targets=["PBA"],
        requested_fields=["formula"],
        required_modalities=["page_text"],
        answer_shape="bullets",
        precision_requirement="exact",
        continuation_mode="fresh",
    )

    updated = apply_conversation_memory_to_contract(contract=contract, session=_session_with_pba_binding())

    assert updated.notes == []
    assert updated.continuation_mode == "fresh"


def test_apply_conversation_memory_to_contract_respects_explicit_clarification_selection() -> None:
    contract = QueryContract(
        clean_query="PBA 的结果如何？",
        interaction_mode="research",
        relation="metric_value_lookup",
        targets=["PBA"],
        requested_fields=["metric_value"],
        required_modalities=["table"],
        answer_shape="table",
        precision_requirement="exact",
        continuation_mode="fresh",
    )

    updated = apply_conversation_memory_to_contract(
        contract=contract,
        session=_session_with_pba_binding(),
        selected_clarification_paper_id="OTHER",
    )

    assert "selected_paper_id=ALIGNX" not in updated.notes
    assert "resolved_from_conversation_memory" not in updated.notes
