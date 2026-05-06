from __future__ import annotations

import json

from app.domain.models import QueryContract, SessionContext, SessionTurn
from app.services.contracts.conversation_memory import (
    active_memory_bindings,
    apply_conversation_memory_to_contract,
    llm_memory_followup_contract,
    memory_followup_contract_from_payload,
    memory_binding_doc_ids,
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


def test_active_memory_bindings_prioritizes_active_targets_then_recent_bindings() -> None:
    session = _session_with_pba_binding()
    session.set_active_research(
        relation="metric_value_lookup",
        targets=["PBA"],
        titles=["AlignX"],
        requested_fields=["metric_value"],
        required_modalities=["table"],
        answer_shape="table",
        precision_requirement="exact",
        clean_query="PBA result",
    )
    session.working_memory["target_bindings"]["ica"] = {"target": "ICA", "paper_id": "ICA"}

    bindings = active_memory_bindings(session)

    assert [binding["target"] for binding in bindings] == ["PBA", "ICA"]


def test_memory_binding_doc_ids_deduplicates_evidence_and_paper_ids() -> None:
    doc_ids = memory_binding_doc_ids(
        [
            {"evidence_ids": ["b1", "b2", "b3"], "paper_id": "P1"},
            {"evidence_ids": ["b1", ""], "paper_id": "P1"},
        ]
    )

    assert doc_ids == ["b1", "b2", "paper::P1"]


def test_memory_followup_contract_from_payload_normalizes_llm_fields() -> None:
    contract = memory_followup_contract_from_payload(
        payload={
            "should_use_memory": True,
            "reason": "previous rationale",
            "targets": [" Paper A ", ""],
            "requested_fields": ["recommendation_reason"],
            "answer_shape": "weird",
        },
        clean_query="为什么推荐它？",
        current_contract=QueryContract(clean_query="为什么推荐它？", answer_shape="table"),
    )

    assert contract is not None
    assert contract.relation == "memory_followup"
    assert contract.targets == ["Paper A"]
    assert contract.requested_fields == ["recommendation_reason"]
    assert contract.answer_shape == "narrative"
    assert contract.notes[-1] == "previous rationale"


def test_llm_memory_followup_contract_invokes_router_with_context() -> None:
    class Clients:
        chat = object()

        def __init__(self) -> None:
            self.payload: dict[str, object] = {}

        def invoke_json(self, *, system_prompt: str, human_prompt: str, fallback: object) -> object:
            assert "会话记忆追问判别器" in system_prompt
            self.payload = json.loads(human_prompt)
            return {
                "should_use_memory": True,
                "reason": "asks about prior tool output",
                "targets": ["A Survey"],
                "requested_fields": ["previous_tool_basis"],
                "answer_shape": "bullets",
            }

    session = SessionContext(session_id="llm-memory")
    session.turns.append(SessionTurn(query="推荐几篇", answer="A Survey", relation="library_recommendation"))
    clients = Clients()

    contract = llm_memory_followup_contract(
        clean_query="为什么推荐第一篇？",
        session=session,
        current_contract=QueryContract(clean_query="为什么推荐第一篇？", relation="library_recommendation"),
        clients=clients,
        conversation_context=lambda _: {"tool_results": [{"tool": "query_library_metadata"}]},
    )

    assert contract is not None
    assert contract.relation == "memory_followup"
    assert contract.answer_shape == "bullets"
    assert clients.payload["conversation_context"] == {"tool_results": [{"tool": "query_library_metadata"}]}


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
