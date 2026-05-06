from __future__ import annotations

from app.domain.models import AssistantCitation, Claim, QueryContract
from app.services.agent.runtime_summary import build_runtime_summary


def test_build_runtime_summary_canonicalizes_tools_and_context() -> None:
    contract = QueryContract(
        clean_query="DPO 的公式是什么？",
        interaction_mode="research",
        relation="formula_lookup",
        targets=["DPO"],
        allow_web_search=True,
        continuation_mode="followup",
        notes=[
            "answer_slot=formula",
            "intent_kind=research_lookup",
            "intent_confidence=0.72",
            "ambiguous_slot=paper",
            "target_alias=Direct Preference Optimization",
            "selected_paper_id=paper-1",
            "low_intent_confidence",
        ],
    )

    summary = build_runtime_summary(
        contract=contract,
        active_research_context={"targets": ["DPO"]},
        tool_plan={"actions": ["ask_human", "search_corpus"]},
        research_plan={"solver_sequence": ["formula_lookup"]},
        execution_steps=[
            {"node": "agent_tool:unknown_internal"},
            {"node": "agent_tool:ask_human"},
            {"node": "compound_task:1"},
        ],
        verification_report={"status": "pass"},
        answer_confidence={"score": 0.82, "basis": "logprobs", "detail": {"token_count": 42}},
        claims=[Claim(claim_type="formula", structured_data={"source": "schema_solver"})],
        citations=[
            AssistantCitation(
                title="DPO",
                file_path="/tmp/dpo.pdf",
                page=1,
                snippet="formula",
            )
        ],
    )

    assert summary["intent"]["kind"] == "research_lookup"
    assert summary["intent"]["answer_slots"] == ["formula"]
    assert summary["intent"]["confidence"] == 0.72
    assert summary["tool_loop"]["planned_tools"] == ["ask_human", "search_corpus"]
    assert summary["tool_loop"]["observed_tools"] == ["ask_human", "compose"]
    assert summary["tool_loop"]["noncanonical_tools"] == ["unknown_internal"]
    assert summary["grounding"]["claim_sources"] == {"schema_solver": 1}
    assert summary["grounding"]["citation_count"] == 1
    assert summary["answer_generation"]["confidence"]["basis"] == "logprobs"
    assert summary["contract_context"]["selected_paper_id"] == "paper-1"
    assert summary["contract_context"]["clarification_reasons"] == ["paper", "low_intent_confidence"]
    assert summary["active_research_context"] == {"targets": ["DPO"]}


def test_build_runtime_summary_defaults_pending_grounding() -> None:
    contract = QueryContract(clean_query="hello", interaction_mode="conversation")

    summary = build_runtime_summary(contract=contract)

    assert summary["intent"]["kind"] == "smalltalk"
    assert summary["grounding"]["verification_status"] == "pending"
    assert "active_research_context" not in summary


def test_build_runtime_summary_labels_unsourced_claims_as_deterministic() -> None:
    contract = QueryContract(clean_query="DPO", interaction_mode="research")

    summary = build_runtime_summary(contract=contract, claims=[Claim(claim_type="answer")])

    assert summary["grounding"]["claim_sources"] == {"deterministic_solver": 1}
