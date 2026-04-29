from __future__ import annotations

from app.domain.models import AssistantCitation, Claim, QueryContract
from app.services.agent_runtime_summary import build_runtime_summary


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
        tool_plan={"actions": ["web_citation_lookup", "search_corpus"]},
        research_plan={"solver_sequence": ["formula_lookup"]},
        execution_steps=[{"node": "agent_tool:web_citation_lookup"}, {"node": "compound_task:1"}],
        verification_report={"status": "pass"},
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
    assert summary["tool_loop"]["planned_tools"] == ["web_search", "search_corpus"]
    assert summary["tool_loop"]["observed_tools"] == ["web_search", "compose"]
    assert summary["tool_loop"]["legacy_tools"] == ["web_citation_lookup"]
    assert summary["grounding"]["claim_sources"] == {"schema_solver": 1}
    assert summary["grounding"]["citation_count"] == 1
    assert summary["contract_context"]["selected_paper_id"] == "paper-1"
    assert summary["contract_context"]["clarification_reasons"] == ["paper", "low_intent_confidence"]
    assert summary["active_research_context"] == {"targets": ["DPO"]}


def test_build_runtime_summary_defaults_pending_grounding() -> None:
    contract = QueryContract(clean_query="hello", interaction_mode="conversation")

    summary = build_runtime_summary(contract=contract)

    assert summary["intent"]["kind"] == "smalltalk"
    assert summary["grounding"]["verification_status"] == "pending"
    assert "active_research_context" not in summary
