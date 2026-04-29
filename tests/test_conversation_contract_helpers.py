from __future__ import annotations

from app.domain.models import CandidatePaper, QueryContract, SessionContext, SessionTurn
from app.services.conversation_contract_helpers import normalize_conversation_tool_contract


def _base_contract(query: str) -> QueryContract:
    return QueryContract(clean_query=query, relation="general_question", requested_fields=["answer"])


def test_normalize_conversation_contract_routes_citation_ranking_followup() -> None:
    session = SessionContext(session_id="citation-context")
    session.turns.append(
        SessionTurn(
            query="论文库里推荐哪篇？",
            answer="默认推荐 A。",
            relation="library_recommendation",
            requested_fields=["library_recommendation"],
        )
    )

    contract = normalize_conversation_tool_contract(
        contract=_base_contract("按引用数呢"),
        clean_query="按引用数呢",
        session=session,
        paper_from_query_hint=lambda _: None,
    )

    assert contract.relation == "library_citation_ranking"
    assert contract.allow_web_search is True
    assert contract.answer_shape == "table"
    assert contract.continuation_mode == "followup"


def test_normalize_conversation_contract_routes_formula_correction_with_paper_hint() -> None:
    session = SessionContext(session_id="formula-correction")
    session.set_active_research(
        relation="formula_lookup",
        targets=["DPO"],
        titles=["Direct Preference Optimization"],
        requested_fields=["formula"],
        required_modalities=["page_text", "table"],
        answer_shape="bullets",
        precision_requirement="exact",
        clean_query="DPO 的公式是什么？",
    )
    paper = CandidatePaper(paper_id="DPO", title="Direct Preference Optimization")

    contract = normalize_conversation_tool_contract(
        contract=_base_contract("我觉得不是这个公式哦"),
        clean_query="我觉得不是这个公式哦",
        session=session,
        paper_from_query_hint=lambda title: paper if title == "Direct Preference Optimization" else None,
    )

    assert contract.relation == "formula_lookup"
    assert contract.requested_fields == ["formula", "variable_explanation", "source"]
    assert "formula_answer_correction" in contract.notes
    assert "selected_paper_id=DPO" in contract.notes


def test_normalize_conversation_contract_routes_formula_interpretation_followup() -> None:
    session = SessionContext(session_id="formula-interpret")
    session.set_active_research(
        relation="formula_lookup",
        targets=["DPO"],
        titles=[],
        requested_fields=["formula"],
        required_modalities=["page_text"],
        answer_shape="bullets",
        precision_requirement="exact",
        clean_query="DPO 的公式是什么？",
    )

    contract = normalize_conversation_tool_contract(
        contract=_base_contract("怎么理解这个公式？"),
        clean_query="怎么理解这个公式？",
        session=session,
        paper_from_query_hint=lambda _: None,
    )

    assert contract.relation == "memory_followup"
    assert contract.requested_fields == ["formula_interpretation"]
    assert contract.notes == ["agent_tool", "formula_interpretation_followup"]


def test_normalize_conversation_contract_routes_language_preference_followup() -> None:
    session = SessionContext(session_id="language")
    session.turns.append(SessionTurn(query="DPO 的公式是什么？", answer="English sentence.", relation="formula_lookup"))

    contract = normalize_conversation_tool_contract(
        contract=_base_contract("我要中文"),
        clean_query="我要中文",
        session=session,
        paper_from_query_hint=lambda _: None,
    )

    assert contract.relation == "memory_followup"
    assert contract.requested_fields == ["answer_language_preference"]
    assert "answer_language_preference" in contract.notes


def test_normalize_conversation_contract_routes_memory_synthesis() -> None:
    session = SessionContext(
        session_id="memory-synthesis",
        working_memory={"last_compound_query": {"subtasks": [{"target": "DPO"}, {"target": "PPO"}]}},
    )
    session.set_active_research(
        relation="compound_query",
        targets=["DPO", "PPO"],
        titles=[],
        requested_fields=["formula"],
        required_modalities=["page_text"],
        answer_shape="narrative",
        precision_requirement="normal",
        clean_query="DPO 和 PPO 公式",
    )

    contract = normalize_conversation_tool_contract(
        contract=_base_contract("两者区别是什么？"),
        clean_query="两者区别是什么？",
        session=session,
        paper_from_query_hint=lambda _: None,
    )

    assert contract.relation == "memory_synthesis"
    assert contract.targets == ["DPO", "PPO"]
    assert contract.answer_shape == "table"


def test_normalize_conversation_contract_marks_existing_conversation_relation() -> None:
    contract = normalize_conversation_tool_contract(
        contract=QueryContract(
            clean_query="你好",
            interaction_mode="research",
            relation="greeting",
            required_modalities=["page_text"],
            notes=["router"],
        ),
        clean_query="你好",
        session=SessionContext(session_id="hello"),
        paper_from_query_hint=lambda _: None,
    )

    assert contract.interaction_mode == "conversation"
    assert contract.required_modalities == []
    assert contract.notes == ["router", "agent_tool"]
