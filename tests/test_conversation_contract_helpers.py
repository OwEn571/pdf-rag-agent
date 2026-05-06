from __future__ import annotations

from app.domain.models import CandidatePaper, QueryContract, SessionContext, SessionTurn
from app.services.contracts.conversation_helpers import normalize_conversation_tool_contract


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


def test_normalize_conversation_contract_routes_metric_definition_followup_to_research() -> None:
    session = SessionContext(session_id="metric-definition")
    session.set_active_research(
        relation="metric_value_lookup",
        targets=["ICA", "PBA"],
        titles=["From 1,000,000 Users to Every User"],
        requested_fields=["metric_value", "setting", "evidence"],
        required_modalities=["table", "caption", "page_text"],
        answer_shape="table",
        precision_requirement="exact",
        clean_query="ICA、PBA 在各数据集上的准确度是多少？",
    )

    contract = normalize_conversation_tool_contract(
        contract=_base_contract("这个准确度是怎么定义的？"),
        clean_query="这个准确度是怎么定义的？",
        session=session,
        paper_from_query_hint=lambda _: None,
    )

    assert contract.interaction_mode == "research"
    assert contract.relation == "metric_value_lookup"
    assert contract.continuation_mode == "followup"
    assert contract.targets == ["ICA", "PBA"]
    assert contract.requested_fields == ["metric_value", "metric_definition", "setting", "evidence"]
    assert contract.required_modalities == ["table", "caption", "page_text"]
    assert "metric_definition_followup" in contract.notes
    assert "answer_slot=metric_definition" in contract.notes


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
