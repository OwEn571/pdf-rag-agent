from __future__ import annotations

from app.domain.models import SessionContext
from app.services.intents.router import (
    LLMIntentRouter,
    ROUTER_TOOLS,
    RouterDecision,
    query_contract_from_router_decision,
    router_miss_clarification_contract,
)


class _RouterClients:
    chat = object()

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def invoke_tool_plan_messages(self, *, tools: list[dict[str, object]], **_: object) -> dict[str, object]:
        self.calls += 1
        assert {item["name"] for item in tools} == {
            "answer_directly",
            "need_conversation_tool",
            "need_corpus_search",
            "need_web",
            "need_clarify",
        }
        return self.payload


class _NoChatClients:
    chat = None


def test_router_tool_schemas_have_real_input_schema() -> None:
    for tool in ROUTER_TOOLS:
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert isinstance(schema["properties"], dict)
        assert schema["additionalProperties"] is False


def test_llm_intent_router_normalizes_corpus_search_decision() -> None:
    clients = _RouterClients(
        {
            "thought": "local paper question",
            "actions": ["need_corpus_search"],
            "tool_call_args": [
                {
                    "name": "need_corpus_search",
                    "args": {
                        "query": "DPO core formula",
                        "targets": ["DPO"],
                        "confidence": 0.87,
                        "rationale": "The user asks about a paper formula.",
                    },
                }
            ],
        }
    )
    router = LLMIntentRouter(
        clients=clients,
        conversation_context=lambda session: {"session_id": session.session_id},
        conversation_messages=lambda session: [{"role": "assistant", "content": "previous answer"}],
    )

    decision = router.route(query="DPO 公式是什么？", session=SessionContext(session_id="demo"))

    assert decision.action == "need_corpus_search"
    assert decision.confidence == 0.87
    assert decision.args["query"] == "DPO core formula"
    assert "target:DPO" in decision.tags
    assert decision.notes()[:2] == ["router_action=need_corpus_search", "router_confidence=0.87"]
    assert clients.calls == 1


def test_llm_intent_router_fills_query_when_tool_args_omit_it() -> None:
    clients = _RouterClients(
        {
            "actions": ["need_web"],
            "tool_call_args": [{"name": "need_web", "args": {"confidence": 0.74, "rationale": "Current fact."}}],
        }
    )
    router = LLMIntentRouter(clients=clients, conversation_context=lambda session: {})

    decision = router.route(query="latest citation count", session=SessionContext(session_id="demo"))

    assert decision.action == "need_web"
    assert decision.args["query"] == "latest citation count"


def test_llm_intent_router_falls_back_to_clarify_without_chat() -> None:
    router = LLMIntentRouter(clients=_NoChatClients(), conversation_context=lambda session: {})

    decision = router.route(query="它怎么样？", session=SessionContext(session_id="demo"))

    assert decision.action == "need_clarify"
    assert decision.confidence == 0.0
    assert "router_unavailable" in decision.tags


def test_router_miss_clarification_contract_marks_removed_fallback() -> None:
    contract = router_miss_clarification_contract(clean_query="它怎么样？")

    assert contract.interaction_mode == "conversation"
    assert contract.relation == "clarify_user_intent"
    assert "router_unavailable" in contract.notes
    assert "legacy_intent_fallback_removed" in contract.notes


def test_llm_intent_router_tags_invalid_tool_plan_payload() -> None:
    router = LLMIntentRouter(
        clients=_RouterClients({"actions": ["not_a_router_tool"], "tool_call_args": []}),
        conversation_context=lambda session: {},
    )

    decision = router.route(query="PBA 准确率多少", session=SessionContext(session_id="demo"))

    assert decision.action == "need_clarify"
    assert decision.confidence == 0.0
    assert decision.rationale == "router_invalid_payload"
    assert "router_unavailable" in decision.tags
    assert "router_invalid_payload" in decision.tags


def test_router_decision_converts_corpus_search_to_query_contract() -> None:
    contract = query_contract_from_router_decision(
        decision=RouterDecision(
            action="need_corpus_search",
            confidence=0.88,
            args={"query": "PBA 准确率多少", "targets": ["PBA"], "rationale": "table metric"},
            rationale="table metric",
            tags=["need_corpus_search", "target:PBA"],
        ),
        clean_query="PBA 准确率多少",
        session=SessionContext(session_id="demo"),
        extracted_targets=[],
        normalize_targets=lambda targets, requested_fields: targets,
    )

    assert contract is not None
    assert contract.relation == "metric_value_lookup"
    assert contract.targets == ["PBA"]
    assert contract.answer_slots == ["metric_value"]
    assert "llm_tool_router" in contract.notes
    assert "intent_confidence=0.88" in contract.notes


def test_router_decision_uses_configured_confidence_floor_for_clarification() -> None:
    decision = RouterDecision(
        action="need_corpus_search",
        confidence=0.65,
        args={"query": "PBA 准确率多少", "targets": ["PBA"], "rationale": "table metric"},
        rationale="table metric",
        tags=["need_corpus_search"],
    )

    research_contract = query_contract_from_router_decision(
        decision=decision,
        clean_query="PBA 准确率多少",
        session=SessionContext(session_id="demo"),
        extracted_targets=[],
        normalize_targets=lambda targets, requested_fields: targets,
        confidence_floor=0.6,
    )
    clarify_contract = query_contract_from_router_decision(
        decision=decision,
        clean_query="PBA 准确率多少",
        session=SessionContext(session_id="demo"),
        extracted_targets=[],
        normalize_targets=lambda targets, requested_fields: targets,
        confidence_floor=0.7,
    )

    assert research_contract is not None
    assert research_contract.relation == "metric_value_lookup"
    assert clarify_contract is not None
    assert clarify_contract.relation == "metric_value_lookup"
    assert clarify_contract.answer_slots == ["metric_value"]
    assert "low_intent_confidence" in clarify_contract.notes
    assert "low_confidence_recovered_research_slot" in clarify_contract.notes


def test_router_decision_extracts_targets_when_tool_args_omit_them() -> None:
    contract = query_contract_from_router_decision(
        decision=RouterDecision(
            action="need_corpus_search",
            confidence=0.84,
            args={"query": "Alignx 是什么？", "rationale": "entity definition"},
            rationale="entity definition",
            tags=["need_corpus_search"],
        ),
        clean_query="Alignx 是什么？",
        session=SessionContext(session_id="demo"),
        extracted_targets=[],
        normalize_targets=lambda targets, requested_fields: targets,
    )

    assert contract is not None
    assert contract.relation == "entity_definition"
    assert contract.targets == ["Alignx"]
    assert contract.answer_slots == ["definition"]


def test_router_decision_preserves_direct_answer_subtype() -> None:
    for answer_style, expected_relation, expected_slot in [
        ("greeting", "greeting", "greeting"),
        ("self_identity", "self_identity", "self_identity"),
        ("capability", "capability", "capability"),
        ("anything else", "general_question", "general_answer"),
    ]:
        contract = query_contract_from_router_decision(
            decision=RouterDecision(
                action="answer_directly",
                confidence=0.91,
                args={"rationale": "direct answer", "answer_style": answer_style},
                rationale="direct answer",
                tags=["answer_directly"],
            ),
            clean_query="你好",
            session=SessionContext(session_id="demo"),
            extracted_targets=[],
            normalize_targets=lambda targets, requested_fields: targets,
        )

        assert contract is not None
        assert contract.interaction_mode == "conversation"
        assert contract.relation == expected_relation
        assert contract.answer_slots == [expected_slot]
        assert f"answer_slot={expected_slot}" in contract.notes


def test_router_decision_converts_conversation_tool_relation() -> None:
    contract = query_contract_from_router_decision(
        decision=RouterDecision(
            action="need_conversation_tool",
            confidence=0.83,
            args={
                "relation": "library_recommendation",
                "requested_fields": ["recommendation_reason"],
                "rationale": "local library recommendation",
                "continuation_mode": "followup",
                "notes": ["avoid_recent_recommendations"],
            },
            rationale="local library recommendation",
            tags=["need_conversation_tool"],
        ),
        clean_query="再推荐一篇别的",
        session=SessionContext(session_id="demo"),
        extracted_targets=[],
        normalize_targets=lambda targets, requested_fields: targets,
    )

    assert contract is not None
    assert contract.interaction_mode == "conversation"
    assert contract.relation == "library_recommendation"
    assert contract.answer_slots == ["library_recommendation"]
    assert contract.requested_fields == ["recommendation_reason"]
    assert contract.continuation_mode == "followup"
    assert "avoid_recent_recommendations" in contract.notes


def test_router_decision_respects_explicit_research_relation() -> None:
    contract = query_contract_from_router_decision(
        decision=RouterDecision(
            action="need_corpus_search",
            confidence=0.86,
            args={
                "query": "POPI的核心结论是什么，实验结果如何？",
                "relation": "paper_summary_results",
                "targets": ["POPI"],
                "requested_fields": ["summary", "results", "evidence"],
                "required_modalities": ["page_text", "paper_card", "table", "caption"],
                "rationale": "paper summary and results",
            },
            rationale="paper summary and results",
            tags=["need_corpus_search", "target:POPI"],
        ),
        clean_query="POPI的核心结论是什么，实验结果如何？",
        session=SessionContext(session_id="demo"),
        extracted_targets=[],
        normalize_targets=lambda targets, requested_fields: targets,
    )

    assert contract is not None
    assert contract.interaction_mode == "research"
    assert contract.relation == "paper_summary_results"
    assert contract.targets == ["POPI"]
    assert contract.answer_slots == ["paper_summary"]
    assert contract.requested_fields == ["summary", "results", "evidence"]
    assert contract.required_modalities == ["page_text", "paper_card", "table", "caption"]


def test_low_confidence_clarify_recovers_named_paper_source_query() -> None:
    session = SessionContext(session_id="demo")
    session.set_active_research(
        relation="figure_question",
        targets=["DeepSeek"],
        titles=["CommunityBench: Benchmarking Community-Level Alignment across Diverse Groups and Tasks"],
        requested_fields=["figure_conclusion"],
        required_modalities=["figure", "caption", "page_text"],
        answer_shape="bullets",
        precision_requirement="high",
        clean_query="DeepSeek论文中有哪些figure",
    )

    contract = query_contract_from_router_decision(
        decision=RouterDecision(
            action="need_clarify",
            confidence=0.34,
            args={"question": "请提供 DeepSeek 的具体论文来源。", "reason": "ambiguous paper source"},
            rationale="ambiguous paper source",
            tags=["need_clarify"],
        ),
        clean_query="我就是让你找DeepSeek是哪篇论文",
        session=session,
        extracted_targets=["DeepSeek"],
        normalize_targets=lambda targets, requested_fields: targets,
    )

    assert contract is not None
    assert contract.interaction_mode == "research"
    assert contract.relation == "origin_lookup"
    assert contract.targets == ["DeepSeek"]
    assert contract.answer_slots == ["origin"]
    assert contract.requested_fields == ["paper_title", "year", "evidence"]
    assert contract.continuation_mode == "context_switch"
    assert "clarify_recovered_research_slot" in contract.notes
    assert "clarify_recovered_from_router" in contract.notes


def test_clarify_decision_recovers_named_summary_results_query() -> None:
    contract = query_contract_from_router_decision(
        decision=RouterDecision(
            action="need_clarify",
            confidence=0.42,
            args={"question": "请说明 AlignX 是什么。", "reason": "ambiguous target"},
            rationale="ambiguous target",
            tags=["need_clarify"],
        ),
        clean_query="AlignX中主要结论是什么？用什么数据支持？",
        session=SessionContext(session_id="demo"),
        extracted_targets=["AlignX"],
        normalize_targets=lambda targets, requested_fields: targets,
    )

    assert contract is not None
    assert contract.interaction_mode == "research"
    assert contract.relation == "paper_summary_results"
    assert contract.targets == ["AlignX"]
    assert contract.answer_slots == ["paper_summary"]
    assert contract.requested_fields == ["summary", "results", "evidence"]
    assert "clarify_recovered_research_slot" in contract.notes


def test_clarify_decision_recovers_named_formula_query() -> None:
    contract = query_contract_from_router_decision(
        decision=RouterDecision(
            action="need_clarify",
            confidence=0.51,
            args={"question": "请说明 DPO 所属领域。", "reason": "ambiguous acronym"},
            rationale="ambiguous acronym",
            tags=["need_clarify"],
        ),
        clean_query="DPO的公式是什么？",
        session=SessionContext(session_id="demo"),
        extracted_targets=["DPO"],
        normalize_targets=lambda targets, requested_fields: targets,
    )

    assert contract is not None
    assert contract.interaction_mode == "research"
    assert contract.relation == "formula_lookup"
    assert contract.targets == ["DPO"]
    assert contract.answer_slots == ["formula"]
    assert contract.requested_fields == ["formula", "variable_explanation", "source"]
    assert "clarify_recovered_research_slot" in contract.notes


def test_direct_answer_decision_recovers_named_formula_query() -> None:
    contract = query_contract_from_router_decision(
        decision=RouterDecision(
            action="answer_directly",
            confidence=0.92,
            args={"answer_style": "general", "rationale": "general knowledge"},
            rationale="general knowledge",
            tags=["answer_directly"],
        ),
        clean_query="DPO的公式是什么？",
        session=SessionContext(session_id="demo"),
        extracted_targets=["DPO"],
        normalize_targets=lambda targets, requested_fields: targets,
    )

    assert contract is not None
    assert contract.interaction_mode == "research"
    assert contract.relation == "formula_lookup"
    assert contract.answer_slots == ["formula"]
    assert "direct_answer_recovered_research_slot" in contract.notes


def test_clarify_decision_recovers_named_comparison_query() -> None:
    contract = query_contract_from_router_decision(
        decision=RouterDecision(
            action="need_clarify",
            confidence=0.44,
            args={"question": "请说明比较范围。", "reason": "ambiguous comparison"},
            rationale="ambiguous comparison",
            tags=["need_clarify"],
        ),
        clean_query="你觉得GRPO、PPO、DPO三者的区别和联系是什么",
        session=SessionContext(session_id="demo"),
        extracted_targets=["GRPO", "PPO", "DPO"],
        normalize_targets=lambda targets, requested_fields: targets,
    )

    assert contract is not None
    assert contract.interaction_mode == "research"
    assert contract.relation == "general_question"
    assert contract.targets == ["GRPO", "PPO", "DPO"]
    assert contract.answer_slots == ["comparison"]
    assert contract.requested_fields == ["comparison", "relationship", "evidence"]
    assert "clarify_recovered_research_slot" in contract.notes


def test_clarify_decision_recovers_named_figure_query() -> None:
    contract = query_contract_from_router_decision(
        decision=RouterDecision(
            action="need_clarify",
            confidence=0.34,
            args={"question": "请提供 DeepSeek 的具体论文来源。", "reason": "ambiguous paper source"},
            rationale="ambiguous paper source",
            tags=["need_clarify"],
        ),
        clean_query="DeepSeek论文中有哪些figure",
        session=SessionContext(session_id="demo"),
        extracted_targets=["DeepSeek"],
        normalize_targets=lambda targets, requested_fields: targets,
    )

    assert contract is not None
    assert contract.interaction_mode == "research"
    assert contract.relation == "figure_question"
    assert contract.targets == ["DeepSeek"]
    assert contract.answer_slots == ["figure"]
    assert contract.requested_fields == ["figure_conclusion", "caption", "evidence"]
    assert "clarify_recovered_research_slot" in contract.notes


def test_router_decision_unavailable_returns_none_for_legacy_fallback() -> None:
    contract = query_contract_from_router_decision(
        decision=RouterDecision(
            action="need_clarify",
            confidence=0.0,
            args={},
            rationale="router_unavailable",
            tags=["router_unavailable"],
        ),
        clean_query="PBA 是什么？",
        session=SessionContext(session_id="demo"),
        extracted_targets=["PBA"],
        normalize_targets=lambda targets, requested_fields: targets,
    )

    assert contract is None
