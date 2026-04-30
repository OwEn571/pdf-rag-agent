from __future__ import annotations

from app.domain.models import SessionContext
from app.services.intent_router import LLMIntentRouter, ROUTER_TOOLS, RouterDecision, query_contract_from_router_decision


class _RouterClients:
    chat = object()

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def invoke_tool_plan_messages(self, *, tools: list[dict[str, object]], **_: object) -> dict[str, object]:
        self.calls += 1
        assert {item["name"] for item in tools} == {
            "answer_directly",
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
