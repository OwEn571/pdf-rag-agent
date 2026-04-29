from __future__ import annotations

from types import SimpleNamespace

from app.domain.models import QueryContract, VerificationReport
from app.services.agent_planner_helpers import (
    JSON_PLANNER_SYSTEM_PROMPT,
    NEXT_ACTION_SYSTEM_PROMPT,
    TOOL_CALL_PLANNER_SYSTEM_PROMPT,
    fallback_plan,
    first_unexecuted_planned_action,
    json_planner_human_prompt,
    json_planner_system_prompt,
    normalize_plan_payload,
    next_action_human_prompt,
    planner_context_payload,
    planner_context_json,
    planner_intent_payload,
    planner_messages_with_user,
    planner_prompt_with_context,
    planner_state_summary,
    should_fallback_to_human,
    tool_call_planner_human_prompt,
    tool_call_planner_system_prompt,
)


def test_planner_intent_and_context_payloads_include_contract_state() -> None:
    contract = QueryContract(
        clean_query="它的表格结果",
        interaction_mode="research",
        continuation_mode="followup",
        targets=["AlignX"],
        requested_fields=["results"],
        required_modalities=["table"],
        answer_shape="table",
        allow_web_search=True,
        notes=[
            "intent_kind=result_lookup",
            "intent_confidence=0.73",
            "ambiguous_slot=paper_title",
            "answer_slot=accuracy",
        ],
    )

    intent = planner_intent_payload(contract)
    context = planner_context_payload(
        contract=contract,
        active_research_context={"targets": ["AlignX"]},
        use_web_search=True,
        include_available_tools=True,
    )

    assert intent["kind"] == "result_lookup"
    assert intent["confidence"] == "0.73"
    assert intent["ambiguous_slots"] == ["paper_title"]
    assert intent["answer_slots"] == ["accuracy"]
    assert context["intent"] == intent
    assert context["web_enabled"] is True
    assert "available_tools" in context


def test_planner_state_summary_keeps_counts_and_verification_payload() -> None:
    state = {
        "candidate_papers": [object(), object()],
        "screened_papers": [object()],
        "evidence": [object(), object(), object()],
        "web_evidence": [],
        "claims": [object()],
        "answer": "",
        "verification": VerificationReport(status="retry", recommended_action="expand_recall"),
    }

    summary = planner_state_summary(state)

    assert summary["candidate_papers"] == 2
    assert summary["screened_papers"] == 1
    assert summary["evidence"] == 3
    assert summary["has_answer"] is False
    assert summary["verification"]["status"] == "retry"


def test_planner_fallback_plan_handles_clarification_conversation_and_research() -> None:
    settings = SimpleNamespace(confidence_floor=0.6)

    assert should_fallback_to_human(
        contract=QueryContract(clean_query="x", notes=["ambiguous_slot=target"]),
        settings=settings,
    ) is True
    assert fallback_plan(
        contract=QueryContract(clean_query="x", notes=["ambiguous_slot=target"]),
        use_web_search=False,
        settings=settings,
        is_negative_correction_query=lambda _: False,
    )["actions"] == ["ask_human"]
    assert fallback_plan(
        contract=QueryContract(clean_query="hello", interaction_mode="conversation", relation="greeting"),
        use_web_search=False,
        settings=settings,
        is_negative_correction_query=lambda _: False,
    )["actions"] == []
    assert fallback_plan(
        contract=QueryContract(clean_query="DPO 是什么", relation="entity_definition"),
        use_web_search=True,
        settings=settings,
        is_negative_correction_query=lambda _: False,
    )["actions"] == []


def test_planner_normalize_plan_payload_filters_unknown_actions_and_preserves_tool_args() -> None:
    fallback = {"thought": "fallback", "actions": [], "stop_conditions": ["fallback_stop"]}

    normalized = normalize_plan_payload(
        payload={
            "actions": ["search_corpus", "not_a_tool", "compose"],
            "tool_call_args": [{"name": "search_corpus", "args": {"query": "DPO"}}],
        },
        fallback=fallback,
    )

    assert normalized == {
        "thought": "fallback",
        "actions": ["search_corpus", "compose"],
        "stop_conditions": ["fallback_stop"],
        "tool_call_args": [{"name": "search_corpus", "args": {"query": "DPO"}}],
    }
    assert normalize_plan_payload(payload={"actions": ["not_a_tool"]}, fallback=fallback) is None
    assert normalize_plan_payload(payload="bad", fallback=fallback) is None


def test_planner_prompts_keep_tool_loop_contract_and_context_suffix() -> None:
    prompt = planner_prompt_with_context(
        system_prompt=TOOL_CALL_PLANNER_SYSTEM_PROMPT,
        context_json='{"web_enabled": false}',
    )

    assert "工具循环控制器" in JSON_PLANNER_SYSTEM_PROMPT
    assert "tool calls" in TOOL_CALL_PLANNER_SYSTEM_PROMPT
    assert "下一步工具" in NEXT_ACTION_SYSTEM_PROMPT
    assert "以下非语言上下文只用于工具选择" in prompt
    assert '{"web_enabled": false}' in prompt


def test_planner_call_helpers_build_json_messages_and_prompts() -> None:
    contract = QueryContract(clean_query="AlignX 的表格结果", targets=["AlignX"], notes=["intent_kind=result_lookup"])
    context_payload = planner_context_payload(
        contract=contract,
        active_research_context={"paper_title": "AlignX"},
        use_web_search=False,
        include_available_tools=False,
    )

    assert planner_context_json({"query": "中文"}) == '{"query": "中文"}'
    assert planner_messages_with_user(
        conversation_messages=[{"role": "assistant", "content": "上一轮"}],
        contract=contract,
    ) == [
        {"role": "assistant", "content": "上一轮"},
        {"role": "user", "content": "AlignX 的表格结果"},
    ]
    json_system = json_planner_system_prompt(context_payload)
    tool_system = tool_call_planner_system_prompt(context_payload)
    json_human = json_planner_human_prompt(
        contract=contract,
        conversation_context={"turns": 1},
        context_payload=context_payload,
    )
    tool_human = tool_call_planner_human_prompt(
        contract=contract,
        conversation_context={"turns": 1},
        context_payload=context_payload,
    )

    assert "以下非语言上下文只用于工具选择" in json_system
    assert "以下非语言上下文只用于工具选择" in tool_system
    assert '"query": "AlignX 的表格结果"' in json_human
    assert '"conversation_context": {"turns": 1}' in tool_human


def test_next_action_payload_and_first_unexecuted_action() -> None:
    contract = QueryContract(
        clean_query="继续",
        interaction_mode="research",
        requested_fields=["results"],
        notes=["intent_kind=result_lookup"],
    )
    prompt = next_action_human_prompt(
        contract=contract,
        state={"evidence": [object(), object()], "answer": ""},
        executed_actions=["search_corpus"],
        conversation_context={"turns": 2},
    )

    assert '"query": "继续"' in prompt
    assert '"executed_actions": ["search_corpus"]' in prompt
    assert '"evidence": 2' in prompt
    assert (
        first_unexecuted_planned_action(
            payload={"actions": ["search_corpus", "compose"]},
            allowed_tools={"search_corpus", "compose"},
            executed_actions=["search_corpus"],
        )
        == "compose"
    )
    assert (
        first_unexecuted_planned_action(
            payload={"actions": ["unknown", "compose"]},
            allowed_tools={"search_corpus"},
            executed_actions=[],
        )
        is None
    )
