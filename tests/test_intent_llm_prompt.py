from __future__ import annotations

from app.services.intent_llm_prompt import intent_router_system_prompt


def test_intent_router_system_prompt_injects_context_and_keeps_core_constraints() -> None:
    prompt = intent_router_system_prompt('{"last_relation":"metric_value_lookup"}')

    assert "只输出 JSON" in prompt
    assert "intent_kind 只能是 smalltalk/meta_library/research/memory_op" in prompt
    assert "不要做 22 个 relation 单选" in prompt
    assert '"last_relation":"metric_value_lookup"' in prompt
