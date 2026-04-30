from __future__ import annotations

from app.services.conversation_intents import protected_conversation_intent, smalltalk_relation_from_slots


def test_protected_conversation_intent_handles_empty_and_greeting() -> None:
    empty = protected_conversation_intent("  ")
    greeting = protected_conversation_intent("你好！")

    assert empty is not None
    assert empty.answer_slots == ["clarify"]
    assert empty.notes == ["local_protected_empty_query"]
    assert greeting is not None
    assert greeting.answer_slots == ["greeting"]
    assert greeting.notes == ["local_protected_greeting"]


def test_protected_conversation_intent_handles_identity_capability_and_short_clarify() -> None:
    identity = protected_conversation_intent("你是谁？")
    capability = protected_conversation_intent("你能做什么")
    short = protected_conversation_intent("什么意思")

    assert identity is not None
    assert identity.answer_slots == ["self_identity"]
    assert capability is not None
    assert capability.answer_slots == ["capability"]
    assert short is not None
    assert short.answer_slots == ["clarify"]
    assert short.ambiguous_slots == ["missing_target"]


def test_smalltalk_relation_from_slots_preserves_compatibility_relations() -> None:
    assert smalltalk_relation_from_slots(["self_identity"]) == "self_identity"
    assert smalltalk_relation_from_slots(["capability"]) == "capability"
    assert smalltalk_relation_from_slots(["clarify"]) == "clarify_user_intent"
    assert smalltalk_relation_from_slots(["greeting"]) == "greeting"
