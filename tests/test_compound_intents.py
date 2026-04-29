from __future__ import annotations

from app.domain.models import SessionContext
from app.services.compound_intents import should_try_compound_decomposition, should_try_compound_decomposition_heuristic


def test_compound_intent_detects_library_count_plus_recommendation() -> None:
    assert should_try_compound_decomposition_heuristic(
        "论文库里有多少篇，推荐哪篇？",
        normalized_query="论文库里有多少篇，推荐哪篇？",
        target_count=0,
        has_memory_context=False,
    )


def test_compound_intent_requires_context_for_bare_comparison() -> None:
    assert should_try_compound_decomposition_heuristic(
        "两者区别是什么？",
        normalized_query="两者区别是什么？",
        target_count=0,
        has_memory_context=True,
    )
    assert not should_try_compound_decomposition_heuristic(
        "两者区别是什么？",
        normalized_query="两者区别是什么？",
        target_count=0,
        has_memory_context=False,
    )


def test_compound_intent_detects_multi_target_task_cues() -> None:
    assert should_try_compound_decomposition_heuristic(
        "DPO和PPO公式分别是什么？",
        normalized_query="dpo和ppo公式分别是什么？",
        target_count=2,
        has_memory_context=False,
    )
    assert not should_try_compound_decomposition_heuristic(
        "PBA公式是什么？",
        normalized_query="pba公式是什么？",
        target_count=1,
        has_memory_context=False,
    )


def test_compound_intent_session_aware_wrapper_uses_active_context_for_comparison() -> None:
    session = SessionContext(session_id="compound-intent")
    session.set_active_research(
        relation="formula_lookup",
        targets=["DPO", "PPO"],
        titles=[],
        requested_fields=["formula"],
        required_modalities=["page_text"],
        answer_shape="bullets",
        precision_requirement="exact",
        clean_query="DPO 和 PPO 公式",
    )

    assert should_try_compound_decomposition("两者区别是什么？", session=session)
    assert not should_try_compound_decomposition("两者区别是什么？", session=SessionContext(session_id="empty"))
