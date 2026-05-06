from __future__ import annotations

from app.domain.models import QueryContract, SessionContext, SessionTurn
from app.services.answers.memory_followup import (
    compose_formula_interpretation_followup_answer,
    compose_language_preference_followup_answer,
    compose_memory_followup_answer,
    compose_memory_synthesis_answer,
)


class NoChatClients:
    chat = None


def _context(*_: object, **__: object) -> dict[str, object]:
    return {}


def test_compose_memory_synthesis_fallback_uses_last_compound_subtasks() -> None:
    session = SessionContext(
        session_id="synthesis",
        working_memory={
            "last_compound_query": {
                "subtasks": [
                    {"targets": ["DPO"], "answer_preview": "Direct Preference Optimization 的核心是偏好对比。"},
                    {"targets": ["PPO"], "answer_preview": "PPO 用 reward model 稳定更新策略。"},
                ]
            }
        },
    )
    answer = compose_memory_synthesis_answer(
        query="比较一下",
        session=session,
        contract=QueryContract(clean_query="比较一下", relation="memory_synthesis"),
        clients=NoChatClients(),
        conversation_context=_context,
        clean_text=lambda text: text,
    )

    assert "DPO" in answer
    assert "PPO" in answer


def test_compose_memory_followup_returns_recent_artifact_reference() -> None:
    session = SessionContext(
        session_id="artifact",
        working_memory={
            "last_displayed_list": {
                "tool": "query_library_metadata",
                "query": "列出最新论文",
                "items": [
                    {"ordinal": 1, "row": {"title": "First Paper", "year": 2024}},
                    {"ordinal": 2, "row": {"title": "Second Paper", "year": 2025}},
                ],
            }
        },
    )
    answer = compose_memory_followup_answer(
        query="第二篇是什么",
        session=session,
        contract=QueryContract(clean_query="第二篇是什么", relation="memory_followup"),
        clients=NoChatClients(),
        conversation_context=_context,
        clean_text=lambda text: text,
    )

    assert "Second Paper" in answer
    assert "第 2 条" in answer


def test_compose_formula_interpretation_fallback_uses_previous_formula_turn() -> None:
    session = SessionContext(session_id="formula")
    session.turns.append(
        SessionTurn(
            query="DPO 的公式是什么？",
            answer=r"$$L_{DPO}(\pi_\theta;\pi_{ref})$$",
            relation="formula_lookup",
            requested_fields=["formula", "variable_explanation"],
        )
    )

    answer = compose_formula_interpretation_followup_answer(
        query="怎么理解这个公式？",
        session=session,
        contract=QueryContract(
            clean_query="怎么理解这个公式？",
            relation="memory_followup",
            requested_fields=["formula_interpretation"],
        ),
        clients=NoChatClients(),
        conversation_context=_context,
        clean_text=lambda text: text,
    )

    assert "偏好回答相对参考策略更可能" in answer
    assert "上一轮公式摘要" in answer


def test_compose_language_preference_fallback_is_chinese_ack() -> None:
    session = SessionContext(session_id="language")
    session.turns.append(
        SessionTurn(
            query="DPO 的公式是什么？",
            answer="Some English explanation.",
            relation="formula_lookup",
        )
    )
    answer = compose_language_preference_followup_answer(
        query="我要中文",
        session=session,
        contract=QueryContract(
            clean_query="我要中文",
            relation="memory_followup",
            requested_fields=["answer_language_preference"],
        ),
        clients=NoChatClients(),
        conversation_context=_context,
        clean_text=lambda text: text,
    )

    assert "中文说明" in answer
