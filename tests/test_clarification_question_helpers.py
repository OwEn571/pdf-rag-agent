from __future__ import annotations

import json

from app.domain.models import QueryContract, SessionContext, SessionTurn
from app.services.clarification.intents import contract_with_ambiguity_options
from app.services.clarification.questions import build_clarification_question


class NoChatClients:
    chat = None


def test_build_clarification_question_prefers_ambiguity_options() -> None:
    contract = contract_with_ambiguity_options(
        contract=QueryContract(clean_query="PBA是什么", relation="entity_definition", targets=["PBA"]),
        options=[
            {
                "title": "AlignX",
                "year": "2025",
                "meaning": "Preference Bridged Alignment",
                "display_label": "推荐候选",
                "display_reason": "title match",
            }
        ],
    )

    question = build_clarification_question(
        contract=contract,
        session=SessionContext(session_id="s1"),
        clients=NoChatClients(),
        conversation_context=lambda _: {},
    )

    assert "`PBA` 在本地论文库里有多个可能含义" in question
    assert "Preference Bridged Alignment" in question


def test_build_clarification_question_handles_formula_correction_without_llm() -> None:
    question = build_clarification_question(
        contract=QueryContract(clean_query="不是这个公式", requested_fields=["formula"], targets=["DPO"]),
        session=SessionContext(session_id="s1"),
        clients=NoChatClients(),
        conversation_context=lambda _: {},
    )

    assert "上一条候选公式不能直接当作 `DPO` 的公式" in question
    assert "未找到" in question


def test_build_clarification_question_uses_llm_when_available() -> None:
    class Clients:
        chat = object()

        def __init__(self) -> None:
            self.human_prompt = ""

        def invoke_text(self, *, system_prompt: str, human_prompt: str, fallback: str) -> str:
            self.human_prompt = human_prompt
            assert "研究澄清问题生成器" in system_prompt
            assert fallback == ""
            return "请指定你说的是哪篇 AlignX 论文。"

    clients = Clients()
    session = SessionContext(
        session_id="s1",
        turns=[SessionTurn(query="AlignX 是什么", answer="answer", targets=["AlignX"])],
    )

    question = build_clarification_question(
        contract=QueryContract(clean_query="这个可靠吗", relation="paper_summary", targets=["AlignX"]),
        session=session,
        clients=clients,
        conversation_context=lambda _: {"recent": "context"},
    )

    payload = json.loads(clients.human_prompt)
    assert question == "请指定你说的是哪篇 AlignX 论文。"
    assert payload["conversation_context"] == {"recent": "context"}
    assert payload["recent_turns"] == [
        {"query": "AlignX 是什么", "targets": ["AlignX"], "requested_fields": []}
    ]


def test_build_clarification_question_falls_back_to_targeted_research_hint() -> None:
    question = build_clarification_question(
        contract=QueryContract(clean_query="PBA是什么", relation="entity_definition", targets=["PBA"]),
        session=SessionContext(session_id="s1"),
        clients=NoChatClients(),
        conversation_context=lambda _: {},
    )

    assert "与 `PBA` 直接相关的证据" in question
