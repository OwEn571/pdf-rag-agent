from __future__ import annotations

from typing import Any

from app.domain.models import AssistantCitation, QueryContract, SessionContext
from app.services.agent_context import AgentRunContext
from app.services.agent_loop import run_conversation_turn


def test_run_conversation_turn_commits_answer_and_response_payload() -> None:
    citation = AssistantCitation(
        doc_id="doc-1",
        paper_id="paper-1",
        title="Demo Paper",
        file_path="/tmp/demo.pdf",
        page=1,
        snippet="demo evidence",
    )
    agent = _FakeAgent(conversation_state={"answer": "hello", "citations": [citation]})
    context = AgentRunContext.create(session_id="demo", session=SessionContext(session_id="demo"))
    contract = QueryContract(clean_query="hello", interaction_mode="conversation", relation="library_status")

    payload = run_conversation_turn(
        agent=agent,
        run_context=context,
        query="hello",
        contract=contract,
        agent_plan={"actions": ["compose"]},
        max_web_results=3,
    )

    assert payload["session_id"] == "demo"
    assert payload["interaction_mode"] == "conversation"
    assert payload["answer"] == "hello"
    assert payload["citations"][0]["title"] == "Demo Paper"
    assert payload["needs_human"] is False
    assert agent.sessions.committed[0]["turn"].answer == "hello"
    assert agent.sessions.committed[0]["active"]["titles"] == ["Demo Paper"]
    assert context.events[0]["data"]["name"] == "compose"
    assert agent.cleared_pending is True


def test_run_conversation_turn_keeps_clarification_payload() -> None:
    agent = _FakeAgent(
        conversation_state={
            "answer": "",
            "verification_report": {
                "status": "clarify",
                "missing_fields": "target",
                "unsupported_claims": ["claim"],
                "recommended_action": "ask_human",
            },
        }
    )
    context = AgentRunContext.create(session_id="demo", session=SessionContext(session_id="demo"))
    contract = QueryContract(clean_query="it", interaction_mode="conversation", relation="general_question")

    payload = run_conversation_turn(
        agent=agent,
        run_context=context,
        query="it",
        contract=contract,
        agent_plan={"actions": ["ask_human"]},
        max_web_results=3,
    )

    assert payload["needs_human"] is True
    assert payload["clarification_question"] == "Which target?"
    assert payload["clarification_options"] == [{"label": "A"}]
    assert agent.stored_pending is True
    assert agent.remembered_verification.missing_fields == ["target"]
    assert agent.remembered_verification.unsupported_claims == ["claim"]


class _FakeRuntime:
    def __init__(self, conversation_state: dict[str, Any]) -> None:
        self.conversation_state = conversation_state

    def execute_conversation_tools(self, **kwargs: Any) -> dict[str, Any]:
        kwargs["emit"]("observation", {"tool": "compose", "summary": "done", "payload": {"answer": "ok"}})
        kwargs["execution_steps"].append({"node": "agent_tool:compose", "summary": "done"})
        return self.conversation_state


class _FakeSessions:
    def __init__(self) -> None:
        self.committed: list[dict[str, Any]] = []

    def commit_turn(self, session: SessionContext, turn: Any, *, active: Any = None) -> None:
        self.committed.append({"session": session, "turn": turn, "active": active})


class _FakeAgent:
    def __init__(self, *, conversation_state: dict[str, Any]) -> None:
        self.runtime = _FakeRuntime(conversation_state)
        self.sessions = _FakeSessions()
        self.stored_pending = False
        self.cleared_pending = False
        self.remembered_verification = None

    @staticmethod
    def _conversation_relation_updates_research_context(relation: str) -> bool:
        return relation == "library_status"

    @staticmethod
    def _make_active_research(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    def _store_pending_clarification(self, **_: Any) -> None:
        self.stored_pending = True

    def _remember_clarification_attempt(self, **kwargs: Any) -> None:
        self.remembered_verification = kwargs["verification"]

    def _clear_pending_clarification(self, _: SessionContext) -> None:
        self.cleared_pending = True

    @staticmethod
    def _reset_clarification_tracking(_: SessionContext) -> None:
        return None

    @staticmethod
    def _runtime_summary(**kwargs: Any) -> dict[str, Any]:
        return {"steps": len(kwargs["execution_steps"])}

    @staticmethod
    def _clarification_question(_: QueryContract, __: SessionContext) -> str:
        return "Which target?"

    @staticmethod
    def _clarification_options(_: QueryContract) -> list[dict[str, str]]:
        return [{"label": "A"}]
