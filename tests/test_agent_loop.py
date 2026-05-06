from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.domain.models import (
    AssistantCitation,
    CandidatePaper,
    Claim,
    EvidenceBlock,
    QueryContract,
    ResearchPlan,
    SessionContext,
    VerificationReport,
)
from app.services.agent.context import AgentRunContext
from app.services.agent.loop import (
    finish_agent_turn,
    run_compound_turn_if_needed,
    run_conversation_turn,
    run_research_turn,
    run_standard_turn,
)


def test_finish_agent_turn_writes_trace_and_returns_events(tmp_path) -> None:
    context = AgentRunContext.create(session_id="demo", session=SessionContext(session_id="demo"))
    context.emit("session", {"session_id": "demo"})
    context.execution_steps.append({"node": "agent_loop", "summary": "done"})
    payload = {"answer": "hello"}

    final_payload, events = finish_agent_turn(
        settings=SimpleNamespace(agent_trace_enabled=True, data_dir=tmp_path),
        run_context=context,
        final_payload=payload,
    )

    assert final_payload is payload
    assert events == context.events
    traces = list((tmp_path / "traces" / "demo").glob("*.jsonl"))
    assert len(traces) == 1
    assert '"answer_preview": "hello"' in traces[0].read_text(encoding="utf-8")


def test_run_compound_turn_if_needed_delegates_with_run_context(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run_compound_query_if_needed(**kwargs: Any) -> dict[str, str]:
        captured.update(kwargs)
        kwargs["emit"]("agent_plan", {"actions": ["Task"]})
        kwargs["execution_steps"].append({"node": "compound_planner", "summary": "fake"})
        return {"answer": "compound"}

    monkeypatch.setattr(
        "app.services.agent.loop.run_compound_query_if_needed",
        fake_run_compound_query_if_needed,
    )
    agent = object()
    context = AgentRunContext.create(session_id="demo", session=SessionContext(session_id="demo"))

    payload = run_compound_turn_if_needed(
        agent=agent,
        run_context=context,
        query="DPO和PPO分别是什么？",
        clarification_choice={"option_id": "a"},
    )

    assert payload == {"answer": "compound"}
    assert captured["agent"] is agent
    assert captured["session_id"] == "demo"
    assert captured["session"] is context.session
    assert captured["clarification_choice"] == {"option_id": "a"}
    assert context.events[0]["event"] == "agent_plan"
    assert context.execution_steps == [{"node": "compound_planner", "summary": "fake"}]


def test_run_standard_turn_extracts_plans_and_routes_conversation() -> None:
    contract = QueryContract(clean_query="hello", interaction_mode="conversation", relation="greeting")
    agent = _FakeAgent(
        conversation_state={"answer": "hello"},
        standard_contract=contract,
    )
    context = AgentRunContext.create(session_id="demo", session=SessionContext(session_id="demo"))

    payload = run_standard_turn(
        agent=agent,
        run_context=context,
        query="hello",
                use_web_search=True,
        max_web_results=3,
        clarification_choice=None,
        stream_answer=False,
    )

    assert payload["answer"] == "hello"
    assert payload["query_contract"]["allow_web_search"] is True
    assert agent.extract_call["query"] == "hello"
    assert agent.plan_contract.allow_web_search is True
    assert [item["event"] for item in context.events[:2]] == ["contract", "agent_plan"]
    assert context.execution_steps[0]["node"] == "query_contract_extractor"


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
    assert agent.sessions.committed[0]["active"].titles == ["Demo Paper"]
    assert context.events[0]["data"]["name"] == "compose"


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
    assert payload["clarification_options"] == []
    assert context.session.clarification_attempts == 1
    assert context.session.last_clarification_key


def test_run_research_turn_composes_commits_and_streams_answer_delta() -> None:
    contract = QueryContract(
        clean_query="DPO是什么",
        interaction_mode="research",
        relation="entity_definition",
        targets=["DPO"],
        requested_fields=["definition"],
    )
    paper = CandidatePaper(paper_id="paper-1", title="DPO Paper")
    evidence = EvidenceBlock(
        doc_id="doc-1",
        paper_id="paper-1",
        title="DPO Paper",
        file_path="/tmp/dpo.pdf",
        page=2,
        block_type="text",
        snippet="DPO is a preference optimization method.",
    )
    claim = Claim(claim_type="definition", entity="DPO", value="preference optimization", evidence_ids=["doc-1"])
    agent = _FakeAgent(
        conversation_state={},
        research_state={
            "contract": contract,
            "plan": ResearchPlan(required_claims=["definition"]),
            "screened_papers": [paper],
            "evidence": [evidence],
            "claims": [claim],
            "verification": VerificationReport(status="pass"),
        },
    )
    context = AgentRunContext.create(session_id="demo", session=SessionContext(session_id="demo"))

    payload = run_research_turn(
        agent=agent,
        run_context=context,
        query="DPO是什么",
        contract=contract,
        agent_plan={"actions": ["search_corpus", "compose"]},
        web_enabled=False,
        explicit_web_search=False,
        max_web_results=3,
        stream_answer=True,
    )

    assert payload["answer"] == "research answer"
    assert payload["research_plan_summary"]["required_claims"] == ["definition"]
    assert payload["verification_report"]["status"] == "pass"
    assert context.session.answered_titles == ["DPO Paper"]
    assert context.session.working_memory["target_bindings"]["dpo"]["title"] == "DPO Paper"
    assert agent.sessions.committed[0]["turn"].answer == payload["answer"]
    assert agent.sessions.committed[0]["active"].titles == ["DPO Paper"]
    assert any(item["event"] == "answer_delta" for item in context.events)


def test_run_research_turn_can_emit_answer_logprob_confidence() -> None:
    contract = QueryContract(
        clean_query="DPO是什么",
        interaction_mode="research",
        relation="entity_definition",
        targets=["DPO"],
        requested_fields=["definition"],
    )
    paper = CandidatePaper(paper_id="paper-1", title="DPO Paper")
    evidence = EvidenceBlock(
        doc_id="doc-1",
        paper_id="paper-1",
        title="DPO Paper",
        file_path="/tmp/dpo.pdf",
        page=2,
        block_type="text",
        snippet="DPO is a preference optimization method.",
    )
    claim = Claim(claim_type="definition", entity="DPO", value="preference optimization", evidence_ids=["doc-1"])
    agent = _FakeAgent(
        conversation_state={},
        research_state={
            "contract": contract,
            "plan": ResearchPlan(required_claims=["definition"]),
            "screened_papers": [paper],
            "evidence": [evidence],
            "claims": [claim],
            "verification": VerificationReport(status="pass"),
        },
    )
    agent.agent_settings = SimpleNamespace(answer_logprobs_enabled=True, answer_logprobs_min_tokens=2)
    context = AgentRunContext.create(session_id="demo", session=SessionContext(session_id="demo"))

    payload = run_research_turn(
        agent=agent,
        run_context=context,
        query="DPO是什么",
        contract=contract,
        agent_plan={"actions": ["search_corpus", "compose"]},
        web_enabled=False,
        explicit_web_search=False,
        max_web_results=3,
        stream_answer=True,
    )

    confidence_events = [
        item["data"]
        for item in context.events
        if item["event"] == "confidence" and item["data"].get("basis") == "logprobs"
    ]
    assert confidence_events
    assert confidence_events[-1]["detail"]["token_count"] == 2
    assert payload["runtime_summary"]["answer_generation"]["confidence"]["basis"] == "logprobs"


def test_run_research_turn_can_emit_answer_self_consistency_confidence() -> None:
    contract = QueryContract(
        clean_query="DPO是什么",
        interaction_mode="research",
        relation="entity_definition",
        targets=["DPO"],
        requested_fields=["definition"],
    )
    paper = CandidatePaper(paper_id="paper-1", title="DPO Paper")
    evidence = EvidenceBlock(
        doc_id="doc-1",
        paper_id="paper-1",
        title="DPO Paper",
        file_path="/tmp/dpo.pdf",
        page=2,
        block_type="text",
        snippet="DPO is a preference optimization method.",
    )
    claim = Claim(claim_type="definition", entity="DPO", value="preference optimization", evidence_ids=["doc-1"])
    agent = _FakeAgent(
        conversation_state={},
        research_state={
            "contract": contract,
            "plan": ResearchPlan(required_claims=["definition"]),
            "screened_papers": [paper],
            "evidence": [evidence],
            "claims": [claim],
            "verification": VerificationReport(status="pass"),
        },
    )
    agent.agent_settings = SimpleNamespace(
        answer_logprobs_enabled=False,
        answer_logprobs_min_tokens=2,
        answer_self_consistency_enabled=True,
        answer_self_consistency_samples=3,
        max_clarification_attempts=3,
    )
    context = AgentRunContext.create(session_id="demo", session=SessionContext(session_id="demo"))

    payload = run_research_turn(
        agent=agent,
        run_context=context,
        query="DPO是什么",
        contract=contract,
        agent_plan={"actions": ["search_corpus", "compose"]},
        web_enabled=False,
        explicit_web_search=False,
        max_web_results=3,
        stream_answer=True,
    )

    confidence_events = [
        item["data"]
        for item in context.events
        if item["event"] == "confidence" and item["data"].get("basis") == "self_consistency"
    ]
    assert confidence_events
    assert confidence_events[-1]["detail"]["sample_count"] == 3
    assert payload["runtime_summary"]["answer_generation"]["confidence"]["basis"] == "self_consistency"


class _FakeRuntime:
    def __init__(self, conversation_state: dict[str, Any], research_state: dict[str, Any] | None = None) -> None:
        self.conversation_state = conversation_state
        self.research_state = research_state

    def execute_conversation_tools(self, **kwargs: Any) -> dict[str, Any]:
        kwargs["emit"]("observation", {"tool": "compose", "summary": "done", "payload": {"answer": "ok"}})
        kwargs["execution_steps"].append({"node": "agent_tool:compose", "summary": "done"})
        return self.conversation_state

    def run_research_agent_loop(self, **kwargs: Any) -> dict[str, Any]:
        kwargs["emit"]("observation", {"tool": "search_corpus", "summary": "done", "payload": {"count": 1}})
        kwargs["execution_steps"].append({"node": "agent_tool:search_corpus", "summary": "done"})
        if self.research_state is None:
            raise AssertionError("missing research state")
        return self.research_state


class _FakeSessions:
    def __init__(self) -> None:
        self.committed: list[dict[str, Any]] = []

    def commit_turn(self, session: SessionContext, turn: Any, *, active: Any = None) -> None:
        self.committed.append({"session": session, "turn": turn, "active": active})


class _FakeAgent:
    def __init__(
        self,
        *,
        conversation_state: dict[str, Any],
        research_state: dict[str, Any] | None = None,
        standard_contract: QueryContract | None = None,
    ) -> None:
        self.runtime = _FakeRuntime(conversation_state, research_state)
        self.sessions = _FakeSessions()
        self.standard_contract = standard_contract
        self.retriever = SimpleNamespace(paper_doc_by_id=lambda _: None)
        self.planner = SimpleNamespace(plan_actions=self._plan_actions)
        self.clients = SimpleNamespace(chat=object(), invoke_text=lambda **_: "Which target?")
        self.settings = SimpleNamespace(data_dir=Path("/tmp/pdf-rag-agent-v4-test-empty"))
        self.extract_call: dict[str, Any] = {}
        self.plan_contract: QueryContract | None = None
        self.agent_settings = SimpleNamespace(
            answer_logprobs_enabled=False,
            answer_logprobs_min_tokens=3,
            max_clarification_attempts=3,
        )

    def query_contract_extractor(self, **kwargs: Any) -> QueryContract:
        self.extract_call = dict(kwargs)
        if self.standard_contract is None:
            raise AssertionError("missing standard contract")
        return self.standard_contract

    def _plan_actions(self, *, contract: QueryContract, **_: Any) -> dict[str, list[str]]:
        self.plan_contract = contract
        return {"actions": ["compose"]}

    @staticmethod
    def _conversation_relation_updates_research_context(relation: str) -> bool:
        return relation == "library_status"

    @staticmethod
    def _compose_answer(**kwargs: Any) -> tuple[str, list[AssistantCitation]]:
        stream_callback = kwargs.get("stream_callback")
        if callable(stream_callback):
            stream_callback("research ")
            stream_callback("answer")
        logprob_callback = kwargs.get("logprob_callback")
        if kwargs.get("request_logprobs") and callable(logprob_callback):
            logprob_callback([-0.1, -0.2])
        return (
            "research answer",
            [
                AssistantCitation(
                    doc_id="doc-1",
                    paper_id="paper-1",
                    title="DPO Paper",
                    file_path="/tmp/dpo.pdf",
                    page=2,
                    snippet="DPO is a preference optimization method.",
                )
            ],
        )

    @staticmethod
    def _candidate_from_paper_id(_: str) -> None:
        return None


# ── LLM-driven loop tests ──────────────────────────────────────────────


def _build_llm_loop_agent() -> tuple[Any, list[dict[str, Any]]]:
    """Build a fake agent that exercises the LLM-driven loop path."""
    tool_calls_log: list[dict[str, Any]] = []
    text_log: list[dict[str, Any]] = []
    call_count = [0]  # mutable counter

    def fake_planner_messages(*, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], fallback: Any) -> dict[str, Any]:
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: return a tool call
            tool_calls_log.append({"messages": len(messages), "tools": len(tools)})
            return {
                "thought": "I need to search.",
                "actions": ["search_corpus"],
                "tool_call_args": [{"name": "search_corpus", "args": {"query": "test"}}],
            }
        # Second call: no more tools — LLM stops
        tool_calls_log.append({"messages": len(messages), "final": True})
        return {"thought": "Final answer computed.", "actions": [], "tool_call_args": []}

    def fake_invoke_text(*, system_prompt: str, human_prompt: str, **__: Any) -> str:
        text_log.append({"system": system_prompt[:80], "human": human_prompt[:80]})
        return "This is the final answer."

    agent = SimpleNamespace(
        clients=SimpleNamespace(
            invoke_tool_plan_messages=fake_planner_messages,
            invoke_text=fake_invoke_text,
        ),
        agent_settings=SimpleNamespace(
            llm_driven_loop_enabled=True,
            max_agent_steps=8,
            max_parallel_tools=2,
            max_calls_per_tool=3,
        ),
        planner=SimpleNamespace(
            choose_next_actions=lambda **_: [],
        ),
    )
    return agent, tool_calls_log, text_log


def test_llm_driven_loop_calls_planner_messages_with_tool_results() -> None:
    """The LLM-driven loop feeds tool results back as message history."""
    from app.services.agent.runtime_helpers import execute_tool_loop, research_runtime_state
    from app.services.agent.tools import AgentToolExecutor, RegisteredAgentTool, research_execution_tool_names

    agent, tool_calls_log, text_log = _build_llm_loop_agent()
    contract = QueryContract(clean_query="test query", interaction_mode="research")
    state = research_runtime_state(
        contract=contract,
        plan=SimpleNamespace(solver_sequence=[], required_claims=[]),
        excluded_titles=set(),
        agent_plan={"actions": ["search_corpus"]},
    )

    def search_corpus_handler(arguments: dict[str, Any] | None = None) -> None:
        pass

    tools: dict[str, RegisteredAgentTool] = {
        "search_corpus": RegisteredAgentTool("search_corpus", search_corpus_handler, accepts_arguments=True),
    }
    executor = AgentToolExecutor(tools)

    events: list[tuple[str, dict[str, Any]]] = []
    execute_tool_loop(
        agent=agent,
        contract=contract,
        session=SessionContext(session_id="test"),
        state=state,
        executor=executor,
        planned_actions=["search_corpus"],
        allowed_tools=research_execution_tool_names(),
        emit=lambda e, d: events.append((e, d)),
        fallback_next=lambda _: None,
        stop_condition=lambda _: False,
    )

    # Verify the LLM was called with messages history
    assert len(tool_calls_log) == 2
    # First call: only user message
    assert tool_calls_log[0]["tools"] > 0
    # Second call: user message + tool result(s)
    assert tool_calls_log[1]["final"] is True
    # The final answer was produced
    assert state.get("answer") == "This is the final answer."
    assert text_log


def test_llm_driven_loop_skips_when_disabled() -> None:
    """When llm_driven_loop_enabled is False, the Python-driven loop runs."""
    from app.services.agent.runtime_helpers import execute_tool_loop, research_runtime_state
    from app.services.agent.tools import AgentToolExecutor, RegisteredAgentTool, research_execution_tool_names

    agent = SimpleNamespace(
        clients=SimpleNamespace(),
        agent_settings=SimpleNamespace(
            llm_driven_loop_enabled=False,
            max_agent_steps=8,
            max_parallel_tools=2,
            max_calls_per_tool=3,
        ),
        planner=SimpleNamespace(
            choose_next_actions=lambda **_: [],
        ),
    )
    contract = QueryContract(clean_query="test", interaction_mode="research")
    state = research_runtime_state(
        contract=contract,
        plan=SimpleNamespace(solver_sequence=[], required_claims=[]),
        excluded_titles=set(),
        agent_plan={"actions": []},
    )
    executor = AgentToolExecutor({})
    events: list[tuple[str, dict[str, Any]]] = []

    # Should not raise — just returns immediately since no actions, no planner
    execute_tool_loop(
        agent=agent,
        contract=contract,
        session=SessionContext(session_id="test"),
        state=state,
        executor=executor,
        planned_actions=[],
        allowed_tools=research_execution_tool_names(),
        emit=lambda e, d: events.append((e, d)),
        fallback_next=lambda _: None,
        stop_condition=lambda _: False,
    )
    # Loop exited cleanly
    assert len(events) == 0
