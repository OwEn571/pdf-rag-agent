from __future__ import annotations

from types import SimpleNamespace

from app.core.agent_settings import AgentSettings
from app.core.config import Settings
from app.domain.models import ActiveResearch, EvidenceBlock, SessionContext, VerificationReport
from app.domain.models import QueryContract
from app.services import agent_tools
from app.services.agent import ResearchAssistantAgentV4
from app.services.agent_planner import AgentPlanner
from app.services.agent_runtime import AgentRuntime
from app.services.agent_tool_registries import (
    build_conversation_tool_registry,
    build_research_tool_registry,
)
from app.services.session_store import InMemorySessionStore
from app.services.agent_tools import (
    AgentToolExecutor,
    agent_tool_manifest,
    agent_tool_manifest_for_names,
    all_agent_tool_names,
    conversation_tool_sequence,
    normalize_plan_actions,
    research_execution_tool_names,
    research_tool_sequence,
)
from app.services.agent_tools import RegisteredAgentTool
from app.services.contract_context import LEGACY_TOOL_NAME_ALIASES, canonical_tools


class _ToolPlanClients:
    chat = object()

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.tool_plan_calls = 0
        self.json_calls = 0

    def invoke_tool_plan(self, *, tools: list[dict[str, object]], **_: object) -> dict[str, object]:
        self.tool_plan_calls += 1
        assert {str(item["name"]) for item in tools} >= {"search_corpus", "compose"}
        return self.payload

    def invoke_json(self, **_: object) -> dict[str, object]:
        self.json_calls += 1
        return {"thought": "json fallback", "actions": ["compose"], "stop_conditions": []}


class _RegistryProbeAgent:
    def __init__(self) -> None:
        self.observations: list[dict[str, object]] = []
        self.runtime = AgentRuntime(agent=self)
        self.settings = SimpleNamespace(paper_limit_default=6, evidence_limit_default=2, llm_retry_budget=1)

    def _record_agent_observation(self, **kwargs: object) -> None:
        self.observations.append(kwargs)
        emit = kwargs.get("emit")
        if callable(emit):
            emit(
                "observation",
                {
                    "tool": kwargs.get("tool", ""),
                    "summary": kwargs.get("summary", ""),
                    "payload": kwargs.get("payload", {}),
                },
            )

    def _emit_agent_tool_call(self, **_: object) -> None:
        return None

    def _compose_conversation_response(self, **_: object) -> str:
        return "hello from runtime"

    def _set_conversation_answer(self, *, state: dict[str, object], answer: str, **_: object) -> None:
        state["answer"] = answer

    def _extract_query_contract(self, *, query: str, **_: object) -> QueryContract:
        return QueryContract(clean_query=query, interaction_mode="conversation", relation="greeting")

    def _plan_agent_actions(self, **_: object) -> dict[str, object]:
        return {"actions": ["compose"], "tool_call_args": []}

    def _excluded_focus_titles(self, **_: object) -> set[str]:
        return set()

    def _is_negative_correction_query(self, query: str) -> bool:
        return False

    def _agent_search_papers(self, *, state: dict[str, object], **_: object) -> None:
        state["candidate_papers"] = ["paper"]
        state["screened_papers"] = ["paper"]

    def _agent_search_evidence(self, *, state: dict[str, object], **_: object) -> None:
        state["evidence"] = ["evidence"]

    def _agent_solve_claims(self, *, state: dict[str, object], **_: object) -> None:
        state["claims"] = ["claim"]

    def _agent_verify_grounding(self, *, state: dict[str, object], **_: object) -> None:
        state["verification"] = VerificationReport(status="pass", recommended_action="probe_pass")

    def _agent_reflect(self, *, state: dict[str, object], **_: object) -> None:
        state["reflection"] = {"checked": True}

    def _merge_evidence(self, existing: list[EvidenceBlock], new: list[EvidenceBlock]) -> list[EvidenceBlock]:
        merged: dict[str, EvidenceBlock] = {item.doc_id: item for item in existing if isinstance(item, EvidenceBlock)}
        for item in new:
            merged[item.doc_id] = item
        return list(merged.values())

    def _candidate_from_paper_id(self, paper_id: str) -> None:
        return None


def _agent_step_payloads(events: list[tuple[str, dict[str, object]]]) -> list[dict[str, object]]:
    return [
        {"action": payload.get("action"), "arguments": payload.get("arguments", {})}
        for event, payload in events
        if event == "agent_step"
    ]


class _RerankProbeRetriever:
    def rerank_evidence(
        self,
        *,
        query: str,
        evidence: list[EvidenceBlock],
        top_k: int,
        focus: list[str] | None = None,
    ) -> list[EvidenceBlock]:
        terms = [item.lower() for item in [query, *list(focus or [])] if item]

        def score(item: EvidenceBlock) -> int:
            text = f"{item.title} {item.snippet}".lower()
            return sum(1 for term in terms if term in text)

        return sorted(evidence, key=score, reverse=True)[:top_k]


class _AtomicSearchProbeRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def hybrid_search(self, **kwargs: object) -> list[EvidenceBlock]:
        self.calls.append(dict(kwargs))
        return [
            EvidenceBlock(
                doc_id="hybrid-1",
                paper_id="paper-1",
                title="DPO paper",
                file_path="",
                page=1,
                block_type="text",
                snippet="DPO objective evidence",
                score=0.9,
                metadata={"search_source": "hybrid_search"},
            )
        ]


def _planner_agent(tmp_path, clients: object) -> ResearchAssistantAgentV4:
    settings = Settings(
        _env_file=None,
        openai_api_key="",
        data_dir=tmp_path / "data",
        paper_store_path=tmp_path / "data" / "papers.jsonl",
        block_store_path=tmp_path / "data" / "blocks.jsonl",
        ingestion_state_path=tmp_path / "data" / "state.json",
        session_store_path=tmp_path / "data" / "sessions.sqlite3",
        eval_cases_path=tmp_path / "evals" / "cases.yaml",
    )
    return ResearchAssistantAgentV4(
        settings=settings,
        retriever=object(),  # type: ignore[arg-type]
        clients=clients,  # type: ignore[arg-type]
        sessions=InMemorySessionStore(),
    )


def test_agent_tool_manifest_and_allowed_sets_share_one_registry() -> None:
    manifest_names = {item["name"] for item in agent_tool_manifest()}

    assert manifest_names == {
        "read_memory",
        "search_corpus",
        "bm25_search",
        "vector_search",
        "hybrid_search",
        "rerank",
        "read_pdf_page",
        "grep_corpus",
        "query_rewrite",
        "summarize",
        "verify_claim",
        "web_search",
        "fetch_url",
        "query_library_metadata",
        "compose",
        "todo_write",
        "remember",
        "propose_tool",
        "Task",
        "ask_human",
    }
    search_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "search_corpus")
    assert search_schema["required"] == ["query"]
    assert search_schema["properties"]["top_k"]["maximum"] == 50
    assert search_schema["properties"]["strategy"]["enum"] == ["auto", "legacy", "bm25", "vector", "hybrid"]
    bm25_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "bm25_search")
    assert bm25_schema["required"] == ["query"]
    hybrid_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "hybrid_search")
    assert hybrid_schema["properties"]["alpha"]["maximum"] == 1.0
    rerank_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "rerank")
    assert rerank_schema["required"] == ["query"]
    assert "candidates" in rerank_schema["properties"]
    read_page_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "read_pdf_page")
    assert read_page_schema["required"] == ["paper_id", "page_from"]
    grep_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "grep_corpus")
    assert grep_schema["required"] == ["regex"]
    query_rewrite_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "query_rewrite")
    assert query_rewrite_schema["required"] == ["query"]
    assert query_rewrite_schema["properties"]["mode"]["enum"] == ["multi_query", "hyde", "step_back"]
    summarize_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "summarize")
    assert summarize_schema["properties"]["target_words"]["maximum"] == 1000
    verify_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "verify_claim")
    assert verify_schema["required"] == ["claim"]
    assert verify_schema["properties"]["min_overlap"]["default"] == 2
    todo_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "todo_write")
    assert todo_schema["required"] == ["items"]
    remember_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "remember")
    assert remember_schema["required"] == ["key", "content"]
    propose_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "propose_tool")
    assert propose_schema["required"] == ["name", "description", "input_schema", "python_code", "rationale"]
    assert next(item for item in agent_tool_manifest() if item["name"] == "propose_tool")["dangerous"] is True
    task_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "Task")
    assert task_schema["required"] == ["description", "prompt"]
    fetch_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "fetch_url")
    assert fetch_schema["required"] == ["url"]
    assert [tool["name"] for tool in agent_tool_manifest_for_names({"compose", "not_a_tool"})] == ["compose"]
    assert "search_corpus" in research_execution_tool_names()
    assert all_agent_tool_names() == manifest_names
    assert research_execution_tool_names() <= all_agent_tool_names()


def test_tool_sequence_policies_are_centralized() -> None:
    assert conversation_tool_sequence(relation="library_status") == []
    assert conversation_tool_sequence(
        relation="memory_followup",
        planned_actions=["read_memory", "compose"],
    ) == ["read_memory", "compose"]
    assert research_tool_sequence(
        planned_actions=["solve_claims"],
        use_web_search=True,
        needs_reflection=True,
    ) == []
    assert research_tool_sequence(planned_actions=[], use_web_search=True, needs_reflection=True) == []
    assert normalize_plan_actions(actions=["search_corpus", "not_a_tool"], allowed=all_agent_tool_names()) == ["search_corpus"]


def test_legacy_solver_aliases_are_not_accepted_as_canonical_tools() -> None:
    assert (
        canonical_tools(
            raw_tools=["search_papers", "search_evidence", "solve_claims", "verify_grounding"],
            aliases=LEGACY_TOOL_NAME_ALIASES,
            canonical_names=all_agent_tool_names(),
        )
        == []
    )


def test_registered_research_tool_executor_runs_dependencies_once() -> None:
    calls: list[str] = []
    tools = {
        "search_papers": RegisteredAgentTool("search_papers", lambda: calls.append("search_papers")),
        "search_evidence": RegisteredAgentTool(
            "search_evidence",
            lambda: calls.append("search_evidence"),
            requires=("search_papers",),
        ),
        "solve_claims": RegisteredAgentTool(
            "solve_claims",
            lambda: calls.append("solve_claims"),
            requires=("search_papers", "search_evidence"),
        ),
    }
    executor = AgentToolExecutor(tools)
    executor.executed.add("search_papers")

    should_stop = executor.run("solve_claims")

    assert should_stop is False
    assert calls == ["search_evidence", "solve_claims"]
    assert executor.executed == {"search_papers", "search_evidence", "solve_claims"}


def test_agent_tool_executor_records_success_latency(monkeypatch) -> None:
    metrics: list[dict[str, object]] = []
    monkeypatch.setattr(agent_tools, "record_tool_execution", lambda **kwargs: metrics.append(kwargs))
    executor = AgentToolExecutor({"compose": RegisteredAgentTool("compose", lambda: None, terminal=True)})

    should_stop = executor.run("compose")

    assert should_stop is True
    assert metrics[0]["name"] == "compose"
    assert metrics[0]["ok"] is True
    assert isinstance(metrics[0]["elapsed_seconds"], float)
    assert metrics[0]["elapsed_seconds"] >= 0


def test_agent_tool_executor_records_failed_call(monkeypatch) -> None:
    metrics: list[dict[str, object]] = []
    monkeypatch.setattr(agent_tools, "record_tool_execution", lambda **kwargs: metrics.append(kwargs))

    def fail() -> None:
        raise RuntimeError("boom")

    executor = AgentToolExecutor({"compose": RegisteredAgentTool("compose", fail, terminal=True)})

    try:
        executor.run("compose")
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("expected tool failure")

    assert executor.executed == set()
    assert metrics[0]["name"] == "compose"
    assert metrics[0]["ok"] is False


def test_tool_registry_builders_are_outside_agent_class() -> None:
    probe = _RegistryProbeAgent()
    contract = QueryContract(clean_query="hello", relation="greeting")
    session = SessionContext(session_id="demo")
    events: list[tuple[str, dict[str, object]]] = []
    steps: list[dict[str, object]] = []

    conversation_tools = build_conversation_tool_registry(
        agent=probe,
        state={"contract": contract, "answer": ""},
        contract=contract,
        query="hello",
        session=session,
        max_web_results=0,
        emit=lambda event, payload: events.append((event, payload)),
        execution_steps=steps,
    )
    research_tools = build_research_tool_registry(
        agent=probe,
        state={"contract": contract, "excluded_titles": set(), "verification": None},
        session=session,
        web_enabled=False,
        explicit_web_search=False,
        max_web_results=0,
        emit=lambda event, payload: events.append((event, payload)),
        execution_steps=steps,
    )

    assert set(conversation_tools) == {
        "read_memory",
        "todo_write",
        "remember",
        "propose_tool",
        "summarize",
        "verify_claim",
        "Task",
        "web_search",
        "fetch_url",
        "query_library_metadata",
        "compose",
        "ask_human",
    }
    assert set(research_tools) == {
        "read_memory",
        "todo_write",
        "remember",
        "propose_tool",
        "search_corpus",
        "bm25_search",
        "vector_search",
        "hybrid_search",
        "rerank",
        "read_pdf_page",
        "grep_corpus",
        "query_rewrite",
        "summarize",
        "verify_claim",
        "Task",
        "web_search",
        "fetch_url",
        "compose",
        "ask_human",
    }
    assert "search_corpus" in research_tools


def test_agent_runtime_runs_conversation_and_research_loops() -> None:
    probe = _RegistryProbeAgent()
    runtime = AgentRuntime(agent=probe)
    session = SessionContext(session_id="demo")
    events: list[tuple[str, dict[str, object]]] = []
    steps: list[dict[str, object]] = []

    conversation_state = runtime.execute_conversation_tools(
        contract=QueryContract(clean_query="hello", interaction_mode="conversation", relation="greeting"),
        query="hello",
        session=session,
        agent_plan={"actions": ["compose"]},
        max_web_results=0,
        emit=lambda event, payload: events.append((event, payload)),
        execution_steps=steps,
    )
    research_state = runtime.run_research_agent_loop(
        contract=QueryContract(clean_query="PPO 公式", relation="formula_lookup", targets=["PPO"]),
        session=session,
        agent_plan={"actions": ["search_corpus", "compose"]},
        web_enabled=False,
        explicit_web_search=False,
        max_web_results=0,
        emit=lambda event, payload: events.append((event, payload)),
        execution_steps=steps,
    )

    assert conversation_state["answer"] == "hello from runtime"
    assert isinstance(research_state["verification"], VerificationReport)
    assert research_state["verification"].status == "pass"
    assert research_state["confidence"]["basis"] == "verifier"
    assert research_state["confidence"]["score"] > 0.8
    assert any(event == "confidence" and payload["basis"] == "verifier" for event, payload in events)
    agent_step_actions = [payload["action"] for payload in _agent_step_payloads(events)]
    assert "search_corpus" in agent_step_actions
    assert "compose" in agent_step_actions


def test_agent_runtime_preserves_structured_tool_arguments() -> None:
    probe = _RegistryProbeAgent()
    runtime = AgentRuntime(agent=probe)
    session = SessionContext(session_id="demo")
    events: list[tuple[str, dict[str, object]]] = []
    steps: list[dict[str, object]] = []

    research_state = runtime.run_research_agent_loop(
        contract=QueryContract(clean_query="PPO 公式", relation="formula_lookup", targets=["PPO"]),
        session=session,
        agent_plan={
            "actions": ["search_corpus", "compose"],
            "tool_call_args": [{"name": "search_corpus", "args": {"query": "custom PPO query", "top_k": 3}}],
        },
        web_enabled=False,
        explicit_web_search=False,
        max_web_results=0,
        emit=lambda event, payload: events.append((event, payload)),
        execution_steps=steps,
    )

    assert research_state["tool_inputs"]["search_corpus"] == {"query": "custom PPO query", "top_k": 3}
    assert {"action": "search_corpus", "arguments": {"query": "custom PPO query", "top_k": 3}} in _agent_step_payloads(events)


def test_summarize_and_verify_claim_tools_run_inside_research_loop() -> None:
    probe = _RegistryProbeAgent()
    runtime = AgentRuntime(agent=probe)
    session = SessionContext(session_id="demo")
    events: list[tuple[str, dict[str, object]]] = []
    steps: list[dict[str, object]] = []

    state = runtime.run_research_agent_loop(
        contract=QueryContract(clean_query="PPO 目标函数", relation="formula_lookup", targets=["PPO"]),
        session=session,
        agent_plan={
            "actions": ["summarize", "verify_claim", "compose"],
            "tool_call_args": [
                {
                    "name": "summarize",
                    "args": {
                        "text": "PPO uses a clipped surrogate objective. This objective limits policy updates.",
                        "target_words": 24,
                        "focus": ["PPO", "clipped"],
                    },
                },
                {
                    "name": "verify_claim",
                    "args": {
                        "claim": "PPO uses a clipped surrogate objective",
                        "evidence": ["The PPO algorithm optimizes a clipped surrogate objective to limit policy updates."],
                    },
                },
            ],
        },
        web_enabled=False,
        explicit_web_search=False,
        max_web_results=0,
        emit=lambda event, payload: events.append((event, payload)),
        execution_steps=steps,
    )

    assert state["summaries"][0]["source_chars"] > 0
    assert "clipped surrogate objective" in state["summaries"][0]["summary"].lower()
    assert state["claim_checks"][0]["status"] == "pass"
    assert state["claim_checks"][0]["supporting_evidence_ids"] == ["inline::1"]
    assert {"action": "verify_claim", "arguments": state["tool_inputs"]["verify_claim"]} in _agent_step_payloads(events)


def test_query_rewrite_tool_runs_inside_research_loop() -> None:
    probe = _RegistryProbeAgent()
    runtime = AgentRuntime(agent=probe)
    session = SessionContext(session_id="demo")
    events: list[tuple[str, dict[str, object]]] = []
    steps: list[dict[str, object]] = []

    state = runtime.run_research_agent_loop(
        contract=QueryContract(clean_query="DPO 核心公式", relation="formula_lookup", targets=["DPO"]),
        session=session,
        agent_plan={
            "actions": ["query_rewrite", "compose"],
            "tool_call_args": [
                {
                    "name": "query_rewrite",
                    "args": {"query": "核心公式", "targets": ["DPO"], "mode": "step_back", "max_queries": 3},
                }
            ],
        },
        web_enabled=False,
        explicit_web_search=False,
        max_web_results=0,
        emit=lambda event, payload: events.append((event, payload)),
        execution_steps=steps,
    )

    assert state["query_rewrites"][0]["mode"] == "step_back"
    assert state["rewritten_queries"][0] == "核心公式"
    assert any(event == "observation" and payload["tool"] == "query_rewrite" for event, payload in events)


def test_rerank_tool_accepts_explicit_candidates() -> None:
    probe = _RegistryProbeAgent()
    probe.settings.evidence_limit_default = 2
    probe.retriever = _RerankProbeRetriever()
    runtime = AgentRuntime(agent=probe)
    session = SessionContext(session_id="demo")
    events: list[tuple[str, dict[str, object]]] = []
    steps: list[dict[str, object]] = []

    state = runtime.run_research_agent_loop(
        contract=QueryContract(clean_query="DPO objective", relation="formula_lookup", targets=["DPO"]),
        session=session,
        agent_plan={
            "actions": ["rerank", "compose"],
            "tool_call_args": [
                {
                    "name": "rerank",
                    "args": {
                        "query": "DPO objective",
                        "top_k": 1,
                        "focus": ["DPO"],
                        "candidates": [
                            {"doc_id": "weak", "title": "Other", "snippet": "general RLHF background"},
                            {
                                "doc_id": "strong",
                                "title": "DPO paper",
                                "snippet": "DPO objective optimizes preference likelihood.",
                            },
                        ],
                    },
                }
            ],
        },
        web_enabled=False,
        explicit_web_search=False,
        max_web_results=0,
        emit=lambda event, payload: events.append((event, payload)),
        execution_steps=steps,
    )

    assert [item.doc_id for item in state["evidence"]] == ["strong"]
    observation = next(payload for event, payload in events if event == "observation" and payload["tool"] == "rerank")
    assert observation["payload"]["used_explicit_candidates"] is True
    assert observation["payload"]["input_candidate_count"] == 2


def test_search_corpus_can_delegate_to_atomic_strategy() -> None:
    probe = _RegistryProbeAgent()
    probe.settings.evidence_limit_default = 2
    retriever = _AtomicSearchProbeRetriever()
    probe.retriever = retriever
    runtime = AgentRuntime(agent=probe)
    session = SessionContext(session_id="demo")
    events: list[tuple[str, dict[str, object]]] = []
    steps: list[dict[str, object]] = []

    state = runtime.run_research_agent_loop(
        contract=QueryContract(clean_query="DPO objective", relation="formula_lookup", targets=["DPO"]),
        session=session,
        agent_plan={
            "actions": ["search_corpus", "compose"],
            "tool_call_args": [
                {
                    "name": "search_corpus",
                    "args": {"strategy": "hybrid", "query": "DPO objective", "top_k": 1},
                }
            ],
        },
        web_enabled=False,
        explicit_web_search=False,
        max_web_results=0,
        emit=lambda event, payload: events.append((event, payload)),
        execution_steps=steps,
    )

    assert retriever.calls[0]["query"] == "DPO objective"
    assert retriever.calls[0]["limit"] == 1
    assert [item.doc_id for item in state["evidence"]] == ["hybrid-1"]
    assert any(event == "observation" and payload["tool"] == "hybrid_search" for event, payload in events)


def test_todo_write_tool_updates_session_memory_and_emits_event() -> None:
    probe = _RegistryProbeAgent()
    runtime = AgentRuntime(agent=probe)
    session = SessionContext(session_id="demo")
    events: list[tuple[str, dict[str, object]]] = []
    steps: list[dict[str, object]] = []
    todos = [
        {"id": "search", "text": "检索本地证据", "status": "doing"},
        {"id": "answer", "text": "整理带引用回答", "status": "pending"},
    ]

    runtime.run_research_agent_loop(
        contract=QueryContract(clean_query="PPO 公式", relation="formula_lookup", targets=["PPO"]),
        session=session,
        agent_plan={
            "actions": ["todo_write", "search_corpus", "compose"],
            "tool_call_args": [{"name": "todo_write", "args": {"items": todos}}],
        },
        web_enabled=False,
        explicit_web_search=False,
        max_web_results=0,
        emit=lambda event, payload: events.append((event, payload)),
        execution_steps=steps,
    )

    assert session.working_memory["todos"] == todos
    assert ("todo_update", {"items": todos}) in events
    assert "todo_write" in [payload["action"] for payload in _agent_step_payloads(events)]


def test_task_tool_runs_subtask_through_conversation_runtime() -> None:
    probe = _RegistryProbeAgent()
    runtime = AgentRuntime(agent=probe)
    session = SessionContext(session_id="demo")
    events: list[tuple[str, dict[str, object]]] = []
    steps: list[dict[str, object]] = []

    state = runtime.execute_conversation_tools(
        contract=QueryContract(clean_query="拆成子任务", interaction_mode="conversation", relation="greeting"),
        query="拆成子任务",
        session=session,
        agent_plan={
            "actions": ["Task", "compose"],
            "tool_call_args": [
                {
                    "name": "Task",
                    "args": {
                        "description": "回答一个子问题",
                        "prompt": "子问题：你好",
                        "tools_allowed": ["compose"],
                    },
                }
            ],
        },
        max_web_results=0,
        emit=lambda event, payload: events.append((event, payload)),
        execution_steps=steps,
    )

    assert state["task_results"][0]["prompt"] == "子问题：你好"
    assert state["task_results"][0]["answer"] == "hello from runtime"
    assert "子问题：你好" in state["answer"]
    assert "hello from runtime" in state["answer"]
    assert any(event == "observation" and payload["tool"] == "Task" for event, payload in events)


def test_task_tool_runs_subtask_through_research_runtime() -> None:
    probe = _RegistryProbeAgent()
    runtime = AgentRuntime(agent=probe)
    session = SessionContext(session_id="demo")
    events: list[tuple[str, dict[str, object]]] = []
    steps: list[dict[str, object]] = []

    state = runtime.run_research_agent_loop(
        contract=QueryContract(clean_query="研究子任务", relation="general_question"),
        session=session,
        agent_plan={
            "actions": ["Task", "compose"],
            "tool_call_args": [
                {
                    "name": "Task",
                    "args": {
                        "description": "研究一个子问题",
                        "prompt": "子问题：DPO 是什么？",
                        "tools_allowed": ["compose"],
                    },
                }
            ],
        },
        web_enabled=False,
        explicit_web_search=False,
        max_web_results=0,
        emit=lambda event, payload: events.append((event, payload)),
        execution_steps=steps,
    )

    assert state["task_results"][0]["prompt"] == "子问题：DPO 是什么？"
    assert state["task_results"][0]["answer"] == "hello from runtime"
    assert any(event == "observation" and payload["tool"] == "Task" for event, payload in events)


def test_agent_tool_executor_detects_dependency_cycles() -> None:
    tools = {
        "a": RegisteredAgentTool("a", lambda: None, requires=("b",)),
        "b": RegisteredAgentTool("b", lambda: None, requires=("a",)),
    }
    executor = AgentToolExecutor(tools)

    try:
        executor.run("a")
    except RuntimeError as exc:
        assert "dependency cycle" in str(exc)
    else:
        raise AssertionError("expected dependency cycle error")


def test_planner_prefers_tool_call_payload_over_json_fallback(tmp_path) -> None:
    clients = _ToolPlanClients(
        {
            "thought": "tool calls",
            "actions": ["search_corpus", "compose"],
            "tool_call_args": [{"name": "search_corpus", "args": {"reason": "local corpus"}}],
            "stop_conditions": ["tool_calls_selected"],
        }
    )
    agent = _planner_agent(tmp_path, clients)

    plan = agent._plan_agent_actions(
        contract=QueryContract(clean_query="PPO 公式", relation="formula_lookup", targets=["PPO"]),
        session=SessionContext(session_id="demo"),
        use_web_search=False,
    )

    assert plan["actions"] == ["search_corpus", "compose"]
    assert plan["tool_call_args"][0]["name"] == "search_corpus"
    assert clients.tool_plan_calls == 1
    assert clients.json_calls == 0


def test_agent_planner_runs_without_agent_instance() -> None:
    clients = _ToolPlanClients(
        {
            "thought": "standalone tool calls",
            "actions": ["search_corpus", "compose"],
            "tool_call_args": [{"name": "search_corpus", "args": {"reason": "start from corpus"}}],
            "stop_conditions": ["tool_calls_selected"],
        }
    )
    planner = AgentPlanner(
        clients=clients,
        conversation_context=lambda session: {"session_id": session.session_id, "turns": []},
        is_negative_correction_query=lambda query: False,
    )

    plan = planner.plan_actions(
        contract=QueryContract(clean_query="PPO 公式", relation="formula_lookup", targets=["PPO"]),
        session=SessionContext(session_id="demo"),
        use_web_search=False,
    )

    assert plan["actions"] == ["search_corpus", "compose"]
    assert clients.tool_plan_calls == 1
    assert clients.json_calls == 0


def test_planner_falls_back_to_json_when_tool_calls_are_empty(tmp_path) -> None:
    clients = _ToolPlanClients({})
    agent = _planner_agent(tmp_path, clients)

    plan = agent._plan_agent_actions(
        contract=QueryContract(clean_query="PPO 公式", relation="formula_lookup", targets=["PPO"]),
        session=SessionContext(session_id="demo"),
        use_web_search=False,
    )

    assert plan["actions"] == ["compose"]
    assert clients.tool_plan_calls == 1
    assert clients.json_calls == 1


def test_session_context_populates_active_research_from_legacy_fields() -> None:
    session = SessionContext(
        session_id="demo",
        active_research_relation="entity_definition",
        active_targets=["GRPO"],
        active_titles=["DeepSeekMath"],
        active_requested_fields=["definition"],
        active_required_modalities=["page_text"],
        active_answer_shape="bullets",
        active_precision_requirement="high",
        active_clean_query="GRPO 是什么",
    )

    assert session.active_research.relation == "entity_definition"
    assert session.active_research.targets == ["GRPO"]
    assert session.active_research.titles == ["DeepSeekMath"]


def test_session_context_syncs_active_research_to_legacy_fields() -> None:
    session = SessionContext(
        session_id="demo",
        active_research=ActiveResearch(
            relation="formula_lookup",
            targets=["PPO"],
            titles=["Proximal Policy Optimization Algorithms"],
            requested_fields=["formula"],
            required_modalities=["page_text"],
            answer_shape="bullets",
            precision_requirement="exact",
            clean_query="PPO 公式",
        ),
    )

    assert session.active_research_relation == "formula_lookup"
    assert session.active_targets == ["PPO"]
    assert session.active_precision_requirement == "exact"


def test_set_active_research_updates_both_models() -> None:
    session = SessionContext(session_id="demo")

    session.set_active_research(
        relation="paper_summary_results",
        targets=["AlignX"],
        titles=["AlignX paper"],
        requested_fields=["results"],
        required_modalities=["table"],
        answer_shape="narrative",
        precision_requirement="high",
        clean_query="AlignX 结果",
    )

    assert session.active_research.relation == "paper_summary_results"
    assert session.active_research_relation == "paper_summary_results"
    assert session.active_titles == ["AlignX paper"]


def test_agent_settings_are_loaded_from_runtime_settings(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="",
        data_dir=tmp_path / "data",
        paper_store_path=tmp_path / "data" / "papers.jsonl",
        block_store_path=tmp_path / "data" / "blocks.jsonl",
        ingestion_state_path=tmp_path / "data" / "state.json",
        session_store_path=tmp_path / "data" / "sessions.sqlite3",
        eval_cases_path=tmp_path / "evals" / "cases.yaml",
        agent_max_steps=12,
        agent_confidence_floor=0.72,
        agent_answer_logprobs_enabled=True,
        agent_answer_logprobs_min_tokens=8,
        agent_legacy_intent_fallback_enabled=False,
        agent_max_clarification_attempts=3,
    )

    agent_settings = AgentSettings.from_settings(settings)

    assert agent_settings.max_agent_steps == 12
    assert agent_settings.confidence_floor == 0.72
    assert agent_settings.answer_logprobs_enabled is True
    assert agent_settings.answer_logprobs_min_tokens == 8
    assert agent_settings.legacy_intent_fallback_enabled is False
    assert agent_settings.max_clarification_attempts == 3
