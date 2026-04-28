from __future__ import annotations

from app.domain.models import ActiveResearch, ResearchPlan, SessionContext, VerificationReport
from app.core.agent_settings import AgentSettings
from app.core.config import Settings
from app.domain.models import QueryContract
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
    all_agent_tool_names,
    conversation_tool_sequence,
    normalize_plan_actions,
    research_execution_tool_names,
    research_tool_sequence,
)
from app.services.agent_tools import RegisteredAgentTool


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
        self.steps: list[str] = []
        self.step_payloads: list[dict[str, object]] = []
        self.runtime = AgentRuntime(agent=self)

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

    def _emit_agent_step(self, *, action: str, arguments: dict[str, object] | None = None, **_: object) -> None:
        self.steps.append(action)
        self.step_payloads.append({"action": action, "arguments": arguments or {}})

    def _compose_conversation_response(self, **_: object) -> str:
        return "hello from runtime"

    def _remember_conversation_tool_result(self, **_: object) -> None:
        return None

    def _set_conversation_answer(self, *, state: dict[str, object], answer: str, **_: object) -> None:
        state["answer"] = answer

    def _extract_query_contract(self, *, query: str, **_: object) -> QueryContract:
        return QueryContract(clean_query=query, interaction_mode="conversation", relation="greeting")

    def _plan_agent_actions(self, **_: object) -> dict[str, object]:
        return {"actions": ["compose"], "tool_call_args": []}

    def _build_research_plan(self, contract: QueryContract) -> ResearchPlan:
        return ResearchPlan(solver_sequence=["probe"], evidence_limit=2)

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
        "web_search",
        "fetch_url",
        "query_library_metadata",
        "compose",
        "todo_write",
        "remember",
        "Task",
        "ask_human",
    }
    search_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "search_corpus")
    assert search_schema["required"] == ["query"]
    assert search_schema["properties"]["top_k"]["maximum"] == 50
    bm25_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "bm25_search")
    assert bm25_schema["required"] == ["query"]
    hybrid_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "hybrid_search")
    assert hybrid_schema["properties"]["alpha"]["maximum"] == 1.0
    rerank_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "rerank")
    assert rerank_schema["required"] == ["query"]
    read_page_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "read_pdf_page")
    assert read_page_schema["required"] == ["paper_id", "page_from"]
    grep_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "grep_corpus")
    assert grep_schema["required"] == ["regex"]
    todo_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "todo_write")
    assert todo_schema["required"] == ["items"]
    remember_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "remember")
    assert remember_schema["required"] == ["key", "content"]
    task_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "Task")
    assert task_schema["required"] == ["description", "prompt"]
    fetch_schema = next(item["input_schema"] for item in agent_tool_manifest() if item["name"] == "fetch_url")
    assert fetch_schema["required"] == ["url"]
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
        "search_corpus",
        "bm25_search",
        "vector_search",
        "hybrid_search",
        "rerank",
        "read_pdf_page",
        "grep_corpus",
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
    assert "search_corpus" in probe.steps
    assert "compose" in probe.steps


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
    assert {"action": "search_corpus", "arguments": {"query": "custom PPO query", "top_k": 3}} in probe.step_payloads


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
    assert "todo_write" in probe.steps


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
        agent_max_clarification_attempts=3,
    )

    agent_settings = AgentSettings.from_settings(settings)

    assert agent_settings.max_agent_steps == 12
    assert agent_settings.confidence_floor == 0.72
    assert agent_settings.max_clarification_attempts == 3
