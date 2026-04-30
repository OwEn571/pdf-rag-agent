from __future__ import annotations

from app.domain.models import QueryContract, SessionContext
from app.services.agent_task import run_task_subagent, task_plan_with_allow_list


def test_task_plan_with_allow_list_filters_actions_and_tool_args() -> None:
    plan = {
        "actions": ["read_memory", "search_corpus", "compose"],
        "tool_call_args": [
            {"name": "read_memory", "args": {"focus": "context"}},
            {"name": "search_corpus", "args": {"query": "DPO"}},
            {"name": "compose", "args": {}},
        ],
    }

    filtered = task_plan_with_allow_list(plan, ["read_memory", "compose"])

    assert filtered["actions"] == ["read_memory", "compose"]
    assert [item["name"] for item in filtered["tool_call_args"]] == ["read_memory", "compose"]
    assert plan["actions"] == ["read_memory", "search_corpus", "compose"]


def test_task_plan_with_allow_list_keeps_plan_without_allow_list() -> None:
    plan = {"actions": ["compose"], "tool_call_args": []}

    assert task_plan_with_allow_list(plan, []) is plan


def test_run_task_subagent_can_use_precomputed_contract() -> None:
    class _Runtime:
        def __init__(self) -> None:
            self.contract = None

        def execute_conversation_tools(self, *, contract, **_: object) -> dict[str, object]:
            self.contract = contract
            return {"answer": "ok", "citations": [], "verification_report": {"status": "pass"}}

    class _Agent:
        def __init__(self) -> None:
            self.runtime = _Runtime()
            self.planner = self

        def _extract_query_contract(self, **_: object) -> QueryContract:
            raise AssertionError("precomputed compound contracts should not be re-extracted")

        @staticmethod
        def plan_actions(**_: object) -> dict[str, object]:
            return {"actions": ["compose"], "tool_call_args": []}

        @staticmethod
        def _emit_agent_tool_call(*, emit, tool: str, arguments: dict[str, object]) -> None:
            emit("tool_call", {"tool": tool, "arguments": arguments})

    contract = QueryContract(clean_query="库状态", interaction_mode="conversation", relation="library_status")
    events: list[tuple[str, dict[str, object]]] = []

    result = run_task_subagent(
        agent=_Agent(),
        prompt="库状态",
        description="查看库状态",
        tools_allowed=[],
        max_steps=8,
        session=SessionContext(session_id="demo"),
        max_web_results=0,
        emit=lambda event, payload: events.append((event, payload)),
        execution_steps=[],
        contract=contract,
    )

    assert result["answer"] == "ok"
    assert result["contract_obj"] is contract
    assert result["contract"]["relation"] == "library_status"
    assert events[0][0] == "agent_plan"
    assert events[1][1]["tool"] == "Task"
