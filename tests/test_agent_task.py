from __future__ import annotations

from app.services.agent_task import task_plan_with_allow_list


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
