from __future__ import annotations

from app.services import agent_metrics
from app.services.agent_events import normalize_agent_event


def test_normalize_tool_call_event_keeps_legacy_fields_and_adds_protocol_fields() -> None:
    payload = normalize_agent_event("tool_call", {"tool": "search_corpus", "arguments": {"query": "PPO"}})

    assert payload["tool"] == "search_corpus"
    assert payload["arguments"] == {"query": "PPO"}
    assert payload["type"] == "tool_use"
    assert payload["name"] == "search_corpus"
    assert payload["input"] == {"query": "PPO"}


def test_normalize_observation_event_adds_tool_result_shape() -> None:
    payload = normalize_agent_event("observation", {"tool": "search_corpus", "summary": "ok", "payload": {"count": 2}})

    assert payload["type"] == "tool_result"
    assert payload["name"] == "search_corpus"
    assert payload["ok"] is True
    assert payload["output"] == {"count": 2}


def test_normalize_plan_verification_and_confidence_events_add_protocol_fields() -> None:
    plan = normalize_agent_event("plan", {"solver_sequence": ["search", "compose"]})
    verification = normalize_agent_event("verification", {"status": "pass", "total_claims": 2})
    confidence = normalize_agent_event("confidence", {"score": "1.2", "basis": "verifier"})
    unknown_confidence = normalize_agent_event("confidence", {"score": "bad"})

    assert plan["type"] == "plan"
    assert plan["payload"] == {"solver_sequence": ["search", "compose"]}
    assert plan["items"] == [
        {"id": "plan-1", "text": "search", "status": "pending"},
        {"id": "plan-2", "text": "compose", "status": "pending"},
    ]
    assert verification["type"] == "verification"
    assert verification["payload"] == {"status": "pass", "total_claims": 2}
    assert verification["status"] == "pass"
    assert confidence["type"] == "confidence"
    assert confidence["value"] == 1.0
    assert confidence["basis"] == "verifier"
    assert unknown_confidence["value"] is None
    assert unknown_confidence["basis"] == "unknown"


def test_normalize_thinking_and_ask_human_events_add_protocol_fields() -> None:
    thinking = normalize_agent_event("thinking_delta", {"delta": "checking evidence"})
    ask_human = normalize_agent_event("ask_human", {"question": "选哪篇？", "options": "bad"})

    assert thinking["type"] == "thinking_delta"
    assert thinking["text"] == "checking evidence"
    assert ask_human["type"] == "ask_human"
    assert ask_human["question"] == "选哪篇？"
    assert ask_human["options"] == []
    assert ask_human["reason"] == ""


def test_normalize_todo_update_event_adds_protocol_items() -> None:
    payload = normalize_agent_event(
        "todo_update",
        {
            "items": [
                {"id": "  step-1 ", "text": "  查找表格证据  ", "status": "doing"},
                {"text": "解释指标定义", "status": "bad"},
                {"id": "empty", "text": ""},
            ]
        },
    )

    assert payload["type"] == "todo_update"
    assert payload["items"] == [
        {"id": "step-1", "text": "查找表格证据", "status": "doing"},
        {"id": "todo-2", "text": "解释指标定义", "status": "pending"},
    ]


def test_observation_events_record_tool_metrics(monkeypatch) -> None:
    calls = _CounterProbe()
    latency = _HistogramProbe()
    monkeypatch.setattr(agent_metrics, "TOOL_CALLS_TOTAL", calls)
    monkeypatch.setattr(agent_metrics, "TOOL_LATENCY_SECONDS", latency)

    normalize_agent_event(
        "observation",
        {"tool": "rerank", "summary": "done", "payload": {}, "ok": False, "took_ms": 250},
    )
    normalize_agent_event(
        "observation",
        {"tool": "compose", "summary": "tool_loop_ready", "payload": {}, "took_ms": 1},
    )

    assert calls.labels_seen == [{"name": "rerank", "ok": "false"}]
    assert calls.inc_count == 1
    assert latency.labels_seen == [{"name": "rerank"}]
    assert latency.observed == [0.25]


def test_tool_execution_context_avoids_double_counting_observations(monkeypatch) -> None:
    calls = _CounterProbe()
    latency = _HistogramProbe()
    monkeypatch.setattr(agent_metrics, "TOOL_CALLS_TOTAL", calls)
    monkeypatch.setattr(agent_metrics, "TOOL_LATENCY_SECONDS", latency)

    token = agent_metrics.begin_tool_execution()
    try:
        normalize_agent_event("observation", {"tool": "rerank", "summary": "done", "payload": {}})
    finally:
        agent_metrics.end_tool_execution(token)
    agent_metrics.record_tool_execution(name="rerank", ok=True, elapsed_seconds=0.5)

    assert calls.labels_seen == [{"name": "rerank", "ok": "true"}]
    assert calls.inc_count == 1
    assert latency.labels_seen == [{"name": "rerank"}]
    assert latency.observed == [0.5]


class _CounterProbe:
    def __init__(self) -> None:
        self.labels_seen: list[dict[str, str]] = []
        self.inc_count = 0

    def labels(self, **labels: str) -> "_CounterProbe":
        self.labels_seen.append(labels)
        return self

    def inc(self) -> None:
        self.inc_count += 1


class _HistogramProbe:
    def __init__(self) -> None:
        self.labels_seen: list[dict[str, str]] = []
        self.observed: list[float] = []

    def labels(self, **labels: str) -> "_HistogramProbe":
        self.labels_seen.append(labels)
        return self

    def observe(self, value: float) -> None:
        self.observed.append(value)
