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
