from __future__ import annotations

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
