from __future__ import annotations

import json

from app.services.agent.trace_diff import diff_agent_traces
from app.services.agent.trace import write_agent_trace


def test_write_agent_trace_records_events_and_compact_final(tmp_path) -> None:
    path = write_agent_trace(
        data_dir=tmp_path,
        session_id="demo/session",
        events=[
            {"event": "tool_call", "data": {"type": "tool_use", "name": "search_corpus"}},
            {"event": "observation", "data": {"type": "tool_result", "name": "search_corpus"}},
        ],
        final_payload={"answer": "hello" * 400, "session_id": "demo/session"},
        execution_steps=[{"node": "agent_loop", "summary": "search_corpus"}],
    )

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert path.parent.name == "demo_session"
    assert lines[0]["event"] == "tool_call"
    assert lines[1]["data"]["type"] == "tool_result"
    assert lines[2]["event"] == "final"
    assert lines[2]["data"]["answer_chars"] == 2000
    assert len(lines[2]["data"]["answer_preview"]) == 1200
    assert lines[2]["data"]["execution_steps"][0]["node"] == "agent_loop"


def test_diff_agent_traces_ignores_volatile_fields() -> None:
    expected = [
        {"index": 1, "event": "tool_call", "data": {"id": "a", "type": "tool_use", "name": "search_corpus"}},
        {
            "index": 2,
            "event": "observation",
            "data": {"id": "a", "type": "tool_result", "name": "search_corpus", "ok": True, "took_ms": 10},
        },
        {
            "index": 3,
            "event": "final",
            "data": {
                "answer_chars": 380,
                "answer_preview": "old",
                "execution_steps": [{"node": "agent_loop", "summary": "old"}],
            },
        },
    ]
    actual = [
        {"index": 1, "event": "tool_call", "data": {"id": "b", "type": "tool_use", "name": "search_corpus"}},
        {
            "index": 2,
            "event": "observation",
            "data": {"id": "b", "type": "tool_result", "name": "search_corpus", "ok": True, "took_ms": 200},
        },
        {
            "index": 3,
            "event": "final",
            "data": {
                "answer_chars": 399,
                "answer_preview": "new",
                "execution_steps": [{"node": "agent_loop", "summary": "new"}],
            },
        },
    ]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is True
    assert diff.differences == []


def test_diff_agent_traces_reports_stable_signal_changes() -> None:
    expected = [
        {"event": "tool_call", "data": {"type": "tool_use", "name": "search_corpus"}},
        {"event": "observation", "data": {"type": "tool_result", "name": "search_corpus", "ok": True}},
    ]
    actual = [
        {"event": "tool_call", "data": {"type": "tool_use", "name": "web_search"}},
        {"event": "observation", "data": {"type": "tool_result", "name": "web_search", "ok": False}},
    ]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is False
    assert "search_corpus" in diff.differences[0]
    assert "web_search" in diff.differences[0]


def test_diff_agent_traces_reports_tool_input_changes() -> None:
    expected = [
        {
            "event": "tool_call",
            "data": {
                "type": "tool_use",
                "name": "search_corpus",
                "input": {"query": "DeepSeek figures", "top_k": 8},
            },
        }
    ]
    actual = [
        {
            "event": "tool_call",
            "data": {
                "type": "tool_use",
                "name": "search_corpus",
                "input": {"query": "CommunityBench figures", "top_k": 8},
            },
        }
    ]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is False
    assert "DeepSeek figures" in diff.differences[0]
    assert "CommunityBench figures" in diff.differences[0]


def test_diff_agent_traces_reports_agent_step_argument_changes() -> None:
    expected = [
        {
            "event": "agent_step",
            "data": {"type": "agent_step", "action": "search_corpus", "arguments": {"query": "AlignX table"}},
        }
    ]
    actual = [
        {
            "event": "agent_step",
            "data": {"type": "agent_step", "action": "search_corpus", "arguments": {"query": "DPO formula"}},
        }
    ]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is False
    assert "AlignX table" in diff.differences[0]
    assert "DPO formula" in diff.differences[0]


def test_diff_agent_traces_reports_ask_human_question_changes() -> None:
    expected = [
        {
            "event": "ask_human",
            "data": {"type": "ask_human", "question": "选哪篇？", "options": [{"label": "A"}]},
        }
    ]
    actual = [
        {
            "event": "ask_human",
            "data": {"type": "ask_human", "question": "确认哪个公式？", "options": []},
        }
    ]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is False
    assert "选哪篇" in diff.differences[0]
    assert "确认哪个公式" in diff.differences[0]


def test_diff_agent_traces_reports_todo_update_changes() -> None:
    expected = [
        {
            "event": "todo_update",
            "data": {"type": "todo_update", "items": [{"id": "1", "text": "查找表格证据", "status": "doing"}]},
        }
    ]
    actual = [
        {
            "event": "todo_update",
            "data": {"type": "todo_update", "items": [{"id": "1", "text": "解释指标定义", "status": "pending"}]},
        }
    ]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is False
    assert "查找表格证据" in diff.differences[0]
    assert "解释指标定义" in diff.differences[0]


def test_diff_agent_traces_reports_confidence_bucket_changes() -> None:
    expected = [{"event": "confidence", "data": {"type": "confidence", "basis": "logprobs", "score": 0.86}}]
    actual = [{"event": "confidence", "data": {"type": "confidence", "basis": "logprobs", "score": 0.52}}]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is False
    assert ">=0.8" in diff.differences[0]
    assert "<0.6" in diff.differences[0]


def test_diff_agent_traces_reports_contract_router_changes() -> None:
    expected = [
        {
            "event": "contract",
            "data": {
                "type": "contract",
                "interaction_mode": "research",
                "relation": "metric_value_lookup",
                "notes": [
                    "llm_tool_router",
                    "intent_kind=research",
                    "router_action=need_corpus_search",
                    "router_tag=need_corpus_search",
                    "router_tag=target:PBA",
                ],
            },
        }
    ]
    actual = [
        {
            "event": "contract",
            "data": {
                "type": "contract",
                "interaction_mode": "conversation",
                "relation": "general_question",
                "notes": ["llm_tool_router", "intent_kind=smalltalk", "router_action=answer_directly"],
            },
        }
    ]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is False
    assert "metric_value_lookup" in diff.differences[0]
    assert "answer_directly" in diff.differences[0]


def test_diff_agent_traces_reports_solver_sequence_changes() -> None:
    expected = [
        {
            "event": "plan",
            "data": {
                "type": "plan",
                "solver_sequence": ["text_solver", "table_solver"],
                "required_claims": ["summary", "metric_value"],
            },
        }
    ]
    actual = [
        {
            "event": "plan",
            "data": {
                "type": "plan",
                "solver_sequence": ["generic_compose"],
                "required_claims": ["summary"],
            },
        }
    ]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is False
    assert "table_solver" in diff.differences[0]
    assert "generic_compose" in diff.differences[0]


def test_diff_agent_traces_reports_claim_signature_changes() -> None:
    expected = [
        {
            "event": "claims",
            "data": {
                "type": "claims",
                "count": 1,
                "items": [
                    {
                        "claim_type": "metric_value",
                        "entity": "PBA",
                        "value": "59.66",
                        "paper_ids": ["paper-a"],
                        "evidence_ids": ["table-1"],
                        "structured_data": {"source": "table_solver"},
                    }
                ],
            },
        }
    ]
    actual = [
        {
            "event": "claims",
            "data": {
                "type": "claims",
                "count": 1,
                "items": [
                    {
                        "claim_type": "summary",
                        "entity": "PBA",
                        "value": "accuracy improves",
                        "paper_ids": ["paper-b"],
                        "evidence_ids": ["page-2"],
                        "structured_data": {"source": "generic_compose"},
                    }
                ],
            },
        }
    ]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is False
    assert "metric_value" in diff.differences[0]
    assert "generic_compose" in diff.differences[0]


def test_diff_agent_traces_reports_solver_shadow_signature_changes() -> None:
    expected = [
        {
            "event": "solver_shadow",
            "data": {
                "mode": "generic_claim_solver_shadow",
                "selected": "deterministic",
                "schema": {
                    "count": 1,
                    "types": ["summary"],
                    "paper_ids": ["paper-a"],
                    "evidence_ids": ["page-2"],
                    "sources": {"schema_claim_solver": 1},
                },
                "deterministic": {
                    "count": 2,
                    "types": ["metric_value", "definition"],
                    "paper_ids": ["paper-a"],
                    "evidence_ids": ["table-1"],
                    "sources": {"table_solver": 1, "text_solver": 1},
                },
            },
        }
    ]
    actual = [
        {
            "event": "solver_shadow",
            "data": {
                "mode": "generic_claim_solver_shadow",
                "selected": "schema",
                "schema": {
                    "count": 2,
                    "types": ["metric_value", "summary"],
                    "paper_ids": ["paper-b"],
                    "evidence_ids": ["table-2"],
                    "sources": {"schema_claim_solver": 2},
                },
                "deterministic": {
                    "count": 1,
                    "types": ["summary"],
                    "paper_ids": ["paper-a"],
                    "evidence_ids": ["page-2"],
                    "sources": {"deterministic_solver": 1},
                },
            },
        }
    ]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is False
    assert "deterministic" in diff.differences[0]
    assert "schema" in diff.differences[0]
    assert "table-2" in diff.differences[0]


def test_diff_agent_traces_reports_evidence_signature_changes() -> None:
    expected = [
        {
            "event": "evidence",
            "data": {
                "type": "evidence",
                "count": 1,
                "items": [
                    {
                        "doc_id": "alignx-table-24",
                        "paper_id": "paper-alignx",
                        "title": "AlignX",
                        "file_path": "/papers/alignx.pdf",
                        "page": 24,
                        "block_type": "table",
                        "snippet": "ICA reaches 57.80 and PBA reaches 59.66 on PPAIR.",
                        "metadata": {"source": "local_index"},
                    }
                ],
            },
        }
    ]
    actual = [
        {
            "event": "evidence",
            "data": {
                "type": "evidence",
                "count": 1,
                "items": [
                    {
                        "doc_id": "alignx-page-6",
                        "paper_id": "paper-alignx",
                        "title": "AlignX",
                        "file_path": "/papers/alignx.pdf",
                        "page": 6,
                        "block_type": "page_text",
                        "snippet": "ICA directly learns from user-profile-enhanced prompts.",
                        "metadata": {"source": "local_index"},
                    }
                ],
            },
        }
    ]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is False
    assert "alignx-table-24" in diff.differences[0]
    assert "alignx-page-6" in diff.differences[0]


def test_diff_agent_traces_reports_final_citation_signature_changes() -> None:
    expected = [
        {
            "event": "final",
            "data": {
                "answer_chars": 720,
                "citations": [
                    {
                        "doc_id": "alignx-table-24",
                        "paper_id": "paper-alignx",
                        "title": "AlignX",
                        "file_path": "/papers/alignx.pdf",
                        "page": 24,
                        "block_type": "table",
                        "snippet": "ICA reaches 57.80 and PBA reaches 59.66.",
                    }
                ],
                "execution_steps": [{"node": "agent_loop"}],
            },
        }
    ]
    actual = [
        {
            "event": "final",
            "data": {
                "answer_chars": 730,
                "citations": [
                    {
                        "doc_id": "alignx-page-6",
                        "paper_id": "paper-alignx",
                        "title": "AlignX",
                        "file_path": "/papers/alignx.pdf",
                        "page": 6,
                        "block_type": "page_text",
                        "snippet": "ICA directly learns from user-profile-enhanced prompts.",
                    }
                ],
                "execution_steps": [{"node": "agent_loop"}],
            },
        }
    ]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is False
    assert "alignx-table-24" in diff.differences[0]
    assert "alignx-page-6" in diff.differences[0]


def test_diff_agent_traces_reports_final_runtime_summary_changes() -> None:
    expected = [
        {
            "event": "final",
            "data": {
                "answer_chars": 720,
                "execution_steps": [{"node": "agent_loop"}],
                "runtime_summary": {
                    "grounding": {
                        "verification_status": "pass",
                        "claim_count": 2,
                        "citation_count": 2,
                        "claim_sources": {"deterministic_solver": 2},
                    },
                    "answer_generation": {"confidence": {"basis": "logprobs", "score": 0.86}},
                },
            },
        }
    ]
    actual = [
        {
            "event": "final",
            "data": {
                "answer_chars": 730,
                "execution_steps": [{"node": "agent_loop"}],
                "runtime_summary": {
                    "grounding": {
                        "verification_status": "retry",
                        "claim_count": 1,
                        "citation_count": 0,
                        "claim_sources": {"schema_claim_solver": 1},
                    },
                    "answer_generation": {"confidence": {"basis": "self_consistency", "score": 0.52}},
                },
            },
        }
    ]

    diff = diff_agent_traces(expected, actual)

    assert diff.ok is False
    assert "deterministic_solver" in diff.differences[0]
    assert "schema_claim_solver" in diff.differences[0]
    assert "self_consistency" in diff.differences[0]
