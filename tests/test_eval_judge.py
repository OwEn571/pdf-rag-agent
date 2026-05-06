from __future__ import annotations

import json
from types import SimpleNamespace

from app.services.agent.trace_diff import TraceDiff
from app.services.eval.judge import (
    coerce_eval_judge_result,
    eval_judge_human_prompt,
    judge_eval_case,
)


class _JudgeClients:
    chat = object()

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.last_human_prompt = ""

    def invoke_json(self, *, system_prompt: str, human_prompt: str, fallback: object) -> object:
        _ = system_prompt, fallback
        self.last_human_prompt = human_prompt
        return self.payload


def test_eval_judge_prompt_includes_expectations_and_trace_diff() -> None:
    trace_diff = TraceDiff(
        ok=False,
        differences=["event[1] changed"],
        expected_signature=[{"event": "tool_call", "name": "search_corpus"}],
        actual_signature=[{"event": "tool_call", "name": "web_search"}],
    )

    payload = json.loads(
        eval_judge_human_prompt(
            case_id="case-1",
            query="PBA 准确率多少",
            answer="PBA 59.66",
            expectations={"must_contain_any": ["PBA"]},
            trace_diff=trace_diff,
        )
    )

    assert payload["case_id"] == "case-1"
    assert payload["expectations"]["must_contain_any"] == ["PBA"]
    assert payload["trace_diff"]["differences"] == ["event[1] changed"]


def test_judge_eval_case_invokes_llm_and_coerces_payload() -> None:
    clients = _JudgeClients(
        {
            "status": "pass",
            "score": "0.92",
            "rationale": "meets expectations",
            "missing": [],
            "unsupported": [" "],
        }
    )

    result = judge_eval_case(
        clients=clients,
        case_id="case-1",
        query="query",
        answer="answer",
        expectations={"min_citations": 1},
    )

    assert result.status == "pass"
    assert result.score == 0.92
    assert result.unsupported == []
    assert json.loads(clients.last_human_prompt)["expectations"]["min_citations"] == 1


def test_judge_eval_case_reports_unavailable_without_chat_model() -> None:
    result = judge_eval_case(
        clients=SimpleNamespace(chat=None),
        case_id="case-1",
        query="query",
        answer="answer",
        expectations={},
    )

    assert result.status == "unavailable"


def test_coerce_eval_judge_result_bounds_unknown_payload() -> None:
    result = coerce_eval_judge_result(
        {
            "status": "maybe",
            "score": 5,
            "rationale": "  needs manual review  ",
            "missing": ["formula"],
            "unsupported": ["invented citation"],
        }
    )

    assert result.status == "needs_review"
    assert result.score == 1.0
    assert result.rationale == "needs manual review"
