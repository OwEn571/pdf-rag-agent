from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from app.services.agent.trace_diff import TraceDiff

EvalJudgeStatus = Literal["pass", "fail", "needs_review", "unavailable"]


@dataclass(frozen=True, slots=True)
class EvalJudgeResult:
    status: EvalJudgeStatus
    score: float
    rationale: str
    missing: list[str]
    unsupported: list[str]


def eval_judge_system_prompt() -> str:
    return (
        "你是 PDF-RAG Agent 的回归评测裁判。"
        "只基于给定 query、answer、expectations 和 trace_diff 判断，不要使用外部知识。"
        "重点检查：是否回答了问题、是否满足 must_contain/match_groups、是否有不支持的论文事实、"
        "以及 trace_diff 是否显示工具参数、证据、claim、citation 或 runtime_summary 发生质量漂移。"
        "只输出 JSON：status, score, rationale, missing, unsupported。"
        "status 必须是 pass、fail 或 needs_review；score 是 0 到 1。"
    )


def eval_judge_human_prompt(
    *,
    case_id: str,
    query: str,
    answer: str,
    expectations: dict[str, Any],
    trace_diff: TraceDiff | None = None,
) -> str:
    return json.dumps(
        {
            "case_id": case_id,
            "query": query,
            "answer": answer[:12000],
            "expectations": expectations,
            "trace_diff": _trace_diff_payload(trace_diff),
        },
        ensure_ascii=False,
    )


def judge_eval_case(
    *,
    clients: Any,
    case_id: str,
    query: str,
    answer: str,
    expectations: dict[str, Any],
    trace_diff: TraceDiff | None = None,
) -> EvalJudgeResult:
    if getattr(clients, "chat", None) is None:
        return EvalJudgeResult(
            status="unavailable",
            score=0.0,
            rationale="LLM judge unavailable because no chat model is configured.",
            missing=[],
            unsupported=[],
        )
    payload = clients.invoke_json(
        system_prompt=eval_judge_system_prompt(),
        human_prompt=eval_judge_human_prompt(
            case_id=case_id,
            query=query,
            answer=answer,
            expectations=expectations,
            trace_diff=trace_diff,
        ),
        fallback={},
    )
    return coerce_eval_judge_result(payload)


def coerce_eval_judge_result(payload: Any) -> EvalJudgeResult:
    if not isinstance(payload, dict):
        return EvalJudgeResult(status="needs_review", score=0.0, rationale="judge_payload_not_object", missing=[], unsupported=[])
    status = str(payload.get("status", "") or "").strip().lower()
    if status not in {"pass", "fail", "needs_review"}:
        status = "needs_review"
    score = _bounded_score(payload.get("score", 0.0))
    return EvalJudgeResult(
        status=status,  # type: ignore[arg-type]
        score=score,
        rationale=" ".join(str(payload.get("rationale", "") or "").split())[:1000],
        missing=[str(item).strip() for item in list(payload.get("missing", []) or []) if str(item).strip()][:20],
        unsupported=[str(item).strip() for item in list(payload.get("unsupported", []) or []) if str(item).strip()][:20],
    )


def _trace_diff_payload(trace_diff: TraceDiff | None) -> dict[str, Any]:
    if trace_diff is None:
        return {}
    return {
        "ok": trace_diff.ok,
        "differences": trace_diff.differences[:20],
        "expected_signature": trace_diff.expected_signature[:40],
        "actual_signature": trace_diff.actual_signature[:40],
    }


def _bounded_score(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))
