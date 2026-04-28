from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import yaml

if str(PROJECT_ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.deps import get_agent  # noqa: E402

DEFAULT_CASES_PATH = PROJECT_ROOT / "evals" / "cases_test_md.yaml"
DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_ROUTE = "/api/v1/v4/chat"

INSUFFICIENT_MARKERS = [
    "当前语料下证据仍然不足",
    "当前语料下可确认的是",
    "证据不足",
    "无法回答",
    "无法确定",
    "未找到",
]


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("\n", " ").split())


def _contains_any(text: str, needles: list[str]) -> bool:
    haystack = _normalize(text)
    return any(_normalize(item) in haystack for item in needles if item)


def _count_group_matches(text: str, groups: list[list[str]]) -> int:
    haystack = _normalize(text)
    matched = 0
    for group in groups:
        if any(_normalize(item) in haystack for item in group if item):
            matched += 1
    return matched


def _evaluate_turn(response: dict[str, Any], expect: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    answer = str(response.get("answer", ""))
    interaction_mode = str(response.get("interaction_mode", ""))
    citations = list(response.get("citations", []))
    expected_mode = str(expect.get("interaction_mode", "")).strip()
    if expected_mode and interaction_mode != expected_mode:
        issues.append(f"interaction_mode={interaction_mode}, expected={expected_mode}")
    for item in expect.get("must_contain_all", []):
        if not _contains_any(answer, [str(item)]):
            issues.append(f"missing required phrase: {item}")
    must_contain_any = [str(item) for item in expect.get("must_contain_any", [])]
    if must_contain_any and not _contains_any(answer, must_contain_any):
        issues.append(f"missing any-of phrases: {must_contain_any}")
    for item in expect.get("must_not_contain", []):
        if _contains_any(answer, [str(item)]):
            issues.append(f"contains forbidden phrase: {item}")
    match_groups = expect.get("match_groups", [])
    if match_groups:
        group_matches = _count_group_matches(answer, [[str(token) for token in group] for group in match_groups])
        if group_matches < int(expect.get("min_group_matches", 1)):
            issues.append(f"group_matches={group_matches}, expected>={int(expect.get('min_group_matches', 1))}")
    min_citations = expect.get("min_citations")
    if min_citations is not None and len(citations) < int(min_citations):
        issues.append(f"citations={len(citations)}, expected>={int(min_citations)}")
    max_citations = expect.get("max_citations")
    if max_citations is not None and len(citations) > int(max_citations):
        issues.append(f"citations={len(citations)}, expected<={int(max_citations)}")
    if not bool(expect.get("allow_insufficient", True)):
        if _contains_any(answer, INSUFFICIENT_MARKERS):
            issues.append("answer marked as insufficient")
    return not issues, issues


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:300]}") from exc
    except URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc
    return json.loads(body)


def _run_case(driver: str, api_base: str, route: str, case: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = api_base.rstrip("/") + route
    session_id = uuid4().hex[:12]
    turn_results: list[dict[str, Any]] = []
    case_passed = True
    for turn in case.get("turns", []):
        payload = {
            "query": str(turn.get("query", "")),
            "session_id": session_id,
            "mode": "auto",
            "use_web_search": False,
            "max_web_results": 3,
        }
        started = time.perf_counter()
        error_message = ""
        response: dict[str, Any] = {}
        try:
            if driver == "http":
                response = _post_json(url, payload, timeout)
            else:
                response = asyncio.run(
                    asyncio.wait_for(
                        get_agent().achat(
                            query=str(payload["query"]),
                            session_id=str(payload["session_id"]),
                            mode="auto",
                            use_web_search=False,
                            max_web_results=3,
                        ),
                        timeout=timeout,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if response.get("session_id"):
            session_id = str(response["session_id"])
        if error_message:
            passed = False
            issues = [error_message]
        else:
            passed, issues = _evaluate_turn(response, dict(turn.get("expect", {})))
        case_passed = case_passed and passed
        turn_results.append(
            {
                "query": payload["query"],
                "passed": passed,
                "issues": issues,
                "latency_ms": elapsed_ms,
                "answer_preview": str(response.get("answer", ""))[:300],
            }
        )
    return {"id": case.get("id", ""), "passed": case_passed, "turns": turn_results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V4 eval cases")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--driver", choices=["local", "http"], default="local")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--route", default=DEFAULT_ROUTE)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = yaml.safe_load(args.cases.read_text(encoding="utf-8"))
    cases = list(payload.get("cases", []))
    results = [_run_case(args.driver, args.api_base, args.route, case, args.timeout) for case in cases]
    latencies = [turn["latency_ms"] for case in results for turn in case["turns"]]
    summary = {
        "cases": len(results),
        "passed_cases": sum(1 for case in results if case["passed"]),
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_ms": sorted(latencies)[int(max(0, len(latencies) * 0.95) - 1)] if latencies else 0.0,
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
