from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if str(PROJECT_ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.agent.trace_diff import diff_agent_trace_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff two V4 agent trace jsonl files")
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--max-differences", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diff = diff_agent_trace_files(
        args.expected,
        args.actual,
        max_differences=max(1, int(args.max_differences)),
    )
    payload = {
        "ok": diff.ok,
        "differences": diff.differences,
        "expected_events": len(diff.expected_signature),
        "actual_events": len(diff.actual_signature),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if diff.ok else 1)


if __name__ == "__main__":
    main()
