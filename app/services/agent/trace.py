from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_agent_trace(
    *,
    data_dir: Path,
    session_id: str,
    events: list[dict[str, Any]],
    final_payload: dict[str, Any],
    execution_steps: list[dict[str, Any]],
) -> Path:
    trace_dir = data_dir / "traces" / _safe_path_segment(session_id)
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(events, start=1):
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "event": item.get("event", "message"),
                        "data": item.get("data", {}),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        handle.write(
            json.dumps(
                {
                    "index": len(events) + 1,
                    "event": "final",
                    "data": {
                        **_compact_final_payload(final_payload),
                        "execution_steps": execution_steps,
                    },
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return path


def _safe_path_segment(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value or "").strip())
    return cleaned or "default"


def _compact_final_payload(payload: dict[str, Any]) -> dict[str, Any]:
    answer = str(payload.get("answer", "") or "")
    compact = {key: value for key, value in payload.items() if key != "answer"}
    compact["answer_chars"] = len(answer)
    compact["answer_preview"] = answer[:1200]
    return compact
