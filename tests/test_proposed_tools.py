from __future__ import annotations

import hashlib
import json

from app.services.proposed_tools import propose_tool


def test_propose_tool_writes_pending_review_record_without_executing_code(tmp_path) -> None:
    proposal = propose_tool(
        data_dir=tmp_path,
        name="extract_metric",
        description="Extract a named metric from evidence.",
        input_schema={
            "type": "object",
            "properties": {"metric": {"type": "string"}},
            "required": ["metric"],
        },
        python_code="async def run(args, ctx, session):\n    raise RuntimeError('must not execute')",
        rationale="The corpus often needs reusable metric extraction.",
    )

    assert proposal.status == "pending_review"
    assert proposal.path.parent == tmp_path / "tools_proposed"
    payload = json.loads(proposal.path.read_text(encoding="utf-8"))
    assert payload["name"] == "extract_metric"
    assert payload["proposal_id"] == proposal.path.stem
    assert payload["status"] == "pending_review"
    assert payload["admin_approval_required"] is True
    assert payload["sandbox_required"] is True
    assert payload["code_sha256"] == hashlib.sha256(payload["python_code"].encode("utf-8")).hexdigest()
    assert payload["safety"]["static_check"] == "pass"
    assert "must not execute" in payload["python_code"]
    assert payload["input_schema"]["additionalProperties"] is False


def test_propose_tool_rejects_unsafe_names(tmp_path) -> None:
    try:
        propose_tool(
            data_dir=tmp_path,
            name="../bad",
            description="bad",
            input_schema={"type": "object"},
            python_code="async def run(args, ctx, session):\n    return None",
            rationale="bad",
        )
    except ValueError as exc:
        assert "tool name" in str(exc)
    else:
        raise AssertionError("expected invalid tool name to be rejected")


def test_propose_tool_rejects_disallowed_imports(tmp_path) -> None:
    try:
        propose_tool(
            data_dir=tmp_path,
            name="shell_tool",
            description="bad",
            input_schema={"type": "object"},
            python_code="import os\n\nasync def run(args, ctx, session):\n    return os.listdir('.')",
            rationale="bad",
        )
    except ValueError as exc:
        assert "import not allowed: os" in str(exc)
    else:
        raise AssertionError("expected unsafe import to be rejected")


def test_propose_tool_requires_async_run_entrypoint(tmp_path) -> None:
    try:
        propose_tool(
            data_dir=tmp_path,
            name="sync_tool",
            description="bad",
            input_schema={"type": "object"},
            python_code="def run(args, ctx, session):\n    return None",
            rationale="bad",
        )
    except ValueError as exc:
        assert "async def run" in str(exc)
    else:
        raise AssertionError("expected missing async run to be rejected")
