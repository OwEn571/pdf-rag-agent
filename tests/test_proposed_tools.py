from __future__ import annotations

import hashlib
import json

from app.services.tools.proposals import (
    load_runtime_tool_manifests,
    propose_tool,
    run_tool_proposal_sandbox,
    runtime_tool_manifest,
    transition_tool_proposal_status,
)


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
    assert payload["scope"] == {"visibility": "deployment", "deployment_id": "local", "session_id": ""}
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


def test_propose_tool_rejects_reflection_builtins(tmp_path) -> None:
    try:
        propose_tool(
            data_dir=tmp_path,
            name="reflect_tool",
            description="bad",
            input_schema={"type": "object"},
            python_code="async def run(args, ctx, session):\n    return getattr(args, '__class__')",
            rationale="bad",
        )
    except ValueError as exc:
        assert "blocked call: getattr" in str(exc)
    else:
        raise AssertionError("expected reflection builtin to be rejected")


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


def test_tool_proposal_status_transition_requires_hash_and_sandbox_report(tmp_path) -> None:
    proposal = propose_tool(
        data_dir=tmp_path,
        name="extract_metric",
        description="Extract a named metric from evidence.",
        input_schema={"type": "object"},
        python_code="async def run(args, ctx, session):\n    return {'ok': True}",
        rationale="Reusable metric extraction.",
    )
    payload = json.loads(proposal.path.read_text(encoding="utf-8"))
    code_hash = payload["code_sha256"]

    sandbox_payload = transition_tool_proposal_status(
        proposal_path=proposal.path,
        next_status="approved_for_sandbox_test",
        code_sha256=code_hash,
        reviewer="local-admin",
        note="static review passed",
    )

    assert sandbox_payload["status"] == "approved_for_sandbox_test"
    assert sandbox_payload["review_log"][-1]["from"] == "pending_review"
    assert sandbox_payload["review_log"][-1]["to"] == "approved_for_sandbox_test"

    try:
        transition_tool_proposal_status(
            proposal_path=proposal.path,
            next_status="approved_for_runtime",
            code_sha256=code_hash,
            reviewer="local-admin",
        )
    except ValueError as exc:
        assert "sandbox_report" in str(exc)
    else:
        raise AssertionError("expected runtime approval without sandbox report to fail")

    try:
        transition_tool_proposal_status(
            proposal_path=proposal.path,
            next_status="approved_for_runtime",
            code_sha256=code_hash,
            reviewer="local-admin",
            sandbox_report={
                "status": "pass",
                "proposal_id": payload["proposal_id"],
                "tool_name": payload["name"],
                "code_sha256": "bad",
            },
        )
    except ValueError as exc:
        assert "sandbox_report code_sha256" in str(exc)
    else:
        raise AssertionError("expected mismatched sandbox report hash to fail")

    valid_report = {
        "status": "pass",
        "policy_version": "tool_sandbox.v1",
        "input_sha256": "0" * 64,
        "duration_ms": 12,
        "proposal_id": payload["proposal_id"],
        "tool_name": payload["name"],
        "code_sha256": code_hash,
        "limits": {
            "timeout_seconds": 2.0,
            "memory_mb": 256,
            "processes": 16,
            "stdout_stderr_chars": 8000,
        },
        "sandbox": {
            "process": "subprocess",
            "isolated_python": True,
            "working_directory": "temporary_empty",
            "network": "deny_by_import_gate",
            "filesystem": "deny_by_static_check_and_restricted_builtins",
            "resource_limits": {
                "address_space_mb": 256,
                "cpu_seconds_soft": 2,
                "cpu_seconds_hard": 3,
                "file_size_bytes": 1024 * 1024,
                "open_files": 32,
                "processes": 16,
            },
        },
    }
    try:
        transition_tool_proposal_status(
            proposal_path=proposal.path,
            next_status="approved_for_runtime",
            code_sha256=code_hash,
            reviewer="local-admin",
            sandbox_report={**valid_report, "policy_version": "tool_sandbox.old"},
        )
    except ValueError as exc:
        assert "policy_version" in str(exc)
    else:
        raise AssertionError("expected mismatched sandbox policy version to fail")

    try:
        transition_tool_proposal_status(
            proposal_path=proposal.path,
            next_status="approved_for_runtime",
            code_sha256=code_hash,
            reviewer="local-admin",
            sandbox_report={**valid_report, "input_sha256": ""},
        )
    except ValueError as exc:
        assert "input_sha256" in str(exc)
    else:
        raise AssertionError("expected missing sandbox input hash to fail")

    try:
        transition_tool_proposal_status(
            proposal_path=proposal.path,
            next_status="approved_for_runtime",
            code_sha256=code_hash,
            reviewer="local-admin",
            sandbox_report={**valid_report, "limits": {**valid_report["limits"], "timeout_seconds": 60}},
        )
    except ValueError as exc:
        assert "timeout limit" in str(exc)
    else:
        raise AssertionError("expected excessive sandbox timeout limit to fail")

    try:
        transition_tool_proposal_status(
            proposal_path=proposal.path,
            next_status="approved_for_runtime",
            code_sha256=code_hash,
            reviewer="local-admin",
            sandbox_report={
                **valid_report,
                "sandbox": {
                    **valid_report["sandbox"],
                    "resource_limits": {
                        **valid_report["sandbox"]["resource_limits"],
                        "open_files": 128,
                    },
                },
            },
        )
    except ValueError as exc:
        assert "open-file resource limit" in str(exc)
    else:
        raise AssertionError("expected excessive sandbox open-file resource limit to fail")

    try:
        transition_tool_proposal_status(
            proposal_path=proposal.path,
            next_status="approved_for_runtime",
            code_sha256=code_hash,
            reviewer="local-admin",
            sandbox_report={**valid_report, "sandbox": {**valid_report["sandbox"], "network": "allow"}},
        )
    except ValueError as exc:
        assert "network policy" in str(exc)
    else:
        raise AssertionError("expected mismatched sandbox network policy to fail")

    runtime_payload = transition_tool_proposal_status(
        proposal_path=proposal.path,
        next_status="approved_for_runtime",
        code_sha256=code_hash,
        reviewer="local-admin",
        sandbox_report=valid_report,
    )

    assert runtime_payload["status"] == "approved_for_runtime"
    assert runtime_payload["sandbox_report"]["status"] == "pass"


def test_run_tool_proposal_sandbox_executes_approved_code_in_subprocess(tmp_path) -> None:
    proposal = propose_tool(
        data_dir=tmp_path,
        name="metric_root",
        description="Compute a simple metric transform.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
        },
        python_code=(
            "import math\n\n"
            "async def run(args, ctx, session):\n"
            "    return {'root': math.sqrt(args['value']), 'has_ctx': bool(ctx)}"
        ),
        rationale="Sandbox smoke test.",
    )
    payload = json.loads(proposal.path.read_text(encoding="utf-8"))
    transition_tool_proposal_status(
        proposal_path=proposal.path,
        next_status="approved_for_sandbox_test",
        code_sha256=payload["code_sha256"],
        reviewer="local-admin",
        note="ready for sandbox",
    )

    report = run_tool_proposal_sandbox(proposal_path=proposal.path, args={"value": 16})

    assert report["status"] == "pass"
    assert report["policy_version"] == "tool_sandbox.v1"
    assert len(report["input_sha256"]) == 64
    assert report["tool_name"] == "metric_root"
    assert report["code_sha256"] == payload["code_sha256"]
    assert report["result"] == {"root": 4.0, "has_ctx": False}
    assert report["sandbox"]["process"] == "subprocess"
    assert report["sandbox"]["working_directory"] == "temporary_empty"
    assert report["sandbox"]["resource_limits"] == {
        "address_space_mb": 256,
        "cpu_seconds_soft": 2,
        "cpu_seconds_hard": 3,
        "file_size_bytes": 1024 * 1024,
        "open_files": 32,
        "processes": 16,
    }


def test_run_tool_proposal_sandbox_rejects_pending_and_bad_args(tmp_path) -> None:
    proposal = propose_tool(
        data_dir=tmp_path,
        name="echo_metric",
        description="Echo a named metric.",
        input_schema={
            "type": "object",
            "properties": {"metric": {"type": "string"}},
            "required": ["metric"],
        },
        python_code="async def run(args, ctx, session):\n    return {'metric': args['metric']}",
        rationale="Sandbox validation test.",
    )
    payload = json.loads(proposal.path.read_text(encoding="utf-8"))

    try:
        run_tool_proposal_sandbox(proposal_path=proposal.path, args={"metric": "accuracy"})
    except ValueError as exc:
        assert "not sandbox-runnable" in str(exc)
    else:
        raise AssertionError("expected pending proposal to be rejected")

    transition_tool_proposal_status(
        proposal_path=proposal.path,
        next_status="approved_for_sandbox_test",
        code_sha256=payload["code_sha256"],
        reviewer="local-admin",
    )
    try:
        run_tool_proposal_sandbox(proposal_path=proposal.path, args={"metric": "accuracy", "extra": True})
    except ValueError as exc:
        assert "unexpected tool args" in str(exc)
    else:
        raise AssertionError("expected extra sandbox args to be rejected")


def test_run_tool_proposal_sandbox_reports_timeout(tmp_path) -> None:
    proposal = propose_tool(
        data_dir=tmp_path,
        name="busy_metric",
        description="Busy loop for sandbox timeout.",
        input_schema={"type": "object"},
        python_code="async def run(args, ctx, session):\n    while True:\n        pass",
        rationale="Sandbox timeout test.",
    )
    payload = json.loads(proposal.path.read_text(encoding="utf-8"))
    transition_tool_proposal_status(
        proposal_path=proposal.path,
        next_status="approved_for_sandbox_test",
        code_sha256=payload["code_sha256"],
        reviewer="local-admin",
    )

    report = run_tool_proposal_sandbox(proposal_path=proposal.path, args={}, timeout_seconds=0.25)

    assert report["status"] == "timeout"
    assert report["error"]["type"] == "TimeoutExpired"


def test_runtime_tool_manifest_requires_runtime_approval_and_passed_sandbox(tmp_path) -> None:
    proposal = propose_tool(
        data_dir=tmp_path,
        name="approved_metric",
        description="Return an approved metric.",
        input_schema={"type": "object", "properties": {"metric": {"type": "string"}}},
        python_code="async def run(args, ctx, session):\n    return {'metric': args.get('metric', '')}",
        rationale="Runtime manifest test.",
    )
    payload = json.loads(proposal.path.read_text(encoding="utf-8"))

    try:
        runtime_tool_manifest(proposal_path=proposal.path)
    except ValueError as exc:
        assert "not approved for runtime" in str(exc)
    else:
        raise AssertionError("expected pending proposal to be rejected")

    transition_tool_proposal_status(
        proposal_path=proposal.path,
        next_status="approved_for_sandbox_test",
        code_sha256=payload["code_sha256"],
        reviewer="local-admin",
    )
    report = run_tool_proposal_sandbox(proposal_path=proposal.path, args={"metric": "accuracy"})
    transition_tool_proposal_status(
        proposal_path=proposal.path,
        next_status="approved_for_runtime",
        code_sha256=payload["code_sha256"],
        reviewer="local-admin",
        sandbox_report=report,
    )

    manifest = runtime_tool_manifest(proposal_path=proposal.path, reserved_names={"compose"})

    assert manifest["name"] == "approved_metric"
    assert manifest["dynamic"] is True
    assert manifest["dangerous"] is True
    assert manifest["input_schema"]["additionalProperties"] is False
    assert manifest["scope"] == {"visibility": "deployment", "deployment_id": "local", "session_id": ""}
    assert manifest["sandbox_report"]["status"] == "pass"
    assert manifest["sandbox_report"]["policy_version"] == "tool_sandbox.v1"
    assert len(manifest["sandbox_report"]["input_sha256"]) == 64
    assert manifest["sandbox_report"]["sandbox"]["resource_limits"]["open_files"] == 32
    assert manifest["runtime_approval"]["reviewer"] == "local-admin"
    assert manifest["runtime_approval"]["code_sha256"] == payload["code_sha256"]
    assert "python_code" not in manifest


def test_runtime_tool_manifest_requires_runtime_approval_review_log(tmp_path) -> None:
    proposal = propose_tool(
        data_dir=tmp_path,
        name="forged_metric",
        description="Return a forged metric.",
        input_schema={"type": "object"},
        python_code="async def run(args, ctx, session):\n    return {'ok': True}",
        rationale="Runtime approval audit test.",
    )
    payload = json.loads(proposal.path.read_text(encoding="utf-8"))
    transition_tool_proposal_status(
        proposal_path=proposal.path,
        next_status="approved_for_sandbox_test",
        code_sha256=payload["code_sha256"],
        reviewer="local-admin",
    )
    report = run_tool_proposal_sandbox(proposal_path=proposal.path, args={})
    forged = json.loads(proposal.path.read_text(encoding="utf-8"))
    forged["status"] = "approved_for_runtime"
    forged["sandbox_report"] = report
    forged["review_log"] = []
    proposal.path.write_text(json.dumps(forged, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        runtime_tool_manifest(proposal_path=proposal.path)
    except ValueError as exc:
        assert "runtime approval review_log" in str(exc)
    else:
        raise AssertionError("expected forged runtime approval without review log to be rejected")


def test_runtime_tool_execution_revalidates_runtime_approval_review_log(tmp_path) -> None:
    proposal = propose_tool(
        data_dir=tmp_path,
        name="forged_runtime_exec",
        description="Return a forged runtime execution result.",
        input_schema={"type": "object"},
        python_code="async def run(args, ctx, session):\n    return {'ok': True}",
        rationale="Runtime execution audit test.",
    )
    payload = json.loads(proposal.path.read_text(encoding="utf-8"))
    transition_tool_proposal_status(
        proposal_path=proposal.path,
        next_status="approved_for_sandbox_test",
        code_sha256=payload["code_sha256"],
        reviewer="local-admin",
    )
    report = run_tool_proposal_sandbox(proposal_path=proposal.path, args={})
    forged = json.loads(proposal.path.read_text(encoding="utf-8"))
    forged["status"] = "approved_for_runtime"
    forged["sandbox_report"] = report
    forged["review_log"] = []
    proposal.path.write_text(json.dumps(forged, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        run_tool_proposal_sandbox(proposal_path=proposal.path, args={})
    except ValueError as exc:
        assert "runtime approval review_log" in str(exc)
    else:
        raise AssertionError("expected forged runtime execution without review log to be rejected")


def test_runtime_tool_manifest_rejects_static_name_collision_and_hash_tamper(tmp_path) -> None:
    proposal = propose_tool(
        data_dir=tmp_path,
        name="compose",
        description="Colliding dynamic tool.",
        input_schema={"type": "object"},
        python_code="async def run(args, ctx, session):\n    return {'ok': True}",
        rationale="Collision test.",
    )
    payload = json.loads(proposal.path.read_text(encoding="utf-8"))
    transition_tool_proposal_status(
        proposal_path=proposal.path,
        next_status="approved_for_sandbox_test",
        code_sha256=payload["code_sha256"],
        reviewer="local-admin",
    )
    report = run_tool_proposal_sandbox(proposal_path=proposal.path, args={})
    transition_tool_proposal_status(
        proposal_path=proposal.path,
        next_status="approved_for_runtime",
        code_sha256=payload["code_sha256"],
        reviewer="local-admin",
        sandbox_report=report,
    )

    try:
        runtime_tool_manifest(proposal_path=proposal.path, reserved_names={"compose"})
    except ValueError as exc:
        assert "collides" in str(exc)
    else:
        raise AssertionError("expected dynamic tool name collision to be rejected")

    tampered = json.loads(proposal.path.read_text(encoding="utf-8"))
    tampered["name"] = "compose_dynamic"
    tampered["python_code"] += "\n# tampered"
    proposal.path.write_text(json.dumps(tampered, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        runtime_tool_manifest(proposal_path=proposal.path, reserved_names={"compose"})
    except ValueError as exc:
        assert "code_sha256" in str(exc)
    else:
        raise AssertionError("expected tampered runtime proposal to be rejected")


def test_load_runtime_tool_manifests_returns_only_runtime_approved_tools(tmp_path) -> None:
    pending = propose_tool(
        data_dir=tmp_path,
        name="pending_metric",
        description="Pending metric.",
        input_schema={"type": "object"},
        python_code="async def run(args, ctx, session):\n    return {'ok': True}",
        rationale="Pending manifest test.",
    )
    approved = propose_tool(
        data_dir=tmp_path,
        name="runtime_metric",
        description="Runtime metric.",
        input_schema={"type": "object"},
        python_code="async def run(args, ctx, session):\n    return {'ok': True}",
        rationale="Runtime manifest test.",
    )
    payload = json.loads(approved.path.read_text(encoding="utf-8"))
    transition_tool_proposal_status(
        proposal_path=approved.path,
        next_status="approved_for_sandbox_test",
        code_sha256=payload["code_sha256"],
        reviewer="local-admin",
    )
    report = run_tool_proposal_sandbox(proposal_path=approved.path, args={})
    transition_tool_proposal_status(
        proposal_path=approved.path,
        next_status="approved_for_runtime",
        code_sha256=payload["code_sha256"],
        reviewer="local-admin",
        sandbox_report=report,
    )

    manifests = load_runtime_tool_manifests(data_dir=tmp_path, reserved_names={"compose"})

    assert [item["name"] for item in manifests] == ["runtime_metric"]
    assert pending.path.exists()


def test_load_runtime_tool_manifests_filters_deployment_scope(tmp_path) -> None:
    proposal = propose_tool(
        data_dir=tmp_path,
        name="scoped_metric",
        description="Runtime metric for one deployment.",
        input_schema={"type": "object"},
        python_code="async def run(args, ctx, session):\n    return {'ok': True}",
        rationale="Scoped runtime manifest test.",
        deployment_id="prod-east",
    )
    payload = json.loads(proposal.path.read_text(encoding="utf-8"))
    transition_tool_proposal_status(
        proposal_path=proposal.path,
        next_status="approved_for_sandbox_test",
        code_sha256=payload["code_sha256"],
        reviewer="local-admin",
    )
    report = run_tool_proposal_sandbox(proposal_path=proposal.path, args={})
    transition_tool_proposal_status(
        proposal_path=proposal.path,
        next_status="approved_for_runtime",
        code_sha256=payload["code_sha256"],
        reviewer="local-admin",
        sandbox_report=report,
    )

    local_manifests = load_runtime_tool_manifests(data_dir=tmp_path, reserved_names={"compose"})
    scoped_manifests = load_runtime_tool_manifests(
        data_dir=tmp_path,
        reserved_names={"compose"},
        deployment_id="prod-east",
    )

    assert local_manifests == []
    assert [item["name"] for item in scoped_manifests] == ["scoped_metric"]
    assert scoped_manifests[0]["scope"]["deployment_id"] == "prod-east"


def test_tool_proposal_status_transition_rejects_skip_and_hash_mismatch(tmp_path) -> None:
    proposal = propose_tool(
        data_dir=tmp_path,
        name="extract_metric",
        description="Extract a named metric from evidence.",
        input_schema={"type": "object"},
        python_code="async def run(args, ctx, session):\n    return {'ok': True}",
        rationale="Reusable metric extraction.",
    )
    payload = json.loads(proposal.path.read_text(encoding="utf-8"))

    try:
        transition_tool_proposal_status(
            proposal_path=proposal.path,
            next_status="approved_for_runtime",
            code_sha256=payload["code_sha256"],
            reviewer="local-admin",
            sandbox_report={"status": "pass"},
        )
    except ValueError as exc:
        assert "pending_review -> approved_for_runtime" in str(exc)
    else:
        raise AssertionError("expected direct runtime approval to fail")

    try:
        transition_tool_proposal_status(
            proposal_path=proposal.path,
            next_status="approved_for_sandbox_test",
            code_sha256="bad",
            reviewer="local-admin",
        )
    except ValueError as exc:
        assert "code_sha256" in str(exc)
    else:
        raise AssertionError("expected mismatched code hash to fail")
