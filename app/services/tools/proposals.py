from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{2,63}$")
_TOOL_PROPOSAL_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,180}$")
_SCOPE_VALUE_RE = re.compile(r"[^A-Za-z0-9_.:-]+")
_TOOL_SCOPE_VISIBILITIES = {"global", "deployment", "session"}
_ALLOWED_IMPORT_ROOTS = {
    "collections",
    "dataclasses",
    "datetime",
    "decimal",
    "functools",
    "itertools",
    "json",
    "math",
    "re",
    "statistics",
    "string",
    "typing",
}
_BLOCKED_CALL_NAMES = {
    "__import__",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "hasattr",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
_BLOCKED_CALL_ROOTS = {
    "httpx",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "urllib",
}
_SANDBOX_RUNNABLE_STATUSES = {"approved_for_sandbox_test", "approved_for_runtime"}
_SANDBOX_STDIO_LIMIT = 8_000
_TOOL_SANDBOX_POLICY_VERSION = "tool_sandbox.v1"
_TOOL_SANDBOX_INPUT_HASH_RE = re.compile(r"^[a-f0-9]{64}$")
_TOOL_SANDBOX_MAX_TIMEOUT_SECONDS = 30.0
_TOOL_SANDBOX_MAX_MEMORY_MB = 1024
_TOOL_SANDBOX_CPU_SECONDS_SOFT = 2
_TOOL_SANDBOX_CPU_SECONDS_HARD = 3
_TOOL_SANDBOX_FILE_SIZE_BYTES = 1024 * 1024
_TOOL_SANDBOX_OPEN_FILES = 32
_TOOL_SANDBOX_PROCESSES = 16
TOOL_PROPOSAL_STATUSES = {
    "pending_review",
    "approved_for_sandbox_test",
    "approved_for_runtime",
    "revoked",
}

_TOOL_SANDBOX_RUNNER = r"""
from __future__ import annotations

import asyncio
import builtins
import json
import sys
import traceback

ALLOWED_IMPORT_ROOTS = set(json.loads(sys.argv[1]))


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level:
        raise ImportError("relative imports are not allowed")
    root = str(name or "").split(".", 1)[0]
    if root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"import not allowed: {name}")
    return builtins.__import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
    "__build_class__": builtins.__build_class__,
    "__import__": safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "object": object,
    "print": print,
    "range": range,
    "repr": repr,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "zip": zip,
}


def jsonable(value):
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [jsonable(item) for item in value]
        return repr(value)


def main():
    payload = json.load(sys.stdin)
    namespace = {"__builtins__": SAFE_BUILTINS, "__name__": "__proposed_tool__"}
    try:
        compiled = compile(str(payload.get("python_code") or ""), "<proposed_tool>", "exec")
        exec(compiled, namespace, namespace)
        run = namespace.get("run")
        if not asyncio.iscoroutinefunction(run):
            raise TypeError("python_code must define async def run(...)")
        result = asyncio.run(run(payload.get("args") or {}, payload.get("ctx") or {}, payload.get("session") or {}))
        print(json.dumps({"ok": True, "result": jsonable(result)}, ensure_ascii=False))
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(limit=4),
                    },
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
"""
TOOL_PROPOSAL_TRANSITIONS = {
    "pending_review": {"approved_for_sandbox_test", "revoked"},
    "approved_for_sandbox_test": {"approved_for_runtime", "revoked"},
    "approved_for_runtime": {"revoked"},
    "revoked": set(),
}


@dataclass(frozen=True, slots=True)
class ToolProposal:
    name: str
    description: str
    input_schema: dict[str, Any]
    python_code: str
    rationale: str
    path: Path
    safety: dict[str, Any]
    scope: dict[str, str]
    status: str = "pending_review"

    def payload(self) -> dict[str, Any]:
        code_sha256 = hashlib.sha256(self.python_code.encode("utf-8")).hexdigest()
        return {
            "proposal_id": self.path.stem,
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "rationale": self.rationale,
            "path": str(self.path),
            "status": self.status,
            "code_chars": len(self.python_code),
            "code_sha256": code_sha256,
            "admin_approval_required": True,
            "sandbox_required": True,
            "scope": dict(self.scope),
            "safety": self.safety,
        }


def propose_tool(
    *,
    data_dir: Path,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    python_code: str,
    rationale: str,
    deployment_id: str = "local",
    session_id: str = "",
) -> ToolProposal:
    clean_name = str(name or "").strip()
    if not _TOOL_NAME_RE.match(clean_name):
        raise ValueError("tool name must match ^[A-Za-z_][A-Za-z0-9_]{2,63}$")
    clean_description = " ".join(str(description or "").split())
    clean_rationale = " ".join(str(rationale or "").split())
    clean_code = str(python_code or "").strip()
    if not clean_description:
        raise ValueError("description is required")
    if not clean_rationale:
        raise ValueError("rationale is required")
    if not clean_code:
        raise ValueError("python_code is required")
    safety = _validate_python_code(clean_code)
    normalized_schema = _normalize_input_schema(input_schema)
    scope = _proposal_scope(deployment_id=deployment_id, session_id=session_id)
    proposal_dir = Path(data_dir) / "tools_proposed"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(f"{clean_name}\n{clean_code}".encode("utf-8")).hexdigest()[:10]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = proposal_dir / f"{timestamp}_{clean_name}_{digest}.json"
    proposal = ToolProposal(
        name=clean_name,
        description=clean_description,
        input_schema=normalized_schema,
        python_code=clean_code,
        rationale=clean_rationale,
        path=path,
        safety=safety,
        scope=scope,
    )
    path.write_text(
        json.dumps(
            {
                **proposal.payload(),
                "python_code": clean_code,
                "created_at": timestamp,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return proposal


def transition_tool_proposal_status(
    *,
    proposal_path: Path,
    next_status: str,
    code_sha256: str,
    reviewer: str,
    note: str = "",
    sandbox_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(proposal_path)
    payload = _read_proposal_payload(path)
    current_status = str(payload.get("status") or "pending_review")
    normalized_next = str(next_status or "").strip()
    if normalized_next not in TOOL_PROPOSAL_STATUSES:
        raise ValueError(f"unknown proposal status: {normalized_next or '<empty>'}")
    allowed = TOOL_PROPOSAL_TRANSITIONS.get(current_status, set())
    if normalized_next not in allowed:
        raise ValueError(f"invalid proposal status transition: {current_status} -> {normalized_next}")
    expected_hash = str(payload.get("code_sha256") or "").strip()
    if not expected_hash or str(code_sha256 or "").strip() != expected_hash:
        raise ValueError("code_sha256 does not match proposal payload")
    reviewer_text = " ".join(str(reviewer or "").split())
    if not reviewer_text:
        raise ValueError("reviewer is required")
    if normalized_next == "approved_for_runtime":
        _validate_sandbox_report_for_payload(sandbox_report, payload=payload)
        payload["sandbox_report"] = dict(sandbox_report)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload["status"] = normalized_next
    payload.setdefault("review_log", []).append(
        {
            "at": timestamp,
            "from": current_status,
            "to": normalized_next,
            "reviewer": reviewer_text,
            "note": " ".join(str(note or "").split()),
            "code_sha256": expected_hash,
        }
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_tool_proposal_sandbox(
    *,
    proposal_path: Path,
    args: dict[str, Any] | None = None,
    timeout_seconds: float = 2.0,
    memory_limit_mb: int = 256,
) -> dict[str, Any]:
    path = Path(proposal_path)
    payload = _read_proposal_payload(path)
    status = str(payload.get("status") or "pending_review")
    if status not in _SANDBOX_RUNNABLE_STATUSES:
        raise ValueError(f"proposal status is not sandbox-runnable: {status}")
    if status == "approved_for_runtime":
        _validate_sandbox_report_for_payload(payload.get("sandbox_report"), payload=payload)
        _runtime_approval_from_payload(payload)
    code = str(payload.get("python_code") or "")
    expected_hash = str(payload.get("code_sha256") or "").strip()
    actual_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError("code_sha256 does not match proposal python_code")
    _validate_python_code(code)
    tool_args = dict(args or {})
    _validate_tool_args(schema=payload.get("input_schema", {}), args=tool_args)
    timeout = max(float(timeout_seconds), 0.05)
    memory_mb = max(int(memory_limit_mb), 64)
    args_json = json.dumps(tool_args, ensure_ascii=False, sort_keys=True)
    sandbox_input = {
        "python_code": code,
        "args": tool_args,
        "ctx": {},
        "session": {},
    }
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="proposed-tool-sandbox-") as sandbox_cwd:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-c",
                    _TOOL_SANDBOX_RUNNER,
                    json.dumps(sorted(_ALLOWED_IMPORT_ROOTS)),
                ],
                input=json.dumps(sandbox_input, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=timeout,
                cwd=sandbox_cwd,
                env={"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"},
                preexec_fn=_sandbox_resource_limiter(memory_mb=memory_mb),
            )
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        return _sandbox_report(
            payload=payload,
            status="timeout",
            duration_ms=duration_ms,
            timeout_seconds=timeout,
            memory_limit_mb=memory_mb,
            input_sha256=hashlib.sha256(args_json.encode("utf-8")).hexdigest(),
            stdout=_truncate_stdio(exc.stdout),
            stderr=_truncate_stdio(exc.stderr),
            error={"type": "TimeoutExpired", "message": f"tool exceeded {timeout:.2f}s wall-clock limit"},
        )
    duration_ms = int((time.perf_counter() - started) * 1000)
    stdout = _truncate_stdio(completed.stdout)
    stderr = _truncate_stdio(completed.stderr)
    result_payload = _parse_sandbox_stdout(stdout)
    if completed.returncode != 0 and result_payload is None:
        return _sandbox_report(
            payload=payload,
            status="fail",
            duration_ms=duration_ms,
            timeout_seconds=timeout,
            memory_limit_mb=memory_mb,
            input_sha256=hashlib.sha256(args_json.encode("utf-8")).hexdigest(),
            stdout=stdout,
            stderr=stderr,
            error={"type": "SandboxProcessFailed", "message": f"exit_code={completed.returncode}"},
        )
    if not result_payload or not result_payload.get("ok"):
        error = dict(result_payload.get("error", {})) if isinstance(result_payload, dict) else {}
        return _sandbox_report(
            payload=payload,
            status="fail",
            duration_ms=duration_ms,
            timeout_seconds=timeout,
            memory_limit_mb=memory_mb,
            input_sha256=hashlib.sha256(args_json.encode("utf-8")).hexdigest(),
            stdout=stdout,
            stderr=stderr,
            error=error or {"type": "SandboxError", "message": "sandbox execution failed"},
        )
    return _sandbox_report(
        payload=payload,
        status="pass",
        duration_ms=duration_ms,
        timeout_seconds=timeout,
        memory_limit_mb=memory_mb,
        input_sha256=hashlib.sha256(args_json.encode("utf-8")).hexdigest(),
        stdout=stdout,
        stderr=stderr,
        result=result_payload.get("result"),
    )


def runtime_tool_manifest(
    *,
    proposal_path: Path,
    reserved_names: set[str] | None = None,
    deployment_id: str = "local",
    session_id: str = "",
) -> dict[str, Any]:
    path = Path(proposal_path)
    payload = _read_proposal_payload(path)
    status = str(payload.get("status") or "pending_review")
    if status != "approved_for_runtime":
        raise ValueError(f"proposal is not approved for runtime: {status}")
    name = str(payload.get("name") or "").strip()
    if not _TOOL_NAME_RE.match(name):
        raise ValueError("runtime proposal has invalid tool name")
    if name in set(reserved_names or set()):
        raise ValueError(f"runtime proposal tool name collides with existing tool: {name}")
    code = str(payload.get("python_code") or "")
    expected_hash = str(payload.get("code_sha256") or "").strip()
    actual_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError("code_sha256 does not match proposal python_code")
    sandbox_report = payload.get("sandbox_report")
    _validate_sandbox_report_for_payload(sandbox_report, payload=payload)
    runtime_approval = _runtime_approval_from_payload(payload)
    scope = _proposal_scope_from_payload(payload.get("scope"))
    if not _proposal_scope_matches(scope, deployment_id=deployment_id, session_id=session_id):
        raise ValueError("runtime proposal scope does not match this agent")
    safety = _validate_python_code(code)
    input_schema = _normalize_input_schema(payload.get("input_schema", {}))
    return {
        "name": name,
        "description": str(payload.get("description") or ""),
        "when": str(payload.get("description") or ""),
        "returns": "Dynamic tool result from approved sandboxed code.",
        "input_schema": input_schema,
        "dangerous": True,
        "dynamic": True,
        "streaming": False,
        "proposal_id": str(payload.get("proposal_id") or path.stem),
        "proposal_path": str(path),
        "code_sha256": expected_hash,
        "scope": scope,
        "safety": safety,
        "sandbox_report": {
            "status": sandbox_report.get("status"),
            "policy_version": sandbox_report.get("policy_version"),
            "input_sha256": sandbox_report.get("input_sha256"),
            "duration_ms": sandbox_report.get("duration_ms"),
            "limits": sandbox_report.get("limits", {}),
            "sandbox": sandbox_report.get("sandbox", {}),
        },
        "runtime_approval": runtime_approval,
    }


def load_runtime_tool_manifests(
    *,
    data_dir: Path,
    reserved_names: set[str] | None = None,
    deployment_id: str = "local",
    session_id: str = "",
) -> list[dict[str, Any]]:
    proposal_dir = Path(data_dir) / "tools_proposed"
    if not proposal_dir.exists():
        return []
    manifests: list[dict[str, Any]] = []
    for path in sorted(proposal_dir.glob("*.json")):
        try:
            payload = _read_proposal_payload(path)
        except ValueError:
            continue
        if str(payload.get("status") or "pending_review") != "approved_for_runtime":
            continue
        try:
            manifests.append(
                runtime_tool_manifest(
                    proposal_path=path,
                    reserved_names=reserved_names,
                    deployment_id=deployment_id,
                    session_id=session_id,
                )
            )
        except ValueError:
            continue
    return manifests


def list_tool_proposals(*, data_dir: Path, include_code: bool = False) -> list[dict[str, Any]]:
    proposal_dir = Path(data_dir) / "tools_proposed"
    if not proposal_dir.exists():
        return []
    proposals: list[dict[str, Any]] = []
    for path in sorted(proposal_dir.glob("*.json")):
        try:
            payload = _read_proposal_payload(path)
        except ValueError:
            continue
        proposals.append(_proposal_admin_payload(payload, path=path, include_code=include_code))
    return proposals


def load_tool_proposal(*, data_dir: Path, proposal_id: str, include_code: bool = True) -> dict[str, Any]:
    path = find_tool_proposal_path(data_dir=data_dir, proposal_id=proposal_id)
    return _proposal_admin_payload(_read_proposal_payload(path), path=path, include_code=include_code)


def find_tool_proposal_path(*, data_dir: Path, proposal_id: str) -> Path:
    clean_id = str(proposal_id or "").strip()
    if _TOOL_PROPOSAL_ID_RE.fullmatch(clean_id) is None:
        raise ValueError("invalid proposal_id")
    proposal_dir = Path(data_dir) / "tools_proposed"
    path = proposal_dir / f"{clean_id}.json"
    try:
        path.relative_to(proposal_dir)
    except ValueError:
        raise ValueError("invalid proposal_id") from None
    if not path.exists():
        raise ValueError(f"proposal not found: {clean_id}")
    return path


def _proposal_admin_payload(payload: dict[str, Any], *, path: Path, include_code: bool) -> dict[str, Any]:
    item = dict(payload)
    item["proposal_id"] = str(item.get("proposal_id") or path.stem)
    item["path"] = str(path)
    if not include_code:
        item.pop("python_code", None)
    return item


def _read_proposal_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"proposal not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"proposal payload is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("proposal payload must be an object")
    return payload


def _validate_sandbox_report_for_payload(sandbox_report: Any, *, payload: dict[str, Any]) -> None:
    if not isinstance(sandbox_report, dict) or not sandbox_report:
        raise ValueError("sandbox_report is required for runtime approval")
    if sandbox_report.get("status") != "pass":
        raise ValueError("sandbox_report status must be pass for runtime approval")
    expected_hash = str(payload.get("code_sha256") or "").strip()
    if not expected_hash or str(sandbox_report.get("code_sha256") or "").strip() != expected_hash:
        raise ValueError("sandbox_report code_sha256 does not match proposal payload")
    expected_proposal_id = str(payload.get("proposal_id") or "").strip()
    if expected_proposal_id and str(sandbox_report.get("proposal_id") or "").strip() != expected_proposal_id:
        raise ValueError("sandbox_report proposal_id does not match proposal payload")
    expected_tool_name = str(payload.get("name") or "").strip()
    if expected_tool_name and str(sandbox_report.get("tool_name") or "").strip() != expected_tool_name:
        raise ValueError("sandbox_report tool_name does not match proposal payload")
    if str(sandbox_report.get("policy_version") or "").strip() != _TOOL_SANDBOX_POLICY_VERSION:
        raise ValueError("sandbox_report policy_version does not match runtime requirements")
    if _TOOL_SANDBOX_INPUT_HASH_RE.fullmatch(str(sandbox_report.get("input_sha256") or "").strip()) is None:
        raise ValueError("sandbox_report input_sha256 is required for runtime approval")
    _validate_sandbox_report_limits(sandbox_report.get("limits"))
    sandbox = sandbox_report.get("sandbox")
    if not isinstance(sandbox, dict):
        raise ValueError("sandbox_report sandbox policy is required for runtime approval")
    if sandbox.get("process") != "subprocess" or sandbox.get("isolated_python") is not True:
        raise ValueError("sandbox_report process policy does not match runtime requirements")
    if sandbox.get("working_directory") != "temporary_empty":
        raise ValueError("sandbox_report working_directory policy does not match runtime requirements")
    if str(sandbox.get("network") or "").split("_", 1)[0] != "deny":
        raise ValueError("sandbox_report network policy does not match runtime requirements")
    if str(sandbox.get("filesystem") or "").split("_", 1)[0] != "deny":
        raise ValueError("sandbox_report filesystem policy does not match runtime requirements")
    _validate_sandbox_resource_limits(sandbox.get("resource_limits"), limits=sandbox_report.get("limits"))


def _runtime_approval_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    review_log = payload.get("review_log")
    if not isinstance(review_log, list):
        raise ValueError("runtime approval review_log is required")
    expected_hash = str(payload.get("code_sha256") or "").strip()
    for item in reversed(review_log):
        if not isinstance(item, dict):
            continue
        if str(item.get("from") or "") != "approved_for_sandbox_test":
            continue
        if str(item.get("to") or "") != "approved_for_runtime":
            continue
        reviewer = " ".join(str(item.get("reviewer") or "").split())
        if not reviewer:
            continue
        if not expected_hash or str(item.get("code_sha256") or "").strip() != expected_hash:
            continue
        return {
            "approved_at": str(item.get("at") or ""),
            "reviewer": reviewer,
            "note": " ".join(str(item.get("note") or "").split()),
            "code_sha256": expected_hash,
        }
    raise ValueError("runtime approval review_log entry is required")


def _validate_sandbox_report_limits(limits: Any) -> None:
    if not isinstance(limits, dict):
        raise ValueError("sandbox_report limits are required for runtime approval")
    try:
        timeout_seconds = float(limits.get("timeout_seconds"))
        memory_mb = int(limits.get("memory_mb"))
        processes = int(limits.get("processes"))
        stdio_chars = int(limits.get("stdout_stderr_chars"))
    except (TypeError, ValueError):
        raise ValueError("sandbox_report limits must be numeric") from None
    if timeout_seconds <= 0 or timeout_seconds > _TOOL_SANDBOX_MAX_TIMEOUT_SECONDS:
        raise ValueError("sandbox_report timeout limit does not match runtime requirements")
    if memory_mb <= 0 or memory_mb > _TOOL_SANDBOX_MAX_MEMORY_MB:
        raise ValueError("sandbox_report memory limit does not match runtime requirements")
    if processes <= 0 or processes > 16:
        raise ValueError("sandbox_report process limit does not match runtime requirements")
    if stdio_chars <= 0 or stdio_chars > _SANDBOX_STDIO_LIMIT:
        raise ValueError("sandbox_report stdio limit does not match runtime requirements")


def _validate_sandbox_resource_limits(resource_limits: Any, *, limits: Any) -> None:
    if not isinstance(resource_limits, dict):
        raise ValueError("sandbox_report OS resource limits are required for runtime approval")
    if not isinstance(limits, dict):
        raise ValueError("sandbox_report limits are required for runtime approval")
    try:
        address_space_mb = int(resource_limits.get("address_space_mb"))
        cpu_seconds_soft = int(resource_limits.get("cpu_seconds_soft"))
        cpu_seconds_hard = int(resource_limits.get("cpu_seconds_hard"))
        file_size_bytes = int(resource_limits.get("file_size_bytes"))
        open_files = int(resource_limits.get("open_files"))
        processes = int(resource_limits.get("processes"))
        expected_memory_mb = int(limits.get("memory_mb"))
        expected_processes = int(limits.get("processes"))
    except (TypeError, ValueError):
        raise ValueError("sandbox_report OS resource limits must be numeric") from None
    if address_space_mb != expected_memory_mb or address_space_mb <= 0 or address_space_mb > _TOOL_SANDBOX_MAX_MEMORY_MB:
        raise ValueError("sandbox_report address-space resource limit does not match runtime requirements")
    if cpu_seconds_soft != _TOOL_SANDBOX_CPU_SECONDS_SOFT or cpu_seconds_hard != _TOOL_SANDBOX_CPU_SECONDS_HARD:
        raise ValueError("sandbox_report CPU resource limit does not match runtime requirements")
    if cpu_seconds_hard < cpu_seconds_soft:
        raise ValueError("sandbox_report CPU resource limit is invalid")
    if file_size_bytes <= 0 or file_size_bytes > _TOOL_SANDBOX_FILE_SIZE_BYTES:
        raise ValueError("sandbox_report file-size resource limit does not match runtime requirements")
    if open_files <= 0 or open_files > _TOOL_SANDBOX_OPEN_FILES:
        raise ValueError("sandbox_report open-file resource limit does not match runtime requirements")
    if processes != expected_processes or processes <= 0 or processes > _TOOL_SANDBOX_PROCESSES:
        raise ValueError("sandbox_report process resource limit does not match runtime requirements")


def _normalize_input_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise ValueError("input_schema must be an object")
    normalized = dict(schema)
    if normalized.get("type") != "object":
        normalized["type"] = "object"
    if not isinstance(normalized.get("properties"), dict):
        normalized["properties"] = {}
    if "required" in normalized and not isinstance(normalized["required"], list):
        normalized["required"] = []
    normalized.setdefault("additionalProperties", False)
    return normalized


def _proposal_scope(
    *,
    deployment_id: str = "local",
    session_id: str = "",
    visibility: str | None = None,
) -> dict[str, str]:
    clean_deployment = _scope_value(deployment_id, default="local")
    clean_session = _scope_value(session_id, default="")
    clean_visibility = str(visibility or ("session" if clean_session else "deployment")).strip().lower()
    if clean_visibility not in _TOOL_SCOPE_VISIBILITIES:
        clean_visibility = "session" if clean_session else "deployment"
    if clean_visibility == "session" and not clean_session:
        clean_visibility = "deployment"
    return {
        "visibility": clean_visibility,
        "deployment_id": clean_deployment,
        "session_id": clean_session if clean_visibility == "session" else "",
    }


def _proposal_scope_from_payload(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return _proposal_scope()
    return _proposal_scope(
        deployment_id=str(value.get("deployment_id") or "local"),
        session_id=str(value.get("session_id") or ""),
        visibility=str(value.get("visibility") or "deployment"),
    )


def _proposal_scope_matches(scope: dict[str, str], *, deployment_id: str, session_id: str = "") -> bool:
    visibility = str(scope.get("visibility") or "deployment")
    if visibility == "global":
        return True
    deployment = _scope_value(deployment_id, default="local")
    if str(scope.get("deployment_id") or "local") != deployment:
        return False
    if visibility != "session":
        return True
    return bool(session_id) and str(scope.get("session_id") or "") == _scope_value(session_id, default="")


def _scope_value(value: Any, *, default: str) -> str:
    text = _SCOPE_VALUE_RE.sub("_", str(value or "").strip())[:128].strip("._:-")
    return text or default


def _validate_tool_args(*, schema: Any, args: dict[str, Any]) -> None:
    normalized = _normalize_input_schema(schema)
    properties = normalized.get("properties", {})
    required = [str(item) for item in normalized.get("required", []) if str(item)]
    missing = [name for name in required if name not in args]
    if missing:
        raise ValueError("missing required tool args: " + ", ".join(missing))
    if normalized.get("additionalProperties") is False:
        allowed = set(properties)
        unexpected = [name for name in args if name not in allowed]
        if unexpected:
            raise ValueError("unexpected tool args: " + ", ".join(unexpected))
    for name, spec in properties.items():
        if name not in args or not isinstance(spec, dict):
            continue
        expected_type = spec.get("type")
        if expected_type and not _tool_arg_matches_type(args[name], str(expected_type)):
            raise ValueError(f"tool arg {name} must be {expected_type}")


def _tool_arg_matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "null":
        return value is None
    return True


def _validate_python_code(code: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"python_code syntax error: {exc.msg}") from exc
    issues: list[str] = []
    run_defs = [node for node in tree.body if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == "run"]
    if not any(isinstance(node, ast.AsyncFunctionDef) for node in run_defs):
        issues.append("python_code must define async def run(...)")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _validate_import_root(alias.name, issues=issues)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                issues.append("relative imports are not allowed")
                continue
            _validate_import_root(node.module or "", issues=issues)
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            root = call_name.split(".", 1)[0]
            if call_name in _BLOCKED_CALL_NAMES or root in _BLOCKED_CALL_ROOTS:
                issues.append(f"blocked call: {call_name}")
        elif isinstance(node, ast.Attribute) and str(node.attr).startswith("__"):
            issues.append(f"dunder attribute access is not allowed: {node.attr}")
    if issues:
        raise ValueError("unsafe python_code: " + "; ".join(dict.fromkeys(issues)))
    return {
        "static_check": "pass",
        "sandbox_policy_version": _TOOL_SANDBOX_POLICY_VERSION,
        "allowed_import_roots": sorted(_ALLOWED_IMPORT_ROOTS),
        "blocked_call_names": sorted(_BLOCKED_CALL_NAMES),
        "blocked_call_roots": sorted(_BLOCKED_CALL_ROOTS),
    }


def _validate_import_root(name: str, *, issues: list[str]) -> None:
    root = str(name or "").split(".", 1)[0]
    if not root or root not in _ALLOWED_IMPORT_ROOTS:
        issues.append(f"import not allowed: {name or '<empty>'}")


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _sandbox_resource_limiter(*, memory_mb: int) -> Any:
    def limit() -> None:
        try:
            import resource

            memory_bytes = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (_TOOL_SANDBOX_CPU_SECONDS_SOFT, _TOOL_SANDBOX_CPU_SECONDS_HARD),
            )
            resource.setrlimit(resource.RLIMIT_FSIZE, (_TOOL_SANDBOX_FILE_SIZE_BYTES, _TOOL_SANDBOX_FILE_SIZE_BYTES))
            resource.setrlimit(resource.RLIMIT_NOFILE, (_TOOL_SANDBOX_OPEN_FILES, _TOOL_SANDBOX_OPEN_FILES))
            if hasattr(resource, "RLIMIT_NPROC"):
                resource.setrlimit(resource.RLIMIT_NPROC, (_TOOL_SANDBOX_PROCESSES, _TOOL_SANDBOX_PROCESSES))
        except Exception:
            return

    return limit


def _parse_sandbox_stdout(stdout: str) -> dict[str, Any] | None:
    for line in reversed([item.strip() for item in str(stdout or "").splitlines() if item.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _truncate_stdio(value: Any) -> str:
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value or "")
    if len(text) <= _SANDBOX_STDIO_LIMIT:
        return text
    return text[:_SANDBOX_STDIO_LIMIT] + "...[truncated]"


def _sandbox_report(
    *,
    payload: dict[str, Any],
    status: str,
    duration_ms: int,
    timeout_seconds: float,
    memory_limit_mb: int,
    input_sha256: str,
    stdout: str,
    stderr: str,
    result: Any = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "status": status,
        "policy_version": _TOOL_SANDBOX_POLICY_VERSION,
        "proposal_id": str(payload.get("proposal_id") or ""),
        "tool_name": str(payload.get("name") or ""),
        "code_sha256": str(payload.get("code_sha256") or ""),
        "input_sha256": input_sha256,
        "duration_ms": max(int(duration_ms), 0),
        "limits": {
            "timeout_seconds": timeout_seconds,
            "memory_mb": memory_limit_mb,
            "processes": 16,
            "stdout_stderr_chars": _SANDBOX_STDIO_LIMIT,
        },
        "sandbox": {
            "process": "subprocess",
            "isolated_python": True,
            "working_directory": "temporary_empty",
            "network": "deny_by_import_gate",
            "filesystem": "deny_by_static_check_and_restricted_builtins",
            "resource_limits": {
                "address_space_mb": memory_limit_mb,
                "cpu_seconds_soft": _TOOL_SANDBOX_CPU_SECONDS_SOFT,
                "cpu_seconds_hard": _TOOL_SANDBOX_CPU_SECONDS_HARD,
                "file_size_bytes": _TOOL_SANDBOX_FILE_SIZE_BYTES,
                "open_files": _TOOL_SANDBOX_OPEN_FILES,
                "processes": _TOOL_SANDBOX_PROCESSES,
            },
        },
        "stdout": stdout,
        "stderr": stderr,
    }
    if status == "pass":
        report["result"] = result
    else:
        report["error"] = dict(error or {})
    return report
