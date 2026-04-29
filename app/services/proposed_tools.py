from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{2,63}$")
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
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
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


@dataclass(frozen=True, slots=True)
class ToolProposal:
    name: str
    description: str
    input_schema: dict[str, Any]
    python_code: str
    rationale: str
    path: Path
    safety: dict[str, Any]
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
