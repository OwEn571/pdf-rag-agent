from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{2,63}$")


@dataclass(frozen=True, slots=True)
class ToolProposal:
    name: str
    description: str
    input_schema: dict[str, Any]
    python_code: str
    rationale: str
    path: Path
    status: str = "pending_review"

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "rationale": self.rationale,
            "path": str(self.path),
            "status": self.status,
            "code_chars": len(self.python_code),
            "admin_approval_required": True,
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
