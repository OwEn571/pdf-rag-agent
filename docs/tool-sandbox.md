# Dynamic Tool Sandbox

This document defines the safety boundary for `propose_tool`.

## Current State

- `propose_tool` records a pending JSON proposal under
  `data/tools_proposed/*.json`.
- Proposed code is not imported, registered, or executed.
- Each proposal must include `name`, `description`, `input_schema`,
  `python_code`, and `rationale`.
- The static checker requires `async def run(...)`.
- The static checker currently rejects unsafe names, relative imports,
  disallowed import roots, dunder attribute access, and calls such as `open`,
  `eval`, `exec`, `__import__`, `os`, `subprocess`, `socket`, `httpx`,
  `urllib`, `pathlib`, and `shutil`.

## Approval Stages

1. `pending_review`: default state written by `propose_tool`.
2. `approved_for_sandbox_test`: an admin has reviewed the schema, rationale,
   code, and expected resource needs.
3. `approved_for_runtime`: the tool passed sandbox tests and is allowed to be
   registered for a specific deployment or session.
4. `revoked`: the tool must not be registered or executed.

No proposal may skip directly from `pending_review` to runtime registration.

## Runtime Sandbox Requirements

Before dynamic code execution is enabled, the runner must provide all of the
following:

- Process isolation from the API server.
- CPU, wall-clock, memory, file count, and output-size limits.
- A clean working directory with no access to project secrets, `.env`, Zotero
  databases, vector stores, caches, SSH keys, or user home files.
- Deny-by-default network access. If a tool needs network access, it must use an
  explicit allowlist that is visible in the proposal.
- Deny-by-default filesystem writes. If writes are needed, they must be limited
  to a dedicated scratch directory that is deleted after execution.
- Structured input validation against the tool `input_schema`.
- Structured output validation and truncation before the result is returned to
  the Agent loop.
- Audit logging for proposal id, code hash, input hash, status, duration,
  resource limits, and sanitized error output.

## Registration Requirements

Runtime registration must persist the following metadata with the tool:

- Proposal path and status.
- Code SHA-256 hash.
- Approved input schema.
- Allowed imports and blocked capabilities from the latest static check.
- Sandbox policy used for execution.
- Approver identity or local admin action reference.
- Expiration or review date.

## Non-Goals For The Current Implementation

- `propose_tool` does not make the Agent self-modifying today.
- Pending proposals are not production tools.
- Static AST checks are not a substitute for runtime sandboxing.
- Admin approval alone is not sufficient without isolated execution.
- Proposed tools must not bypass existing security boundaries such as PDF access
  control, SSRF-safe URL fetching, prompt-injection wrapping, or trace logging.
