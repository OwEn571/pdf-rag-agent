# Dynamic Tool Sandbox

This document defines the safety boundary for `propose_tool`.

## Current State

- `propose_tool` records a pending JSON proposal under
  `data/tools_proposed/*.json`.
- Proposed code is not imported or registered into the Agent runtime.
- Proposed code can run only through the explicit sandbox-test API after the
  proposal reaches `approved_for_sandbox_test` or `approved_for_runtime`.
- Each proposal must include `name`, `description`, `input_schema`,
  `python_code`, and `rationale`.
- The static checker requires `async def run(...)`.
- The static checker currently rejects unsafe names, relative imports,
  disallowed import roots, dunder attribute access, and calls such as `open`,
  `eval`, `exec`, `__import__`, `getattr`, `hasattr`, `setattr`, `delattr`,
  `os`, `subprocess`, `socket`, `httpx`, `urllib`, `pathlib`, and `shutil`.
- Pending proposal payloads include a stable `proposal_id` and full
  `code_sha256` hash for later audit and approval checks.
- Pending proposal payloads include a deployment/session `scope`. New proposals
  default to deployment scope with `deployment_id=local`; session scope is
  reserved for tools approved for one session only.
- Proposal status transitions are enforced by code: a reviewer must provide the
  matching `code_sha256`, transitions cannot skip directly to runtime approval,
  and `approved_for_runtime` requires a passing recorded sandbox report.
- Sandbox tests execute proposed code in an isolated Python subprocess with
  restricted builtins/imports, input-schema checks, wall-clock timeout, memory
  limit, process-count limit, file-size limit, an empty temporary working
  directory, truncated stdout/stderr, input hash, resource limits, and the
  current sandbox policy version in the audit report.
- Runtime manifest loading is fail-closed: only `approved_for_runtime` proposals
  with matching code hash, passing sandbox report, matching sandbox policy
  version, valid input hash, acceptable resource limits, valid schema, and no
  static tool-name collision can produce dynamic tool metadata for the current
  deployment/session scope. The metadata does not include `python_code`.
- Dynamic tool runtime registration is behind `agent_dynamic_tools_enabled`,
  which defaults to `false`. When enabled, only validated runtime manifests are
  exposed to the planner and registered into the tool loop; execution still goes
  through the sandbox-test runner for each call.
- Admin-only proposal review endpoints now exist under
  `/api/v1/v4/admin/tools/proposals`. They require `ADMIN_API_KEY` via
  `Authorization: Bearer ...` or `X-API-Key`, can list/read proposal payloads,
  run sandbox tests, and transition proposal status. List responses omit
  `python_code` unless explicitly requested.

## Approval Stages

1. `pending_review`: default state written by `propose_tool`.
2. `approved_for_sandbox_test`: an admin has reviewed the schema, rationale,
   code, and expected resource needs.
3. `approved_for_runtime`: the tool passed sandbox tests and is allowed to be
   registered for a specific deployment or session.
4. `revoked`: the tool must not be registered or executed.

No proposal may skip directly from `pending_review` to runtime registration.
`revoked` is terminal.

## Runtime Sandbox Requirements

The current sandbox-test runner provides process isolation, restricted
imports/builtins, basic JSON-schema-like argument validation, wall-clock timeout,
memory/process/file-size limits, empty temporary cwd execution, output
truncation, and structured audit reports.
The current report policy version is `tool_sandbox.v1`; runtime approval rejects
reports from any other policy version. Runtime approval also requires a
64-character `input_sha256` and bounded report limits for timeout, memory,
process count, and stdout/stderr size.

Before dynamic runtime registration is enabled, the system still needs all of
the following:

- A clean working directory with no access to project secrets, `.env`, Zotero
  databases, vector stores, caches, SSH keys, or user home files.
- OS-level deny-by-default network enforcement. If a tool needs network access,
  it must use an explicit allowlist that is visible in the proposal.
- OS-level deny-by-default filesystem writes. If writes are needed, they must be
  limited to a dedicated scratch directory that is deleted after execution.
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
- Deployment/session scope used for registration.
- Approver identity or local admin action reference.
- Expiration or review date.

The current Agent consumes this metadata only when `agent_dynamic_tools_enabled`
is true. Runtime execution remains sandboxed and does not import dynamic code
into the API process.

## Admin API

- `GET /api/v1/v4/admin/tools/proposals`: list proposal metadata.
- `GET /api/v1/v4/admin/tools/proposals/{proposal_id}`: inspect one proposal,
  including code for review by default.
- `POST /api/v1/v4/admin/tools/proposals/{proposal_id}/sandbox`: run the
  proposal through the sandbox runner with reviewed input args.
- `POST /api/v1/v4/admin/tools/proposals/{proposal_id}/status`: transition
  status with reviewer, code hash, optional note, and runtime sandbox report.

These endpoints do not enable dynamic tools by themselves; runtime exposure
still requires `approved_for_runtime`, a valid audit chain, a matching sandbox
report, deployment/session scope match, and `agent_dynamic_tools_enabled=true`.

## Non-Goals For The Current Implementation

- `propose_tool` does not make the Agent self-modifying today.
- Pending proposals are not production tools.
- Passing sandbox tests do not automatically register production tools.
- Runtime manifests are not exposed to the planner or tool registry unless the
  explicit dynamic-tool setting is enabled.
- Admin approval alone is not sufficient without isolated execution.
- Proposed tools must not bypass existing security boundaries such as PDF access
  control, SSRF-safe URL fetching, prompt-injection wrapping, or trace logging.
