# Review Refactor Progress Snapshot - 2026-04-30

This note tracks the safe refactor slices completed against
`docs/review-2026-04-28.md` and names the remaining high-risk work that should
not be changed without a rollback point and test-backed slices.

## Current Baseline

- Latest pushed local baseline before this snapshot: `b46601d`.
- `app/services/agent.py` has been reduced from the reviewed 7400-line monolith
  to about 1758 lines.
- `app/services/intent.py` has been reduced to about 510 lines after the legacy
  recognizer fallback was split into adapter/helper modules.
- The latest validated full test suite before this snapshot collected and passed
  585 tests in the `zotero-paper-rag` conda environment.
- Published branch target remains `publish/main`
  (`git@github.com:OwEn571/pdf-rag-agent.git`).
- Rollback marker for the high-risk migration path:
  `rollback-review-high-risk-start-2026-04-30`.

## Completed Safe Slices

- M1 agent split: turn state, emit/trace handling, conversation/research turn
  assembly, compound compatibility, runtime summaries, step messages, and PDF
  rendering have been moved out of the giant `ResearchAssistantAgentV4` body.
- Tool schema work: tool definitions now preserve real `input_schema`; research
  and conversation registries expose structured arguments for corpus search,
  BM25/vector/hybrid search, rerank, PDF page reads, grep, URL fetch, summarize,
  verify, todo, remember, propose_tool, ask_human, and Task.
- Event/trace protocol: canonical event normalization covers tool use/results,
  answer/thinking deltas, verification, confidence, plan payloads, and
  `ask_human`, `todo_update`, `final`, and error-path final events; trace diff
  now includes stable `ask_human` question/options and todo item changes.
- Confidence and clarification: centralized settings and confidence helpers now
  drive clarification thresholds while preserving the existing user-facing
  behavior.
- Retrieval/tooling migration: query rewrite, strategy-selectable
  `search_corpus`, atomic search tools, rerank inputs, formula-heavy retrieval
  flag, SSRF-safe `fetch_url`, and prompt-injection wrapping are implemented.
- Dynamic extension foundation: `propose_tool` writes pending proposals only
  after static safety checks; pending proposals include `proposal_id` and
  `code_sha256`; arbitrary proposed Python code is not executed.
- Persistent learning foundation: `remember` persists learnings and conversation
  context can inject them into later turns.
- Security boundary: PDF page rendering is isolated in `pdf_rendering.py`, with a
  bare-command allowlist for `pdftoppm`; `agent.py` no longer owns subprocess
  command construction.
- Marker consolidation: most intent, follow-up, retrieval, composer, verifier,
  entity, concept, solver-origin, and Zotero webpage marker lists have been
  centralized behind marker profiles or helper modules rather than scattered
  inline across the Agent.
- Metric follow-up routing: a user asking how a reported accuracy/metric is
  defined after a table-backed answer now stays on the research path, inherits
  the active metric context, and triggers both text and table evidence planning.
- Sandbox documentation: `docs/tool-sandbox.md` defines the approval stages,
  deny-by-default runtime requirements, audit fields, and non-goals for dynamic
  tool execution.
- LLM intent router migration: `LLMIntentRouter` now runs before the legacy
  `IntentRecognizer`; router decisions are converted into compatible
  `QueryContract` objects, including fallback target recovery when tool-call
  args omit targets.
- Legacy intent recognizer thinning: research slot profiles, relation/slot
  compatibility, target shaping, protected conversation intents, fallback
  payload assembly, and legacy relation-style payload conversion have been
  extracted into focused helper/adapter modules.
- Agent compatibility shell cleanup: test-only and dead wrapper methods have
  been removed, compound planning now calls helper modules directly from
  `agent_compound.py`, and citation ranking tools now call citation helper
  functions directly from the registry instead of through Agent wrapper methods.

## Remaining High-Risk Work

These items are still aligned with the review, but they remain core behavior
changes and should continue as small rollback-safe slices:

- Fully demoting `QueryContract` from control-flow driver to read-only tags.
  Many stable workflows still depend on relation/notes compatibility, active
  paper binding, ambiguity state, and follow-up memory.
- Deleting the remaining `IntentRecognizer` fallback path and forcing all
  routing through `LLMIntentRouter`. The fallback is now much thinner, but full
  removal can still regress offline behavior and deterministic precision tasks.
- Removing `agent_mixins/solver_pipeline.py` and replacing the 15+ specialized
  solvers with a single generic compose loop. This is the largest remaining
  behavior risk because citation-grounded paper QA currently relies on those
  solvers for exact evidence selection.
- Executing dynamically proposed tools in a real sandbox. The proposal and
  safety-review pipeline exists, but running user/agent-written code requires
  an explicit sandbox, resource limits, and approval UI.
- Enabling provider-level logprobs or multi-sample self-consistency by default.
  The confidence interfaces exist, but production defaults need provider support
  and latency/cost decisions.
- Removing remaining legacy alias and compatibility shells that are still used
  by older frontend/eval traces or by tool registry/subtask monkeypatch tests.

## Next Safe Direction

Continue with small, test-backed slices only where behavior can stay compatible:

- Add regression coverage for LLM router misses before fully deleting
  deterministic fallback paths.
- Continue moving compatibility mappings out of the Agent/Recognizer into
  typed adapters, then delete wrapper code once callers are migrated.
- Improve trace/eval tooling around router decisions and solver/composer output
  so generic replacements can be measured before deleting specialized paths.
