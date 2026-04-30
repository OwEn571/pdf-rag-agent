# Review Refactor Progress Snapshot - 2026-04-30

This note tracks the safe refactor slices completed against
`docs/review-2026-04-28.md` and names the remaining high-risk work that should
not be changed without a rollback point and test-backed slices.

## Current Baseline

- Latest pushed local baseline before this snapshot: `4c638d0`.
- `app/services/agent.py` has been reduced from the reviewed 7400-line monolith
  to about 1558 lines.
- `app/services/intent.py` has been reduced to about 510 lines after the legacy
  recognizer fallback was split into adapter/helper modules.
- The latest validated full test suite before this snapshot collected and passed
  596 tests in the `zotero-paper-rag` conda environment.
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
- Answer confidence wiring: provider streaming can optionally request token
  logprobs; when enabled for streamed research answers, answer-level
  `Confidence(basis="logprobs")` is emitted and retained in runtime summary
  without changing default latency/cost behavior.
- Router fallback hardening: invalid LLM router tool-plan payloads are tagged as
  `router_invalid_payload`, and the legacy `IntentRecognizer` fallback is now
  behind a default-on compatibility switch so it can be disabled in eval/gray
  runs without deleting the code path yet.
- Trace diff coverage: stable signatures now include contract routing signals
  (`interaction_mode`, `relation`, `intent_kind`, router action/tags) and
  confidence score buckets, so router/composer/solver drift is visible in trace
  comparisons before risky deletions.
- Agent shell cleanup continued: memory answer helpers, follow-up validator and
  selected-candidate assessment, clarification contract/tracking helpers,
  figure helpers, compound subtask execution, agent-step emit, and paper
  recommendation reason generation are now called directly from their owning
  modules or registries instead of through Agent wrapper methods.
- Persistent learning context now loads through the session context boundary
  directly; the Agent no longer keeps a separate learnings-only forwarding
  method while preserving the same failure logging and empty-context fallback.
- Session LLM history rendering now uses the `session_context_helpers` helper
  directly from planner, compound planning, answer composition, and follow-up
  refinement call sites; the Agent no longer keeps a history-message forwarding
  method.
- Web evidence collection and web-claim construction now call the
  `web_evidence.py` helpers directly from Agent/test call sites instead of
  keeping Agent forwarding methods.
- Web-search enablement now runs directly through
  `query_shaping.should_use_web_search` in the turn loop; the Agent no longer
  exposes a pure forwarding method for that decision.
- Clarification option rendering, pending clarification clearing/reset, and
  active research construction now call their owning helper modules directly
  from the turn/compound loops; the Agent no longer exposes those session-state
  forwarding methods.
- Research plan construction now lives at the runtime boundary through
  `research_planning.build_research_plan`; the Agent no longer exposes a
  plan-building forwarding method.
- Excluded-focus title filtering now calls `agent_runtime_helpers` directly
  from runtime and disambiguation paths; the Agent no longer exposes a separate
  forwarding method for that helper.

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
  Provider logprob collection is now wired behind a default-off switch, but
  production defaults still need latency/cost decisions and provider capability
  checks.
- Removing remaining legacy alias and compatibility shells that are still used
  by older frontend/eval traces or by runtime/registry integration boundaries.

## Next Safe Direction

Continue with small, test-backed slices only where behavior can stay compatible:

- Run eval traces with `agent_legacy_intent_fallback_enabled=false` to identify
  the remaining routes that still depend on deterministic fallback.
- Continue moving compatibility mappings out of the Agent/Recognizer into typed
  adapters, then delete wrapper code once callers are migrated.
- Add solver/composer parity traces before replacing specialized solver outputs
  with generic compose-loop behavior.
- Treat the remaining Agent methods that execute core tools
  (`_agent_search_*`, `_agent_solve_claims`, `_agent_verify_grounding`,
  retry/reflection and disambiguation refresh) as behavior-bearing boundaries;
  move them only with broader integration tests.
