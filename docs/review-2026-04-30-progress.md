# Review Refactor Progress Snapshot - 2026-04-30

This note tracks the safe refactor slices completed against
`docs/review-2026-04-28.md` and names the remaining high-risk work that should
not be changed mechanically.

## Current Baseline

- Latest pushed local baseline before this snapshot: `481e34a`.
- `app/services/agent.py` has been reduced from the reviewed 7400-line monolith
  to about 1929 lines.
- The latest validated full test suite before this snapshot collected and passed
  566 tests in the `zotero-paper-rag` conda environment.
- Published branch target remains `publish/main`
  (`git@github.com:OwEn571/pdf-rag-agent.git`).

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

## Remaining High-Risk Work

These items are still aligned with the review, but they change core behavior
enough that they should stop for explicit product/architecture confirmation:

- Fully demoting `QueryContract` from control-flow driver to read-only tags.
  Many stable workflows still depend on relation/notes compatibility, active
  paper binding, ambiguity state, and follow-up memory.
- Deleting the remaining `IntentRecognizer` fallback path and forcing all
  routing through `LLMIntentRouter`. This can improve generality, but it may
  regress latency, offline behavior, and deterministic precision tasks.
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
- Removing legacy alias and compatibility shells that are still used by older
  frontend/eval traces.

## Next Safe Direction

Continue with small, test-backed slices only where behavior can stay compatible:

- Add more regression coverage around user-visible paper QA failures before
  deleting deterministic compatibility paths.
- Replace remaining inline marker fragments only when they are clearly isolated
  formatting or routing helpers.
- Improve trace/eval tooling so high-risk architectural replacements can be
  measured before deleting existing deterministic paths.
