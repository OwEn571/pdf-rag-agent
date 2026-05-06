# Review Refactor Progress Snapshot - 2026-04-30

This note tracks the safe refactor slices completed against
`docs/review-2026-04-28.md` and names the remaining high-risk work that should
not be changed without a rollback point and test-backed slices.

## Current Baseline

- Latest pushed local baseline before this snapshot: `5e1f6cd`.
- `app/services/agent.py` has been reduced from the reviewed 7400-line monolith
  to about 196 lines.
- `app/services/intent.py` and the legacy intent fallback adapter/helper modules
  have been removed from the runtime tree.
- The latest validated full test suite before this snapshot collected and passed
  629 tests in the `zotero-paper-rag` conda environment.
- Published branch target remains `publish/main`
  (`git@github.com:OwEn571/pdf-rag-agent.git`).
- Rollback marker for the high-risk migration path:
  `rollback-review-high-risk-start-2026-04-30`.
- Rollback marker for legacy fallback removal work:
  `rollback-legacy-fallback-start-2026-04-30`.

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
  now includes stable `ask_human` question/options, todo item changes, and plan
  solver/claim requirements.
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
  functions directly from the registry instead of through Agent wrapper methods;
  tool-call and observation emission now run through `agent_tool_events.py`
  instead of Agent forwarding methods.
- Answer confidence wiring: provider streaming can optionally request token
  logprobs; when enabled for streamed research answers, answer-level
  `Confidence(basis="logprobs")` is emitted and retained in runtime summary
  without changing default latency/cost behavior.
- Answer self-consistency wiring now has a default-off runtime switch. When
  enabled, research answers take a bounded number of extra final-answer samples,
  emit `Confidence(basis="self_consistency")`, and retain the conservative
  answer confidence in runtime summary.
- Router fallback hardening: invalid LLM router tool-plan payloads are tagged as
  `router_invalid_payload`, and the legacy `IntentRecognizer` fallback is now
  behind a default-on compatibility switch so it can be disabled in eval/gray
  runs without deleting the code path yet.
- Router clarification recovery now handles named paper-source questions such as
  "DeepSeek 是哪篇论文": low-confidence or clarify router decisions are recovered
  into `origin_lookup` when the query has an explicit target and source-paper
  intent, preventing stale active-paper context from trapping correction turns.
- Figure-specific paper screening now requires title/alias identity matches for
  explicit targets, so a paper that merely mentions a model such as DeepSeek is
  not treated as the target paper for figure questions. Formula lookup keeps its
  fallback behavior for method/acronym targets that may not be paper titles.
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
- Contract target normalization now calls `contract_normalization` directly
  from router, follow-up, compound, and test call sites; the Agent no longer
  exposes a target-normalization forwarding method.
- Runtime summary construction now calls `agent_runtime_summary.build_runtime_summary`
  directly from standard and compound turn loops; the Agent no longer exposes a
  runtime-summary forwarding method.
- Citation resolution now calls `evidence_presentation.citations_from_doc_ids`
  directly from answer composition, tool registries, and tests; the Agent no
  longer exposes a citation-resolution forwarding method.
- Claim focus title selection now calls `agent_runtime_helpers.claim_focus_titles`
  directly from research turn and reflection paths; the Agent no longer exposes
  a focus-title forwarding method.
- Pending clarification storage and clarification-attempt tracking now call
  `clarification_intents` helpers directly from standard and compound turn
  loops; the Agent no longer exposes those clarification forwarding methods.
- Identity-match paper filtering is now passed inline to screening/retry helper
  callbacks; the Agent no longer exposes a one-off identity-filter forwarding
  method.
- Agent planning now calls `AgentPlanner.plan_actions` directly from the
  standard turn loop and Task subagent; the Agent no longer exposes a planner
  forwarding method.
- LLM router direct-answer contracts now preserve protected conversation
  subtypes through `answer_style` (`greeting`, `self_identity`, `capability`)
  instead of collapsing every direct answer into `general_question`, reducing
  dependence on the legacy recognizer for fallback-off runs.
- LLM router schemas now include a `need_conversation_tool` action for library,
  citation-ranking, and memory conversation tools, and corpus-search router
  decisions can carry explicit relation/field/modality hints; this lets
  fallback-off runs preserve structured contracts instead of re-inferring every
  relation from query text.
- The `agent_legacy_intent_fallback_enabled` default is now off. The main
  `test_agent_v4.py` suite passes with fallback disabled after the test router
  stub was upgraded to emit tool-plan payloads for protected conversation tools,
  contextual paper follow-ups, origin repair, metric/formula follow-ups, and
  local-library recommendation routes.
- The Agent runtime no longer instantiates or calls `IntentRecognizer` as a
  fallback path. Router misses now produce an explicit clarification contract,
  and the legacy fallback setting was removed from runtime configuration.
- The unused `IntentRecognizer`, legacy fallback helpers, legacy contract
  adapter, and legacy intent prompt modules were deleted along with tests that
  only covered those dead paths.
- Dynamic tool proposals now have a code-backed approval state machine:
  transitions require a matching `code_sha256`, cannot skip directly from
  `pending_review` to runtime approval, and require a sandbox report before
  `approved_for_runtime`.
- Dynamic tool proposals now have an explicit sandbox-test runner: approved
  proposals execute in an isolated Python subprocess with restricted
  imports/builtins, input-schema checks, timeout/resource limits, code-hash
  verification, and structured audit reports.
- Approved dynamic tool proposals now have a fail-closed runtime manifest
  loader. It exposes only metadata for `approved_for_runtime` proposals with a
  passing sandbox report, matching code hash, valid schema, and no collision
  with reserved static tool names; it does not expose `python_code` or register
  tools into the Agent loop.
- The dynamic tool sandbox-test runner now executes in an empty temporary
  working directory and records a process-count limit in its audit report,
  further reducing accidental access to project files during sandbox tests.
- Dynamic tool runtime consumption is now wired behind a default-off setting:
  when `agent_dynamic_tools_enabled` is true, validated runtime manifests are
  exposed to the planner, admitted into runtime action filtering, registered in
  conversation/research tool registries, and executed per call through the
  sandbox runner.
- Dynamic tool static checks and sandbox builtins now block reflection helpers
  such as `getattr`, `hasattr`, `setattr`, and `delattr`, closing an easy path
  around dunder-attribute restrictions.
- Dynamic tool sandbox reports now carry an explicit `tool_sandbox.v1` policy
  version. Runtime approval and manifest loading reject reports from any other
  policy version, so future sandbox-policy upgrades cannot accidentally reuse
  stale pass reports.
- Dynamic tool runtime approval now also requires sandbox reports to carry a
  valid `input_sha256` and bounded resource limits for timeout, memory,
  process count, and stdout/stderr size. Runtime manifests expose the input
  hash as audit metadata without exposing `python_code`.
- Dynamic tool sandbox reports now also expose OS-level `resource_limits`
  for address space, CPU, file size, open files, and process count. Runtime
  approval rejects missing or over-broad resource-limit audit data.
- Dynamic tool proposal review now has admin-only API endpoints for listing
  proposals, inspecting one proposal, running sandbox tests, and transitioning
  status. The endpoints require `ADMIN_API_KEY` and do not enable runtime
  dynamic tools unless the existing runtime approval and default-off settings
  are also satisfied.
- The public `search_corpus` tool schema no longer exposes the historical
  `legacy` retrieval strategy. Old plan payloads that still send `legacy` are
  normalized to `auto`, while the model-facing enum is now `auto|bm25|vector|hybrid`.
- Router target recovery helpers now use neutral names
  (`query_target_candidates` and `loss_notation_target_aliases`) instead of
  legacy fallback naming.
- Runtime tool event canonicalization now uses `INTERNAL_TOOL_STAGE_ALIASES`
  for internal helper-stage names, and runtime summaries report
  `noncanonical_tools` instead of `legacy_tools`, reducing legacy alias surface
  while preserving canonical event normalization.
- Research event payload stages no longer use removed internal tool-alias names
  such as `search_papers`, `search_evidence`, `solve_claims`, or
  `verify_grounding`; the emitted stage labels now describe internal phases
  (`paper_discovery`, `evidence_retrieval`, `claim_composition`, and
  `grounding_verification`) while the public tool remains canonical.
- Verification retry now emits the canonical `search_corpus` tool directly with
  `stage=research_retry`; the obsolete `retry_research -> search_corpus` alias
  entry was removed from `INTERNAL_TOOL_STAGE_ALIASES`.
- Evidence disambiguation now emits canonical `compose` or `ask_human`
  observations with `stage=ambiguity_resolution` or
  `stage=ambiguity_detection`; the obsolete `resolve_ambiguity` and
  `detect_ambiguity` alias entries were removed.
- Conversation answer, library status, and library recommendation steps now
  emit canonical `compose` tool events with explicit stage labels, while
  preserving their memory artifact tool types for follow-up context. The
  obsolete compose aliases for `answer_conversation`, `get_library_status`, and
  `get_library_recommendation` were removed.
- Memory-stage events now emit canonical `read_memory` with explicit stages for
  intent understanding, conversation memory reads, memory follow-up answers,
  memory synthesis, previous-answer reflection, and recommendation-candidate
  recovery. The corresponding read-memory alias entries were removed while
  preserving artifact types needed for follow-up context.
- Citation-count lookup now emits canonical `web_search` with
  `stage=citation_count_lookup`, and citation-count ranking emits canonical
  `compose` with `stage=citation_count_ranking`; the corresponding web citation
  alias entries were removed while preserving memory artifact types.
- The remaining `compose_or_ask_human` dead registry functions were removed,
  clarification-limit observations now emit canonical `ask_human` with
  `stage=clarification_limit`, and `INTERNAL_TOOL_STAGE_ALIASES` is now empty.
- The empty `INTERNAL_TOOL_STAGE_ALIASES` compatibility constant was deleted;
  runtime event canonicalization and runtime summaries now pass explicit empty
  alias maps instead of depending on a named compatibility table.
- QueryContract note parsing has been further centralized: planner intent
  payloads, answer composition, claim verification, solver goals, research
  planning, and retrieval goal/alias extraction now use `contract_context`
  helpers for `answer_slot`, intent, ambiguity, confidence, and target-alias
  note values instead of each module hand-parsing the magic strings.
- Trace diff signatures, confidence scoring, and conversation clarification
  reports also use the shared `contract_context` note helpers instead of local
  prefix parsers.
- Follow-up relationship candidate-title notes now use shared
  `contract_context.note_value` and `notes_without_prefixes`, reducing another
  hand-written `candidate_title=` parser before further QueryContract demotion.
- Contract note reads now also have contract-shaped adapters
  (`contract_note_value(s)`, `contract_note_json_value`, and
  `contract_notes_without_prefixes`). Answer composition, clarification
  disambiguation, conversation tool summaries, and follow-up candidate title
  selection use these adapters instead of directly iterating `contract.notes`.
- Planner context payloads and runtime summary construction also use
  contract-shaped note adapters, including `contract_note_float`, for intent
  confidence, intent kind, ambiguity slots, target aliases, active topic, and
  notes payloads.
- Most remaining service-layer raw `contract.notes` access has been replaced
  with `contract_notes`/contract note adapters across follow-up routing,
  conversation-memory binding, clarification-limit promotion, reflection
  next-action, retrieval target aliases, schema claim prompts,
  contextual/conversation contract helpers, and compound task merging. Direct
  raw note reads are now effectively localized to `contract_context` and the
  `research_planning` compatibility adapter.
- An architecture boundary test now fails if service modules reintroduce direct
  `contract.notes` reads outside `contract_context` and the `research_planning`
  compatibility adapter.
- Answer composition no longer maintains a parallel local requested-field to
  goal alias table for web-answer gating; it reads goals through
  `ResearchPlanContext`, keeping composer behavior aligned with planning.
- Exact boolean note checks such as `exclude_previous_focus`,
  `needs_contextual_refine`, `citation_count_requires_web`, and
  `auto_resolved_by_llm_judge` now go through `contract_has_note`, reducing
  direct control-flow reads from raw `contract.notes`.
- Deterministic solver dispatch now has a tested pure-function boundary in
  `solver_dispatch.deterministic_solver_stages`; `solver_pipeline.py` consumes
  that ordered stage list before invoking the existing specialized solvers,
  making the next composer/solver replacement work easier to verify.
- Research plan trace signatures now include `solver_sequence` and
  `required_claims`, so future solver/composer replacement can detect plan-path
  drift before answer synthesis changes leak into user-facing behavior.
- Tool-call and observation event emission has been split into
  `agent_tool_events.py` and wired directly from Agent/runtime registry/task
  helpers. `ResearchAssistantAgentV4` no longer carries `_emit_agent_tool_call`
  or `_record_agent_observation` forwarding shells.
- Paper summary lookup now lives in `paper_summary_helpers.py`; Agent,
  concept/entity mixins, follow-up ranking, and solver claim assembly use the
  helper directly instead of depending on an Agent summary forwarding method.
- PDF page image rendering for table/figure VLM now calls the `pdf_rendering.py`
  boundary directly from `solver_pipeline.py`, while keeping the Agent-owned
  render cache; the Agent no longer exposes a page-render forwarding method.
- LLM router miss handling now lives in `intent_router.py`, and
  `_extract_query_contract` calls the router, router-decision adapter, and
  conversation-tool normalizer directly instead of through Agent wrapper
  methods.
- Research and compound outcome memory writes now call `research_memory.py`
  directly from `agent_loop.py` and `agent_compound.py`; the Agent no longer
  exposes memory-write forwarding methods.
- Conversation answer state writes now live in `conversation_answer_state.py`;
  tool registries and registry helpers set the answer and emit answer deltas
  directly instead of calling an Agent forwarding method.
- Dynamic tool runtime manifest loading now lives in `dynamic_tool_context.py`
  with explicit default-disabled and fail-closed invalid-manifest behavior; the
  Agent no longer owns that initialization wrapper.
- Session history compression is now coordinated by
  `session_context_helpers.compress_session_history_if_needed`; the Agent no
  longer owns compression window/prompt/payload/apply orchestration.
- Contextual research contract refinement now lives in
  `contextual_contract_resolver.py`, covering formula correction, formula
  location follow-up, active-paper binding, and paper-scope correction outside
  the Agent.
- Disambiguation option construction now calls `clarification_intents` acronym
  evidence/options and judge-payload helpers directly; the Agent no longer
  carries those local forwarding methods.
- Paper-title hint lookup is now a local injected callable inside contract
  extraction rather than an Agent method, keeping contextual/conversation
  adapters explicit about their lookup dependency.
- Agent-facing conversation context now lives in
  `session_context_helpers.agent_session_conversation_context`, including
  persistent learning injection; planner/router/compound/tools/mixins call that
  helper directly instead of an Agent forwarding method.
- Agent clarification question construction now runs through
  `clarification_question_helpers.build_agent_clarification_question`; loop,
  compound, registry, and answer-composer paths no longer call an Agent
  clarification forwarding method.
- Follow-up research candidate seed/expand/rank logic is now invoked directly
  inside `solver_pipeline.py`; the Agent no longer owns the three follow-up
  candidate forwarding methods.
- Agent reflection now builds its local focus-title/ambiguity context directly
  in the reflect tool step and calls `reflect_agent_state_decision` there; the
  separate reflection-state forwarding method was removed.
- Evidence-driven disambiguation runtime now lives in
  `agent_disambiguation_runtime.py`, covering option construction, LLM judge
  parsing, and selected-ambiguity material refresh outside the Agent.
- Clarification-limit best-effort promotion now lives in
  `clarification_limit_runtime.py`, so the research loop can trigger the retry
  path without calling an Agent forwarding method.
- Compound Task subagent execution now lives in `agent_compound.py`; compound
  decomposition no longer calls back into an Agent private method for each
  subtask.
- Research search handlers now live in `agent_research_search_handlers.py`,
  covering paper search/screening, evidence search, and web evidence merge
  outside the Agent.
- Research verification and retry handlers now live in
  `agent_research_verification_handlers.py`; the registry verifies grounding
  without calling Agent private verification/retry methods.
- Research compose/claim solving now lives in
  `agent_research_compose_handlers.py`; the registry composes claims,
  performs evidence-disambiguation resolution, and emits compose observations
  without calling an Agent private solve method.
- Research reflection now lives in `agent_research_reflection_handlers.py`;
  runtime finalization runs the reflection decision without calling an Agent
  private reflect method.
- Chat turn orchestration now lives in `agent_chat_runtime.py`; synchronous and
  streaming Agent entrypoints call the runtime function directly instead of
  routing through an Agent private `_run` method.
- Query contract extraction now lives in `agent_contract_extraction.py`;
  standard turns, Task subagents, and contract-routing tests use
  `extract_agent_query_contract` directly, and the Agent class no longer owns a
  private query-contract method.
- Trace diff parity now includes `claims` event signatures (claim type, entity,
  value preview, paper/evidence ids, and claim source), giving solver/composer
  replacement work a stable claim-level drift signal before answer text changes.
- Runtime summaries now label unsourced deterministic claims as
  `deterministic_solver` instead of `legacy_solver`, and the composer flag for
  deterministic claims no longer uses legacy terminology.
- Trace diff parity now includes evidence event signatures (doc/paper ids,
  title, page, block type, snippet preview, source, and path), so future
  solver/composer replacement can detect retrieval drift before claim or final
  answer text changes.
- Dynamic tool proposals and runtime manifests now carry deployment/session
  scope metadata. Runtime manifest loading filters by the configured
  `agent_dynamic_tool_deployment_id`, so approved tools are not exposed outside
  their deployment boundary while the default-disabled behavior stays unchanged.
- Dynamic tool runtime approval now validates that the submitted sandbox report
  belongs to the same proposal id, tool name, and code hash as the proposal
  being approved; runtime manifest loading repeats that binding check.
- Dynamic tool runtime approval also validates the sandbox policy recorded in
  the report: subprocess execution, isolated Python, temporary empty working
  directory, and deny-mode network/filesystem policy are required.
- Active research memory updates now map `QueryContract` fields through
  `session_context_helpers` instead of rebuilding that mapping directly inside
  `agent_loop.py`, reducing the main turn loop's dependency on contract field
  shape while preserving session behavior.
- Trace diff parity now includes final citation signatures, so composer changes
  can detect citation doc/page/source drift even when the final answer length
  stays in the same bucket.
- Trace diff parity now also includes final runtime-summary signatures:
  grounding status, claim/citation counts, claim-source counts, and answer
  confidence basis/score bucket are compared before and after solver/composer
  changes.
- A default-off `agent_generic_claim_solver_enabled` setting now lets eval or
  gray runs try the schema-based generic claim solver before deterministic
  specialized solvers. The default path remains unchanged, but there is now a
  controlled bridge for replacing the 15+ solver-specific branches.
- A default-off `agent_generic_claim_solver_shadow_enabled` setting now runs
  the schema-based generic claim solver side-by-side with the deterministic
  solver and emits a `solver_shadow` trace plus observation summary. This gives
  eval/gray runs claim-count/source/type parity data before changing the
  default selected solver path.
- Trace diff parity now includes `solver_shadow` signatures, covering selected
  solver path plus schema/deterministic claim counts, types, paper/evidence ids,
  and source counts. Generic solver gray runs can now fail loudly on shadow
  drift without relying on ad hoc log inspection.
- `selected_ambiguity_option` JSON note parsing now lives in
  `contract_context.note_json_value(s)`, and clarification/answer-composer paths
  use that shared helper instead of local split/json parsing.
- Ambiguity-note prefix filtering now uses
  `contract_context.notes_without_prefixes`, so auto-resolve and refreshed
  ambiguity-option contracts remove stale note prefixes through one shared
  adapter.
- Research plan construction now has a typed `ResearchPlanContext` and a
  `build_research_plan_from_context` entrypoint. The public QueryContract path
  remains compatible, but plan limits, solver sequence, recall mode, and
  required claims can now be driven from explicit planning tags instead of
  reading QueryContract directly inside the planner.
- Query shaping now reuses `ResearchPlanContext`: paper/evidence query text and
  concept-evidence routing have context-based entrypoints, while the public
  QueryContract wrappers only adapt into that typed context.
- Deterministic solver dispatch now has a typed `SolverDispatchContext` and a
  context-based dispatch entrypoint. The old goals/modalities helper remains as
  a compatibility wrapper, while `solver_pipeline.py` consumes the typed context
  before selecting specialized solver stages.
- Solver goal inference now has a typed `ClaimGoalContext`. The deterministic
  solver pipeline and schema-solver eligibility check consume that context,
  while the old QueryContract/ResearchPlan goal helper remains only as a
  compatibility wrapper.
- Web evidence query/domain selection and web-claim append decisions now have
  `ResearchPlanContext` entrypoints. The public QueryContract API remains, but
  web evidence no longer needs to re-read QueryContract directly to infer goals.
- Conversation-memory target binding, contextual formula/paper repair, and
  follow-up relationship inheritance now read research goals through
  `ResearchPlanContext` rather than calling the QueryContract goal inference
  helper directly.
- Clarification question fallback and evidence-disambiguation routing now read
  goals through `ResearchPlanContext`, reducing another set of direct
  QueryContract-driven branch points.
- Follow-up refinement and runtime material-screening/retry helpers now also
  read goals through `ResearchPlanContext`. Outside `research_planning`'s own
  compatibility adapter, service modules no longer call
  `research_plan_goals(contract)` directly.
- A review architecture boundary test now fails if service modules reintroduce
  direct `research_plan_goals(contract)` calls outside the research-planning
  compatibility adapter.
- The browser workspace now has a V5 shell: `/` redirects to `/v5`, `/v4` is
  preserved as a compatibility entry, the UI is rebranded to Paper Agent V5,
  runtime Intent/Evidence/Status state is visible in the main work area, the
  right panel is an Evidence Board, and frontend local-storage keys moved to
  the v5 namespace.
- Schema-based generic claim solving now lives behind
  `generic_claim_solver.solve_claims_with_generic_schema`; the solver pipeline
  only supplies context and dispatches to the helper, which uses the shared
  `ModelClients.invoke_json` adapter and keeps the no-chat/no-evidence empty
  result behavior compatible.
- Generic solver shadow/parity summary construction now also lives in
  `generic_claim_solver`, so `solver_pipeline.py` no longer owns the trace
  comparison payload shape for schema-vs-deterministic claim outputs.
- The V5 browser Run panel now renders TODO updates and consumes normalized SSE
  event types (`tool_use`, `tool_result`, `todo_update`, `confidence`, and
  `ask_human`) while keeping the old event-name fallback for compatibility.
- Solver composition no longer accepts unused web-search arguments:
  `_run_solvers`, claim composition, and grounding retry now operate only on
  contract/plan/papers/evidence/session, leaving web retrieval decisions in the
  material search layer.
- Deterministic solver stage invocation now lives in
  `deterministic_solver_runner.run_deterministic_solver_stage`, moving the
  stage-name-to-specialized-method dispatch table out of `solver_pipeline.py`
  while preserving the same selected solver behavior.
- The old `_solve_text` private compatibility shell was removed; tests now call
  the deterministic fallback solver entrypoint directly.

## Remaining High-Risk Work

These items are still aligned with the review, but they remain core behavior
changes and should continue as small rollback-safe slices:

- Fully demoting `QueryContract` from control-flow driver to read-only tags.
  Many stable workflows still depend on relation/notes compatibility, active
  paper binding, ambiguity state, and follow-up memory, though note parsing and
  research-plan, query-shaping, web-evidence, clarification, contextual repair,
  runtime material screening, follow-up refinement, and conversation-memory
  construction now have typed adapter boundaries.
- Removing `agent_mixins/solver_pipeline.py` and replacing the 15+ specialized
  solvers with a single generic compose loop. This is the largest remaining
  behavior risk because citation-grounded paper QA currently relies on those
  solvers for exact evidence selection, though deterministic dispatch is now
  isolated, typed, test-covered, and shadow-comparable against the schema
  solver; solver goal inference is also behind a typed context boundary.
- Hardening production use of approved dynamic tools. Proposal capture, static
  checks, code-hash audit fields, approval transitions, sandbox-test execution,
  admin review APIs, runtime manifest loading, and default-off planner/registry
  consumption exist, and runtime manifests now require an auditable
  `approved_for_runtime` review log entry as well as a matching sandbox report.
  Runtime execution also revalidates the same approval chain before running an
  approved tool; production enablement still needs a browser-facing approval UI
  and stricter OS-level filesystem/network isolation.
- Enabling provider-level logprobs or multi-sample self-consistency by default.
  Provider logprob collection and self-consistency sampling are now wired
  behind default-off switches, but production defaults still need latency/cost
  decisions and provider capability checks.
## Next Safe Direction

Continue with small, test-backed slices only where behavior can stay compatible:

- Continue moving compatibility mappings out of runtime boundaries into typed
  adapters, then delete wrapper code once callers are migrated.
- Add solver/composer parity traces before replacing specialized solver outputs
  with generic compose-loop behavior.
- The remaining Agent class is now a thin public entrypoint and dependency
  container; further review work should target solver/composer parity,
  QueryContract control-flow demotion, and dynamic-tool production hardening.
