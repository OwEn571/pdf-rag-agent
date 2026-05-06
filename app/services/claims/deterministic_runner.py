from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, SessionContext


SolverHandler = Callable[..., list[Claim]]


@dataclass(frozen=True)
class DeterministicSolverHandlers:
    origin_lookup: SolverHandler | None = None
    formula: SolverHandler | None = None
    followup_research: SolverHandler | None = None
    figure: SolverHandler | None = None
    table: SolverHandler | None = None
    metric_context: SolverHandler | None = None
    paper_recommendation: SolverHandler | None = None
    topology_recommendation: SolverHandler | None = None
    topology_discovery: SolverHandler | None = None
    paper_summary_results: SolverHandler | None = None
    default_text: SolverHandler | None = None
    entity_definition: SolverHandler | None = None
    concept_definition: SolverHandler | None = None


def run_deterministic_solver_stage(
    *,
    handlers: DeterministicSolverHandlers,
    stage: str,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    session: SessionContext,
    claims: list[Claim],
) -> list[Claim]:
    handler, needs_session = _handler_for_stage(handlers, stage)
    if handler is not None:
        return _run_handler(
            handler,
            needs_session=needs_session,
            contract=contract,
            papers=papers,
            evidence=evidence,
            session=session,
        )
    if stage == "table_metric":
        return _run_table_metric_stage(
            handlers=handlers,
            contract=contract,
            papers=papers,
            evidence=evidence,
            session=session,
            claims=claims,
        )
    return []


# Map stage name → (handler attribute, needs_session).
# To register a new deterministic solver stage:
#   1. Add a field to DeterministicSolverHandlers.
#   2. Add an entry to this table.
#   3. Wire the handler in SolverPipelineMixin._deterministic_solver_handlers().
_STAGE_HANDLER_TABLE: dict[str, tuple[str, bool]] = {
    "origin_lookup": ("origin_lookup", True),
    "formula": ("formula", False),
    "followup_research": ("followup_research", True),
    "figure": ("figure", False),
    "paper_recommendation": ("paper_recommendation", True),
    "topology_recommendation": ("topology_recommendation", True),
    "topology_discovery": ("topology_discovery", True),
    "paper_summary_results": ("paper_summary_results", True),
    "default_text": ("default_text", True),
    "entity_definition": ("entity_definition", True),
    "concept_definition": ("concept_definition", True),
}


def _handler_for_stage(handlers: DeterministicSolverHandlers, stage: str) -> tuple[SolverHandler | None, bool]:
    entry = _STAGE_HANDLER_TABLE.get(stage)
    if entry is None:
        return None, False
    attr_name, needs_session = entry
    return getattr(handlers, attr_name, None), needs_session


def _run_handler(
    handler: SolverHandler,
    *,
    needs_session: bool,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    session: SessionContext,
) -> list[Claim]:
    kwargs = {"contract": contract, "papers": papers, "evidence": evidence}
    if needs_session:
        kwargs["session"] = session
    return handler(**kwargs)


def _run_table_metric_stage(
    *,
    handlers: DeterministicSolverHandlers,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    session: SessionContext,
    claims: list[Claim],
) -> list[Claim]:
    if handlers.table is None:
        return []
    metric_claims = handlers.table(contract=contract, papers=papers, evidence=evidence)
    if any(claim.claim_type == "metric_value" for claim in [*claims, *metric_claims]):
        return metric_claims
    if handlers.metric_context is None:
        return metric_claims
    return [
        *metric_claims,
        *handlers.metric_context(contract=contract, papers=papers, evidence=evidence, session=session),
    ]
