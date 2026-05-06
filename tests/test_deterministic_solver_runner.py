from __future__ import annotations

from app.domain.models import Claim, QueryContract, SessionContext
from app.services.claims.deterministic_runner import DeterministicSolverHandlers, run_deterministic_solver_stage


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def origin_lookup(self, **_: object) -> list[Claim]:
        self.calls.append("origin_lookup")
        return [Claim(claim_type="origin")]

    def formula(self, **_: object) -> list[Claim]:
        self.calls.append("formula")
        return [Claim(claim_type="formula")]

    def table(self, **_: object) -> list[Claim]:
        self.calls.append("table_metric")
        return [Claim(claim_type="table_summary")]

    def metric_context(self, **_: object) -> list[Claim]:
        self.calls.append("metric_context")
        return [Claim(claim_type="metric_context")]


def _handlers(recorder: _Recorder) -> DeterministicSolverHandlers:
    return DeterministicSolverHandlers(
        origin_lookup=recorder.origin_lookup,
        formula=recorder.formula,
        table=recorder.table,
        metric_context=recorder.metric_context,
    )


def test_run_deterministic_solver_stage_dispatches_named_stage() -> None:
    recorder = _Recorder()

    claims = run_deterministic_solver_stage(
        handlers=_handlers(recorder),
        stage="origin_lookup",
        contract=QueryContract(clean_query="DPO"),
        papers=[],
        evidence=[],
        session=SessionContext(session_id="solver"),
        claims=[],
    )

    assert recorder.calls == ["origin_lookup"]
    assert claims[0].claim_type == "origin"


def test_run_deterministic_solver_stage_adds_metric_context_when_table_has_no_metric_value() -> None:
    recorder = _Recorder()

    claims = run_deterministic_solver_stage(
        handlers=_handlers(recorder),
        stage="table_metric",
        contract=QueryContract(clean_query="DPO"),
        papers=[],
        evidence=[],
        session=SessionContext(session_id="solver"),
        claims=[],
    )

    assert recorder.calls == ["table_metric", "metric_context"]
    assert [claim.claim_type for claim in claims] == ["table_summary", "metric_context"]


def test_run_deterministic_solver_stage_skips_metric_context_when_metric_value_exists() -> None:
    recorder = _Recorder()

    claims = run_deterministic_solver_stage(
        handlers=_handlers(recorder),
        stage="table_metric",
        contract=QueryContract(clean_query="DPO"),
        papers=[],
        evidence=[],
        session=SessionContext(session_id="solver"),
        claims=[Claim(claim_type="metric_value")],
    )

    assert recorder.calls == ["table_metric"]
    assert [claim.claim_type for claim in claims] == ["table_summary"]
