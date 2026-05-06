from __future__ import annotations

from app.domain.models import Claim, QueryContract, ResearchPlan, SessionContext
from app.services.claims.deterministic_solver import solve_claims_with_deterministic_fallback
from app.services.claims.deterministic_runner import DeterministicSolverHandlers


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def formula(self, **_: object) -> list[Claim]:
        self.calls.append("formula")
        return [Claim(claim_type="formula", value="formula")]

    def default_text(self, **_: object) -> list[Claim]:
        self.calls.append("default_text")
        return [Claim(claim_type="summary", value="default")]


def test_deterministic_claim_solver_dispatches_goal_stage() -> None:
    recorder = _Recorder()

    claims = solve_claims_with_deterministic_fallback(
        handlers=DeterministicSolverHandlers(formula=recorder.formula, default_text=recorder.default_text),
        contract=QueryContract(clean_query="DPO 核心公式", requested_fields=["formula"]),
        plan=ResearchPlan(required_claims=["formula"]),
        papers=[],
        evidence=[],
        session=SessionContext(session_id="deterministic"),
    )

    assert recorder.calls == ["formula"]
    assert claims[0].claim_type == "formula"


def test_deterministic_claim_solver_uses_default_when_no_stage_claims() -> None:
    recorder = _Recorder()

    claims = solve_claims_with_deterministic_fallback(
        handlers=DeterministicSolverHandlers(formula=recorder.formula, default_text=recorder.default_text),
        contract=QueryContract(clean_query="随便聊聊"),
        plan=ResearchPlan(required_claims=["answer"]),
        papers=[],
        evidence=[],
        session=SessionContext(session_id="deterministic"),
    )

    assert recorder.calls == ["default_text"]
    assert claims[0].value == "default"
