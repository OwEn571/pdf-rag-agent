from __future__ import annotations

from types import SimpleNamespace

from app.domain.models import Claim, QueryContract, ResearchPlan, SessionContext
from app.services.agent_mixins.solver_pipeline import SolverPipelineMixin


class _SolverProbe(SolverPipelineMixin):
    def __init__(
        self,
        *,
        generic_claim_solver_enabled: bool,
        generic_claim_solver_shadow_enabled: bool = False,
    ) -> None:
        self.agent_settings = SimpleNamespace(
            generic_claim_solver_enabled=generic_claim_solver_enabled,
            generic_claim_solver_shadow_enabled=generic_claim_solver_shadow_enabled,
        )
        self.calls: list[str] = []

    def _run_schema_claim_solver(self, **_: object) -> list[Claim]:
        self.calls.append("schema")
        return [Claim(claim_type="summary", value="schema")]

    def _run_deterministic_claim_solver(self, **_: object) -> list[Claim]:
        self.calls.append("deterministic")
        return [Claim(claim_type="summary", value="deterministic")]


def test_generic_claim_solver_flag_tries_schema_before_deterministic_fallback() -> None:
    enabled = _SolverProbe(generic_claim_solver_enabled=True)
    disabled = _SolverProbe(generic_claim_solver_enabled=False)
    kwargs = {
        "contract": QueryContract(clean_query="DPO", relation="formula_lookup"),
        "plan": ResearchPlan(required_claims=["formula"]),
        "papers": [],
        "evidence": [],
        "session": SessionContext(session_id="solver"),
    }

    enabled_claims = enabled._run_solvers(**kwargs)
    disabled_claims = disabled._run_solvers(**kwargs)

    assert enabled.calls == ["schema"]
    assert enabled_claims[0].value == "schema"
    assert disabled.calls == ["deterministic"]
    assert disabled_claims[0].value == "deterministic"


def test_generic_claim_solver_shadow_keeps_default_path_and_records_comparison() -> None:
    probe = _SolverProbe(
        generic_claim_solver_enabled=False,
        generic_claim_solver_shadow_enabled=True,
    )

    claims = probe._run_solvers(
        contract=QueryContract(clean_query="DPO", relation="formula_lookup"),
        plan=ResearchPlan(required_claims=["formula"]),
        papers=[],
        evidence=[],
        session=SessionContext(session_id="solver"),
    )

    assert probe.calls == ["schema", "deterministic"]
    assert claims[0].value == "deterministic"
    shadow = probe._last_generic_claim_solver_shadow
    assert shadow["mode"] == "generic_claim_solver_shadow"
    assert shadow["selected"] == "deterministic"
    assert shadow["schema"]["count"] == 1
    assert shadow["deterministic"]["count"] == 1
