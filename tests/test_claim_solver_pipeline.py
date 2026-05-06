from __future__ import annotations

from app.domain.models import Claim
from app.services.claims.solver_pipeline import run_claim_solver_pipeline


def test_claim_solver_pipeline_uses_schema_when_enabled() -> None:
    calls: list[str] = []

    result = run_claim_solver_pipeline(
        schema_allowed=False,
        generic_enabled=True,
        shadow_enabled=False,
        solve_schema=lambda: calls.append("schema") or [Claim(claim_type="summary", value="schema")],
        solve_deterministic=lambda: calls.append("deterministic") or [Claim(claim_type="summary", value="deterministic")],
    )

    assert calls == ["schema"]
    assert result.claims[0].value == "schema"
    assert result.shadow == {}


def test_claim_solver_pipeline_shadow_keeps_deterministic_path() -> None:
    calls: list[str] = []

    result = run_claim_solver_pipeline(
        schema_allowed=False,
        generic_enabled=False,
        shadow_enabled=True,
        solve_schema=lambda: calls.append("schema") or [Claim(claim_type="summary", value="schema")],
        solve_deterministic=lambda: calls.append("deterministic") or [Claim(claim_type="summary", value="deterministic")],
    )

    assert calls == ["schema", "deterministic"]
    assert result.claims[0].value == "deterministic"
    assert result.shadow["selected"] == "deterministic"
    assert result.shadow["schema"]["count"] == 1
    assert result.shadow["deterministic"]["count"] == 1
