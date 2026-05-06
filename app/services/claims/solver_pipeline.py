from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.domain.models import Claim
from app.services.claims.generic_solver import claim_solver_shadow_summary

ClaimSolverFn = Callable[[], list[Claim]]


@dataclass(frozen=True, slots=True)
class ClaimSolverPipelineResult:
    claims: list[Claim]
    shadow: dict[str, Any]
    selected: str = "deterministic"  # "schema" or "deterministic"


def run_claim_solver_pipeline(
    *,
    schema_allowed: bool,
    generic_enabled: bool,
    shadow_enabled: bool,
    solve_schema: ClaimSolverFn,
    solve_deterministic: ClaimSolverFn,
) -> ClaimSolverPipelineResult:
    use_schema_claim_solver = schema_allowed or generic_enabled
    schema_claims: list[Claim] = []
    if use_schema_claim_solver or shadow_enabled:
        schema_claims = solve_schema()
        if schema_claims and use_schema_claim_solver and not shadow_enabled:
            return ClaimSolverPipelineResult(claims=schema_claims, shadow={}, selected="schema")

    deterministic_claims = solve_deterministic()
    shadow: dict[str, Any] = {}
    selected = "deterministic"
    if use_schema_claim_solver and schema_claims:
        selected = "schema"
    if shadow_enabled:
        shadow = claim_solver_shadow_summary(
            selected=selected,
            schema_claims=schema_claims,
            deterministic_claims=deterministic_claims,
        )
    return ClaimSolverPipelineResult(
        claims=schema_claims if (use_schema_claim_solver and schema_claims) else deterministic_claims,
        shadow=shadow,
        selected=selected,
    )
