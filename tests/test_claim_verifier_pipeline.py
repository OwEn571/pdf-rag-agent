from __future__ import annotations

from app.domain.models import QueryContract, ResearchPlan, VerificationReport
from app.services.claims.verifier_pipeline import claim_verifier_checks_for_goals, verify_claims_with_generic_fallback


def _checks(calls: list[str]) -> dict[str, object]:
    def make_check(name: str, report: VerificationReport | None = None):
        def check(**_: object) -> VerificationReport | None:
            calls.append(name)
            return report

        check._test_name = name  # type: ignore[attr-defined]
        return check

    return {
        "origin": make_check("origin"),
        "entity": make_check("entity"),
        "followup": make_check("followup"),
        "paper_recommendation": make_check("paper_recommendation"),
        "topology": make_check("topology"),
        "figure": make_check("figure"),
        "metric": make_check("metric"),
        "formula": make_check("formula"),
        "general": make_check("general", VerificationReport(status="pass")),
        "concept": make_check("concept"),
    }


def test_claim_verifier_checks_for_goals_orders_specific_checks() -> None:
    calls: list[str] = []
    checks = _checks(calls)  # type: ignore[assignment]

    selected = claim_verifier_checks_for_goals(
        {"origin", "entity_type", "metric_value", "formula"},
        checks=checks,
    )

    assert [check._test_name for check in selected] == ["origin", "entity", "metric", "formula"]  # type: ignore[attr-defined]


def test_verify_claims_with_generic_fallback_runs_until_first_report() -> None:
    calls: list[str] = []
    checks = _checks(calls)  # type: ignore[assignment]

    report = verify_claims_with_generic_fallback(
        contract=QueryContract(clean_query="需要 reward model 吗"),
        plan=ResearchPlan(required_claims=[]),
        claims=[],
        papers=[],
        evidence=[],
        goals={"reward_model_requirement"},
        checks=checks,
    )

    assert report is not None
    assert report.status == "pass"
    assert calls == ["general"]


def test_claim_verifier_checks_default_to_general_for_unknown_goals() -> None:
    calls: list[str] = []
    checks = _checks(calls)  # type: ignore[assignment]

    selected = claim_verifier_checks_for_goals({"unmapped_goal"}, checks=checks)

    assert len(selected) == 1
    selected[0]()
    assert calls == ["general"]
