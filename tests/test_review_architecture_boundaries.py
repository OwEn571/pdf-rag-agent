from __future__ import annotations

from pathlib import Path


def test_service_modules_do_not_bypass_research_plan_context_for_goals() -> None:
    services_dir = Path(__file__).resolve().parents[1] / "app" / "services"
    offenders: list[str] = []

    for path in services_dir.rglob("*.py"):
        if str(path.relative_to(services_dir)) == "planning/research.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "research_plan_goals(contract)" in text:
            offenders.append(str(path.relative_to(services_dir)))

    assert offenders == []


def test_service_modules_do_not_read_raw_contract_notes_outside_adapters() -> None:
    services_dir = Path(__file__).resolve().parents[1] / "app" / "services"
    allowed = {
        "contracts/context.py",
        "planning/research.py",
    }
    offenders: list[str] = []

    for path in services_dir.rglob("*.py"):
        relative = str(path.relative_to(services_dir))
        if relative in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "contract.notes" in text:
            offenders.append(relative)

    assert offenders == []
