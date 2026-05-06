from __future__ import annotations

from app.domain.models import QueryContract, SessionContext
from app.services.contracts.followup_relationship import (
    inherit_followup_relationship_contract,
    normalize_followup_direction_contract,
)


def _normalize_targets(targets: list[str], _: list[str]) -> list[str]:
    return [item.strip() for item in targets if item.strip()]


def test_normalize_followup_direction_contract_extracts_seed_and_candidate() -> None:
    contract = QueryContract(
        clean_query="Candidate Paper 是否是 Seed Method 的严格后续工作？",
        relation="general_question",
        requested_fields=["answer"],
        required_modalities=["paper_card"],
    )

    updated = normalize_followup_direction_contract(contract=contract, normalize_targets=_normalize_targets)

    assert updated.relation == "followup_research"
    assert updated.targets == ["Seed Method"]
    assert "candidate_relationship" in updated.requested_fields
    assert "evidence" in updated.requested_fields
    assert "paper_card" in updated.required_modalities
    assert "page_text" in updated.required_modalities
    assert "candidate_title=Candidate Paper" in updated.notes
    assert "followup_direction_resolved" in updated.notes


def test_normalize_followup_direction_contract_keeps_unmatched_query() -> None:
    contract = QueryContract(clean_query="有哪些后续论文？", relation="followup_research")

    assert normalize_followup_direction_contract(contract=contract, normalize_targets=_normalize_targets) is contract


def test_inherit_followup_relationship_contract_builds_strict_recheck() -> None:
    session = SessionContext(
        session_id="followup-contract",
        working_memory={
            "last_followup_relationship": {
                "seed_target": "Seed Method",
                "candidate_title": "Candidate Paper",
            }
        },
    )
    contract = QueryContract(
        clean_query="确认一下是不是严格后续工作",
        relation="followup_research",
        continuation_mode="followup",
        notes=["candidate_title=Old Candidate"],
    )

    updated = inherit_followup_relationship_contract(
        contract=contract,
        session=session,
        normalize_targets=_normalize_targets,
    )

    assert updated.relation == "followup_research"
    assert updated.clean_query == "Candidate Paper 是否是 Seed Method 的严格后续工作？"
    assert updated.targets == ["Seed Method"]
    assert updated.requested_fields == ["candidate_relationship", "strict_followup", "evidence"]
    assert updated.required_modalities == ["paper_card", "page_text"]
    assert "candidate_title=Candidate Paper" in updated.notes
    assert "candidate_title=Old Candidate" not in updated.notes
    assert "inherited_followup_relationship" in updated.notes
    assert "strict_followup_validation" in updated.notes


def test_inherit_followup_relationship_contract_ignores_non_recheck_query() -> None:
    session = SessionContext(
        session_id="followup-contract-skip",
        working_memory={
            "last_followup_relationship": {
                "seed_target": "Seed Method",
                "candidate_title": "Candidate Paper",
            }
        },
    )
    contract = QueryContract(clean_query="有哪些后续论文？", relation="followup_research")

    assert (
        inherit_followup_relationship_contract(
            contract=contract,
            session=session,
            normalize_targets=_normalize_targets,
        )
        is contract
    )
