from __future__ import annotations

from app.domain.models import Claim, QueryContract
from app.services.followup_relationship_memory import followup_relationship_memory


def test_followup_relationship_memory_selects_explicit_candidate_row() -> None:
    contract = QueryContract(
        clean_query="B 是否是 A 的后续工作？",
        relation="followup_research",
        targets=["A"],
    )
    claim = Claim(
        claim_type="followup_research",
        entity="A",
        structured_data={
            "selected_candidate_title": "Candidate Paper",
            "seed_papers": [{"title": "Seed Paper", "paper_id": "seed-1"}],
            "followup_titles": [
                {"title": "Other Paper", "paper_id": "other"},
                {
                    "title": "Candidate Paper: Full Title",
                    "paper_id": "candidate-1",
                    "relationship_strength": "strong",
                    "relation_type": "extension",
                    "strict_followup": True,
                },
            ],
        },
    )

    memory = followup_relationship_memory(contract=contract, claims=[claim], answer="answer")

    assert memory["seed_target"] == "A"
    assert memory["seed_title"] == "Seed Paper"
    assert memory["candidate_title"] == "Candidate Paper"
    assert memory["candidate_paper_id"] == "candidate-1"
    assert memory["relationship_strength"] == "strong"
    assert memory["strict_followup"] is True


def test_followup_relationship_memory_uses_first_row_without_explicit_candidate() -> None:
    contract = QueryContract(clean_query="后续工作？", relation="followup_research", targets=[])
    claim = Claim(
        claim_type="followup_research",
        entity="Seed Entity",
        structured_data={"followup_titles": [{"title": "First Candidate", "paper_id": "p1"}]},
    )

    memory = followup_relationship_memory(contract=contract, claims=[claim], answer="x" * 1200)

    assert memory["seed_target"] == "Seed Entity"
    assert memory["candidate_title"] == "First Candidate"
    assert len(memory["answer_preview"]) <= 900


def test_followup_relationship_memory_ignores_missing_followup_claim() -> None:
    contract = QueryContract(clean_query="普通问题", relation="general_question")
    assert followup_relationship_memory(contract=contract, claims=[], answer="answer") == {}
