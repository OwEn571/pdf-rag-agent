from __future__ import annotations

from app.domain.models import CandidatePaper
from app.services.followup_claim_helpers import followup_research_claim


def test_followup_research_claim_builds_payload_and_dedupes_ids() -> None:
    seed = CandidatePaper(paper_id="seed", title="Seed Paper", year="2024", doc_ids=["seed-doc"])
    followup = CandidatePaper(paper_id="next", title="Next Paper", year="2025", doc_ids=["next-doc"])

    claim = followup_research_claim(
        entity="AlignX",
        seed_papers=[seed],
        followups=[
            {
                "paper": followup,
                "relation_type": "extends",
                "reason": "builds on seed",
                "relationship_strength": "strong",
                "strict_followup": True,
                "classification": "direct",
                "evidence_ids": ["seed-doc", "ev-next"],
                "confidence": 0.9,
            }
        ],
        selected_candidate_title="Next Paper",
    )

    assert claim.claim_type == "followup_research"
    assert claim.value == "Next Paper (2025)"
    assert claim.evidence_ids == ["seed-doc", "ev-next", "next-doc"]
    assert claim.paper_ids == ["seed", "next"]
    assert claim.confidence == 0.9
    assert claim.structured_data["mode"] == "candidate_validation"
    assert claim.structured_data["followup_titles"][0]["strict_followup"] is True


def test_followup_research_claim_limits_followups_and_averages_confidence() -> None:
    seed = CandidatePaper(paper_id="seed", title="Seed Paper", year="2024")
    followups = [
        {"paper": CandidatePaper(paper_id=f"p{idx}", title=f"Paper {idx}", year="2025"), "confidence": 0.5 + idx / 10}
        for idx in range(4)
    ]

    claim = followup_research_claim(
        entity="",
        seed_papers=[seed],
        followups=followups,
        selected_candidate_title="",
        limit=2,
    )

    assert claim.structured_data["mode"] == "followup_discovery"
    assert [item["paper_id"] for item in claim.structured_data["followup_titles"]] == ["p0", "p1"]
    assert claim.confidence == 0.55
