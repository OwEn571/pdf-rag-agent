from __future__ import annotations

from app.domain.models import CandidatePaper, QueryContract
from app.services.followup_candidate_helpers import (
    candidate_title_matches,
    filter_followup_candidates,
    followup_target_aliases,
    selected_followup_candidate_title,
)


def test_selected_followup_candidate_title_reads_contract_note() -> None:
    contract = QueryContract(
        clean_query="确认一下",
        notes=["candidate_title=Candidate Paper", "strict_followup_validation"],
    )

    assert selected_followup_candidate_title(contract) == "Candidate Paper"


def test_candidate_title_matches_title_or_alias() -> None:
    paper = CandidatePaper(
        paper_id="p1",
        title="A Different Surface Title",
        metadata={"aliases": "Candidate Paper||CP"},
    )

    assert candidate_title_matches(paper, "candidate paper")
    assert not candidate_title_matches(paper, "Other Paper")


def test_followup_target_aliases_combines_targets_seed_aliases_and_anchor() -> None:
    contract = QueryContract(clean_query="后续工作", targets=["AlignX"])
    seed = CandidatePaper(
        paper_id="alignx",
        title="From 1,000,000 Users to Every User",
        metadata={"aliases": "AlignX||Personalized Preference Dataset,AlignX"},
    )

    aliases = followup_target_aliases(
        contract=contract,
        seed_papers=[seed],
        paper_anchor_text=lambda _: "AlignX Benchmark",
    )

    assert aliases == ["AlignX", "Personalized Preference Dataset", "AlignX Benchmark"]


def test_filter_followup_candidates_keeps_domain_related_without_literal_target() -> None:
    contract = QueryContract(
        clean_query="AlignX数据集有后续工作吗？",
        relation="followup_research",
        targets=["AlignX"],
    )
    candidates = [
        CandidatePaper(
            paper_id="ALIGNX",
            title="From 1,000,000 Users to Every User",
            metadata={"paper_card_text": "This paper introduces AlignX."},
        ),
        CandidatePaper(
            paper_id="PERSONADUAL",
            title="PersonaDual: Balancing Personalization and Objectivity via Adaptive Reasoning",
            metadata={"paper_card_text": "A personalized alignment method using user preferences and persona reasoning."},
        ),
    ]

    filtered = filter_followup_candidates(
        contract=contract,
        candidates=candidates,
        paper_summary_text=lambda _: "",
    )

    assert [item.paper_id for item in filtered] == ["ALIGNX", "PERSONADUAL"]
