from __future__ import annotations

from types import SimpleNamespace

from app.domain.models import CandidatePaper, QueryContract
from app.services.claims.origin_selection import (
    origin_lookup_claim,
    origin_display_entity,
    origin_paper_text,
    origin_target_aliases,
    origin_target_definition_score,
    origin_target_intro_score,
    paper_has_origin_intro_support,
    pick_origin_paper,
    select_origin_paper,
)


def test_origin_target_aliases_strip_domain_suffixes_and_split_camel_case() -> None:
    aliases = origin_target_aliases(["pdfRagAgent", "AlignX 方法"])

    assert "pdfRagAgent" in aliases
    assert "pdf Rag Agent" in aliases
    assert "AlignX" in aliases


def test_origin_intro_score_rewards_direct_intro_cues() -> None:
    aliases = ["AlignX"]

    assert origin_target_intro_score("In this paper, we propose AlignX for user-level alignment.", aliases) >= 6.0
    assert origin_target_intro_score("Prior work compares against AlignX in experiments.", aliases) == 0.0


def test_origin_intro_score_rewards_parenthetical_acronym_named_by_authors() -> None:
    aliases = ["PPO"]

    score = origin_target_intro_score(
        "The new methods, which we call proximal policy optimization (PPO), have several benefits.",
        aliases,
    )

    assert score >= 6.0


def test_origin_intro_score_does_not_treat_variant_reference_as_origin() -> None:
    aliases = ["PPO"]

    score = origin_target_intro_score(
        "Second, we introduce Group Relative Policy Optimization (GRPO), "
        "a variant of Proximal Policy Optimization (PPO).",
        aliases,
    )

    assert score == 0.0


def test_origin_definition_score_rewards_entity_definition_context() -> None:
    aliases = ["AlignX"]

    score = origin_target_definition_score("AlignX is a benchmark for personalized preference alignment.", aliases)

    assert score >= 3.0


def test_origin_display_entity_uses_paper_text_casing_before_fallback() -> None:
    paper = CandidatePaper(
        paper_id="p1",
        title="From 1,000,000 Users to Every User",
        metadata={"generated_summary": "This paper introduced AlignX for user-level alignment."},
    )

    assert origin_display_entity(targets=["alignx"], paper=paper) == "AlignX"


def test_origin_lookup_claim_uses_contract_target_and_doc_fallback() -> None:
    contract = QueryContract(clean_query="AlignX 最早是哪篇论文提出的", targets=["alignx"])
    paper = CandidatePaper(
        paper_id="p1",
        title="From 1,000,000 Users to Every User",
        year="2025",
        doc_ids=["doc-1"],
        metadata={"generated_summary": "This paper introduced AlignX for user-level alignment."},
    )

    claim = origin_lookup_claim(contract=contract, paper=paper, evidence_ids=[])

    assert claim.claim_type == "origin"
    assert claim.entity == "AlignX"
    assert claim.value == "From 1,000,000 Users to Every User"
    assert claim.structured_data == {"year": "2025", "paper_title": "From 1,000,000 Users to Every User"}
    assert claim.evidence_ids == ["doc-1"]
    assert claim.paper_ids == ["p1"]
    assert claim.confidence == 0.94


def test_paper_has_origin_intro_support_uses_combined_paper_text() -> None:
    paper = CandidatePaper(
        paper_id="p1",
        title="AlignX",
        metadata={"abstract_note": "We introduce AlignX, a personalized preference alignment dataset."},
    )

    assert "We introduce AlignX" in origin_paper_text(paper)
    assert paper_has_origin_intro_support(paper=paper, targets=["AlignX"])


def test_pick_origin_paper_prefers_earliest_year_then_score() -> None:
    early = CandidatePaper(paper_id="early", title="Early", year="2020", score=0.2)
    stronger_same_year = CandidatePaper(paper_id="strong", title="Strong", year="2020", score=0.9)
    recent = CandidatePaper(paper_id="recent", title="Recent", year="2024", score=2.0)

    assert pick_origin_paper([recent, early, stronger_same_year]) == stronger_same_year
    assert pick_origin_paper([]) is None


def test_select_origin_paper_requires_target_before_picking_oldest_paper() -> None:
    contract = QueryContract(clean_query="最早是哪篇论文？", relation="origin_lookup", targets=[])
    paper = CandidatePaper(paper_id="early", title="ImageNet Classification", year="2012", score=1.0)

    selected = select_origin_paper(
        contract=contract,
        papers=[paper],
        evidence=[],
        paper_documents=[],
        candidate_from_paper_id=lambda _: None,
        paper_identity_matches_targets=lambda *_: False,
        target_matcher=lambda *_: False,
    )

    assert selected is None


def test_select_origin_paper_recovers_origin_candidate_from_paper_corpus_without_initial_papers() -> None:
    contract = QueryContract(clean_query="AlignX 最早是哪篇论文提出的", relation="origin_lookup", targets=["AlignX"])
    paper = CandidatePaper(paper_id="origin", title="AlignX Paper", year="2024", score=0.1)
    paper_doc = SimpleNamespace(
        metadata={"paper_id": "origin", "abstract_note": "We introduce AlignX, a benchmark for user-level alignment."},
        page_content="This paper introduces AlignX.",
    )

    selected = select_origin_paper(
        contract=contract,
        papers=[],
        evidence=[],
        paper_documents=[paper_doc],
        candidate_from_paper_id=lambda paper_id: paper if paper_id == "origin" else None,
        paper_identity_matches_targets=lambda *_: False,
        target_matcher=lambda text, target: target.lower() in str(text).lower(),
    )

    assert selected is not None
    assert selected.paper_id == "origin"
    assert selected.score >= 6.0
    assert "introduces AlignX" in str(selected.metadata.get("paper_card_text", ""))


def test_select_origin_paper_prefers_paper_level_acronym_origin_over_repeated_later_mentions() -> None:
    contract = QueryContract(clean_query="PPO 最早由什么论文提出", relation="origin_lookup", targets=["PPO"])
    source = CandidatePaper(
        paper_id="source",
        title="Proximal Policy Optimization Algorithms",
        year="2017",
        score=0.9,
        metadata={
            "abstract_note": (
                "We propose a new family of policy gradient methods. "
                "The new methods, which we call proximal policy optimization (PPO), are simpler to implement."
            )
        },
    )
    later = CandidatePaper(
        paper_id="later",
        title="Secrets of RLHF in Large Language Models Part I: PPO",
        year="2023",
        score=3.4,
        metadata={"abstract_note": "This paper studies PPO implementation details for RLHF."},
    )
    evidence = [
        *[
            SimpleNamespace(
                doc_id=f"later-{index}",
                paper_id="later",
                title=later.title,
                caption="",
                snippet="This paper introduces practical PPO details for RLHF.",
                score=5.0,
            )
            for index in range(10)
        ],
        SimpleNamespace(
            doc_id="source-1",
            paper_id="source",
            title=source.title,
            caption="",
            snippet="The new methods, which we call proximal policy optimization (PPO), are simple.",
            score=2.5,
        ),
    ]

    selected = select_origin_paper(
        contract=contract,
        papers=[later, source],
        evidence=evidence,
        paper_documents=[],
        candidate_from_paper_id=lambda _: None,
        paper_identity_matches_targets=lambda *_: False,
        target_matcher=lambda text, target: target.lower() in str(text).lower(),
    )

    assert selected == source
