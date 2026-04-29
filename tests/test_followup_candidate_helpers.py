from __future__ import annotations

import json

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services.followup_candidate_helpers import (
    candidate_title_matches,
    filter_followup_candidates,
    followup_relationship_validator_human_prompt,
    followup_relationship_validator_system_prompt,
    followup_target_aliases,
    followup_validator_assessment_from_payload,
    merge_followup_rankings,
    paper_relationship_brief,
    relationship_evidence_ids_from_payload,
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


def test_merge_followup_rankings_deduplicates_by_paper_id() -> None:
    paper_a = CandidatePaper(paper_id="a", title="A")
    paper_b = CandidatePaper(paper_id="b", title="B")

    merged = merge_followup_rankings(
        primary=[{"paper": paper_a, "source": "primary"}],
        secondary=[{"paper": paper_a, "source": "secondary"}, {"paper": paper_b, "source": "secondary"}],
    )

    assert [item["paper"].paper_id for item in merged] == ["a", "b"]
    assert merged[0]["source"] == "primary"


def test_relationship_evidence_ids_from_payload_filters_unknown_ids() -> None:
    evidence = [
        EvidenceBlock(doc_id="doc-1", paper_id="p1", title="P1", file_path="p1.pdf", page=1, block_type="page_text", snippet="one"),
        EvidenceBlock(doc_id="doc-2", paper_id="p2", title="P2", file_path="p2.pdf", page=2, block_type="page_text", snippet="two"),
    ]

    selected = relationship_evidence_ids_from_payload(
        payload={"evidence_ids": ["doc-2", "missing", "doc-1"]},
        relationship_evidence=evidence,
    )
    fallback = relationship_evidence_ids_from_payload(payload={"evidence_ids": ["missing"]}, relationship_evidence=evidence)

    assert selected == ["doc-2", "doc-1"]
    assert fallback == ["doc-1", "doc-2"]


def test_paper_relationship_brief_truncates_card_and_injects_summary() -> None:
    paper = CandidatePaper(
        paper_id="p1",
        title="Paper One",
        year="2026",
        metadata={
            "authors": "A; B",
            "aliases": "P1",
            "paper_card_text": "x" * 1900,
            "tags": "alignment",
        },
    )

    brief = paper_relationship_brief(paper=paper, paper_summary_text=lambda paper_id: f"summary:{paper_id}")

    assert brief["paper_id"] == "p1"
    assert brief["summary"] == "summary:p1"
    assert len(brief["paper_card_text"]) == 1800


def test_followup_relationship_validator_prompts_build_role_payload() -> None:
    contract = QueryContract(clean_query="Candidate 是否是 Seed 的严格后续工作？", targets=["Seed"])
    seed = CandidatePaper(paper_id="seed", title="Seed Paper", metadata={"paper_card_text": "seed card"})
    candidate = CandidatePaper(paper_id="candidate", title="Candidate Paper", metadata={"paper_card_text": "candidate card"})
    evidence = [
        EvidenceBlock(
            doc_id="seed-doc",
            paper_id="seed",
            title="Seed Paper",
            file_path="seed.pdf",
            page=1,
            block_type="page_text",
            snippet="seed evidence",
        ),
        EvidenceBlock(
            doc_id="candidate-doc",
            paper_id="candidate",
            title="Candidate Paper",
            file_path="candidate.pdf",
            page=2,
            block_type="page_text",
            snippet="x" * 950,
        ),
    ]

    assert "strict_followup" in followup_relationship_validator_system_prompt()
    payload = json.loads(
        followup_relationship_validator_human_prompt(
            contract=contract,
            seed_papers=[seed],
            paper=candidate,
            relationship_evidence=evidence,
            paper_summary_text=lambda paper_id: f"summary:{paper_id}",
        )
    )

    assert payload["seed_papers"][0]["summary"] == "summary:seed"
    assert payload["candidate_paper"]["paper_id"] == "candidate"
    assert [item["role"] for item in payload["relationship_evidence"]] == ["seed", "candidate"]
    assert len(payload["relationship_evidence"][1]["snippet"]) == 900


def test_followup_validator_assessment_from_payload_normalizes_defaults() -> None:
    evidence = [
        EvidenceBlock(
            doc_id="candidate-doc",
            paper_id="candidate",
            title="Candidate Paper",
            file_path="candidate.pdf",
            page=2,
            block_type="page_text",
            snippet="candidate evidence",
        )
    ]

    assessment = followup_validator_assessment_from_payload(
        payload={
            "classification": "strict_followup",
            "strict_followup": True,
            "relationship_strength": "unexpected",
            "reason": "  uses the seed benchmark  ",
            "confidence": "0.91",
            "evidence_ids": ["candidate-doc", "missing"],
        },
        relationship_evidence=evidence,
        coerce_confidence=lambda value: float(value),
    )

    assert assessment["relationship_strength"] == "direct"
    assert assessment["relation_type"] == "严格后续/直接使用证据"
    assert assessment["reason"] == "uses the seed benchmark"
    assert assessment["confidence"] == 0.91
    assert assessment["evidence_ids"] == ["candidate-doc"]
