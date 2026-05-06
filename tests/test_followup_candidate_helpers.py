from __future__ import annotations

import json

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services.followup.candidates import (
    FOLLOWUP_CANDIDATE_MARKERS,
    candidate_title_matches,
    expand_followup_candidate_pool,
    extract_followup_keyphrases,
    filter_followup_candidates,
    followup_candidate_ranker_human_prompt,
    followup_candidate_ranker_system_prompt,
    followup_expansion_terms,
    followup_relationship_evidence,
    followup_relationship_validator_human_prompt,
    followup_relationship_validator_system_prompt,
    followup_reason_fallback,
    followup_relationship_assessment,
    followup_seed_selector_human_prompt,
    followup_seed_selector_system_prompt,
    followup_seed_score,
    followup_target_aliases,
    followup_validator_assessment_from_payload,
    infer_followup_relation_type,
    llm_validate_followup_candidate,
    merge_followup_rankings,
    paper_anchor_text,
    paper_brief,
    paper_author_tokens,
    paper_keyword_set,
    paper_relationship_brief,
    rank_followup_candidates,
    rank_followup_candidates_fallback,
    relationship_evidence_ids_from_payload,
    resolve_followup_seed_papers,
    selected_followup_candidate_assessment,
    selected_followup_candidate_title,
)


def _evidence(
    doc_id: str,
    paper_id: str,
    *,
    score: float = 1.0,
    page: int = 1,
    snippet: str = "snippet",
) -> EvidenceBlock:
    return EvidenceBlock(
        doc_id=doc_id,
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        file_path=f"{paper_id}.pdf",
        page=page,
        block_type="page_text",
        snippet=snippet,
        score=score,
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


def test_paper_anchor_text_uses_short_title_prefix() -> None:
    assert paper_anchor_text(CandidatePaper(paper_id="p1", title="AlignX: Scaling Preferences")) == "AlignX"
    assert paper_anchor_text(CandidatePaper(paper_id="p2", title="Plain Title")) == "Plain Title"


def test_paper_brief_injects_summary_and_metadata() -> None:
    paper = CandidatePaper(
        paper_id="p1",
        title="Paper One",
        year="2026",
        metadata={"authors": "A; B", "aliases": "P1"},
    )

    brief = paper_brief(paper=paper, paper_summary_text=lambda paper_id: f"summary:{paper_id}")

    assert brief == {
        "paper_id": "p1",
        "title": "Paper One",
        "year": "2026",
        "authors": "A; B",
        "aliases": "P1",
        "summary": "summary:p1",
    }


def test_followup_seed_selector_prompts_build_bounded_candidate_payload() -> None:
    contract = QueryContract(clean_query="AlignX 后续工作", targets=["AlignX"])
    candidates = [CandidatePaper(paper_id=f"p{index}", title=f"Paper {index}") for index in range(10)]

    assert "seed_paper_ids" in followup_seed_selector_system_prompt()
    payload = json.loads(
        followup_seed_selector_human_prompt(
            contract=contract,
            active_titles=["Active Paper"],
            candidates=candidates,
            paper_summary_text=lambda paper_id: f"summary:{paper_id}",
        )
    )

    assert payload["query"] == "AlignX 后续工作"
    assert payload["active_titles"] == ["Active Paper"]
    assert len(payload["candidates"]) == 8
    assert payload["candidates"][0]["summary"] == "summary:p0"


def test_followup_candidate_ranker_prompts_build_seed_and_candidate_payload() -> None:
    contract = QueryContract(clean_query="AlignX 后续工作", targets=["AlignX"])
    seed = CandidatePaper(paper_id="seed", title="Seed Paper")
    candidates = [CandidatePaper(paper_id=f"p{index}", title=f"Paper {index}") for index in range(12)]

    assert "followups" in followup_candidate_ranker_system_prompt()
    payload = json.loads(
        followup_candidate_ranker_human_prompt(
            contract=contract,
            seed_papers=[seed],
            candidates=candidates,
            paper_summary_text=lambda paper_id: f"summary:{paper_id}",
        )
    )

    assert payload["seed_papers"][0]["paper_id"] == "seed"
    assert len(payload["candidates"]) == 10
    assert payload["candidates"][-1]["paper_id"] == "p9"


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


def test_extract_followup_keyphrases_keeps_topic_terms_and_repeated_words() -> None:
    phrases = extract_followup_keyphrases(
        "Personalized preference inference uses profile signals. Profile evidence supports preference inference."
    )

    assert "personalized preference" in phrases
    assert "preference inference" in phrases
    assert "profile" in phrases


def test_followup_expansion_terms_are_topic_based_not_example_titles() -> None:
    paper = CandidatePaper(
        paper_id="ALIGNX",
        title="From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
        metadata={
            "paper_card_text": (
                "This work introduces a user-level alignment benchmark for personalized preference, "
                "preference inference, and conditioned generation."
            )
        },
    )

    terms = followup_expansion_terms(paper=paper, paper_summary_text=lambda _: "")

    assert "preference inference" in terms
    assert "conditioned generation" in terms
    assert "POPI" not in terms
    assert "PersonaDual" not in terms
    assert "Text as a Universal Interface" not in terms


def test_followup_reason_fallback_truncates_summary_and_mentions_seed() -> None:
    seed = CandidatePaper(paper_id="seed", title="Seed Paper")
    paper = CandidatePaper(paper_id="candidate", title="Candidate Paper")
    reason = followup_reason_fallback(
        seed_papers=[seed],
        paper=paper,
        paper_summary_text=lambda _: "x" * 140,
    )

    assert reason.startswith("它延续了《Seed Paper》相关主题")
    assert reason.endswith("...")


def test_infer_followup_relation_type_uses_summary_and_strict_flag() -> None:
    paper = CandidatePaper(paper_id="candidate", title="Candidate Paper")

    assert "strict_direct" in FOLLOWUP_CANDIDATE_MARKERS
    assert (
        infer_followup_relation_type(
            paper=paper,
            paper_summary_text=lambda _: "The candidate uses the benchmark dataset.",
            strict=True,
        )
        == "直接使用/评测证据"
    )
    assert (
        infer_followup_relation_type(
            paper=paper,
            paper_summary_text=lambda _: "A transfer method for cross-task personalization.",
        )
        == "transfer extension"
    )


def test_paper_keyword_set_normalizes_tokens_and_filters_stopwords() -> None:
    paper = CandidatePaper(
        paper_id="p1",
        title="Personalized Preferences with Large Language Models",
    )

    keywords = paper_keyword_set([paper], paper_summary_text=lambda _: "Applications across user studies and profiles")

    assert "personalized" in keywords
    assert "preference" in keywords
    assert "study" in keywords
    assert "large" not in keywords
    assert "application" not in keywords


def test_paper_author_tokens_excludes_connector_tokens() -> None:
    paper = CandidatePaper(paper_id="p1", title="Paper", metadata={"authors": "Ada Lovelace and Alan Turing et al"})

    assert paper_author_tokens([paper]) == {"ada", "lovelace", "alan", "turing"}


def test_followup_seed_score_boosts_active_target_intro_and_older_year() -> None:
    contract = QueryContract(clean_query="AlignX 后续工作", targets=["AlignX"])
    paper = CandidatePaper(
        paper_id="alignx",
        title="AlignX",
        year="2025",
        score=1.0,
        metadata={"paper_card_text": "We introduce the AlignX benchmark dataset."},
    )

    score = followup_seed_score(
        contract=contract,
        paper=paper,
        active_titles=["AlignX"],
        paper_summary_text=lambda _: "This paper introduces AlignX for personalized preference inference.",
    )

    assert score > 5.0


def test_resolve_followup_seed_papers_uses_llm_selection_when_available() -> None:
    class Clients:
        chat = object()

        def invoke_json(self, *, system_prompt: str, human_prompt: str, fallback: object) -> object:
            assert "种子论文定位器" in system_prompt
            assert "seed-new" in human_prompt
            return {"seed_paper_ids": ["seed-old", "missing"]}

    seed_old = CandidatePaper(paper_id="seed-old", title="Old Seed", year="2020", score=0.1)
    seed_new = CandidatePaper(paper_id="seed-new", title="New Candidate", year="2025", score=9.0)

    selected = resolve_followup_seed_papers(
        contract=QueryContract(clean_query="Old Seed 后续工作", targets=["Old Seed"]),
        candidates=[seed_new, seed_old],
        active_titles=[],
        clients=Clients(),
        paper_summary_text=lambda paper_id: f"summary {paper_id}",
    )

    assert selected == [seed_old]


def test_resolve_followup_seed_papers_falls_back_to_seed_score() -> None:
    class Clients:
        chat = None

    weak = CandidatePaper(paper_id="weak", title="Weak Candidate", year="2025", score=3.0)
    intro = CandidatePaper(
        paper_id="intro",
        title="AlignX",
        year="2024",
        score=1.0,
        metadata={"paper_card_text": "We introduce AlignX."},
    )

    selected = resolve_followup_seed_papers(
        contract=QueryContract(clean_query="AlignX 后续工作", targets=["AlignX"]),
        candidates=[weak, intro],
        active_titles=["AlignX"],
        clients=Clients(),
        paper_summary_text=lambda paper_id: "This paper introduces AlignX." if paper_id == "intro" else "",
    )

    assert selected == [intro]


def test_expand_followup_candidate_pool_searches_with_seed_context_and_dedupes() -> None:
    contract = QueryContract(clean_query="AlignX 后续工作", targets=["AlignX"])
    seed = CandidatePaper(
        paper_id="seed",
        title="AlignX: Scaling Personalized Preference",
        year="2024",
        score=2.0,
    )
    initial = CandidatePaper(paper_id="initial", title="Initial Candidate", year="2026", score=1.0)
    expanded = CandidatePaper(paper_id="expanded", title="Expanded Candidate", year="2025", score=5.0)
    calls: list[tuple[str, QueryContract, int]] = []

    def search(query: str, search_contract: QueryContract, limit: int) -> list[CandidatePaper]:
        calls.append((query, search_contract, limit))
        return [expanded, initial]

    ranked = expand_followup_candidate_pool(
        contract=contract,
        seed_papers=[seed],
        initial_candidates=[initial],
        paper_limit_default=4,
        paper_summary_text=lambda _: "This paper introduces AlignX and personalized preference inference.",
        search_papers=search,
    )

    assert [paper.paper_id for paper in ranked] == ["expanded", "initial"]
    assert "AlignX" in calls[0][0]
    assert calls[0][1].continuation_mode == "fresh"
    assert calls[0][2] == 16


def test_rank_followup_candidates_uses_explicit_selected_candidate_assessment() -> None:
    class Clients:
        chat = object()

        def invoke_json(self, **_: object) -> object:
            raise AssertionError("LLM ranker should not run for explicit selected candidate")

    seed = CandidatePaper(paper_id="seed", title="Seed Paper")
    candidate = CandidatePaper(paper_id="cand", title="Candidate Paper")
    other = CandidatePaper(paper_id="other", title="Other Paper")
    contract = QueryContract(
        clean_query="Candidate Paper 是后续工作吗？",
        targets=["Seed"],
        notes=["candidate_title=Candidate Paper"],
    )

    ranked = rank_followup_candidates(
        contract=contract,
        seed_papers=[seed],
        candidates=[seed, other, candidate],
        evidence=[],
        clients=Clients(),
        paper_summary_text=lambda _: "",
        selected_candidate_assessment=lambda paper: {
            "relation_type": "后续工作",
            "reason": f"{paper.title} extends seed.",
            "confidence": 0.8,
            "relationship_strength": "direct",
        },
        coerce_confidence=lambda value: float(value),
    )

    assert len(ranked) == 1
    assert ranked[0]["paper"] == candidate
    assert ranked[0]["relation_type"] == "后续工作"


def test_followup_relationship_assessment_does_not_mark_loose_topic_as_direct() -> None:
    contract = QueryContract(
        clean_query="AlignX数据集有后续工作吗？",
        relation="followup_research",
        targets=["AlignX"],
    )
    seed = CandidatePaper(
        paper_id="ALIGNX",
        title="From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
        year="2025",
        metadata={
            "authors": "Li; Chen",
            "aliases": "AlignX",
            "paper_card_text": "This paper introduces AlignX, a dataset and benchmark for user-level alignment.",
        },
    )
    candidate = CandidatePaper(
        paper_id="RELATED",
        title="Personalized Alignment with User Profiles",
        year="2026",
        metadata={
            "authors": "Li; Zhang",
            "paper_card_text": "A personalized alignment method for user preference inference.",
        },
    )

    assessment = followup_relationship_assessment(
        contract=contract,
        seed_papers=[seed],
        paper=candidate,
        paper_summary_text=lambda paper_id: {
            "ALIGNX": "AlignX introduces a personalized preference benchmark dataset.",
            "RELATED": "A personalized alignment method for user preference inference.",
        }.get(paper_id, ""),
    )

    assert assessment["strength"] != "direct"
    assert "证据不足" in assessment["reason"]


def test_followup_relationship_assessment_marks_direct_target_relation() -> None:
    contract = QueryContract(clean_query="AlignX 后续", targets=["AlignX"])
    seed = CandidatePaper(paper_id="ALIGNX", title="AlignX", metadata={"aliases": "AlignX"})
    candidate = CandidatePaper(
        paper_id="CANDIDATE",
        title="Candidate Benchmark",
        metadata={"paper_card_text": "The benchmark evaluates and extends AlignX for user-level alignment."},
    )

    assessment = followup_relationship_assessment(
        contract=contract,
        seed_papers=[seed],
        paper=candidate,
        paper_summary_text=lambda paper_id: {
            "ALIGNX": "AlignX introduces a dataset.",
            "CANDIDATE": "This work uses the AlignX benchmark dataset.",
        }.get(paper_id, ""),
    )

    assert assessment["strength"] == "direct"
    assert assessment["confidence"] == 0.86
    assert "AlignX" in assessment["reason"]


def test_rank_followup_candidates_fallback_prioritizes_personalization_continuations() -> None:
    contract = QueryContract(
        clean_query="AlignX数据集有后续工作吗？",
        relation="followup_research",
        targets=["AlignX"],
    )
    seed = CandidatePaper(
        paper_id="ALIGNX",
        title="From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
        year="2025",
        metadata={"paper_card_text": "This paper introduces AlignX."},
    )
    candidates = [
        seed,
        CandidatePaper(
            paper_id="COMMUNITY",
            title="CommunityBench: Benchmarking Community-Level Alignment across Diverse Groups and Tasks",
            year="2026",
            score=1.2,
            metadata={"paper_card_text": "A benchmark for community-level alignment."},
        ),
        CandidatePaper(
            paper_id="POPI",
            title="POPI: Personalizing LLMs via Optimized Preference Inference",
            year="2026",
            score=0.4,
            metadata={"paper_card_text": "Modular personalization with preference inference and conditioned generation."},
        ),
    ]

    ranked = rank_followup_candidates_fallback(
        contract=contract,
        seed_papers=[seed],
        candidates=candidates,
        paper_summary_text=lambda _: "",
    )

    assert ranked
    assert ranked[0]["paper"].paper_id == "POPI"
    assert all(item["paper"].paper_id != "ALIGNX" for item in ranked)


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


def test_llm_validate_followup_candidate_invokes_validator_and_normalizes_payload() -> None:
    class Clients:
        chat = object()

        def invoke_json(self, *, system_prompt: str, human_prompt: str, fallback: object) -> object:
            assert "strict_followup" in system_prompt
            payload = json.loads(human_prompt)
            assert payload["candidate_paper"]["paper_id"] == "candidate"
            return {
                "classification": "strict_followup",
                "strict_followup": True,
                "reason": "uses benchmark",
                "confidence": 0.86,
                "evidence_ids": ["candidate-doc"],
            }

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
    assessment = llm_validate_followup_candidate(
        contract=QueryContract(clean_query="关系", targets=["Seed"]),
        seed_papers=[CandidatePaper(paper_id="seed", title="Seed Paper")],
        paper=CandidatePaper(paper_id="candidate", title="Candidate Paper"),
        relationship_evidence=evidence,
        clients=Clients(),
        paper_summary_text=lambda paper_id: f"summary:{paper_id}",
        coerce_confidence=lambda value: float(value),
    )

    assert assessment["relation_type"] == "严格后续/直接使用证据"
    assert assessment["confidence"] == 0.86
    assert assessment["evidence_ids"] == ["candidate-doc"]


def test_selected_followup_candidate_assessment_falls_back_when_llm_disabled() -> None:
    class Clients:
        chat = None

    evidence = [_evidence("candidate-doc", "candidate", snippet="Candidate uses the Seed benchmark.")]
    assessment = selected_followup_candidate_assessment(
        contract=QueryContract(clean_query="Seed 后续工作", targets=["Seed"]),
        seed_papers=[
            CandidatePaper(
                paper_id="seed",
                title="Seed Paper",
                metadata={"paper_card_text": "We introduce the Seed benchmark."},
            )
        ],
        paper=CandidatePaper(
            paper_id="candidate",
            title="Candidate Paper",
            metadata={"paper_card_text": "Candidate uses the Seed benchmark for evaluation."},
        ),
        evidence=evidence,
        clients=Clients(),
        expand_evidence=lambda *_: [],
        paper_summary_text=lambda paper_id: "uses Seed benchmark" if paper_id == "candidate" else "introduces Seed benchmark",
        coerce_confidence=lambda value: float(value),
    )

    assert assessment["relationship_strength"] in {"direct", "strong_related"}
    assert assessment["evidence_ids"] == ["candidate-doc"]


def test_followup_relationship_evidence_filters_and_sorts_pair_without_expansion() -> None:
    contract = QueryContract(clean_query="后续关系", targets=["Seed"])
    seed = CandidatePaper(paper_id="seed", title="Seed Paper")
    candidate = CandidatePaper(paper_id="candidate", title="Candidate Paper")
    evidence = [
        *[_evidence(f"seed-{index}", "seed", score=0.5 + index, page=index) for index in range(1, 4)],
        *[_evidence(f"candidate-{index}", "candidate", score=0.7 + index, page=index) for index in range(1, 4)],
        _evidence("other-1", "other"),
    ]

    selected = followup_relationship_evidence(
        contract=contract,
        seed_papers=[seed],
        paper=candidate,
        evidence=evidence,
        expand_evidence=lambda *_: (_ for _ in ()).throw(AssertionError("unexpected expansion")),
    )

    assert [item.paper_id for item in selected] == ["candidate", "candidate", "candidate", "seed", "seed", "seed"]
    assert selected[0].doc_id == "candidate-3"


def test_followup_relationship_evidence_expands_when_pair_evidence_is_sparse() -> None:
    contract = QueryContract(clean_query="后续关系", targets=["Seed"])
    seed = CandidatePaper(paper_id="seed", title="Seed Paper")
    candidate = CandidatePaper(paper_id="candidate", title="Candidate Paper")
    calls: list[tuple[list[str], str, QueryContract, int]] = []

    def _expand(paper_ids: list[str], query: str, evidence_contract: QueryContract, limit: int) -> list[EvidenceBlock]:
        calls.append((paper_ids, query, evidence_contract, limit))
        return [
            _evidence("candidate-1", "candidate", score=3.0),
            _evidence("expanded-seed", "seed", score=2.0),
        ]

    selected = followup_relationship_evidence(
        contract=contract,
        seed_papers=[seed],
        paper=candidate,
        evidence=[_evidence("candidate-1", "candidate", score=1.0)],
        expand_evidence=_expand,
    )

    assert calls
    assert calls[0][0] == ["seed", "candidate"]
    assert "uses evaluates benchmark" in calls[0][1]
    assert calls[0][2].required_modalities == ["page_text", "paper_card"]
    assert calls[0][3] == 12
    assert [item.doc_id for item in selected] == ["candidate-1", "expanded-seed"]
