from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.domain.models import (
    AssistantCitation,
    CandidatePaper,
    Claim,
    EvidenceBlock,
    QueryContract,
    ResearchPlan,
    SessionContext,
    VerificationReport,
)
from app.services.answers.citation_whitelist import (
    audit_answer_citations,
    build_answer_whitelist,
)
from app.services.answers.evidence_presentation import citations_from_doc_ids
from app.services.agent_mixins.claim_verifier import ClaimVerifierMixin
from app.services.memory.research import remember_research_outcome
from app.services.planning.compound_tasks import _rule_based_compound_split


# ── helpers ────────────────────────────────────────────────────────────

def _evidence(
    doc_id: str,
    *,
    paper_id: str = "P1",
    title: str = "Paper One",
    page: int = 1,
    snippet: str = "snippet",
) -> EvidenceBlock:
    return EvidenceBlock(
        doc_id=doc_id,
        paper_id=paper_id,
        title=title,
        file_path="/tmp/paper.pdf",
        page=page,
        block_type="page_text",
        snippet=snippet,
        metadata={"authors": "A. Author", "year": "2024", "tags": "rag||agent"},
    )


def _citation(title: str = "Paper One", paper_id: str = "P1") -> AssistantCitation:
    return AssistantCitation(
        doc_id=f"doc-{paper_id}",
        paper_id=paper_id,
        title=title,
        file_path="/tmp/paper.pdf",
        page=1,
        snippet="snippet text",
        authors="A. Author",
        year="2024",
    )


def _paper(paper_id: str = "P1", title: str = "Paper One") -> CandidatePaper:
    return CandidatePaper(
        paper_id=paper_id,
        title=title,
        year="2024",
        doc_ids=[f"doc-{paper_id}"],
    )


def _claim(
    claim_type: str = "summary",
    *,
    evidence_ids: list[str] | None = None,
    paper_ids: list[str] | None = None,
    entity: str = "",
) -> Claim:
    return Claim(
        claim_type=claim_type,
        entity=entity,
        evidence_ids=evidence_ids or [],
        paper_ids=paper_ids or [],
    )


def _candidate_lookup(paper_id: str) -> CandidatePaper | None:
    mapping = {"P1": _paper("P1", "Paper One"), "P2": _paper("P2", "Paper Two")}
    return mapping.get(paper_id)


# ── P0-1: Citation whitelist ───────────────────────────────────────────

def test_build_whitelist_from_evidence_citations_and_papers() -> None:
    allowed = build_answer_whitelist(
        evidence=[_evidence("doc-1", paper_id="P1", title="Alpha Theory")],
        citations=[_citation(title="Beta Survey", paper_id="P2")],
        screened_papers=[_paper(paper_id="P3", title="Gamma Method")],
    )
    # Normalized form: all lowercased and whitespace-collapsed
    assert "alpha theory" in allowed
    assert "beta survey" in allowed
    assert "gamma method" in allowed
    assert "p1" in allowed
    assert "p2" in allowed
    assert "p3" in allowed


def test_audit_passes_when_all_titles_in_whitelist() -> None:
    allowed = {"attention is all you need", "bert: pre-training of deep bidirectional transformers"}
    answer = "《Attention Is All You Need》提出了自注意力机制，而《BERT: Pre-training of Deep Bidirectional Transformers》基于此改进。"
    assert audit_answer_citations(answer=answer, allowed_titles=allowed) == []


def test_audit_detects_titles_outside_whitelist() -> None:
    allowed = {"attention is all you need"}
    answer = "《Attention Is All You Need》很好，但《GPT-4 Technical Report》更好。"
    violations = audit_answer_citations(answer=answer, allowed_titles=allowed)
    assert len(violations) >= 1
    assert any("GPT-4 Technical Report" in v for v in violations)


def test_audit_detects_english_italic_titles() -> None:
    allowed = {"attention is all you need"}
    answer = "We refer to *Attention Is All You Need* for details, but *Scaling Laws for Neural Language Models* disagrees."
    violations = audit_answer_citations(answer=answer, allowed_titles=allowed)
    assert len(violations) >= 1
    assert any("Scaling Laws" in v for v in violations)


def test_audit_detects_bracket_refs_out_of_range() -> None:
    allowed = {"paper a"}
    answer = "See [1] and [5] for details."
    violations = audit_answer_citations(answer=answer, allowed_titles=allowed, max_citation_index=3)
    assert len(violations) >= 1
    assert any("5" in v for v in violations)


def test_audit_passes_valid_bracket_refs() -> None:
    allowed = {"paper a"}
    answer = "See [1] and [3] for details."
    violations = audit_answer_citations(answer=answer, allowed_titles=allowed, max_citation_index=3)
    assert len(violations) == 0


def test_audit_handles_empty_answer() -> None:
    assert audit_answer_citations(answer="", allowed_titles={"paper a"}) == []
    assert audit_answer_citations(answer="没有引用任何论文", allowed_titles={"paper a"}) == []


# ── P0-2: citations_from_doc_ids screened_paper_ids filter ──────────────

def test_citations_from_doc_ids_restricts_fallback_to_screened_papers() -> None:
    evidence: list[EvidenceBlock] = []
    fallback_doc = SimpleNamespace(
        page_content="paper card text",
        metadata={
            "doc_id": "paper::P2",
            "paper_id": "P2",
            "title": "Paper Two",
            "authors": "B. Author",
            "year": "2025",
            "file_path": "/tmp/p2.pdf",
        },
    )

    # Without screened_paper_ids, P2 would be included via fallback
    full_citations = citations_from_doc_ids(
        ["paper::P2"], evidence,
        paper_doc_lookup=lambda pid: fallback_doc if pid == "P2" else None,
    )
    assert len(full_citations) == 1

    # With screened_paper_ids restricting to P1 only, P2 is excluded
    restricted = citations_from_doc_ids(
        ["paper::P2"], evidence,
        paper_doc_lookup=lambda pid: fallback_doc if pid == "P2" else None,
        screened_paper_ids={"P1"},
    )
    assert len(restricted) == 0


def test_citations_from_doc_ids_allows_screened_paper() -> None:
    evidence: list[EvidenceBlock] = []
    fallback_doc = SimpleNamespace(
        page_content="paper card text",
        metadata={
            "doc_id": "paper::P1",
            "paper_id": "P1",
            "title": "Paper One",
            "authors": "A. Author",
            "year": "2024",
            "file_path": "/tmp/p1.pdf",
        },
    )
    citations = citations_from_doc_ids(
        ["paper::P1"], evidence,
        paper_doc_lookup=lambda pid: fallback_doc if pid == "P1" else None,
        screened_paper_ids={"P1"},
    )
    assert len(citations) == 1
    assert citations[0].title == "Paper One"


def test_citations_from_doc_ids_evidence_direct_lookup_unaffected_by_filter() -> None:
    """Evidence-direct lookups (item in evidence dict) should always work,
    regardless of screened_paper_ids."""
    evidence = [_evidence("doc-a", paper_id="P1", title="Paper One")]
    citations = citations_from_doc_ids(
        ["doc-a"], evidence,
        screened_paper_ids=set(),  # empty filter
    )
    # Still found because it's in evidence directly, not via fallback
    assert len(citations) == 1
    assert citations[0].doc_id == "doc-a"


# ── P0-3: claim verifier issubset audit ────────────────────────────────

class _FakeClaimVerifier(ClaimVerifierMixin):
    """Minimal concrete class to exercise the verifier mixin."""
    settings: object = SimpleNamespace()

    def __init__(self) -> None:
        self.settings = SimpleNamespace()
        self.clients = SimpleNamespace(chat=None)  # skip LLM schema verifier

    def _verify_claims_with_schema(self, **kwargs: object) -> None:
        return None  # skip LLM path for deterministic audit test

    def _verify_claims_with_generic_fallback(self, **kwargs: object) -> VerificationReport | None:
        return None  # fall through to default pass


def _make_plan(*, required_claims: list[str] | None = None) -> ResearchPlan:
    return ResearchPlan(
        required_claims=required_claims or ["summary"],
        solver_sequence=["generic_solver"],
    )


def test_claim_verifier_rejects_mixed_real_and_fake_ids() -> None:
    """[real_id_1, fake_id_2] should be rejected with issubset, not pass with intersection."""
    verifier = _FakeClaimVerifier()
    claims = [_claim("summary", evidence_ids=["doc-real", "doc-fake"])]
    papers: list[CandidatePaper] = []
    evidence = [_evidence("doc-real", paper_id="P1", title="Paper One")]

    report = verifier._verify_claims(
        contract=QueryContract(
            clean_query="test query",
            interaction_mode="research",
            relation="general_question",
            targets=["test"],
            requested_fields=["summary"],
            required_modalities=["page_text"],
        ),
        plan=_make_plan(),
        claims=claims,
        papers=papers,
        evidence=evidence,
    )

    # With issubset: {doc-real, doc-fake} is NOT a subset of {doc-real}
    assert report.status == "clarify"
    assert len(report.unsupported_claims) >= 1


def test_claim_verifier_passes_when_all_ids_in_evidence() -> None:
    verifier = _FakeClaimVerifier()
    claims = [_claim("summary", evidence_ids=["doc-real"])]
    papers: list[CandidatePaper] = []
    evidence = [_evidence("doc-real", paper_id="P1", title="Paper One")]

    report = verifier._verify_claims(
        contract=QueryContract(
            clean_query="test query",
            interaction_mode="research",
            relation="general_question",
            targets=["test"],
            requested_fields=["summary"],
            required_modalities=["page_text"],
        ),
        plan=_make_plan(),
        claims=claims,
        papers=papers,
        evidence=evidence,
    )

    # issubset({doc-real}, {doc-real}) is True — passes the audit (gets past deterministic check)
    # The schema verifier may return clarify for other reasons, but unsupported_claims should be empty
    assert not report.unsupported_claims


def test_claim_verifier_rejects_paper_ids_not_in_papers() -> None:
    """Claims with paper_ids not in the actual papers set should be caught."""
    verifier = _FakeClaimVerifier()
    claims = [_claim("summary", evidence_ids=["doc-real"], paper_ids=["fake-paper-id"])]
    papers = [_paper(paper_id="P1", title="Paper One")]
    evidence = [_evidence("doc-real", paper_id="P1", title="Paper One")]

    report = verifier._verify_claims(
        contract=QueryContract(
            clean_query="test query",
            interaction_mode="research",
            relation="general_question",
            targets=["test"],
            requested_fields=["summary"],
            required_modalities=["page_text"],
        ),
        plan=_make_plan(),
        claims=claims,
        papers=papers,
        evidence=evidence,
    )

    # paper_ids not in real_doc_ids set — the deterministic audit should catch the orphan
    # If the schema verifier catches it first, unsupported_claims may not be set but status should be clarify
    assert report.status == "clarify" or report.unsupported_claims


# ── P0-6: Rule-based compound split ────────────────────────────────────

def test_rule_based_split_detects_comparison_pattern() -> None:
    contracts = _rule_based_compound_split(
        clean_query="比较 DPO 和 PPO 的目标函数哪个更好",
    )
    assert len(contracts) >= 2
    relations = {c.relation for c in contracts}
    assert "comparison_synthesis" in relations
    # All contracts should be marked as rule-based
    for c in contracts:
        assert "rule_based_compound_split" in c.notes


def test_rule_based_split_returns_empty_for_single_target() -> None:
    contracts = _rule_based_compound_split(
        clean_query="DPO 是什么",
    )
    assert len(contracts) == 0


def test_rule_based_split_returns_empty_when_no_connectors() -> None:
    contracts = _rule_based_compound_split(
        clean_query="总结这篇论文的核心结论",
    )
    assert len(contracts) == 0


def test_rule_based_split_detects_and_connector() -> None:
    contracts = _rule_based_compound_split(
        clean_query="RLHF 和 DPO 分别怎么训练的",
    )
    assert len(contracts) >= 2


def test_rule_based_split_handles_english_targets() -> None:
    contracts = _rule_based_compound_split(
        clean_query="compare PPO and DPO training objectives",
    )
    assert len(contracts) >= 2


# ── P0-8: best_effort does not pollute target_bindings ─────────────────

def test_remember_research_outcome_skips_bindings_for_best_effort() -> None:
    session = SessionContext(session_id="test-best-effort", working_memory={
        "target_bindings": {"existing_target": {"paper_id": "old", "title": "Old Paper"}}
    })
    contract = QueryContract(
        clean_query="test query",
        interaction_mode="research",
        relation="general_question",
        targets=["Test Target"],
        requested_fields=["summary"],
        required_modalities=["page_text"],
    )
    verification = VerificationReport(
        status="pass",
        original_status="best_effort",
        recommended_action="best_effort_after_clarification_limit",
    )

    remember_research_outcome(
        session=session,
        contract=contract,
        answer="weak best-effort answer",
        claims=[_claim("summary", evidence_ids=["doc-real"], paper_ids=["P1"])],
        papers=[_paper("P1", "Paper One")],
        evidence=[_evidence("doc-real", paper_id="P1", title="Paper One")],
        citations=[_citation("Paper One", "P1")],
        candidate_lookup=_candidate_lookup,
        verification=verification,
    )

    # Existing bindings should be preserved (best_effort does not overwrite)
    assert "existing_target" in session.working_memory["target_bindings"]
    # The new target should NOT appear in target_bindings
    normalized_key = "test target"
    assert normalized_key not in session.working_memory["target_bindings"]
    # But it should appear in the temp best_effort bindings
    assert normalized_key in session.working_memory.get("_temp_best_effort_bindings", {})


def test_remember_research_outcome_writes_bindings_for_normal_pass() -> None:
    session = SessionContext(session_id="test-normal-pass")
    contract = QueryContract(
        clean_query="test query",
        interaction_mode="research",
        relation="general_question",
        targets=["Test Target"],
        requested_fields=["summary"],
        required_modalities=["page_text"],
    )
    verification = VerificationReport(status="pass")  # no original_status

    remember_research_outcome(
        session=session,
        contract=contract,
        answer="solid answer",
        claims=[_claim("summary", evidence_ids=["doc-real"], paper_ids=["P1"])],
        papers=[_paper("P1", "Paper One")],
        evidence=[_evidence("doc-real", paper_id="P1", title="Paper One")],
        citations=[_citation("Paper One", "P1")],
        candidate_lookup=_candidate_lookup,
        verification=verification,
    )

    normalized_key = "test target"
    assert normalized_key in session.working_memory.get("target_bindings", {})
    assert session.working_memory["target_bindings"][normalized_key]["paper_id"] == "P1"
    assert "created_turn_index" in session.working_memory["target_bindings"][normalized_key]


def test_verification_report_original_status_field() -> None:
    report = VerificationReport(status="pass")
    assert report.original_status == ""

    best_effort_report = VerificationReport(
        status="pass",
        recommended_action="best_effort_after_clarification_limit",
        original_status="best_effort",
    )
    assert best_effort_report.original_status == "best_effort"
