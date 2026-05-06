from __future__ import annotations

from app.domain.models import AssistantCitation, CandidatePaper, Claim, EvidenceBlock, QueryContract, SessionContext
from app.services.memory.research import remember_compound_outcome, remember_research_outcome


def _paper(paper_id: str, title: str = "DPO Paper") -> CandidatePaper:
    return CandidatePaper(paper_id=paper_id, title=title, year="2025")


def test_remember_research_outcome_stores_target_binding_from_claim() -> None:
    session = SessionContext(session_id="research-memory")
    contract = QueryContract(
        clean_query="DPO 的公式是什么？",
        relation="formula_lookup",
        targets=["DPO"],
        requested_fields=["formula"],
        required_modalities=["page_text"],
    )
    paper = _paper("paper-1")
    evidence = [
        EvidenceBlock(
            doc_id="block-1",
            paper_id="paper-1",
            title=paper.title,
            file_path="/tmp/dpo.pdf",
            page=2,
            block_type="page_text",
            snippet="DPO objective.",
        )
    ]

    remember_research_outcome(
        session=session,
        contract=contract,
        answer="DPO objective",
        claims=[Claim(claim_type="formula", entity="DPO", paper_ids=["paper-1"], evidence_ids=["block-1"])],
        papers=[paper],
        evidence=evidence,
        citations=[],
        candidate_lookup=lambda _: None,
    )

    binding = session.working_memory["target_bindings"]["dpo"]
    assert binding["paper_id"] == "paper-1"
    assert binding["evidence_ids"] == ["block-1"]
    assert binding["requested_fields"] == ["formula"]
    assert session.working_memory["last_successful_research"]["titles"] == ["DPO Paper"]


def test_remember_research_outcome_uses_candidate_lookup_for_citation_fallback() -> None:
    session = SessionContext(session_id="research-memory-citation")
    contract = QueryContract(clean_query="PPO 是什么？", relation="entity_definition", targets=["PPO"])
    citation = AssistantCitation(
        doc_id="paper::ppo-paper",
        paper_id="ppo-paper",
        title="PPO Paper",
        file_path="/tmp/ppo.pdf",
        page=1,
        snippet="PPO paper",
    )

    remember_research_outcome(
        session=session,
        contract=contract,
        answer="PPO answer",
        claims=[],
        papers=[],
        evidence=[],
        citations=[citation],
        candidate_lookup=lambda paper_id: _paper(paper_id, "PPO Paper") if paper_id == "ppo-paper" else None,
    )

    assert session.working_memory["target_bindings"]["ppo"]["paper_id"] == "ppo-paper"
    assert session.working_memory["target_bindings"]["ppo"]["support_titles"] == ["PPO Paper"]


def test_remember_research_outcome_stores_followup_relationship_memory() -> None:
    session = SessionContext(session_id="research-memory-followup")
    contract = QueryContract(clean_query="B 是否是 A 的后续？", relation="followup_research", targets=["A"])
    claim = Claim(
        claim_type="followup_research",
        entity="A",
        structured_data={
            "selected_candidate_title": "Candidate",
            "followup_titles": [{"title": "Candidate", "paper_id": "candidate-1"}],
        },
    )

    remember_research_outcome(
        session=session,
        contract=contract,
        answer="Candidate is related",
        claims=[claim],
        papers=[],
        evidence=[],
        citations=[],
        candidate_lookup=lambda _: None,
    )

    assert session.working_memory["last_followup_relationship"]["candidate_paper_id"] == "candidate-1"


def test_remember_compound_outcome_stores_subtasks_and_target_bindings() -> None:
    session = SessionContext(session_id="compound-memory")
    papers = {"dpo-paper": _paper("dpo-paper", "DPO Paper")}
    contract = QueryContract(
        clean_query="DPO 公式",
        relation="formula_lookup",
        targets=["DPO"],
        requested_fields=["formula"],
    )
    citation = AssistantCitation(
        doc_id="paper::dpo-paper",
        paper_id="dpo-paper",
        title="DPO Paper",
        file_path="/tmp/dpo.pdf",
        page=1,
        snippet="DPO citation",
    )

    remember_compound_outcome(
        session=session,
        clean_query="DPO 和 PPO 公式分别是什么？",
        subtask_results=[
            {
                "contract": contract,
                "claims": [Claim(claim_type="formula", entity="DPO", paper_ids=["dpo-paper"])],
                "evidence": [],
                "citations": [citation],
                "answer": "DPO answer",
            }
        ],
        candidate_lookup=lambda paper_id: papers.get(paper_id),
    )

    assert session.working_memory["target_bindings"]["dpo"]["paper_id"] == "dpo-paper"
    compound = session.working_memory["last_compound_query"]
    assert compound["query"] == "DPO 和 PPO 公式分别是什么？"
    assert compound["subtasks"][0]["relation"] == "formula_lookup"
    assert compound["subtasks"][0]["citation_titles"] == ["DPO Paper"]
