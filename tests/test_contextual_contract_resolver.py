from __future__ import annotations

from types import SimpleNamespace

from app.domain.models import ActiveResearch, CandidatePaper, QueryContract, SessionContext
from app.services.contracts.contextual_resolver import resolve_contextual_research_contract


def test_resolve_contextual_research_contract_binds_active_paper_reference() -> None:
    paper = CandidatePaper(paper_id="p1", title="DPO Paper")
    session = SessionContext(
        session_id="s1",
        active_research=ActiveResearch(titles=["DPO Paper"], targets=["DPO"], relation="entity_definition"),
    )
    contract = QueryContract(clean_query="这篇论文的贡献是什么？", interaction_mode="research", targets=["DPO"])

    resolved = resolve_contextual_research_contract(
        contract=contract,
        session=session,
        paper_from_query_hint=lambda _: paper,
        block_documents_for_paper=lambda _paper_id, _limit: [],
    )

    assert resolved.continuation_mode == "followup"
    assert "active_paper_reference" in resolved.notes
    assert "selected_paper_id=p1" in resolved.notes


def test_resolve_contextual_research_contract_binds_formula_to_active_paper() -> None:
    paper = CandidatePaper(paper_id="p1", title="AlignX Paper", metadata={"aliases": "AlignX"})
    session = SessionContext(
        session_id="s1",
        active_research=ActiveResearch(titles=["AlignX Paper"], targets=["PBA"], relation="paper_summary_results"),
    )
    contract = QueryContract(
        clean_query="这篇论文里 PBA 的公式是什么？",
        interaction_mode="research",
        targets=["PBA"],
        requested_fields=["formula"],
        required_modalities=["page_text"],
    )

    resolved = resolve_contextual_research_contract(
        contract=contract,
        session=session,
        paper_from_query_hint=lambda _: paper,
        block_documents_for_paper=lambda _paper_id, _limit: [
            SimpleNamespace(page_content="PBA objective formula and loss", metadata={"formula_hint": 1})
        ],
    )

    assert resolved.relation == "formula_lookup"
    assert resolved.clean_query == "PBA 的公式是什么？限定在论文《AlignX Paper》中查找。"
    assert "formula_contextual_paper_binding" in resolved.notes
    assert "selected_paper_id=p1" in resolved.notes
