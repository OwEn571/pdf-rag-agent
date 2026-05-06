from __future__ import annotations

from app.domain.models import QueryContract
from app.services.planning.research import ResearchPlanContext
from app.services.planning.query_shaping import (
    evidence_query_text,
    evidence_query_text_from_context,
    extract_targets,
    is_short_acronym,
    loss_notation_target_aliases,
    matches_target,
    paper_query_text,
    paper_query_text_from_context,
    query_target_candidates,
    should_use_concept_evidence,
    should_use_concept_evidence_from_context,
    should_use_web_search,
)


def test_query_shaping_extracts_quoted_and_identifier_targets() -> None:
    assert extract_targets('解释 "Direct Preference Optimization" 和 DPO-v2 的区别') == [
        "Direct Preference Optimization",
        "DPO",
        "DPO-v2",
    ]


def test_query_shaping_target_candidates_filter_stopwords_and_titlecase_entities() -> None:
    assert query_target_candidates("Which paper first introduced AlignX and PBA?") == ["AlignX", "PBA"]
    assert query_target_candidates("what is DPO") == ["DPO"]
    assert query_target_candidates("first proposed this paper") == []


def test_query_shaping_loss_notation_target_aliases_cover_loss_notation() -> None:
    assert loss_notation_target_aliases(["PBA", "AlignX"]) == [
        "L_PBA",
        "LPBA",
        "L_{PBA}",
        "L_{\\mathrm{PBA}}",
    ]


def test_query_shaping_prefers_target_for_single_entity_definition() -> None:
    contract = QueryContract(
        clean_query="PBA 是什么机制？",
        relation="entity_definition",
        targets=["PBA"],
        requested_fields=["entity_type", "mechanism"],
    )

    assert paper_query_text(contract) == "PBA"
    shaped = evidence_query_text(contract)
    assert shaped.startswith("PBA PBA 是什么机制？")
    assert "objective" in shaped
    assert "奖励" in shaped
    assert not should_use_concept_evidence(contract)


def test_query_shaping_expands_concept_definition_queries() -> None:
    contract = QueryContract(
        clean_query="RAG 的定义和例子是什么？",
        relation="concept_definition",
        targets=["RAG"],
        requested_fields=["definition", "examples"],
    )

    assert paper_query_text(contract) == "RAG"
    assert "example" in evidence_query_text(contract)
    assert should_use_concept_evidence(contract)


def test_query_shaping_accepts_typed_research_plan_context() -> None:
    context = ResearchPlanContext(
        clean_query="RAG 的定义和例子是什么？",
        targets=("RAG",),
        requested_fields=("definition", "examples"),
        goals=frozenset({"definition", "mechanism", "examples"}),
        required_modalities=("page_text",),
    )

    assert paper_query_text_from_context(context) == "RAG"
    assert "example" in evidence_query_text_from_context(context)
    assert should_use_concept_evidence_from_context(context)


def test_query_shaping_web_flag_and_acronym_detection() -> None:
    latest_contract = QueryContract(
        clean_query="最新的多模态 RAG 论文有哪些？",
        relation="paper_recommendation",
    )
    local_contract = QueryContract(clean_query="这篇论文的主要结论是什么？", relation="paper_summary_results")

    assert is_short_acronym("DPO")
    assert is_short_acronym("PBA-2")
    assert not is_short_acronym("Direct Preference Optimization")
    assert should_use_web_search(use_web_search=False, contract=latest_contract)
    assert should_use_web_search(use_web_search=True, contract=local_contract)
    assert not should_use_web_search(use_web_search=False, contract=local_contract)


def test_query_shaping_matches_short_targets_with_boundaries_and_loss_aliases() -> None:
    assert matches_target("Direct Preference Optimization (DPO) improves preference tuning.", "DPO")
    assert not matches_target("ADPO is a different token.", "DPO")
    assert matches_target(r"The objective includes L_{\mathrm{DPO}} terms.", "DPO")
    assert matches_target("Direct Preference Optimization", "Direct Preference")
