from __future__ import annotations

from app.domain.models import QueryContract, SessionContext, SessionTurn
from app.services.library_intents import (
    citation_ranking_has_library_context,
    is_citation_ranking_query,
    is_library_count_query,
    is_library_status_query,
    is_scoped_library_recommendation_query,
    library_recommendation_contract,
    library_query_prefers_previous_candidates,
    library_status_contract,
)


def test_library_status_and_recommendation_intents_share_classifier() -> None:
    assert is_library_status_query("你的论文库里有多少论文？")
    assert is_library_count_query("how many papers in zotero?")
    assert is_scoped_library_recommendation_query("你的知识库里哪篇最值得一看")
    assert not is_scoped_library_recommendation_query("再推荐一篇别的")
    assert library_query_prefers_previous_candidates("再推荐一篇")
    assert not library_query_prefers_previous_candidates("全库里推荐一篇")


def test_citation_ranking_uses_recent_library_context() -> None:
    session = SessionContext(session_id="demo")
    session.turns.append(
        SessionTurn.from_contract(
            query="知识库里哪篇最值得看？",
            answer="默认推荐 A Survey on LLM-as-a-Judge。",
            contract=QueryContract(clean_query="知识库里哪篇最值得看？", relation="library_recommendation"),
            interaction_mode="conversation",
        )
    )

    assert is_citation_ranking_query("按引用数排序")
    assert citation_ranking_has_library_context(clean_query="按引用数排序", session=session)


def test_library_status_contract_uses_dynamic_stats_notes() -> None:
    contract = library_status_contract("你的论文库里有多少论文？")

    assert contract.interaction_mode == "conversation"
    assert contract.relation == "library_status"
    assert contract.answer_shape == "bullets"
    assert contract.precision_requirement == "exact"
    assert contract.notes == ["self_knowledge", "dynamic_library_stats"]


def test_library_recommendation_contract_uses_dynamic_recommendation_notes() -> None:
    contract = library_recommendation_contract("你的知识库里哪篇最值得一看？")

    assert contract.interaction_mode == "conversation"
    assert contract.relation == "library_recommendation"
    assert contract.answer_shape == "bullets"
    assert contract.precision_requirement == "normal"
    assert contract.notes == ["self_knowledge", "dynamic_library_recommendation"]
