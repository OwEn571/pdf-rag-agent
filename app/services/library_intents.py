from __future__ import annotations

from app.domain.models import QueryContract, SessionContext


def normalized_query_text(query: str) -> tuple[str, str]:
    normalized = " ".join(str(query or "").lower().split())
    return normalized, normalized.replace(" ", "")


def has_library_scope(query: str) -> bool:
    normalized, compact = normalized_query_text(query)
    scope_markers = ["知识库", "论文库", "库中", "库里", "zotero", "本地论文", "我的论文", "你有的论文"]
    return any(marker in normalized or marker in compact for marker in scope_markers)


def is_library_status_query(query: str) -> bool:
    normalized, compact = normalized_query_text(query)
    scope_markers = ["论文", "paper", "papers", "知识库", "库里", "zotero", "pdf"]
    list_markers = ["有哪些论文", "有哪些文章", "论文列表", "文章列表", "列出论文", "列出文章", "list papers"]
    has_scope = any(marker in normalized or marker in compact for marker in scope_markers)
    asks_list = any(marker in normalized or marker in compact for marker in list_markers)
    return has_scope and (is_library_count_query(query) or asks_list)


def is_library_count_query(query: str) -> bool:
    normalized, compact = normalized_query_text(query)
    count_markers = [
        "多少",
        "几篇",
        "一共",
        "总共",
        "总计",
        "数量",
        "规模",
        "count",
        "how many",
        "total",
    ]
    return any(marker in normalized or marker in compact for marker in count_markers)


def is_library_recommendation_query(query: str) -> bool:
    normalized, compact = normalized_query_text(query)
    markers = [
        "最值得",
        "值得一读",
        "值得读",
        "值得一看",
        "值得看",
        "再推荐",
        "换一篇",
        "推荐",
        "哪篇",
        "哪几篇",
        "must read",
        "worth reading",
        "recommend",
    ]
    return any(marker in normalized or marker in compact for marker in markers)


def is_scoped_library_recommendation_query(query: str) -> bool:
    return is_library_recommendation_query(query) and has_library_scope(query)


def library_status_contract(clean_query: str) -> QueryContract:
    return QueryContract(
        clean_query=clean_query,
        interaction_mode="conversation",
        relation="library_status",
        targets=[],
        requested_fields=[],
        required_modalities=[],
        answer_shape="bullets",
        precision_requirement="exact",
        continuation_mode="fresh",
        notes=["self_knowledge", "dynamic_library_stats"],
    )


def library_recommendation_contract(clean_query: str) -> QueryContract:
    return QueryContract(
        clean_query=clean_query,
        interaction_mode="conversation",
        relation="library_recommendation",
        targets=[],
        requested_fields=[],
        required_modalities=[],
        answer_shape="bullets",
        precision_requirement="normal",
        continuation_mode="fresh",
        notes=["self_knowledge", "dynamic_library_recommendation"],
    )


def library_query_prefers_previous_candidates(query: str) -> bool:
    normalized, compact = normalized_query_text(query)
    return not any(marker in normalized or marker in compact for marker in ["全库", "所有", "全部"])


def is_citation_query(query: str) -> bool:
    normalized, compact = normalized_query_text(query)
    return any(marker in normalized or marker in compact for marker in ["引用数", "引用量", "被引", "citation", "citations"])


def is_citation_ranking_query(query: str) -> bool:
    normalized, compact = normalized_query_text(query)
    citation_markers = [
        "引用数",
        "引用量",
        "被引",
        "按引用",
        "citation",
        "citations",
        "cited by",
        "citation count",
    ]
    rank_markers = [
        "按",
        "排序",
        "排行",
        "排名",
        "最高",
        "最多",
        "哪篇",
        "推荐",
        "最值得",
        "rank",
        "sort",
        "most cited",
    ]
    has_citation = any(marker in normalized or marker in compact for marker in citation_markers)
    has_rank = any(marker in normalized or marker in compact for marker in rank_markers)
    return has_citation and (has_rank or len(compact) <= 20)


def citation_ranking_has_library_context(*, clean_query: str, session: SessionContext) -> bool:
    if has_library_scope(clean_query):
        return True
    if is_library_recommendation_query(clean_query) or is_library_status_query(clean_query):
        return True
    for turn in reversed(session.turns[-4:]):
        if turn.relation == "library_recommendation":
            return True
        if turn.relation == "compound_query" and (
            "library_recommendation" in turn.requested_fields
            or "默认推荐" in turn.answer
            or "最值得" in turn.query
            or "推荐" in turn.answer
        ):
            return True
        if turn.relation == "library_citation_ranking":
            return True
    return False
