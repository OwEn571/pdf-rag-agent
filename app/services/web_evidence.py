from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.domain.models import Claim, EvidenceBlock, QueryContract
from app.services.intent_marker_matching import MarkerProfile, query_matches_any
from app.services.research_planning import research_plan_goals


WEB_RESEARCH_DOMAINS = [
    "arxiv.org",
    "openreview.net",
    "semanticscholar.org",
    "aclanthology.org",
    "proceedings.mlr.press",
    "papers.nips.cc",
    "thecvf.com",
]

WEB_EVIDENCE_MARKERS: dict[str, MarkerProfile] = {
    "news_topic": ("新闻", "news", "today", "昨天", "今天"),
    "research_domain": ("论文", "paper", "arxiv", "研究", "publication"),
}

WebEvidenceCollector = Callable[[QueryContract, bool, int, str], list[EvidenceBlock]]
WebClaimBuilder = Callable[[QueryContract, list[EvidenceBlock]], Claim]
ClaimSolver = Callable[[], list[Claim]]


@dataclass(frozen=True)
class AgentWebEvidenceResult:
    query: str
    max_results: int
    web_evidence: list[EvidenceBlock]
    merged_evidence: list[EvidenceBlock]
    tool_call_arguments: dict[str, Any]
    observation_summary: str
    observation_payload: dict[str, Any]


def coerce_web_result_limit(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(20, parsed))


def web_query_text(contract: QueryContract) -> str:
    target_text = " ".join(contract.targets).strip()
    query = contract.clean_query.strip()
    goals = research_plan_goals(contract)
    parts = [query]
    if target_text and target_text.lower() not in query.lower():
        parts.insert(0, target_text)
    if goals & {"recommended_papers", "followup_papers", "summary", "results", "answer", "general_answer"}:
        parts.extend(["paper", "arXiv", "publication"])
    return " ".join(dict.fromkeys(part for part in parts if part)).strip()


def web_search_topic(query: str) -> str:
    lowered = str(query or "").lower()
    if query_matches_any(lowered, "", WEB_EVIDENCE_MARKERS["news_topic"]):
        return "news"
    return "general"


def web_include_domains(contract: QueryContract) -> list[str]:
    query = contract.clean_query.lower()
    goals = research_plan_goals(contract)
    if goals & {"recommended_papers", "followup_papers", "summary", "results", "answer", "general_answer"}:
        if query_matches_any(query, "", WEB_EVIDENCE_MARKERS["research_domain"]):
            return list(WEB_RESEARCH_DOMAINS)
    return []


def merge_evidence(local_evidence: list[EvidenceBlock], web_evidence: list[EvidenceBlock]) -> list[EvidenceBlock]:
    merged: list[EvidenceBlock] = []
    seen: set[str] = set()
    for item in [*local_evidence, *web_evidence]:
        if item.doc_id in seen:
            continue
        seen.add(item.doc_id)
        merged.append(item)
    return merged


def collect_web_evidence(
    *,
    web_search: Any,
    contract: QueryContract,
    use_web_search: bool,
    max_web_results: int,
    query_override: str = "",
) -> list[EvidenceBlock]:
    if not use_web_search or not getattr(web_search, "is_configured", False):
        return []
    search_query = str(query_override or "").strip() or web_query_text(contract)
    return web_search.search(
        query=search_query,
        max_results=max_web_results,
        topic=web_search_topic(search_query or contract.clean_query),
        include_domains=web_include_domains(contract),
    )


def search_agent_web_evidence(
    *,
    contract: QueryContract,
    existing_evidence: list[EvidenceBlock],
    tool_input: dict[str, Any],
    web_enabled: bool,
    max_web_results: int,
    collect: WebEvidenceCollector,
) -> AgentWebEvidenceResult:
    web_query = str(tool_input.get("query", "") or "").strip() or web_query_text(contract)
    result_limit = coerce_web_result_limit(
        tool_input.get("max_results", max_web_results),
        default=max_web_results,
    )
    web_evidence = collect(contract, web_enabled, result_limit, web_query)
    merged_evidence = merge_evidence(existing_evidence, web_evidence) if web_evidence else existing_evidence
    return AgentWebEvidenceResult(
        query=web_query,
        max_results=result_limit,
        web_evidence=web_evidence,
        merged_evidence=merged_evidence,
        tool_call_arguments={
            "query": web_query,
            "max_results": result_limit,
            "enabled": web_enabled,
        },
        observation_summary=f"web_evidence={len(web_evidence)}",
        observation_payload={"web_evidence_count": len(web_evidence)},
    )


def should_add_web_claim(*, contract: QueryContract, claims: list[Claim], explicit_web: bool) -> bool:
    if explicit_web:
        return True
    goals = research_plan_goals(contract)
    if contract.allow_web_search and goals & {"answer", "general_answer", "recommended_papers", "followup_papers"}:
        return True
    return not claims


def claims_with_web_research_claim(
    *,
    contract: QueryContract,
    claims: list[Claim],
    web_evidence: list[EvidenceBlock],
    explicit_web: bool,
    build_claim: WebClaimBuilder,
) -> list[Claim]:
    if web_evidence and should_add_web_claim(
        contract=contract,
        claims=claims,
        explicit_web=explicit_web,
    ):
        return [*claims, build_claim(contract, web_evidence)]
    return claims


def solve_claims_with_web_research(
    *,
    contract: QueryContract,
    web_evidence: list[EvidenceBlock],
    explicit_web: bool,
    solve_claims: ClaimSolver,
    build_claim: WebClaimBuilder,
) -> list[Claim]:
    claims = solve_claims()
    return claims_with_web_research_claim(
        contract=contract,
        claims=claims,
        web_evidence=web_evidence,
        explicit_web=explicit_web,
        build_claim=build_claim,
    )


def build_web_research_claim(*, contract: QueryContract, web_evidence: list[EvidenceBlock]) -> Claim:
    results = [
        {
            "title": item.title,
            "url": item.file_path,
            "snippet": item.snippet[:500],
            "score": item.score,
        }
        for item in web_evidence
    ]
    return Claim(
        claim_type="web_research",
        entity=" ".join(contract.targets) if contract.targets else contract.clean_query,
        value="web_search_results",
        structured_data={"web_results": results},
        evidence_ids=[item.doc_id for item in web_evidence],
        paper_ids=[item.paper_id for item in web_evidence],
        confidence=0.72,
    )
