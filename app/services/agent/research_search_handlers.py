from __future__ import annotations

from typing import Any, Callable

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract, ResearchPlan, SessionContext
from app.services.agent.runtime_helpers import run_agent_paper_search, screen_agent_papers, search_agent_evidence
from app.services.agent.tool_events import (
    emit_agent_tool_call as emit_agent_tool_call_event,
    record_agent_observation as record_agent_observation_event,
)
from app.services.claims.paper_summary import paper_summary_text
from app.services.tools.registry_helpers import tool_input_from_state
from app.services.retrieval.web_evidence import collect_web_evidence, search_agent_web_evidence


def _trim_paper_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Trim heavy metadata fields to keep SSE events lightweight."""
    trimmed = {k: v for k, v in meta.items() if k != "vector"}  # strip embedding vector
    for key in ("body_acronyms",):
        value = str(trimmed.get(key, ""))
        if len(value) > 240:
            trimmed[key] = value[:240] + "..."
    for key in ("paper_card_text", "generated_summary", "abstract_note"):
        value = str(trimmed.get(key, ""))
        if len(value) > 400:
            trimmed[key] = value[:400] + "..."
    authors = str(trimmed.get("authors", ""))
    if len(authors) > 200:
        first_three = authors.split(",")[:3]
        trimmed["authors"] = ",".join(a.strip() for a in first_three) + " 等"
    return trimmed


def _trim_candidate_for_sse(item: CandidatePaper) -> dict[str, Any]:
    payload = item.model_dump()
    payload["metadata"] = _trim_paper_metadata(payload.get("metadata", {}))
    return payload


def _trim_evidence_for_sse(item: EvidenceBlock) -> dict[str, Any]:
    payload = item.model_dump()
    snippet = str(payload.get("snippet", ""))
    if len(snippet) > 360:
        payload["snippet"] = snippet[:360] + "..."
    payload["metadata"] = _trim_paper_metadata(payload.get("metadata", {}))
    return payload


EmitFn = Callable[[str, dict[str, Any]], None]


def screen_agent_candidate_papers(
    *,
    agent: Any,
    contract: QueryContract,
    plan: ResearchPlan,
    candidate_papers: list[CandidatePaper],
    excluded_titles: set[str],
) -> tuple[list[CandidatePaper], list[EvidenceBlock] | None]:
    return screen_agent_papers(
        contract=contract,
        plan=plan,
        candidate_papers=candidate_papers,
        excluded_titles=excluded_titles,
        paper_lookup=agent._candidate_from_paper_id,
        paper_summary_text=lambda paper_id: paper_summary_text(
            paper_id,
            paper_doc_lookup=agent.retriever.paper_doc_by_id,
        ),
        prefer_identity_matching_papers=lambda candidates, targets: [
            item for item in candidates if agent._paper_identity_matches_targets(paper=item, targets=targets)
        ],
        search_entity_evidence=lambda query, search_contract, limit: agent.retriever.search_entity_evidence(
            query=query,
            contract=search_contract,
            limit=limit,
        ),
        ground_entity_papers=lambda candidates, evidence, limit: agent._ground_entity_papers(
            candidates=candidates,
            evidence=evidence,
            limit=limit,
        ),
    )


def agent_search_papers(
    *,
    agent: Any,
    state: dict[str, Any],
    session: SessionContext,
    emit: EmitFn,
    execution_steps: list[dict[str, Any]],
    tool_input: dict[str, Any] | None = None,
) -> None:
    contract: QueryContract = state["contract"]
    plan: ResearchPlan = state["plan"]
    tool_input = dict(tool_input or {}) or tool_input_from_state(state, "search_corpus")
    excluded_titles: set[str] = state["excluded_titles"]
    active = session.effective_active_research()
    result = run_agent_paper_search(
        contract=contract,
        plan=plan,
        tool_input=tool_input,
        active_targets=list(active.targets),
        excluded_titles=excluded_titles,
        search_papers=lambda query, search_contract, limit: agent.retriever.search_papers(
            query=query,
            contract=search_contract,
            limit=limit,
        ),
        paper_lookup=agent._candidate_from_paper_id,
        screen_papers=lambda search_contract, search_plan, candidates, search_excluded_titles: screen_agent_candidate_papers(
            agent=agent,
            contract=search_contract,
            plan=search_plan,
            candidate_papers=candidates,
            excluded_titles=search_excluded_titles,
        ),
    )
    emit_agent_tool_call_event(emit=emit, tool="search_corpus", arguments=result.tool_call_arguments)
    state["contract"] = result.contract
    candidate_papers = result.candidate_papers
    screened_papers = result.screened_papers
    state["candidate_papers"] = candidate_papers
    state["screened_papers"] = screened_papers
    state["precomputed_evidence"] = result.precomputed_evidence
    emit("candidate_papers", {"count": len(candidate_papers), "items": [_trim_candidate_for_sse(item) for item in candidate_papers]})
    emit("screened_papers", {"count": len(screened_papers), "items": [_trim_candidate_for_sse(item) for item in screened_papers]})
    record_agent_observation_event(
        emit=emit,
        execution_steps=execution_steps,
        tool="search_corpus",
        summary=result.observation_summary,
        payload=result.observation_payload,
    )


def agent_search_evidence(
    *,
    agent: Any,
    state: dict[str, Any],
    emit: EmitFn,
    execution_steps: list[dict[str, Any]],
    tool_input: dict[str, Any] | None = None,
) -> None:
    contract: QueryContract = state["contract"]
    plan: ResearchPlan = state["plan"]
    tool_input = dict(tool_input or {}) or tool_input_from_state(state, "search_corpus")
    screened_papers: list[CandidatePaper] = state["screened_papers"]
    excluded_titles: set[str] = state["excluded_titles"]
    result = search_agent_evidence(
        contract=contract,
        plan=plan,
        tool_input=tool_input,
        screened_papers=screened_papers,
        precomputed_evidence=state.get("precomputed_evidence"),
        excluded_titles=excluded_titles,
        search_concept_evidence=lambda query, search_contract, paper_ids, limit: agent.retriever.search_concept_evidence(
            query=query,
            contract=search_contract,
            paper_ids=paper_ids,
            limit=limit,
        ),
        expand_evidence=lambda paper_ids, query, search_contract, limit: agent.retriever.expand_evidence(
            paper_ids=paper_ids,
            query=query,
            contract=search_contract,
            limit=limit,
        ),
    )
    emit_agent_tool_call_event(emit=emit, tool="search_corpus", arguments=result.tool_call_arguments)
    evidence = result.evidence
    state["evidence"] = evidence
    emit("evidence", {"count": len(evidence), "items": [_trim_evidence_for_sse(item) for item in evidence]})
    record_agent_observation_event(
        emit=emit,
        execution_steps=execution_steps,
        tool="search_corpus",
        summary=result.observation_summary,
        payload=result.observation_payload,
    )


def agent_web_search(
    *,
    agent: Any,
    state: dict[str, Any],
    web_enabled: bool,
    max_web_results: int,
    emit: EmitFn,
    execution_steps: list[dict[str, Any]],
    tool_input: dict[str, Any] | None = None,
) -> None:
    contract: QueryContract = state["contract"]
    tool_input = dict(tool_input or {}) or tool_input_from_state(state, "web_search")
    result = search_agent_web_evidence(
        contract=contract,
        existing_evidence=state["evidence"],
        tool_input=tool_input,
        web_enabled=web_enabled,
        max_web_results=max_web_results,
        collect=lambda search_contract, enabled, limit, query: collect_web_evidence(
            web_search=agent.web_search,
            contract=search_contract,
            use_web_search=enabled,
            max_web_results=limit,
            query_override=query,
        ),
    )
    emit_agent_tool_call_event(emit=emit, tool="web_search", arguments=result.tool_call_arguments)
    web_evidence = result.web_evidence
    state["web_evidence"] = web_evidence
    if web_evidence:
        state["evidence"] = result.merged_evidence
        emit("web_search", {"count": len(web_evidence), "items": [_trim_evidence_for_sse(item) for item in web_evidence]})
        emit("evidence", {"count": len(state["evidence"]), "items": [_trim_evidence_for_sse(item) for item in state["evidence"]]})
    record_agent_observation_event(
        emit=emit,
        execution_steps=execution_steps,
        tool="web_search",
        summary=result.observation_summary,
        payload=result.observation_payload,
    )
