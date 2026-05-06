from __future__ import annotations

from typing import Any

from app.domain.models import CandidatePaper, QueryContract, SessionContext
from app.services.clarification.intents import contract_from_pending_clarification, selected_clarification_paper_id
from app.services.contracts.normalization import normalize_contract_targets
from app.services.contracts.contextual_helpers import paper_from_query_hint
from app.services.contracts.contextual_resolver import resolve_contextual_research_contract
from app.services.contracts.conversation_helpers import normalize_conversation_tool_contract
from app.services.contracts.conversation_memory import apply_conversation_memory_to_contract
from app.services.contracts.followup_relationship import (
    inherit_followup_relationship_contract,
    normalize_followup_direction_contract,
)
from app.services.intents.router import query_contract_from_router_decision, router_miss_clarification_contract
from app.services.planning.query_shaping import extract_targets


def extract_agent_query_contract(
    *,
    agent: Any,
    query: str,
    session: SessionContext,
    clarification_choice: dict[str, Any] | None = None,
) -> QueryContract:
    custom_extractor = getattr(agent, "query_contract_extractor", None)
    if callable(custom_extractor):
        return custom_extractor(
            query=query,
            session=session,
            clarification_choice=clarification_choice,
        )

    clean_query = " ".join(query.strip().split())
    clarified_contract = contract_from_pending_clarification(
        clean_query=clean_query,
        session=session,
        clarification_choice=clarification_choice,
    )
    if clarified_contract is not None:
        return clarified_contract
    targets = extract_targets(clean_query)
    decision = agent.llm_intent_router.route(query=clean_query, session=session)
    contract = query_contract_from_router_decision(
        decision=decision,
        clean_query=clean_query,
        session=session,
        extracted_targets=targets,
        normalize_targets=lambda raw_targets, requested_fields: normalize_contract_targets(
            targets=raw_targets,
            requested_fields=requested_fields,
            canonicalize_targets=agent.retriever.canonicalize_targets,
        ),
        confidence_floor=getattr(getattr(agent, "agent_settings", None), "confidence_floor", 0.6),
    )
    if contract is None:
        contract = router_miss_clarification_contract(clean_query=clean_query)

    def paper_hint_lookup(query_text: str) -> CandidatePaper | None:
        return paper_from_query_hint(
            query_text,
            paper_documents=agent.retriever.paper_documents(),
            candidate_lookup=agent._candidate_from_paper_id,
        )

    contract = normalize_conversation_tool_contract(
        contract=contract,
        clean_query=clean_query,
        session=session,
        paper_from_query_hint=paper_hint_lookup,
    )
    if contract.interaction_mode == "conversation":
        return contract
    refined_contract = agent._refine_followup_contract(contract=contract, session=session)
    refined_contract = resolve_contextual_research_contract(
        contract=refined_contract,
        session=session,
        paper_from_query_hint=paper_hint_lookup,
        block_documents_for_paper=lambda paper_id, limit: agent.retriever.block_documents_for_paper(
            paper_id,
            limit=limit,
        ),
    )
    refined_contract = inherit_followup_relationship_contract(
        contract=refined_contract,
        session=session,
        normalize_targets=lambda targets, requested_fields: normalize_contract_targets(
            targets=targets,
            requested_fields=requested_fields,
            canonicalize_targets=agent.retriever.canonicalize_targets,
        ),
    )
    refined_contract = normalize_followup_direction_contract(
        contract=refined_contract,
        normalize_targets=lambda targets, requested_fields: normalize_contract_targets(
            targets=targets,
            requested_fields=requested_fields,
            canonicalize_targets=agent.retriever.canonicalize_targets,
        ),
    )
    return apply_conversation_memory_to_contract(
        contract=refined_contract,
        session=session,
        selected_clarification_paper_id=selected_clarification_paper_id(refined_contract),
    )
