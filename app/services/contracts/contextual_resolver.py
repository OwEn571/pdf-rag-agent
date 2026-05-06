from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from app.domain.models import CandidatePaper, QueryContract, SessionContext
from app.services.clarification.intents import selected_clarification_paper_id
from app.services.contracts.contextual_helpers import (
    contextual_active_paper_contract,
    formula_answer_correction_contract,
    formula_contextual_paper_contract,
    formula_followup_target,
    formula_location_followup_contract,
    formula_query_allows_paper_context,
    paper_context_supports_formula_target,
    paper_scope_correction_contract,
)
from app.services.contracts.context import contract_has_note
from app.services.intents.followup import (
    is_negative_correction_query,
    looks_like_active_paper_reference,
    looks_like_formula_answer_correction,
    looks_like_formula_location_correction,
    looks_like_paper_scope_correction,
)
from app.services.planning.research import research_plan_context_from_contract

PaperHintLookup = Callable[[str], CandidatePaper | None]
BlockDocumentsForPaper = Callable[[str, int], Iterable[Any]]


def resolve_contextual_research_contract(
    *,
    contract: QueryContract,
    session: SessionContext,
    paper_from_query_hint: PaperHintLookup,
    block_documents_for_paper: BlockDocumentsForPaper,
) -> QueryContract:
    refined = _resolve_formula_answer_correction_contract(
        contract=contract,
        session=session,
        paper_from_query_hint=paper_from_query_hint,
    )
    refined = _resolve_formula_contextual_paper_contract(
        contract=refined,
        session=session,
        paper_from_query_hint=paper_from_query_hint,
        block_documents_for_paper=block_documents_for_paper,
    )
    refined = _resolve_formula_location_followup_contract(
        contract=refined,
        session=session,
        paper_from_query_hint=paper_from_query_hint,
    )
    refined = _resolve_paper_scope_correction_contract(
        contract=refined,
        session=session,
        paper_from_query_hint=paper_from_query_hint,
    )
    return _resolve_contextual_active_paper_contract(
        contract=refined,
        session=session,
        paper_from_query_hint=paper_from_query_hint,
    )


def _resolve_formula_answer_correction_contract(
    *,
    contract: QueryContract,
    session: SessionContext,
    paper_from_query_hint: PaperHintLookup,
) -> QueryContract:
    active = session.effective_active_research()
    active_formula = active.relation == "formula_lookup" or "formula" in {str(field) for field in active.requested_fields}
    if not active_formula or not active.targets:
        return contract
    if not looks_like_formula_answer_correction(contract.clean_query):
        return contract
    title = active.titles[0] if active.titles else ""
    paper = paper_from_query_hint(title) if title else None
    return formula_answer_correction_contract(contract=contract, active=active, paper=paper)


def _resolve_formula_location_followup_contract(
    *,
    contract: QueryContract,
    session: SessionContext,
    paper_from_query_hint: PaperHintLookup,
) -> QueryContract:
    active = session.effective_active_research()
    active_formula = (
        active.relation == "formula_lookup"
        or "formula" in {str(field) for field in active.requested_fields}
    )
    if not active_formula or not active.targets:
        return contract
    if contract.interaction_mode == "conversation":
        return contract
    if not looks_like_formula_location_correction(contract.clean_query):
        return contract
    paper = paper_from_query_hint(contract.clean_query)
    if paper is None:
        return contract
    target = formula_followup_target(
        contract=contract,
        active=session.effective_active_research(),
        paper=paper,
    )
    if not target:
        return contract
    return formula_location_followup_contract(contract=contract, paper=paper, target=target)


def _resolve_formula_contextual_paper_contract(
    *,
    contract: QueryContract,
    session: SessionContext,
    paper_from_query_hint: PaperHintLookup,
    block_documents_for_paper: BlockDocumentsForPaper,
) -> QueryContract:
    goals = set(research_plan_context_from_contract(contract).goals)
    if contract.interaction_mode != "research" or "formula" not in goals or not contract.targets:
        return contract
    if selected_clarification_paper_id(contract) or contract_has_note(contract, "exclude_previous_focus"):
        return contract
    active = session.effective_active_research()
    context_text = " ".join([*active.titles, *active.targets]).strip()
    if not context_text:
        return contract
    paper = paper_from_query_hint(context_text)
    if paper is None:
        return contract
    if not formula_query_allows_paper_context(
        contract=contract,
        active=session.effective_active_research(),
        paper=paper,
    ):
        return contract
    target = str(contract.targets[0] or "").strip()
    if not target or not paper_context_supports_formula_target(
        block_documents=block_documents_for_paper(paper.paper_id, 256),
        target=target,
    ):
        return contract
    return formula_contextual_paper_contract(contract=contract, paper=paper, target=target)


def _resolve_paper_scope_correction_contract(
    *,
    contract: QueryContract,
    session: SessionContext,
    paper_from_query_hint: PaperHintLookup,
) -> QueryContract:
    if contract.interaction_mode != "research" or selected_clarification_paper_id(contract):
        return contract
    active = session.effective_active_research()
    if not active.has_content() or not active.targets:
        return contract
    if not looks_like_paper_scope_correction(contract.clean_query):
        return contract
    paper = paper_from_query_hint(contract.clean_query)
    if paper is None:
        return contract
    return paper_scope_correction_contract(contract=contract, active=active, paper=paper)


def _resolve_contextual_active_paper_contract(
    *,
    contract: QueryContract,
    session: SessionContext,
    paper_from_query_hint: PaperHintLookup,
) -> QueryContract:
    if contract.interaction_mode != "research" or selected_clarification_paper_id(contract):
        return contract
    if contract_has_note(contract, "exclude_previous_focus") or is_negative_correction_query(contract.clean_query):
        return contract
    if not looks_like_active_paper_reference(contract.clean_query):
        return contract
    active = session.effective_active_research()
    if not active.titles:
        return contract
    paper = paper_from_query_hint(" ".join(active.titles))
    if paper is None:
        return contract
    return contextual_active_paper_contract(contract=contract, paper=paper)
