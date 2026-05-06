from __future__ import annotations

from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, SessionContext
from app.services.infra.confidence import coerce_claim_confidence
from app.services.followup.candidates import (
    expand_followup_candidate_pool,
    rank_followup_candidates,
    resolve_followup_seed_papers,
    selected_followup_candidate_assessment,
    selected_followup_candidate_title,
)
from app.services.claims.followup_helpers import followup_research_claim
from app.services.claims.paper_summary import paper_summary_text


def solve_followup_research_claims(
    *,
    clients: Any,
    retriever: Any,
    paper_limit_default: int,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    session: SessionContext,
) -> list[Claim]:
    if not papers:
        return []
    seed_papers = resolve_followup_seed_papers(
        contract=contract,
        candidates=papers,
        active_titles=session.effective_active_research().titles,
        clients=clients,
        paper_summary_text=lambda paper_id: paper_summary_text(
            paper_id,
            paper_doc_lookup=retriever.paper_doc_by_id,
        ),
    )
    selected_candidate_title = selected_followup_candidate_title(contract)
    candidate_pool = expand_followup_candidate_pool(
        contract=contract,
        seed_papers=seed_papers,
        initial_candidates=papers,
        paper_limit_default=paper_limit_default,
        paper_summary_text=lambda paper_id: paper_summary_text(
            paper_id,
            paper_doc_lookup=retriever.paper_doc_by_id,
        ),
        search_papers=lambda query, search_contract, limit: retriever.search_papers(
            query=query,
            contract=search_contract,
            limit=limit,
        ),
    )
    followups = rank_followup_candidates(
        contract=contract,
        seed_papers=seed_papers,
        candidates=candidate_pool,
        evidence=evidence,
        clients=clients,
        paper_summary_text=lambda paper_id: paper_summary_text(
            paper_id,
            paper_doc_lookup=retriever.paper_doc_by_id,
        ),
        selected_candidate_assessment=lambda paper: selected_followup_candidate_assessment(
            contract=contract,
            seed_papers=seed_papers,
            paper=paper,
            evidence=evidence,
            clients=clients,
            expand_evidence=lambda paper_ids, query, evidence_contract, limit: retriever.expand_evidence(
                paper_ids=paper_ids,
                query=query,
                contract=evidence_contract,
                limit=limit,
            ),
            paper_summary_text=lambda paper_id: paper_summary_text(
                paper_id,
                paper_doc_lookup=retriever.paper_doc_by_id,
            ),
            coerce_confidence=coerce_claim_confidence,
        ),
        coerce_confidence=coerce_claim_confidence,
    )
    if not followups:
        return []
    return [
        followup_research_claim(
            entity=contract.targets[0] if contract.targets else "",
            seed_papers=seed_papers,
            followups=followups,
            selected_candidate_title=selected_candidate_title,
        )
    ]
