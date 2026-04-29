from __future__ import annotations

from collections.abc import Callable

from app.domain.models import AssistantCitation, CandidatePaper, Claim, EvidenceBlock, QueryContract, SessionContext
from app.services.contract_normalization import normalize_lookup_text
from app.services.followup_relationship_memory import followup_relationship_memory
from app.services.session_context_helpers import truncate_context_text


def remember_research_outcome(
    *,
    session: SessionContext,
    contract: QueryContract,
    answer: str,
    claims: list[Claim],
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    citations: list[AssistantCitation],
    candidate_lookup: Callable[[str], CandidatePaper | None],
) -> None:
    memory = dict(session.working_memory or {})
    bindings = dict(memory.get("target_bindings", {}) or {})
    paper_by_id = {paper.paper_id: paper for paper in papers}
    citation_by_paper_id = {citation.paper_id: citation for citation in citations if citation.paper_id}
    fallback_paper = papers[0] if papers else None
    for target in contract.targets:
        target = str(target or "").strip()
        if not target:
            continue
        key = normalize_lookup_text(target)
        if not key:
            continue
        paper: CandidatePaper | None = None
        evidence_ids: list[str] = []
        for claim in claims:
            claim_target = str(claim.entity or target).strip()
            if claim_target and normalize_lookup_text(claim_target) not in {key, ""}:
                continue
            if claim.paper_ids:
                paper = paper_by_id.get(claim.paper_ids[0]) or candidate_lookup(claim.paper_ids[0])
            evidence_ids = list(claim.evidence_ids[:4])
            if paper is not None:
                break
        if paper is None:
            citation = citation_by_paper_id.get(target) or (citations[0] if citations else None)
            if citation is not None and citation.paper_id:
                paper = paper_by_id.get(citation.paper_id) or candidate_lookup(citation.paper_id)
        if paper is None:
            paper = fallback_paper
        if paper is None:
            continue
        support_titles = list(dict.fromkeys([paper.title, *[item.title for item in citations if item.title]]))
        bindings[key] = {
            "target": target,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "year": paper.year,
            "relation": contract.relation,
            "requested_fields": list(contract.requested_fields),
            "required_modalities": list(contract.required_modalities),
            "clean_query": contract.clean_query,
            "answer_preview": truncate_context_text(answer, limit=900),
            "evidence_ids": evidence_ids or [item.doc_id for item in evidence if item.paper_id == paper.paper_id][:4],
            "support_titles": support_titles[:4],
        }
    memory["target_bindings"] = bindings
    memory["last_successful_research"] = {
        "relation": contract.relation,
        "targets": list(contract.targets),
        "requested_fields": list(contract.requested_fields),
        "titles": [paper.title for paper in papers[:4]],
        "clean_query": contract.clean_query,
        "answer_preview": truncate_context_text(answer, limit=1200),
    }
    if any(claim.claim_type == "followup_research" for claim in claims):
        relationship = followup_relationship_memory(contract=contract, claims=claims, answer=answer)
        if relationship:
            memory["last_followup_relationship"] = relationship
    session.working_memory = memory
