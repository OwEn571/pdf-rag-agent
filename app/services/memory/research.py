from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.domain.models import AssistantCitation, CandidatePaper, Claim, EvidenceBlock, QueryContract, SessionContext, VerificationReport
from app.services.contracts.normalization import normalize_lookup_text
from app.services.followup.relationship_memory import followup_relationship_memory
from app.services.contracts.session_context import truncate_context_text


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
    verification: VerificationReport | None = None,
) -> None:
    memory = dict(session.working_memory or {})
    # P0-8: Don't write target_bindings for best_effort results — they pollute
    # future turns with weak bindings that lock targets to wrong papers.
    skip_bindings = (
        verification is not None
        and getattr(verification, "original_status", "") == "best_effort"
    )
    bindings = dict(memory.get("target_bindings", {}) or {}) if not skip_bindings else {}
    if skip_bindings:
        # Write a temporary turn-scoped binding only
        memory["_last_best_effort_bindings"] = memory.get("target_bindings", {})
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
        entry = {
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
            "created_turn_index": int(getattr(session, "turn_index", 0) or 0),
        }
        if not skip_bindings:
            bindings[key] = entry
        else:
            # P0-8: store as temporary turn-scoped binding only
            memory.setdefault("_temp_best_effort_bindings", {})[key] = entry
    if not skip_bindings:
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


def remember_compound_outcome(
    *,
    session: SessionContext,
    clean_query: str,
    subtask_results: list[dict[str, Any]],
    candidate_lookup: Callable[[str], CandidatePaper | None],
) -> None:
    subtasks: list[dict[str, Any]] = []
    for result in subtask_results:
        contract = result.get("contract")
        if not isinstance(contract, QueryContract):
            continue
        claims = [item for item in list(result.get("claims", []) or []) if isinstance(item, Claim)]
        evidence = [item for item in list(result.get("evidence", []) or []) if isinstance(item, EvidenceBlock)]
        citations = [item for item in list(result.get("citations", []) or []) if isinstance(item, AssistantCitation)]
        paper_ids = list(dict.fromkeys(pid for claim in claims for pid in claim.paper_ids))
        papers = [paper for paper_id in paper_ids if (paper := candidate_lookup(paper_id)) is not None]
        if not papers:
            papers = [paper for citation in citations if (paper := candidate_lookup(citation.paper_id)) is not None]
        remember_research_outcome(
            session=session,
            contract=contract,
            answer=str(result.get("answer", "")),
            claims=claims,
            papers=papers,
            evidence=evidence,
            citations=citations,
            candidate_lookup=candidate_lookup,
        )
        subtasks.append(
            {
                "relation": contract.relation,
                "targets": list(contract.targets),
                "requested_fields": list(contract.requested_fields),
                "clean_query": contract.clean_query,
                "answer_preview": truncate_context_text(str(result.get("answer", "")), limit=900),
                "citation_titles": [citation.title for citation in citations[:4]],
            }
        )
    memory = dict(session.working_memory or {})
    memory["last_compound_query"] = {
        "query": clean_query,
        "subtasks": subtasks,
    }
    session.working_memory = memory
