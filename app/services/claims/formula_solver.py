from __future__ import annotations

from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract
from app.services.claims import formula_text as formula_helpers
from app.services.answers.evidence_presentation import evidence_ids_for_paper, formula_terms
from app.services.planning.query_shaping import matches_target


def solve_formula_claims(
    *,
    clients: Any,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    retrieval_formula_token_weights: dict[str, float],
) -> list[Claim]:
    if not papers:
        return []
    claims: list[Claim] = []
    target_terms = formula_helpers.formula_target_terms(contract)
    for paper in papers:
        paper_evidence = [item for item in evidence if item.paper_id == paper.paper_id]
        if not paper_evidence:
            continue
        matched_targets = formula_helpers.formula_matched_targets(
            paper=paper,
            evidence=paper_evidence,
            target_terms=target_terms,
            target_matcher=matches_target,
        )
        if target_terms and not matched_targets:
            continue
        formula_blocks = formula_helpers.select_formula_blocks(
            paper_evidence,
            block_scorer=lambda text: formula_helpers.formula_block_score(
                text,
                query=contract.clean_query,
                token_weights=retrieval_formula_token_weights,
            ),
        )
        formula_payload = extract_formula_claim_payload(
            clients=clients,
            contract=contract,
            formula_blocks=formula_blocks,
            fallback_evidence=paper_evidence,
        )
        claim = formula_helpers.formula_claim_from_payload(
            contract=contract,
            paper=paper,
            matched_targets=matched_targets,
            formula_payload=formula_payload,
            formula_blocks=formula_blocks,
            fallback_evidence_ids=evidence_ids_for_paper(evidence, paper.paper_id, limit=3),
            fallback_term_text="\n".join(item.snippet for item in paper_evidence[:3]),
            term_extractor=formula_terms,
        )
        if claim is None:
            continue
        claims.append(claim)
        if len(claims) >= 6:
            break
    return claims


def extract_formula_claim_payload(
    *,
    clients: Any,
    contract: QueryContract,
    formula_blocks: list[EvidenceBlock],
    fallback_evidence: list[EvidenceBlock],
) -> dict[str, Any]:
    selected_evidence = formula_blocks[:4] or fallback_evidence[:4]
    llm_payload = llm_extract_formula_claim_payload(clients=clients, contract=contract, evidence=selected_evidence)
    if llm_payload:
        return llm_payload
    return formula_helpers.fallback_formula_payload(selected_evidence, term_extractor=formula_terms)


def llm_extract_formula_claim_payload(
    *,
    clients: Any,
    contract: QueryContract,
    evidence: list[EvidenceBlock],
) -> dict[str, Any]:
    if getattr(clients, "chat", None) is None or not evidence:
        return {}
    payload = clients.invoke_json(
        system_prompt=formula_helpers.formula_extractor_system_prompt(),
        human_prompt=formula_helpers.formula_extractor_human_prompt(contract=contract, evidence=evidence),
        fallback={},
    )
    return formula_helpers.llm_formula_payload_from_response(
        payload,
        allowed_evidence_ids={item.doc_id for item in evidence},
        term_extractor=formula_terms,
    )
