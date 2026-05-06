from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract
from app.services.claims import metric_text as metric_helpers
from app.services.claims import origin_selection as origin_helpers
from app.services.answers.evidence_presentation import (
    evidence_ids_for_paper,
    extract_topology_terms,
    paper_recommendation_reason,
)
from app.services.claims.paper_helpers import default_text_claims, paper_recommendation_claim, paper_summary_claims
from app.services.claims.paper_summary import paper_summary_text
from app.services.answers.topology import (
    topology_discovery_claim,
    topology_recommendation_from_payload,
    topology_recommendation_human_prompt,
    topology_recommendation_claim,
    topology_recommendation_system_prompt,
)


def solve_topology_discovery_claims(*, papers: list[CandidatePaper], evidence: list[EvidenceBlock]) -> list[Claim]:
    topology_terms = extract_topology_terms(evidence)
    claim = topology_discovery_claim(
        papers=papers,
        topology_terms=topology_terms,
        evidence_ids_for_paper=lambda paper_id: evidence_ids_for_paper(evidence, paper_id, limit=2),
    )
    return [claim] if claim is not None else []


def solve_topology_recommendation_claims(*, clients: Any, evidence: list[EvidenceBlock]) -> list[Claim]:
    topology_terms = extract_topology_terms(evidence)
    if not topology_terms and not evidence:
        return []
    payload = clients.invoke_json(
        system_prompt=topology_recommendation_system_prompt(),
        human_prompt=topology_recommendation_human_prompt(topology_terms=topology_terms, evidence=evidence),
        fallback={},
    )
    recommendation = topology_recommendation_from_payload(payload, topology_terms=topology_terms)
    return [topology_recommendation_claim(recommendation=recommendation, topology_terms=topology_terms, evidence=evidence)]


def solve_paper_summary_results_claims(
    *,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    solver_metric_token_weights: dict[str, float],
    paper_doc_lookup: Callable[[str], Any],
    paper_identity_matches_targets: Callable[[CandidatePaper, list[str]], bool],
) -> list[Claim]:
    if not papers:
        return []
    focused_papers = list(papers)
    if contract.targets:
        focused = [
            paper
            for paper in focused_papers
            if paper_identity_matches_targets(paper, contract.targets)
            or origin_helpers.paper_has_origin_intro_support(paper=paper, targets=contract.targets)
        ]
        if focused:
            focused_papers = focused
    return paper_summary_claims(
        entity=contract.targets[0] if contract.targets else "",
        papers=focused_papers,
        metric_lines=metric_helpers.extract_metric_lines(
            evidence,
            token_weights=solver_metric_token_weights,
        ),
        summary_for_paper=lambda paper_id: paper_summary_text(paper_id, paper_doc_lookup=paper_doc_lookup),
        evidence_ids_for_paper=lambda paper_id, limit: evidence_ids_for_paper(evidence, paper_id, limit=limit),
    )


def solve_paper_recommendation_claims(
    *,
    contract: QueryContract,
    papers: list[CandidatePaper],
    paper_doc_lookup: Callable[[str], Any],
) -> list[Claim]:
    claim = paper_recommendation_claim(
        entity=contract.targets[0] if contract.targets else contract.clean_query,
        papers=papers,
        reason_for_paper=lambda paper: paper_recommendation_reason(
            paper_summary_text(paper.paper_id, paper_doc_lookup=paper_doc_lookup)
        ),
    )
    return [claim] if claim is not None else []


def solve_metric_context_claims(
    *,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    solver_metric_token_weights: dict[str, float],
    paper_identity_matches_targets: Callable[[CandidatePaper, list[str]], bool],
) -> list[Claim]:
    if not papers and not evidence:
        return []
    metric_evidence = metric_helpers.ranked_metric_context_evidence(
        contract=contract,
        papers=papers,
        evidence=evidence,
        token_weights=solver_metric_token_weights,
        paper_target_matcher=paper_identity_matches_targets,
    )
    selected_paper, selected_papers, paper_ids = metric_helpers.metric_paper_selection(
        papers=papers,
        ranked_evidence=metric_evidence,
    )
    if selected_paper is None:
        return []
    return [
        metric_helpers.metric_context_claim(
            entity=contract.targets[0] if contract.targets else selected_paper.title,
            selected_paper=selected_paper,
            selected_papers=selected_papers,
            metric_lines=metric_helpers.extract_metric_lines(
                metric_evidence or evidence,
                token_weights=solver_metric_token_weights,
            ),
            metric_evidence=metric_evidence,
            fallback_evidence_ids=evidence_ids_for_paper(evidence, selected_paper.paper_id, limit=4),
            paper_ids=paper_ids or [selected_paper.paper_id],
        )
    ]


def solve_default_text_claims(
    *,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    paper_doc_lookup: Callable[[str], Any],
) -> list[Claim]:
    if not papers:
        return []
    return default_text_claims(
        entity=contract.targets[0] if contract.targets else "",
        papers=papers,
        summary_for_paper=lambda paper_id: paper_summary_text(paper_id, paper_doc_lookup=paper_doc_lookup),
        evidence_ids_for_paper=lambda paper_id, limit: evidence_ids_for_paper(evidence, paper_id, limit=limit),
    )
