from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan, VerificationReport
from app.services.claims import origin_selection as origin_helpers
from app.services.answers.evidence_presentation import extract_topology_terms

PaperLookupFn = Callable[[str], CandidatePaper | None]
PaperDocLookupFn = Callable[[str], Any]
TargetSupportFn = Callable[[list[str], list[CandidatePaper], list[EvidenceBlock]], bool]
PaperIdentityFn = Callable[[CandidatePaper, list[str]], bool]
FormulaValueFn = Callable[[str], bool]
FormulaTargetFn = Callable[[QueryContract, Claim, list[CandidatePaper], list[EvidenceBlock]], bool]
FormulaLlmVerifierFn = Callable[
    [QueryContract, list[Claim], list[CandidatePaper], list[EvidenceBlock]],
    VerificationReport | None,
]


def verify_origin_lookup_claims(
    *,
    contract: QueryContract,
    claims: list[Claim],
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    origin_supports_claim: Callable[[QueryContract, Claim, list[CandidatePaper], list[EvidenceBlock]], bool],
) -> VerificationReport | None:
    claim = next((item for item in claims if item.claim_type == "origin"), None)
    if claim is None or not claim.evidence_ids or not claim.paper_ids:
        return VerificationReport(status="retry", missing_fields=["paper_title"], recommended_action="retry_origin")
    if not origin_supports_claim(contract, claim, papers, evidence):
        return VerificationReport(
            status="retry",
            missing_fields=["origin_evidence"],
            unsupported_claims=["origin claim lacks an introduction/proposal cue near the requested target"],
            recommended_action="retry_origin",
        )
    return None


def origin_claim_has_intro_support(
    *,
    contract: QueryContract,
    claim: Claim,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    paper_lookup: PaperLookupFn,
    paper_doc_lookup: PaperDocLookupFn,
) -> bool:
    targets = list(contract.targets or [])
    if claim.entity:
        targets.append(claim.entity)
    aliases = origin_helpers.origin_target_aliases(targets)
    if not aliases:
        return bool(claim.evidence_ids and claim.paper_ids)
    claim_paper_ids = {str(item) for item in claim.paper_ids if str(item)}
    claim_evidence_ids = {str(item) for item in claim.evidence_ids if str(item)}
    paper_by_id = {item.paper_id: item for item in papers}
    for paper_id in list(claim_paper_ids):
        paper = paper_by_id.get(paper_id) or paper_lookup(paper_id)
        if paper is not None and origin_helpers.origin_target_intro_score(origin_helpers.origin_paper_text(paper), aliases) > 0:
            return True
        paper_doc = paper_doc_lookup(paper_id)
        if paper_doc is not None and origin_helpers.origin_target_intro_score(str(getattr(paper_doc, "page_content", "") or ""), aliases) > 0:
            return True
    for item in evidence:
        if item.doc_id not in claim_evidence_ids and item.paper_id not in claim_paper_ids:
            continue
        text = "\n".join([item.title, item.caption, item.snippet])
        if origin_helpers.origin_target_intro_score(text, aliases) > 0:
            return True
    return False


def verify_entity_definition_claims(
    *,
    contract: QueryContract,
    claims: list[Claim],
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    targets_supported_fn: TargetSupportFn,
) -> VerificationReport | None:
    claim = next((item for item in claims if item.claim_type == "entity_definition"), None)
    if claim is None or not str(claim.value or "").strip() or not claim.paper_ids or not claim.evidence_ids:
        return VerificationReport(status="retry", missing_fields=["entity_type"], recommended_action="retry_entity")
    claim_paper_ids = set(claim.paper_ids)
    claim_evidence_ids = set(claim.evidence_ids)
    claim_papers = [item for item in papers if item.paper_id in claim_paper_ids]
    claim_evidence = [item for item in evidence if item.doc_id in claim_evidence_ids or item.paper_id in claim_paper_ids]
    if contract.targets and not targets_supported_fn(contract.targets, claim_papers, claim_evidence):
        if targets_supported_fn(contract.targets, papers, evidence):
            return VerificationReport(status="retry", missing_fields=["supporting_paper"], recommended_action="retry_entity")
        return VerificationReport(status="clarify", missing_fields=["relevant_evidence"], recommended_action="clarify_target")
    return None


def verify_followup_research_claims(*, claims: list[Claim]) -> VerificationReport | None:
    claim = next((item for item in claims if item.claim_type == "followup_research"), None)
    followup_titles = list(claim.structured_data.get("followup_titles", [])) if claim else []
    seed_ids = {str(item.get("paper_id", "")) for item in list(claim.structured_data.get("seed_papers", []))} if claim else set()
    if len(followup_titles) < 1:
        return VerificationReport(status="retry", missing_fields=["followup_papers"], recommended_action="broaden_followup")
    if any(str(item.get("paper_id", "")) in seed_ids for item in followup_titles):
        return VerificationReport(status="retry", missing_fields=["followup_papers"], recommended_action="exclude_seed_paper")
    return None


def verify_paper_recommendation_claims(*, claims: list[Claim]) -> VerificationReport | None:
    claim = next((item for item in claims if item.claim_type == "paper_recommendation"), None)
    recommended = list(claim.structured_data.get("recommended_papers", [])) if claim else []
    if len(recommended) < 1:
        return VerificationReport(status="retry", missing_fields=["recommended_papers"], recommended_action="broaden_recommendation")
    return None


def verify_topology_recommendation_claims(
    *,
    claims: list[Claim],
    evidence: list[EvidenceBlock],
) -> VerificationReport | None:
    claim = next((item for item in claims if item.claim_type == "topology_recommendation"), None)
    if claim is None:
        return VerificationReport(
            status="retry",
            missing_fields=["best_topology", "langgraph_recommendation"],
            recommended_action="retry_topology_recommendation",
        )
    structured = dict(claim.structured_data or {})
    topology_terms = [str(item).strip() for item in structured.get("topology_terms", []) if str(item).strip()]
    if not topology_terms:
        topology_terms = extract_topology_terms(evidence)
    if not claim.evidence_ids and not topology_terms:
        return VerificationReport(
            status="retry",
            missing_fields=["topology_evidence"],
            recommended_action="retry_topology_recommendation",
        )
    return None


def verify_figure_question_claims(
    *,
    contract: QueryContract,
    claims: list[Claim],
    papers: list[CandidatePaper],
    paper_identity_matches_targets: PaperIdentityFn,
) -> VerificationReport | None:
    claim = next((item for item in claims if item.claim_type == "figure_conclusion"), None)
    if claim is None or not claim.evidence_ids:
        return VerificationReport(status="retry", missing_fields=["figure_conclusion"], recommended_action="retry_figure")
    claim_paper_ids = set(claim.paper_ids)
    target_papers = [paper for paper in papers if not claim_paper_ids or paper.paper_id in claim_paper_ids]
    if contract.targets and target_papers and not any(
        paper_identity_matches_targets(paper, contract.targets)
        for paper in target_papers
    ):
        return VerificationReport(status="clarify", missing_fields=["target_paper"], recommended_action="clarify_target")
    return None


def verify_metric_value_lookup_claims(*, claims: list[Claim]) -> VerificationReport | None:
    claim = next((item for item in claims if item.claim_type == "metric_value"), None)
    if claim is None:
        return VerificationReport(status="retry", missing_fields=["metric_value"], recommended_action="retry_table")
    return None


def verify_formula_lookup_claims(
    *,
    contract: QueryContract,
    claims: list[Claim],
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    claim_value_looks_like_formula: FormulaValueFn,
    verify_formula_claims_with_llm: FormulaLlmVerifierFn,
    formula_claim_matches_target: FormulaTargetFn,
) -> VerificationReport | None:
    formula_claims = [item for item in claims if item.claim_type == "formula"]
    if not formula_claims:
        return VerificationReport(status="retry", missing_fields=["formula"], recommended_action="retry_formula")
    invalid_values = [
        str(claim.entity or claim.value or "formula")
        for claim in formula_claims
        if not str(claim.value or "").strip()
        or str(claim.value).startswith("已定位到")
        or not claim_value_looks_like_formula(str(claim.value or ""))
    ]
    if invalid_values:
        return VerificationReport(status="retry", missing_fields=["formula"], unsupported_claims=invalid_values, recommended_action="retry_formula")
    llm_report = verify_formula_claims_with_llm(contract, formula_claims, papers, evidence)
    if llm_report is not None:
        return llm_report
    if contract.targets:
        misaligned = [
            str(dict(claim.structured_data or {}).get("paper_title") or claim.entity or claim.value)
            for claim in formula_claims
            if not formula_claim_matches_target(contract, claim, papers, evidence)
        ]
        if misaligned:
            return VerificationReport(
                status="retry",
                missing_fields=["target_aligned_formula"],
                unsupported_claims=misaligned,
                recommended_action="retry_formula_target_alignment",
            )
    return None


def verify_concept_definition_claims(
    *,
    contract: QueryContract,
    claims: list[Claim],
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    targets_supported_fn: TargetSupportFn,
) -> VerificationReport | None:
    claim = next((item for item in claims if item.claim_type == "concept_definition"), None)
    if claim is None or not claim.evidence_ids:
        return VerificationReport(status="retry", missing_fields=["definition"], recommended_action="retry_definition")
    if contract.targets and not targets_supported_fn(contract.targets, papers, evidence):
        if papers or evidence:
            return VerificationReport(status="retry", missing_fields=["relevant_evidence"], recommended_action="retry_definition")
        return VerificationReport(status="clarify", missing_fields=["relevant_evidence"], recommended_action="clarify_target")
    return None


def verify_general_question_claims(
    *,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    targets_supported_fn: TargetSupportFn,
) -> VerificationReport | None:
    if contract.targets and not targets_supported_fn(contract.targets, papers, evidence):
        if papers or evidence:
            return VerificationReport(status="retry", missing_fields=["relevant_evidence"], recommended_action="expand_recall")
        return VerificationReport(status="clarify", missing_fields=["relevant_evidence"], recommended_action="clarify_target")
    return None
