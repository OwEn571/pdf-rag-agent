from __future__ import annotations

import re
from collections.abc import Callable

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan
from app.services.contracts.context import contract_answer_slots
from app.services.contracts.normalization import is_structural_target_reference, normalize_lookup_text
from app.services.intents.marker_matching import MarkerProfile, query_matches_any
from app.services.planning.query_shaping import is_short_acronym, matches_target
from app.services.planning.solver_goals import looks_like_metric_goal

NormalizeTextFn = Callable[[str], str]


CLAIM_VERIFIER_MARKERS: dict[str, MarkerProfile] = {
    "claim_formula": (
        "=",
        "\\frac",
        "\\sum",
        "\\mathbb",
        "\\mathcal",
        "∑",
        "π",
        "sigma",
        "loss",
        "objective",
    ),
    "evidence_formula": (
        "=",
        "formula",
        "objective",
        "loss",
        "目标函数",
        "公式",
        "∑",
        "π",
        "\\pi",
        "sigma",
        "σ",
    ),
}


def verification_goals(*, contract: QueryContract, plan: ResearchPlan) -> set[str]:
    goals = {
        str(item).strip()
        for item in [
            *list(plan.required_claims or []),
            *contract_answer_slots(contract),
            *list(contract.requested_fields or []),
        ]
        if str(item).strip()
    }
    for modality in contract.required_modalities:
        if modality == "figure":
            goals.add("figure_conclusion")
        elif modality in {"table", "caption"} and looks_like_metric_verification_goal(contract.clean_query, goals):
            goals.add("metric_value")
    return goals


def looks_like_metric_verification_goal(query: str, goals: set[str]) -> bool:
    return looks_like_metric_goal(query, goals)


def claim_value_looks_like_formula(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    return query_matches_any(lowered, text, CLAIM_VERIFIER_MARKERS["claim_formula"])


def formula_claim_matches_target(
    *,
    contract: QueryContract,
    claim: Claim,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
) -> bool:
    claim_evidence_ids = set(claim.evidence_ids)
    claim_paper_ids = set(claim.paper_ids)
    claim_evidence = [item for item in evidence if item.doc_id in claim_evidence_ids or item.paper_id in claim_paper_ids]
    claim_papers = [item for item in papers if item.paper_id in claim_paper_ids]
    context = "\n".join(
        [
            str(claim.value or ""),
            str(claim.entity or ""),
            *[item.title for item in claim_papers],
            *[item.snippet for item in claim_evidence[:6]],
        ]
    )
    normalized_context = normalize_lookup_text(context)
    entity_targets = [
        part.strip()
        for part in re.split(r"[/,;、]", str(claim.entity or ""))
        if part.strip()
    ]
    candidate_targets = list(dict.fromkeys([*entity_targets, *list(contract.targets or [])]))
    if not candidate_targets:
        return True
    for target in candidate_targets:
        target_key = normalize_lookup_text(target)
        if not target_key:
            continue
        formula_text = f"{claim.value}\n{claim.entity}"
        if matches_target(formula_text, target):
            return True
        if formula_evidence_supports_target(target=target, evidence=claim_evidence):
            return True
        if not is_short_acronym(target) and matches_target(normalized_context, target):
            return True
    return False


def formula_evidence_supports_target(*, target: str, evidence: list[EvidenceBlock]) -> bool:
    for item in evidence[:8]:
        text = "\n".join([item.title, item.caption, item.snippet])
        if not matches_target(text, target):
            continue
        lowered = text.lower()
        if query_matches_any(lowered, text, CLAIM_VERIFIER_MARKERS["evidence_formula"]):
            return True
    return False


def targets_supported(
    *,
    targets: list[str],
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
) -> bool:
    normalized_targets = [normalize_lookup_text(item) for item in targets if item]
    if not normalized_targets:
        return True
    haystacks = [normalize_lookup_text(item.snippet) for item in evidence[:8]]
    for paper in papers[:4]:
        haystacks.append(normalize_lookup_text(paper.title))
        haystacks.append(normalize_lookup_text(str(paper.metadata.get("paper_card_text", ""))))
    return all(
        any(matches_target(haystack, target) for haystack in haystacks if haystack)
        for target in normalized_targets
    )


def paper_identity_matches_targets(
    *,
    paper: CandidatePaper,
    targets: list[str],
    canonicalize_target: NormalizeTextFn,
    normalize_entity_text: NormalizeTextFn,
) -> bool:
    normalized_targets = [
        str(item).strip()
        for item in targets
        if str(item).strip() and not is_structural_target_reference(item)
    ]
    if not normalized_targets:
        return True
    aliases = [alias.strip() for alias in str(paper.metadata.get("aliases", "")).split("||") if alias.strip()]
    candidate_names = [paper.title, *aliases]
    if paper.title:
        for separator in [":", " - ", " — ", " – "]:
            if separator in paper.title:
                head = paper.title.split(separator, 1)[0].strip()
                if head and head not in candidate_names:
                    candidate_names.append(head)
    for target in normalized_targets:
        canonical_target = canonicalize_target(target)
        normalized_target = normalize_entity_text(canonical_target)
        if not normalized_target:
            continue
        for candidate_name in candidate_names:
            if is_initialism_alias_match(candidate_name=candidate_name, target=canonical_target):
                return True
            candidate = normalize_entity_text(candidate_name)
            if not candidate:
                continue
            if is_identity_alias_match(candidate=candidate, target=normalized_target):
                return True
    return False


def is_identity_alias_match(*, candidate: str, target: str) -> bool:
    if not candidate or not target:
        return False
    if candidate == target:
        return True
    if candidate.startswith(target) and len(target) >= 4:
        remainder = candidate[len(target) :]
        if remainder and remainder[0] in {"-", " ", "/", ":"}:
            return True
    if target.startswith(candidate) and len(candidate) >= 4:
        remainder = target[len(candidate) :]
        if remainder and remainder[0] in {"-", " ", "/", ":"}:
            return True
    return False


def is_initialism_alias_match(*, candidate_name: str, target: str) -> bool:
    normalized_target = re.sub(r"[^A-Za-z0-9]", "", str(target or "")).upper()
    if not (2 <= len(normalized_target) <= 10) or not normalized_target.isupper():
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", str(candidate_name or ""))
    if len(words) < 2:
        return False
    stopwords = {"a", "an", "and", "are", "for", "in", "is", "of", "on", "the", "to", "via", "with", "your"}
    initials = "".join(word[0].upper() for word in words if word.lower() not in stopwords)
    return initials == normalized_target
