from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services.contract_normalization import normalize_lookup_text
from app.services.followup_relationship_intents import has_followup_domain_signal

PaperText = Callable[[str], str]
PaperAnchor = Callable[[CandidatePaper], str]


def selected_followup_candidate_title(contract: QueryContract) -> str:
    for note in contract.notes:
        raw = str(note or "")
        if raw.startswith("candidate_title="):
            return raw.split("=", 1)[1].strip()
    return ""


def candidate_title_matches(paper: CandidatePaper, selected_title: str) -> bool:
    selected_key = normalize_lookup_text(selected_title)
    if not selected_key:
        return False
    title_key = normalize_lookup_text(paper.title)
    aliases = normalize_lookup_text(str(paper.metadata.get("aliases", "")))
    return selected_key in title_key or title_key in selected_key or selected_key in aliases


def followup_target_aliases(
    *,
    contract: QueryContract,
    seed_papers: list[CandidatePaper],
    paper_anchor_text: PaperAnchor,
) -> list[str]:
    aliases: list[str] = []
    for target in contract.targets:
        target = str(target or "").strip()
        if target:
            aliases.append(target)
    for paper in seed_papers[:2]:
        raw_aliases = str(paper.metadata.get("aliases", ""))
        for alias in re.split(r"\|\||[,;/]", raw_aliases):
            alias = alias.strip()
            if alias and len(alias) <= 48:
                aliases.append(alias)
        anchor = paper_anchor_text(paper)
        if anchor and len(anchor) <= 48:
            aliases.append(anchor)
    normalized: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        key = alias.lower()
        if key and key not in seen:
            seen.add(key)
            normalized.append(alias)
    return normalized


def filter_followup_candidates(
    *,
    contract: QueryContract,
    candidates: list[CandidatePaper],
    paper_summary_text: PaperText,
) -> list[CandidatePaper]:
    if not candidates:
        return []
    target = contract.targets[0].lower() if contract.targets else ""
    filtered: list[CandidatePaper] = []
    for item in candidates:
        title_lower = item.title.lower()
        card_text = str(item.metadata.get("paper_card_text", "")).lower()
        summary_lower = str(item.metadata.get("generated_summary", "") or paper_summary_text(item.paper_id)).lower()
        if (
            target
            and target not in title_lower
            and target not in summary_lower
            and target not in str(item.metadata.get("abstract_note", "")).lower()
            and target not in card_text
            and not has_followup_domain_signal(title_lower + "\n" + card_text + "\n" + summary_lower)
        ):
            continue
        filtered.append(item)
    return filtered or candidates[: min(8, len(candidates))]


def merge_followup_rankings(
    *,
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in (primary, secondary):
        for item in source:
            paper = item.get("paper")
            paper_id = getattr(paper, "paper_id", "")
            if not paper_id or paper_id in seen:
                continue
            seen.add(paper_id)
            merged.append(item)
    return merged


def relationship_evidence_ids_from_payload(
    *,
    payload: dict[str, Any],
    relationship_evidence: list[EvidenceBlock],
) -> list[str]:
    available = {item.doc_id for item in relationship_evidence}
    raw_ids = payload.get("evidence_ids", [])
    selected: list[str] = []
    if isinstance(raw_ids, list):
        selected = [str(item).strip() for item in raw_ids if str(item).strip() in available]
    if selected:
        return selected[:6]
    return [item.doc_id for item in relationship_evidence[:4]]


def paper_relationship_brief(
    *,
    paper: CandidatePaper,
    paper_summary_text: PaperText,
) -> dict[str, Any]:
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "year": paper.year,
        "authors": str(paper.metadata.get("authors", "")),
        "aliases": str(paper.metadata.get("aliases", "")),
        "summary": paper_summary_text(paper.paper_id),
        "paper_card_text": str(paper.metadata.get("paper_card_text", ""))[:1800],
        "tags": str(paper.metadata.get("tags", "")),
    }
