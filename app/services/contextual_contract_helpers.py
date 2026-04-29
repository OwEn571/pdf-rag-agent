from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from app.domain.models import ActiveResearch, CandidatePaper, QueryContract
from app.services.contract_normalization import normalize_lookup_text
from app.services.followup_intents import (
    formula_query_allows_active_paper_context,
    looks_like_contextual_metric_query,
)
from app.services.intent_marker_matching import MarkerProfile, query_matches_any
from app.services.query_shaping import is_short_acronym, matches_target


CONTEXTUAL_CONTRACT_MARKERS: dict[str, MarkerProfile] = {
    "formula_context": ("objective", "loss", "formula", "log σ", "log sigma", "lpba", "l pba"),
}


def active_paper_reference_notes(*, notes: list[str], paper: CandidatePaper, marker: str) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *notes,
                marker,
                "resolved_from_conversation_memory",
                f"selected_paper_id={paper.paper_id}",
                "memory_title=" + paper.title,
            ]
        )
    )


def paper_hint_names(paper: CandidatePaper) -> list[str]:
    names: list[str] = []
    title = str(paper.title or "").strip()
    aliases = [alias.strip() for alias in str(paper.metadata.get("aliases", "")).split("||") if alias.strip()]
    for item in [title, *aliases]:
        if item and item not in names:
            names.append(item)
    if title:
        for separator in [":", " - ", " — ", " – "]:
            if separator in title:
                head = title.split(separator, 1)[0].strip()
                if head and head not in names:
                    names.append(head)
    return names


def normalize_entity_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def formula_query_allows_paper_context(
    *,
    contract: QueryContract,
    active: ActiveResearch,
    paper: CandidatePaper,
) -> bool:
    active_names = [*active.targets, *paper_hint_names(paper)]
    return formula_query_allows_active_paper_context(
        contract.clean_query,
        active_names=active_names,
        normalize_entity_key=normalize_entity_key,
    )


def formula_followup_target(*, contract: QueryContract, active: ActiveResearch, paper: CandidatePaper) -> str:
    paper_name_keys = {normalize_entity_key(name) for name in paper_hint_names(paper) if name}
    active_keys = {normalize_lookup_text(item) for item in active.targets}
    for target in contract.targets:
        candidate = str(target or "").strip()
        if not candidate:
            continue
        if normalize_entity_key(candidate) in paper_name_keys:
            continue
        if is_short_acronym(candidate) or normalize_lookup_text(candidate) in active_keys:
            return candidate
    return str(active.targets[0] or "").strip()


def paper_from_query_hint(
    query: str,
    *,
    paper_documents: Iterable[Any],
    candidate_lookup: Callable[[str], CandidatePaper | None],
) -> CandidatePaper | None:
    query_text = str(query or "").strip()
    if not query_text:
        return None
    query_key = normalize_entity_key(query_text)
    query_words = normalize_lookup_text(query_text)
    query_hints = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", query_text)
        if any(ch.isupper() for ch in token[1:]) or any(ch.isdigit() for ch in token)
    ]
    scored: list[tuple[int, CandidatePaper]] = []
    for doc in paper_documents:
        meta = dict(getattr(doc, "metadata", {}) or {})
        paper_id = str(meta.get("paper_id", "") or "").strip()
        if not paper_id:
            continue
        paper = candidate_lookup(paper_id)
        if paper is None:
            continue
        best = 0
        for name in paper_hint_names(paper):
            name = str(name or "").strip()
            if not name:
                continue
            name_key = normalize_entity_key(name)
            if len(name_key) < 4:
                continue
            if name_key in query_key:
                best = max(best, min(200, len(name_key)))
                continue
            if matches_target(query_words, name.lower()):
                best = max(best, min(160, len(name_key)))
        paper_context = "\n".join(
            [
                paper.title,
                str(paper.metadata.get("aliases", "")),
                str(paper.metadata.get("abstract_note", "")),
                str(paper.metadata.get("generated_summary", "")),
                str(paper.metadata.get("paper_card_text", "")),
                str(getattr(doc, "page_content", "") or ""),
            ]
        )
        for hint in query_hints:
            if matches_target(paper_context, hint):
                best = max(best, 80 + min(40, len(hint)))
        if best:
            scored.append((best, paper))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], -len(item[1].title), item[1].title))
    return scored[0][1]


def paper_context_supports_formula_target(*, block_documents: Iterable[Any], target: str) -> bool:
    target = str(target or "").strip()
    if not target:
        return False
    for doc in block_documents:
        text = str(getattr(doc, "page_content", "") or "")
        if not matches_target(text, target):
            continue
        meta = dict(getattr(doc, "metadata", {}) or {})
        lowered = text.lower()
        if int(meta.get("formula_hint", 0) or 0):
            return True
        if query_matches_any(lowered, "", CONTEXTUAL_CONTRACT_MARKERS["formula_context"]):
            return True
    return False


def paper_scope_correction_contract(
    *,
    contract: QueryContract,
    active: ActiveResearch,
    paper: CandidatePaper,
) -> QueryContract:
    inherited_query = active.clean_query or contract.clean_query
    notes = active_paper_reference_notes(
        notes=contract.notes,
        paper=paper,
        marker="paper_scope_correction",
    )
    scoped = QueryContract(
        clean_query=f"限定在论文《{paper.title}》中回答：{inherited_query}",
        interaction_mode="research",
        relation=active.relation or contract.relation,
        targets=list(active.targets),
        answer_slots=list(contract.answer_slots),
        requested_fields=list(active.requested_fields or contract.requested_fields or ["answer"]),
        required_modalities=list(active.required_modalities or contract.required_modalities or ["page_text"]),
        answer_shape=active.answer_shape or contract.answer_shape,
        precision_requirement=active.precision_requirement or contract.precision_requirement,
        continuation_mode="followup",
        allow_web_search=contract.allow_web_search,
        notes=notes,
    )
    return promote_contextual_metric_contract(scoped)


def contextual_active_paper_contract(*, contract: QueryContract, paper: CandidatePaper) -> QueryContract:
    notes = active_paper_reference_notes(
        notes=contract.notes,
        paper=paper,
        marker="active_paper_reference",
    )
    clean_query = contract.clean_query
    if normalize_entity_key(paper.title) not in normalize_entity_key(clean_query):
        clean_query = f"限定在论文《{paper.title}》中回答：{clean_query}"
    scoped = contract.model_copy(
        update={
            "clean_query": clean_query,
            "continuation_mode": "followup",
            "notes": notes,
        }
    )
    return promote_contextual_metric_contract(scoped)


def promote_contextual_metric_contract(contract: QueryContract) -> QueryContract:
    if contract.relation == "metric_value_lookup":
        return contract
    if not looks_like_contextual_metric_query(
        contract.clean_query,
        targets=list(contract.targets),
        is_short_acronym=is_short_acronym,
    ):
        return contract
    requested_fields = list(dict.fromkeys([*contract.requested_fields, "metric_value", "setting", "evidence"]))
    required_modalities = list(dict.fromkeys([*contract.required_modalities, "table", "caption", "page_text"]))
    answer_slots = list(dict.fromkeys([*contract.answer_slots, "metric_value"]))
    notes = list(dict.fromkeys([*contract.notes, "contextual_metric_query", "answer_slot=metric_value"]))
    return contract.model_copy(
        update={
            "relation": "metric_value_lookup",
            "answer_slots": answer_slots,
            "requested_fields": requested_fields,
            "required_modalities": required_modalities,
            "answer_shape": "narrative",
            "precision_requirement": "exact",
            "notes": notes,
        }
    )


def formula_answer_correction_contract(
    *,
    contract: QueryContract,
    active: ActiveResearch,
    paper: CandidatePaper | None,
) -> QueryContract:
    notes = list(
        dict.fromkeys(
            [
                *contract.notes,
                "formula_answer_correction",
                "prefer_scalar_objective",
                "answer_slot=formula",
            ]
        )
    )
    if paper is not None:
        notes = list(dict.fromkeys([*notes, f"selected_paper_id={paper.paper_id}", "memory_title=" + paper.title]))
    target = active.targets[0] if active.targets else (contract.targets[0] if contract.targets else "当前目标")
    scope = f"限定在论文《{paper.title}》中" if paper is not None else "沿用上一轮论文上下文"
    return QueryContract(
        clean_query=f"{target} 的公式是什么？{scope}重新查找目标函数或损失函数；上一条候选公式可能是梯度/推导式，不要优先返回梯度公式。",
        interaction_mode="research",
        relation="formula_lookup",
        targets=list(active.targets or contract.targets or [target]),
        requested_fields=["formula", "variable_explanation", "source"],
        required_modalities=["page_text", "table"],
        answer_shape="bullets",
        precision_requirement="exact",
        continuation_mode="followup",
        allow_web_search=contract.allow_web_search,
        notes=notes,
    )


def formula_location_followup_contract(
    *,
    contract: QueryContract,
    paper: CandidatePaper,
    target: str,
) -> QueryContract:
    notes = list(
        dict.fromkeys(
            [
                *contract.notes,
                "formula_location_followup",
                "resolved_from_user_paper_hint",
                f"selected_paper_id={paper.paper_id}",
                "memory_title=" + paper.title,
                "answer_slot=formula",
            ]
        )
    )
    return QueryContract(
        clean_query=f"{target} 的公式是什么？限定在论文《{paper.title}》中查找。",
        interaction_mode="research",
        relation="formula_lookup",
        targets=[target],
        requested_fields=["formula", "variable_explanation", "source"],
        required_modalities=["page_text", "table"],
        answer_shape="bullets",
        precision_requirement="exact",
        continuation_mode="followup",
        allow_web_search=contract.allow_web_search,
        notes=notes,
    )


def formula_contextual_paper_contract(
    *,
    contract: QueryContract,
    paper: CandidatePaper,
    target: str,
) -> QueryContract:
    notes = list(
        dict.fromkeys(
            [
                *contract.notes,
                "formula_contextual_paper_binding",
                f"selected_paper_id={paper.paper_id}",
                "memory_title=" + paper.title,
            ]
        )
    )
    return contract.model_copy(
        update={
            "clean_query": f"{target} 的公式是什么？限定在论文《{paper.title}》中查找。",
            "relation": "formula_lookup",
            "requested_fields": ["formula", "variable_explanation", "source"],
            "required_modalities": ["page_text", "table"],
            "answer_shape": "bullets",
            "precision_requirement": "exact",
            "continuation_mode": "followup",
            "notes": notes,
        }
    )
