from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from app.domain.models import QueryContract
from app.services.contract_normalization import normalize_lookup_text


CLARIFICATION_OPTION_SCHEMA_VERSION = "clarification_option.v1"

CLARIFICATION_CHOICE_MARKERS = [
    "我说",
    "选",
    "选择",
    "就是",
    "应该是",
    "指的是",
    "the one",
    "choose",
    "select",
]

CLARIFICATION_ORDINAL_PATTERNS = [
    (
        0,
        ["第一个", "第一项", "第1个", "第 1 个", "第1项", "第 1 项", "first", "the first"],
    ),
    (
        1,
        ["第二个", "第二项", "第2个", "第 2 个", "第2项", "第 2 项", "second", "the second"],
    ),
    (
        2,
        ["第三个", "第三项", "第3个", "第 3 个", "第3项", "第 3 项", "third", "the third"],
    ),
    (
        3,
        ["第四个", "第四项", "第4个", "第 4 个", "第4项", "第 4 项", "fourth", "the fourth"],
    ),
]


def looks_like_clarification_choice_text(normalized_query: str) -> bool:
    return any(marker in normalized_query for marker in CLARIFICATION_CHOICE_MARKERS)


def pending_clarification_selection_index(query: str) -> int | None:
    compact = " ".join(str(query or "").strip().lower().split())
    if not compact:
        return None
    digit_match = re.search(r"(?<!\d)([1-9])(?!\d)", compact)
    if digit_match:
        return int(digit_match.group(1)) - 1
    for index, markers in CLARIFICATION_ORDINAL_PATTERNS:
        if any(marker in compact for marker in markers):
            return index
    return None


def clarification_option_public_payload(option: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": option.get("schema_version", CLARIFICATION_OPTION_SCHEMA_VERSION),
        "option_id": option.get("option_id", ""),
        "kind": option.get("kind", ""),
        "target": option.get("target", ""),
        "label": option.get("label", ""),
        "description": option.get("description", ""),
        "paper_id": option.get("paper_id", ""),
        "title": option.get("title", ""),
        "year": option.get("year", ""),
        "meaning": option.get("meaning", ""),
        "snippet": option.get("snippet", ""),
        "source": option.get("source", ""),
        "source_relation": option.get("source_relation", ""),
        "source_requested_fields": option.get("source_requested_fields", []),
        "source_answer_slots": option.get("source_answer_slots", []),
    }
    for key in [
        "display_title",
        "display_label",
        "display_reason",
        "judge_recommended",
        "disambiguation_confidence",
        "source_required_modalities",
    ]:
        if key in option:
            payload[key] = option.get(key)
    return payload


def clarification_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def clarification_option_description(option: dict[str, Any], *, title: str, year: str) -> str:
    meta = " · ".join(item for item in [title, year] if item)
    context = str(option.get("context_text", "") or option.get("snippet", "") or "").strip()
    context = " ".join(context.split())
    return context or meta


def clarification_option_id(
    *,
    kind: str,
    target: str,
    label: str,
    paper_id: str,
    title: str,
    index: int,
) -> str:
    seed = json.dumps(
        {
            "kind": kind,
            "target": target,
            "label": label,
            "paper_id": paper_id,
            "title": title,
            "index": index,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    prefix = re.sub(r"[^a-z0-9]+", "-", f"{kind}-{target}".lower()).strip("-") or "clarification"
    return f"{prefix}-{digest}"


def ambiguity_options_from_notes(notes: list[str]) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for note in notes:
        raw = str(note or "")
        if not raw.startswith("ambiguity_option="):
            continue
        try:
            payload = json.loads(raw.split("=", 1)[1])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("title"):
            options.append(payload)
    return options


def selected_clarification_paper_id(contract: QueryContract) -> str:
    for note in contract.notes:
        raw = str(note or "")
        if raw.startswith("selected_paper_id="):
            return raw.split("=", 1)[1].strip()
        if not raw.startswith("selected_ambiguity_option="):
            continue
        try:
            payload = json.loads(raw.split("=", 1)[1])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            paper_id = str(payload.get("paper_id", "") or "").strip()
            if paper_id:
                return paper_id
    return ""


def option_from_clarification_choice(
    choice: dict[str, Any] | None,
    options: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(choice, dict) or not options:
        return None
    option_id = str(choice.get("option_id", "") or "").strip()
    if option_id:
        for option in options:
            if str(option.get("option_id", "") or "").strip() == option_id:
                return option
    raw_index = choice.get("index")
    try:
        index = int(raw_index)
    except (TypeError, ValueError):
        index = -1
    if 0 <= index < len(options):
        return options[index]
    paper_id = str(choice.get("paper_id", "") or "").strip()
    meaning = str(choice.get("meaning", "") or "").strip().lower()
    label = str(choice.get("label", "") or "").strip().lower()
    for option in options:
        if paper_id and str(option.get("paper_id", "") or "").strip() == paper_id:
            return option
        if meaning and str(option.get("meaning", "") or "").strip().lower() == meaning:
            return option
        if label and str(option.get("label", "") or "").strip().lower() == label:
            return option
    return None


def select_pending_clarification_option(
    *,
    clean_query: str,
    options: list[dict[str, Any]],
) -> dict[str, Any] | None:
    index = pending_clarification_selection_index(clean_query)
    if index is not None and 0 <= index < len(options):
        return options[index]
    normalized_query = normalize_lookup_text(clean_query)
    if not normalized_query:
        return None
    for option in options:
        meaning = normalize_lookup_text(str(option.get("meaning", "")))
        label = normalize_lookup_text(str(option.get("label", "")))
        title = normalize_lookup_text(str(option.get("title", "")))
        if meaning and normalized_query == meaning:
            return option
        if label and normalized_query == label:
            return option
        if (
            meaning
            and len(meaning) >= 10
            and meaning in normalized_query
            and looks_like_clarification_choice_text(normalized_query)
        ):
            return option
        if (
            label
            and len(label) >= 10
            and label in normalized_query
            and looks_like_clarification_choice_text(normalized_query)
        ):
            return option
        if title and len(normalized_query) >= 6 and normalized_query in title:
            return option
    return None


def contract_with_ambiguity_options(*, contract: QueryContract, options: list[dict[str, Any]]) -> QueryContract:
    notes = [note for note in contract.notes if not str(note).startswith("ambiguity_option=")]
    for option in options[:4]:
        payload = clarification_option_public_payload(option)
        notes.append("ambiguity_option=" + json.dumps(payload, ensure_ascii=False))
    return contract.model_copy(update={"notes": notes})


def contract_from_selected_clarification_option(
    *,
    clean_query: str,
    target: str,
    selected: dict[str, Any],
    notes_extra: list[str] | None = None,
    resolution_note: str = "resolved_human_choice",
    resolution_subject: str = "用户选择的含义是",
) -> QueryContract:
    selected_target = str(selected.get("target", "") or "").strip()
    target = target or selected_target
    meaning = str(selected.get("meaning", "") or selected.get("label", "") or target).strip()
    title = str(selected.get("title", "") or "").strip()
    notes = [
        resolution_note,
        "selected_ambiguity_option=" + json.dumps(selected, ensure_ascii=False),
    ]
    notes.extend(notes_extra or [])
    paper_id = str(selected.get("paper_id", "") or "").strip()
    if paper_id:
        notes.append(f"selected_paper_id={paper_id}")
    raw_requested = selected.get("source_requested_fields", [])
    source_requested = [str(item).strip() for item in raw_requested if str(item).strip()] if isinstance(raw_requested, list) else []
    raw_slots = selected.get("source_answer_slots", [])
    source_answer_slots = [str(item).strip() for item in raw_slots if str(item).strip()] if isinstance(raw_slots, list) else []
    source_relation = str(selected.get("source_relation", "") or selected.get("relation", "") or "").strip()
    is_formula_choice = source_relation == "formula_lookup" or "formula" in source_requested or "formula" in source_answer_slots
    if is_formula_choice:
        answer_slots = source_answer_slots or (["formula"] if "formula" in source_requested else [])
        rewritten = f"{target} 的公式是什么？{resolution_subject} {meaning}"
        if title:
            rewritten += f"，来源论文是《{title}》"
        return QueryContract(
            clean_query=rewritten,
            interaction_mode="research",
            relation="formula_lookup",
            targets=[target] if target else [],
            answer_slots=answer_slots,
            requested_fields=["formula", "variable_explanation", "source"],
            required_modalities=["page_text", "table"],
            answer_shape="bullets",
            precision_requirement="exact",
            continuation_mode="followup",
            notes=notes,
        )
    rewritten = f"{target} 是什么？{resolution_subject} {meaning}"
    if title:
        rewritten += f"，来源论文是《{title}》"
    return QueryContract(
        clean_query=rewritten,
        interaction_mode="research",
        relation="entity_definition",
        targets=[target] if target else [],
        requested_fields=["definition", "mechanism", "role_in_context"],
        required_modalities=["page_text", "paper_card", "table"],
        answer_shape="narrative",
        precision_requirement="high",
        continuation_mode="followup",
        notes=notes,
    )
