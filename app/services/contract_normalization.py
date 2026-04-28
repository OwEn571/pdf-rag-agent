from __future__ import annotations

from collections.abc import Callable
import re

from app.services.research_planning import goals_from_relation_compatibility


TargetCanonicalizer = Callable[[list[str]], list[str]]


def normalize_contract_targets(
    *,
    targets: list[str],
    requested_fields: list[str],
    canonicalize_targets: TargetCanonicalizer,
) -> list[str]:
    raw_targets = [clean_contract_target_text(str(item).strip()) for item in targets if str(item).strip()]
    canonical_targets = canonicalize_targets([item for item in raw_targets if item])
    requested_keys = {normalize_lookup_text(item) for item in requested_fields if item}
    cleaned: list[str] = []
    seen: set[str] = set()
    for target in canonical_targets:
        normalized = str(target or "").strip()
        if not normalized or is_structural_target_reference(normalized):
            continue
        target_key = normalize_lookup_text(normalized)
        if not target_key or target_key in requested_keys or target_key in seen:
            continue
        seen.add(target_key)
        cleaned.append(normalized)
    return cleaned


def clean_contract_target_text(text: str) -> str:
    raw = " ".join(str(text or "").strip().split())
    if not raw:
        return ""
    original = raw
    task_suffixes = (
        "公式",
        "目标函数",
        "损失函数",
        "定义",
        "是什么",
        "算法",
        "方法",
        "模型",
        "数据集",
        "论文",
        "formula",
        "objective",
        "loss",
        "definition",
        "algorithm",
        "method",
        "model",
        "dataset",
        "benchmark",
        "paper",
    )
    suffix_pattern = "|".join(re.escape(item) for item in task_suffixes)
    previous = ""
    while previous != raw:
        previous = raw
        raw = re.sub(rf"\s*(?:的)?(?:{suffix_pattern})\s*$", "", raw, flags=re.I).strip()
    acronym_match = re.match(rf"^([A-Z][A-Z0-9-]{{1,15}})\s*(?:{suffix_pattern})\b.*$", original, flags=re.I)
    if acronym_match:
        head = acronym_match.group(1).strip()
        if head and head.upper() == head:
            return head
    compact_match = re.match(rf"^([A-Z][A-Z0-9-]{{1,15}})(?:{suffix_pattern})$", original, flags=re.I)
    if compact_match:
        head = compact_match.group(1).strip()
        if head and head.upper() == head:
            return head
    return raw or original


def is_structural_target_reference(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    patterns = [
        r"^(fig(?:ure)?)[\s._-]*\d+[a-z]?$",
        r"^(table|tab)[\s._-]*\d+[a-z]?$",
        r"^(eq(?:uation)?|formula)[\s._-]*\d+[a-z]?$",
        r"^[图表式]\s*\d+[a-z]?$",
    ]
    return any(re.fullmatch(pattern, normalized) for pattern in patterns)


def normalize_modalities(modalities: list[str], *, relation: str) -> list[str]:
    normalized: list[str] = []
    alias_map = {
        "text": ["page_text"],
        "textual": ["page_text"],
        "page_text": ["page_text"],
        "paper": ["paper_card"],
        "paper_card": ["paper_card"],
        "table": ["table"],
        "tabular": ["table"],
        "caption": ["caption"],
        "figure": ["figure"],
        "visual": ["figure", "caption"],
        "image": ["figure", "caption"],
        "vision": ["figure", "caption"],
    }
    for modality in modalities:
        key = str(modality or "").strip().lower()
        for item in alias_map.get(key, [key]):
            if item in {"page_text", "paper_card", "table", "caption", "figure"} and item not in normalized:
                normalized.append(item)
    if not normalized:
        goal_defaults = goals_from_relation_compatibility(relation)
        if "figure_conclusion" in goal_defaults:
            normalized = ["figure", "caption", "page_text"]
        elif "formula" in goal_defaults:
            normalized = ["page_text", "table"]
        elif "metric_value" in goal_defaults:
            normalized = ["table", "caption", "page_text"]
        else:
            normalized = ["page_text", "paper_card"]
    elif "figure" in normalized and "page_text" not in normalized:
        normalized.append("page_text")
    return normalized


def normalize_lookup_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())
