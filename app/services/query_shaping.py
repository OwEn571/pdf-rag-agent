from __future__ import annotations

import re

from app.domain.models import QueryContract
from app.services.research_intents import query_needs_external_search
from app.services.research_planning import research_plan_goals


def extract_targets(query: str) -> list[str]:
    targets: list[str] = []
    for pattern in [r"[\"“](.+?)[\"”]", r"[‘'](.+?)[’']"]:
        for match in re.finditer(pattern, query):
            candidate = str(match.group(1) or "").strip()
            if candidate and candidate not in targets:
                targets.append(candidate)
    upper_tokens = re.findall(r"[A-Z][A-Z0-9\-]{1,}", query)
    for raw_token in upper_tokens:
        token = raw_token.strip("-")
        if token not in targets:
            targets.append(token)
    mixed_tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", query)
    for token in mixed_tokens:
        if any(ch.isupper() for ch in token[1:]) or "-" in token or any(ch.isdigit() for ch in token):
            if token not in targets:
                targets.append(token)
    return targets


def paper_query_text(contract: QueryContract) -> str:
    target_text = " ".join(contract.targets).strip()
    goals = research_plan_goals(contract)
    if target_text and goals & {"definition", "entity_type", "role_in_context", "mechanism", "examples"}:
        if len(contract.targets) > 1:
            return f"{target_text} {contract.clean_query}".strip()
        return target_text
    if target_text and target_text.lower() not in contract.clean_query.lower():
        return f"{target_text} {contract.clean_query}".strip()
    return contract.clean_query


def evidence_query_text(contract: QueryContract) -> str:
    target_text = " ".join(contract.targets).strip()
    requested_fields = {" ".join(str(item or "").lower().split()) for item in contract.requested_fields if item}
    goals = research_plan_goals(contract)
    parts: list[str] = [target_text or contract.clean_query]
    if target_text and contract.clean_query and contract.clean_query.lower() not in target_text.lower():
        parts.append(contract.clean_query)
    if goals & {"entity_type", "role_in_context"}:
        if goals & {"mechanism", "formula", "variable_explanation"} or requested_fields & {
            "workflow",
            "objective",
            "reward_signal",
            "training_signal",
        }:
            parts.extend(["algorithm", "objective", "reward", "advantage", "workflow", "机制", "目标", "奖励"])
        elif requested_fields & {"definition", "applications", "key_features"}:
            parts.extend(["definition", "method", "algorithm", "benchmark", "定义", "方法"])
    if goals & {"definition", "mechanism", "examples"} and "entity_type" not in goals:
        parts.extend(["definition", "mechanism", "example", "定义", "原理"])
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = " ".join(str(part or "").split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return " ".join(deduped).strip()


def should_use_concept_evidence(contract: QueryContract) -> bool:
    goals = research_plan_goals(contract)
    return bool(goals & {"definition", "mechanism", "examples"}) and "entity_type" not in goals


def is_short_acronym(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9\-]{1,7}", str(text or "").strip()))


def should_use_web_search(*, use_web_search: bool, contract: QueryContract) -> bool:
    return bool(use_web_search or query_needs_external_search(contract.clean_query))
