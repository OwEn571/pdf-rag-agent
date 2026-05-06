from __future__ import annotations

import re

from app.domain.models import QueryContract
from app.services.intents.research import query_needs_external_search
from app.services.planning.research import ResearchPlanContext, research_plan_context_from_contract


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


def query_target_candidates(query: str) -> list[str]:
    """Conservative target extraction used when router tool args omit targets."""
    targets: list[str] = []
    for pattern in [r"[\"“](.+?)[\"”]", r"[‘'](.+?)[’']"]:
        for match in re.finditer(pattern, query):
            candidate = str(match.group(1) or "").strip()
            if candidate and candidate not in targets:
                targets.append(candidate)
    stopwords = {
        "the",
        "this",
        "that",
        "paper",
        "first",
        "which",
        "what",
        "who",
        "origin",
        "proposed",
        "introduced",
    }
    for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,}", query):
        key = token.lower()
        if key in stopwords:
            continue
        looks_like_target = bool(
            re.fullmatch(r"[A-Z][A-Z0-9\-]{1,8}", token)
            or any(ch.isupper() for ch in token[1:])
            or any(ch.isdigit() for ch in token)
            or re.fullmatch(r"[A-Z][a-z][A-Za-z0-9\-]{2,}", token)
        )
        if looks_like_target and token not in targets:
            targets.append(token)
    return targets


def loss_notation_target_aliases(targets: list[str]) -> list[str]:
    aliases: list[str] = []
    for target in targets:
        raw = str(target or "").strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9\-]{1,8}", raw):
            continue
        aliases.extend([f"L_{raw}", f"L{raw}", f"L_{{{raw}}}", f"L_{{\\mathrm{{{raw}}}}}"])
    deduped: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        key = alias.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(alias)
    return deduped


def paper_query_text(contract: QueryContract) -> str:
    return paper_query_text_from_context(research_plan_context_from_contract(contract))


def paper_query_text_from_context(context: ResearchPlanContext) -> str:
    target_text = " ".join(context.targets).strip()
    goals = set(context.goals)
    clean_query = str(context.clean_query or "")
    if target_text and goals & {"definition", "entity_type", "role_in_context", "mechanism", "examples"}:
        if len(context.targets) > 1:
            return f"{target_text} {clean_query}".strip()
        return target_text
    if target_text and target_text.lower() not in clean_query.lower():
        return f"{target_text} {clean_query}".strip()
    return clean_query


def evidence_query_text(contract: QueryContract) -> str:
    return evidence_query_text_from_context(research_plan_context_from_contract(contract))


def evidence_query_text_from_context(context: ResearchPlanContext) -> str:
    target_text = " ".join(context.targets).strip()
    requested_fields = {" ".join(str(item or "").lower().split()) for item in context.requested_fields if item}
    goals = set(context.goals)
    clean_query = str(context.clean_query or "")
    parts: list[str] = [target_text or clean_query]
    if target_text and clean_query and clean_query.lower() not in target_text.lower():
        parts.append(clean_query)
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
    return should_use_concept_evidence_from_context(research_plan_context_from_contract(contract))


def should_use_concept_evidence_from_context(context: ResearchPlanContext) -> bool:
    goals = set(context.goals)
    return bool(goals & {"definition", "mechanism", "examples"}) and "entity_type" not in goals


def is_short_acronym(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9\-]{1,7}", str(text or "").strip()))


def matches_target(text: str, target: str) -> bool:
    raw_text = str(text or "")
    raw_target = str(target or "").strip()
    if not raw_text or not raw_target:
        return False
    if " " not in raw_target and len(raw_target) <= 16 and re.fullmatch(r"[a-z0-9\-]{2,}", raw_target.lower()):
        lowered_text = raw_text.lower()
        target_key = raw_target.lower()
        pattern = re.compile(rf"(?<![a-z0-9\-]){re.escape(target_key)}(?![a-z0-9\-])")
        if pattern.search(lowered_text) is not None:
            return True
        if raw_target.upper() == raw_target and len(raw_target) <= 10:
            loss_patterns = [
                rf"\bl[_\{{\s]*(?:\\mathrm\{{?)?\s*{re.escape(target_key)}\b",
                rf"\bl{re.escape(target_key)}\b",
            ]
            return any(re.search(loss_pattern, lowered_text) for loss_pattern in loss_patterns)
        return False
    return raw_target in raw_text


def should_use_web_search(*, use_web_search: bool, contract: QueryContract) -> bool:
    return bool(use_web_search or query_needs_external_search(contract.clean_query))
