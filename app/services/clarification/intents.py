from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from app.domain.models import CandidatePaper, DisambiguationJudgeDecision, EvidenceBlock, QueryContract, SessionContext, VerificationReport
from app.services.contracts.context import (
    contract_answer_slots,
    contract_has_note,
    contract_note_json_value,
    contract_note_value,
    contract_note_values,
    contract_notes,
    contract_notes_without_prefixes,
)
from app.services.contracts.normalization import normalize_lookup_text
from app.services.intents.marker_matching import MarkerProfile, query_matches_any
from app.services.planning.query_shaping import extract_targets, is_short_acronym, matches_target
from app.services.planning.research import research_plan_context_from_contract
from app.services.contracts.session_context import truncate_context_text


CLARIFICATION_OPTION_SCHEMA_VERSION = "clarification_option.v1"
AMBIGUITY_RESOLUTION_STAGE = "ambiguity_resolution"
AMBIGUITY_DETECTION_STAGE = "ambiguity_detection"

CLARIFICATION_CHOICE_MARKERS: MarkerProfile = (
    "我说",
    "选",
    "选择",
    "就是",
    "应该是",
    "指的是",
    "the one",
    "choose",
    "select",
)

CLARIFICATION_ORDINAL_PATTERNS: tuple[tuple[int, MarkerProfile], ...] = (
    (
        0,
        ("第一个", "第一项", "第1个", "第 1 个", "第1项", "第 1 项", "first", "the first"),
    ),
    (
        1,
        ("第二个", "第二项", "第2个", "第 2 个", "第2项", "第 2 项", "second", "the second"),
    ),
    (
        2,
        ("第三个", "第三项", "第3个", "第 3 个", "第3项", "第 3 项", "third", "the third"),
    ),
    (
        3,
        ("第四个", "第四项", "第4个", "第 4 个", "第4项", "第 4 项", "fourth", "the fourth"),
    ),
)

CLARIFICATION_INTENT_MARKERS: dict[str, MarkerProfile] = {
    "choice": CLARIFICATION_CHOICE_MARKERS,
}


@dataclass(frozen=True)
class DisambiguationResolutionDecision:
    auto_resolve: bool
    selected_option: dict[str, Any] | None
    contract: QueryContract
    options: list[dict[str, Any]]
    verification: VerificationReport | None
    observation_tool: str
    observation_summary: str
    observation_payload: dict[str, Any]


def looks_like_clarification_choice_text(normalized_query: str) -> bool:
    return query_matches_any(normalized_query, "", CLARIFICATION_INTENT_MARKERS["choice"])


def pending_clarification_selection_index(query: str) -> int | None:
    compact = " ".join(str(query or "").strip().lower().split())
    if not compact:
        return None
    digit_match = re.search(r"(?<!\d)([1-9])(?!\d)", compact)
    if digit_match:
        return int(digit_match.group(1)) - 1
    for index, markers in CLARIFICATION_ORDINAL_PATTERNS:
        if query_matches_any(compact, "", markers):
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


def disambiguation_content_tokens(text: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "its",
        "main",
        "of",
        "on",
        "or",
        "our",
        "the",
        "this",
        "to",
        "we",
        "with",
    }
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return [token for token in tokens if len(token) > 1 and token not in stopwords]


def candidate_title_alignment_score(
    *,
    target: str,
    label: str,
    meaning: str,
    title_alias_text: str,
) -> float:
    title_key = normalize_lookup_text(title_alias_text)
    if not title_key:
        return 0.0
    probes = [target, label, meaning]
    scores: list[float] = []
    for probe in probes:
        probe_key = normalize_lookup_text(probe)
        if not probe_key:
            continue
        if probe_key and probe_key in title_key:
            scores.append(1.0)
            continue
        probe_tokens = disambiguation_content_tokens(probe_key)
        if not probe_tokens:
            continue
        title_tokens = set(disambiguation_content_tokens(title_key))
        if not title_tokens:
            continue
        overlap = len([token for token in probe_tokens if token in title_tokens])
        if overlap:
            scores.append(overlap / max(1, len(probe_tokens)))
    return max(scores or [0.0])


def candidate_origin_signal_score(text: str) -> float:
    lowered = str(text or "").lower()
    patterns = [
        r"\bour\s+main\s+contribution\b",
        r"\bwe\s+(?:propose|proposed|present|introduce|introduced|derive|develop)\b",
        r"\bthis\s+paper\s+(?:proposes|introduces|presents|derives|develops)\b",
        r"\bmain\s+contribution\b",
        r"\bpropose\s+(?:a|an|the)?\b",
        r"\bintroduce\s+(?:a|an|the)?\b",
    ]
    score = 0.0
    for pattern in patterns:
        if re.search(pattern, lowered):
            score += 0.35
    return min(score, 1.0)


def candidate_usage_signal_score(text: str) -> float:
    lowered = str(text or "").lower()
    patterns = [
        r"\badopt(?:s|ed|ing)?\b",
        r"\buse(?:s|d|ing)?\b",
        r"\bfollowing\b",
        r"\bbased\s+on\b",
        r"\bextends?\b",
        r"\bvariant\s+of\b",
        r"\bin\s+recent\s+work\b",
        r"\bproposed\s+by\b",
        r"\bet\s+al\.\s+(?:proposed|introduced)\b",
        r"\binclude(?:s|d|ing)?\b",
    ]
    score = 0.0
    for pattern in patterns:
        if re.search(pattern, lowered):
            score += 0.25
    return min(score, 1.0)


def disambiguation_ranking_signals(
    *,
    option: dict[str, Any],
    paper: CandidatePaper | None,
) -> dict[str, Any]:
    title = str(option.get("title", "") or "").strip()
    target = str(option.get("target", "") or "").strip()
    label = str(option.get("label", "") or "").strip()
    meaning = str(option.get("meaning", "") or "").strip()
    snippet = str(option.get("snippet", "") or "").strip()
    aliases = str((paper.metadata or {}).get("aliases", "") or "") if paper is not None else ""
    summary = (
        str((paper.metadata or {}).get("paper_card_text", "") or "")
        or str((paper.metadata or {}).get("generated_summary", "") or "")
        or str((paper.metadata or {}).get("abstract_note", "") or "")
        if paper is not None
        else ""
    )
    context = "\n".join([title, aliases, label, meaning, snippet, summary])
    title_alias_text = "\n".join([title, aliases])
    title_alignment = candidate_title_alignment_score(
        target=target,
        label=label,
        meaning=meaning,
        title_alias_text=title_alias_text,
    )
    origin_score = candidate_origin_signal_score(context)
    usage_score = candidate_usage_signal_score(context)
    role = "ambiguous"
    if title_alignment >= 0.75 and origin_score >= 0.35:
        role = "direct_definition_or_origin"
    elif usage_score >= max(0.35, origin_score):
        role = "related_usage_or_citation"
    elif title_alignment >= 0.75:
        role = "strong_title_or_alias_alignment"
    return {
        "title_or_alias_alignment": round(title_alignment, 3),
        "origin_or_direct_definition_signal": round(origin_score, 3),
        "usage_or_citation_signal": round(usage_score, 3),
        "candidate_role_hint": role,
    }


def disambiguation_judge_option_payload(*, option: dict[str, Any], paper: CandidatePaper | None) -> dict[str, Any]:
    paper_id = str(option.get("paper_id", "") or "").strip()
    metadata = dict(paper.metadata or {}) if paper is not None else {}
    signals = disambiguation_ranking_signals(option=option, paper=paper)
    return {
        "option_id": str(option.get("option_id", "") or "").strip(),
        "index": option.get("index"),
        "kind": str(option.get("kind", "") or "").strip(),
        "target": str(option.get("target", "") or "").strip(),
        "label": str(option.get("label", "") or "").strip(),
        "meaning": str(option.get("meaning", "") or "").strip(),
        "paper_id": paper_id,
        "title": str(option.get("title", "") or "").strip(),
        "year": str(option.get("year", "") or "").strip(),
        "snippet": truncate_context_text(str(option.get("snippet", "") or ""), limit=420),
        "match_reason": str(option.get("match_reason", "") or (paper.match_reason if paper is not None else "") or "").strip(),
        "evidence_relation": str(option.get("source_relation", "") or option.get("source", "") or "").strip(),
        "source_requested_fields": clarification_string_list(option.get("source_requested_fields")),
        "source_answer_slots": clarification_string_list(option.get("source_answer_slots")),
        "paper_aliases": truncate_context_text(str(metadata.get("aliases", "") or ""), limit=220),
        "paper_summary": truncate_context_text(
            str(
                metadata.get("paper_card_text", "")
                or metadata.get("generated_summary", "")
                or metadata.get("abstract_note", "")
                or ""
            ),
            limit=420,
        ),
        "ranking_signals": signals,
    }


def selected_option_from_judge_decision(
    *,
    decision: DisambiguationJudgeDecision | None,
    options: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if decision is None:
        return None
    selected_option_id = str(decision.selected_option_id or "").strip()
    if selected_option_id:
        for option in options:
            if str(option.get("option_id", "") or "").strip() == selected_option_id:
                return option
    selected_paper_id = str(decision.selected_paper_id or "").strip()
    if selected_paper_id:
        for option in options:
            if str(option.get("paper_id", "") or "").strip() == selected_paper_id:
                return option
    return None


def judge_allows_auto_resolve(
    decision: DisambiguationJudgeDecision | None,
    *,
    threshold: float,
) -> bool:
    return (
        decision is not None
        and decision.decision == "auto_resolve"
        and float(decision.confidence) >= threshold
    )


def apply_disambiguation_judge_recommendation(
    *,
    options: list[dict[str, Any]],
    decision: DisambiguationJudgeDecision | None,
    recommend_threshold: float,
) -> list[dict[str, Any]]:
    selected = selected_option_from_judge_decision(decision=decision, options=options)
    if selected is None or decision is None or float(decision.confidence) < recommend_threshold:
        return options
    rejected_reasons = {
        str(item.option_id or "").strip(): str(item.reason or "").strip()
        for item in decision.rejected_options
        if str(item.option_id or "").strip()
    }
    selected_id = str(selected.get("option_id", "") or "").strip()
    annotated: list[dict[str, Any]] = []
    for option in options:
        payload = dict(option)
        option_id = str(payload.get("option_id", "") or "").strip()
        payload["display_title"] = str(payload.get("display_title", "") or payload.get("title", "") or "").strip()
        if option_id == selected_id:
            payload["display_label"] = str(payload.get("display_label", "") or "推荐候选").strip()
            payload["display_reason"] = truncate_context_text(str(decision.reason or ""), limit=180)
            payload["judge_recommended"] = True
            payload["disambiguation_confidence"] = round(float(decision.confidence), 3)
        elif option_id in rejected_reasons:
            payload["display_reason"] = truncate_context_text(rejected_reasons[option_id], limit=180)
        annotated.append(payload)
    annotated.sort(
        key=lambda item: (
            str(item.get("option_id", "") or "") != selected_id,
            int(item.get("index", 9999) if isinstance(item.get("index"), int) else 9999),
        )
    )
    return annotated


def contract_with_auto_resolved_ambiguity(
    *,
    contract: QueryContract,
    selected: dict[str, Any],
    decision: DisambiguationJudgeDecision | None,
) -> QueryContract:
    notes = contract_notes_without_prefixes(
        contract,
        prefixes={
            "ambiguity_option=",
            "selected_ambiguity_option=",
            "selected_paper_id=",
            "disambiguation_judge_",
        },
    )
    selected_payload = clarification_option_public_payload(selected)
    notes.append("auto_resolved_by_llm_judge")
    notes.append("selected_ambiguity_option=" + json.dumps(selected_payload, ensure_ascii=False))
    paper_id = str(selected.get("paper_id", "") or "").strip()
    if paper_id:
        notes.append(f"selected_paper_id={paper_id}")
    if decision is not None:
        notes.append(f"disambiguation_judge_confidence={float(decision.confidence):.3f}")
        reason = truncate_context_text(str(decision.reason or ""), limit=220)
        if reason:
            notes.append(f"disambiguation_judge_reason={reason}")
    return contract.model_copy(update={"notes": list(dict.fromkeys(notes))})


def disambiguation_judge_summary(
    *,
    options: list[dict[str, Any]],
    judge_decision: DisambiguationJudgeDecision | None,
) -> str:
    if judge_decision is None:
        return f"options={len(options)}, judge=unavailable"
    return (
        f"options={len(options)}, judge={judge_decision.decision}, "
        f"confidence={float(judge_decision.confidence):.2f}"
    )


def resolve_disambiguation_judge_decision(
    *,
    contract: QueryContract,
    options: list[dict[str, Any]],
    judge_decision: DisambiguationJudgeDecision | None,
    auto_resolve_threshold: float,
    recommend_threshold: float,
) -> DisambiguationResolutionDecision:
    selected_option = selected_option_from_judge_decision(
        decision=judge_decision,
        options=options,
    )
    auto_resolve = selected_option is not None and judge_allows_auto_resolve(
        judge_decision,
        threshold=auto_resolve_threshold,
    )
    observation_payload = {
        "stage": AMBIGUITY_RESOLUTION_STAGE,
        "options": options[:4],
        "judge_decision": judge_decision.model_dump() if judge_decision is not None else {},
    }
    if auto_resolve and selected_option is not None:
        resolved_contract = contract_with_auto_resolved_ambiguity(
            contract=contract,
            selected=selected_option,
            decision=judge_decision,
        )
        return DisambiguationResolutionDecision(
            auto_resolve=True,
            selected_option=selected_option,
            contract=resolved_contract,
            options=options,
            verification=None,
            observation_tool="compose",
            observation_summary=disambiguation_judge_summary(options=options, judge_decision=judge_decision),
            observation_payload=observation_payload,
        )
    annotated_options = apply_disambiguation_judge_recommendation(
        options=options,
        decision=judge_decision,
        recommend_threshold=recommend_threshold,
    )
    clarified_contract = contract_with_ambiguity_options(contract=contract, options=annotated_options)
    return DisambiguationResolutionDecision(
        auto_resolve=False,
        selected_option=selected_option,
        contract=clarified_contract,
        options=annotated_options,
        verification=VerificationReport(
            status="clarify",
            missing_fields=disambiguation_missing_fields(clarified_contract),
            recommended_action="clarify_ambiguous_entity",
        ),
        observation_tool="ask_human",
        observation_summary=disambiguation_judge_summary(options=options, judge_decision=judge_decision),
        observation_payload={**observation_payload, "stage": AMBIGUITY_DETECTION_STAGE},
    )


def disambiguation_judge_system_prompt() -> str:
    return (
        "你是通用论文/实体候选消歧裁判器。"
        "你的唯一任务是在用户 query、QueryContract 和候选 metadata/snippet 之间判断哪个候选最符合用户真实意图。"
        "不要硬编码任何具体缩写、方法名、论文标题或 paper_id 的默认答案；只能基于输入字段做判断。"
        "不要生成最终研究答案，不要补充外部知识，只决定是否可自动绑定候选。"
        "当 query 明确指向某个候选的标题、上下文、方法原始提出/直接定义证据，且其他候选只是引用、应用、比较或弱相关时，可以 auto_resolve。"
        "输入中的 ranking_signals 是由候选自身 title/snippet/metadata 计算出的通用线索，不是某个缩写的白名单；"
        "当 direct_definition_or_origin、strong_title_or_alias_alignment 明显集中在同一个候选，"
        "而其他候选主要是 related_usage_or_citation 时，不要因为候选数量多而保守降到 0.70，应给出 >=0.85 的自动消歧。"
        "如果证据不足、候选关系接近、query 可能指向多篇论文，必须 ask_human。"
        "输出必须是 JSON，字段为 decision(auto_resolve|ask_human), selected_option_id, selected_paper_id, confidence, reason, rejected_options。"
        "confidence >= 0.85 才表示可自动消歧；0.65 到 0.85 只表示可作为推荐项；低于 0.65 不要默认推荐。"
    )


def disambiguation_judge_human_prompt(
    *,
    contract: QueryContract,
    candidate_options: list[dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "user_query": contract.clean_query,
            "query_contract": {
                "relation": contract.relation,
                "targets": contract.targets,
                "answer_slots": contract_answer_slots(contract),
                "requested_fields": contract.requested_fields,
                "required_modalities": contract.required_modalities,
                "answer_shape": contract.answer_shape,
                "precision_requirement": contract.precision_requirement,
                "continuation_mode": contract.continuation_mode,
                "notes": [
                    note
                    for note in contract_notes(contract)
                    if not str(note).startswith("ambiguity_option=")
                ][:12],
            },
            "candidate_options": candidate_options,
            "output_schema": {
                "decision": "auto_resolve | ask_human",
                "selected_option_id": "string|null",
                "selected_paper_id": "string|null",
                "confidence": "number in [0,1]",
                "reason": "short reason based only on provided metadata/snippets",
                "rejected_options": [{"option_id": "string", "reason": "string"}],
            },
        },
        ensure_ascii=False,
    )


def disambiguation_goal_markers() -> set[str]:
    return {"definition", "entity_type", "role_in_context", "mechanism", "formula"}


def contract_needs_evidence_disambiguation(contract: QueryContract) -> bool:
    if not contract.targets:
        return False
    target = str(contract.targets[0] or "").strip()
    if not is_short_acronym(target):
        return False
    if contract_note_values(contract, prefix="ambiguous_slot="):
        return True
    return bool(set(research_plan_context_from_contract(contract).goals) & disambiguation_goal_markers())


def disambiguation_missing_fields(contract: QueryContract) -> list[str]:
    ambiguous_slots = contract_note_values(contract, prefix="ambiguous_slot=")
    return ambiguous_slots or ["disambiguation"]


def extract_acronym_expansion_from_text(*, text: str, acronym: str) -> str:
    compact = " ".join(str(text or "").split())
    if not compact or not acronym:
        return ""
    patterns = [
        rf"([A-Za-z][A-Za-z\-/]+(?:\s+[A-Za-z][A-Za-z\-/]+){{1,8}})\s*\(\s*{re.escape(acronym)}\s*\)",
        rf"{re.escape(acronym)}\s*(?:stands for|means|refers to|is short for)\s*([A-Za-z][A-Za-z\-/]+(?:\s+[A-Za-z][A-Za-z\-/]+){{1,8}})",
    ]
    stopwords = {"and", "or", "the", "with", "from", "into", "using", "based"}
    for pattern in patterns:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if not match:
            continue
        expansion = " ".join(match.group(1).strip(" ,.;:-").split())
        expansion = re.sub(r"^and(?=[A-Z])", "", expansion).strip()
        expansion = re.sub(r"^and\s+", "", expansion, flags=re.IGNORECASE).strip()
        words = expansion.split()
        while words and words[0].lower() in stopwords:
            words.pop(0)
        expansion = " ".join(words)
        if len(expansion) >= 6:
            return expansion
    return ""


def normalize_acronym_meaning(text: str) -> str:
    normalized = str(text or "").lower().replace("behaviour", "behavior")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def ambiguity_option_context_text(option: dict[str, Any], *, paper: CandidatePaper | None = None) -> str:
    parts = [
        str(option.get("meaning", "")),
        str(option.get("title", "")),
        str(option.get("snippet", "")),
    ]
    if paper is not None:
        parts.extend(
            [
                str(paper.metadata.get("aliases", "")),
                str(paper.metadata.get("paper_card_text", "")),
                str(paper.metadata.get("generated_summary", "")),
                str(paper.metadata.get("abstract_note", "")),
            ]
        )
    return "\n".join(part for part in parts if part)


def ambiguity_option_matches_context(*, option: dict[str, Any], context_targets: list[str]) -> bool:
    text = str(option.get("context_text", ""))
    return any(matches_target(text, str(target)) for target in context_targets if str(target).strip())


def acronym_options_from_evidence(
    *,
    target: str,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    paper_lookup: Callable[[str], CandidatePaper | None],
) -> list[dict[str, Any]]:
    paper_by_id = {item.paper_id: item for item in papers}
    buckets: dict[str, dict[str, Any]] = {}
    target_key = target.lower()
    for item in evidence:
        if not any(matches_target(haystack, target) for haystack in [item.snippet, item.caption, item.title] if haystack):
            continue
        paper = paper_by_id.get(item.paper_id) or paper_lookup(item.paper_id)
        if paper is None:
            continue
        text = " ".join([item.snippet, item.caption, item.title])
        expansion = extract_acronym_expansion_from_text(text=text, acronym=target)
        option_key = normalize_acronym_meaning(expansion) if expansion else normalize_lookup_text(f"{target_key}:{paper.paper_id}")
        if not option_key:
            option_key = f"{target_key}:{paper.paper_id}"
        bucket = buckets.setdefault(
            option_key,
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "meaning": expansion or target,
                "snippet": "",
                "score": 0.0,
                "paper_ids": [],
                "titles": [],
            },
        )
        bucket["score"] = float(bucket.get("score", 0.0)) + float(item.score) + (5.0 if expansion else 0.0)
        if not bucket.get("snippet"):
            bucket["snippet"] = " ".join(item.snippet.split())[:220]
        if paper.paper_id not in bucket["paper_ids"]:
            bucket["paper_ids"].append(paper.paper_id)
        if paper.title not in bucket["titles"]:
            bucket["titles"].append(paper.title)
        if expansion and len(expansion) > len(str(bucket.get("meaning", ""))):
            bucket["meaning"] = expansion
    options = list(buckets.values())
    expanded_papers = {
        str(option.get("paper_id", ""))
        for option in options
        if str(option.get("meaning", "")).strip().lower() != target.lower()
    }
    options = [
        option
        for option in options
        if not (
            str(option.get("paper_id", "")) in expanded_papers
            and str(option.get("meaning", "")).strip().lower() == target.lower()
        )
    ]
    if any(str(option.get("meaning", "")).strip().lower() != target.lower() for option in options):
        options = [option for option in options if str(option.get("meaning", "")).strip().lower() != target.lower()]
    for option in options:
        paper_id = str(option.get("paper_id", "") or "")
        paper = paper_by_id.get(paper_id) or paper_lookup(paper_id)
        option["context_text"] = ambiguity_option_context_text(option, paper=paper)
    options.sort(key=lambda item: (-float(item.get("score", 0.0)), str(item.get("title", ""))))
    return options


def acronym_evidence_from_corpus(
    *,
    target: str,
    limit: int,
    paper_documents: Callable[[], list[Any]],
    block_documents_for_paper: Callable[[str, int], list[Any]],
) -> list[EvidenceBlock]:
    evidence: list[EvidenceBlock] = []
    for paper_doc in paper_documents():
        paper_id = str((paper_doc.metadata or {}).get("paper_id", "") or "").strip()
        if not paper_id:
            continue
        for doc in block_documents_for_paper(paper_id, 320):
            meta = dict(doc.metadata or {})
            text = str(doc.page_content or "")
            if not matches_target(text, target):
                continue
            score = 1.0
            expansion = extract_acronym_expansion_from_text(text=text, acronym=target)
            if expansion:
                score += 6.0
            if int(meta.get("formula_hint", 0) or 0):
                score += 2.0
            evidence.append(
                EvidenceBlock(
                    doc_id=str(meta.get("doc_id", "")),
                    paper_id=paper_id,
                    title=str(meta.get("title", "")),
                    file_path=str(meta.get("file_path", "")),
                    page=int(meta.get("page", 0) or 0),
                    block_type=str(meta.get("block_type", "")),
                    caption=str(meta.get("caption", "")),
                    bbox=str(meta.get("bbox", "")),
                    snippet=text[:900],
                    score=score,
                    metadata=meta,
                )
            )
            if len(evidence) >= limit:
                return evidence
    evidence.sort(key=lambda item: (-item.score, item.title, item.page))
    return evidence[:limit]


def finalize_acronym_disambiguation_options(
    *,
    options: list[dict[str, Any]],
    contract: QueryContract,
    target: str,
    excluded_titles: set[str],
) -> list[dict[str, Any]]:
    if excluded_titles:
        options = [
            option
            for option in options
            if normalize_lookup_text(str(option.get("title", ""))) not in excluded_titles
        ]
    if len(options) < 2:
        return []
    context_targets = [item for item in contract.targets[1:] if str(item).strip()]
    if context_targets:
        matched = [
            option
            for option in options
            if ambiguity_option_matches_context(option=option, context_targets=context_targets)
        ]
        if len(matched) <= 1:
            return []
        options = matched
    if contract_has_note(contract, "exclude_previous_focus") and len(options) <= 1:
        return []
    return normalize_clarification_options(
        options[:4],
        contract=contract,
        target=target,
        kind="acronym_meaning",
        source="evidence_disambiguation",
    )


def evidence_disambiguation_options(
    *,
    contract: QueryContract,
    target_binding_exists: bool,
    is_negative_correction: bool,
    initial_options: Callable[[], list[dict[str, Any]]],
    broad_options: Callable[[], list[dict[str, Any]]],
    corpus_options: Callable[[], list[dict[str, Any]]],
    excluded_titles: set[str],
) -> list[dict[str, Any]]:
    if not contract_needs_evidence_disambiguation(contract):
        return []
    if contract_has_note(contract, "resolved_human_choice") or selected_clarification_paper_id(contract):
        return []
    target = str(contract.targets[0] or "").strip()
    if not is_negative_correction and not contract_has_note(contract, "exclude_previous_focus") and target_binding_exists:
        return []
    options = initial_options()
    goals = set(research_plan_context_from_contract(contract).goals)
    if len(options) < 2 and "formula" in goals:
        candidates = broad_options()
        if len(candidates) > len(options):
            options = candidates
    if len(options) < 2 and "formula" in goals:
        candidates = corpus_options()
        if len(candidates) > len(options):
            options = candidates
    return finalize_acronym_disambiguation_options(
        options=options,
        contract=contract,
        target=target,
        excluded_titles=excluded_titles,
    )


def normalize_clarification_option(
    option: dict[str, Any],
    *,
    index: int,
    contract: QueryContract,
    target: str = "",
    kind: str = "paper_choice",
    source: str = "clarification",
) -> dict[str, Any]:
    payload = dict(option)
    resolved_target = str(payload.get("target", "") or target or (contract.targets[0] if contract.targets else "") or "").strip()
    resolved_kind = str(payload.get("kind", "") or kind or "paper_choice").strip()
    meaning = str(payload.get("meaning", "") or "").strip()
    title = str(payload.get("title", "") or "").strip()
    year = str(payload.get("year", "") or "").strip()
    label = str(payload.get("label", "") or meaning or title or resolved_target or f"option {index + 1}").strip()
    description = str(payload.get("description", "") or "").strip()
    if not description:
        description = clarification_option_description(payload, title=title, year=year)
    payload["schema_version"] = CLARIFICATION_OPTION_SCHEMA_VERSION
    payload["index"] = index
    payload["kind"] = resolved_kind
    payload["target"] = resolved_target
    payload["label"] = label
    payload["description"] = truncate_context_text(description, limit=260) if description else ""
    payload.setdefault("meaning", meaning or label)
    payload.setdefault("title", title)
    payload.setdefault("year", year)
    payload["display_title"] = str(payload.get("display_title", "") or title).strip()
    payload["display_label"] = str(payload.get("display_label", "") or "").strip()
    payload["display_reason"] = truncate_context_text(str(payload.get("display_reason", "") or ""), limit=220)
    if "disambiguation_confidence" in payload:
        try:
            payload["disambiguation_confidence"] = round(float(payload.get("disambiguation_confidence") or 0.0), 3)
        except (TypeError, ValueError):
            payload.pop("disambiguation_confidence", None)
    payload.setdefault("source", source)
    payload["source_relation"] = str(payload.get("source_relation", "") or contract.relation)
    payload["source_requested_fields"] = clarification_string_list(payload.get("source_requested_fields") or contract.requested_fields)
    payload["source_required_modalities"] = clarification_string_list(payload.get("source_required_modalities") or contract.required_modalities)
    payload["source_answer_slots"] = clarification_string_list(payload.get("source_answer_slots") or contract.answer_slots)
    payload["paper_ids"] = clarification_string_list(payload.get("paper_ids"))
    payload["titles"] = clarification_string_list(payload.get("titles"))
    payload["evidence_ids"] = clarification_string_list(payload.get("evidence_ids"))
    payload["option_id"] = str(payload.get("option_id", "") or "").strip() or clarification_option_id(
        kind=resolved_kind,
        target=resolved_target,
        label=label,
        paper_id=str(payload.get("paper_id", "") or ""),
        title=title,
        index=index,
    )
    return payload


def normalize_clarification_options(
    options: list[dict[str, Any]],
    *,
    contract: QueryContract,
    target: str = "",
    kind: str = "paper_choice",
    source: str = "clarification",
) -> list[dict[str, Any]]:
    return [
        normalize_clarification_option(
            option,
            index=index,
            contract=contract,
            target=target,
            kind=kind,
            source=source,
        )
        for index, option in enumerate(options)
    ]


def clarification_options_from_contract_notes(contract: QueryContract) -> list[dict[str, Any]]:
    target = contract.targets[0] if contract.targets else ""
    return normalize_clarification_options(
        ambiguity_options_from_notes(contract_notes(contract)),
        contract=contract,
        target=target,
        kind="acronym_meaning",
        source="contract_notes",
    )


def clarification_tracking_key(
    *,
    contract: QueryContract,
    verification: VerificationReport,
    options: list[dict[str, Any]],
) -> str:
    option_key = "|".join(
        str(option.get("option_id") or option.get("paper_id") or option.get("meaning") or option.get("title") or "")
        for option in options[:4]
    )
    target_key = ",".join(normalize_lookup_text(item) for item in contract.targets if item)
    missing_key = ",".join(str(item) for item in verification.missing_fields)
    return "|".join(
        [
            contract.relation,
            target_key,
            verification.recommended_action,
            missing_key,
            option_key,
        ]
    )


def next_clarification_attempt(*, session: SessionContext, key: str) -> int:
    if key and key == session.last_clarification_key:
        return session.clarification_attempts + 1
    return 1


def remember_clarification_attempt(*, session: SessionContext, key: str) -> None:
    if key and key == session.last_clarification_key:
        session.clarification_attempts += 1
    else:
        session.last_clarification_key = key
        session.clarification_attempts = 1


def reset_clarification_tracking(session: SessionContext) -> None:
    session.last_clarification_key = ""
    session.clarification_attempts = 0


def store_pending_clarification(*, session: SessionContext, contract: QueryContract, options: list[dict[str, Any]]) -> None:
    if options:
        session.pending_clarification_type = "ambiguity"
        session.pending_clarification_target = contract.targets[0] if contract.targets else ""
        session.pending_clarification_options = options
    else:
        clear_pending_clarification(session)


def clear_pending_clarification(session: SessionContext) -> None:
    session.pending_clarification_type = ""
    session.pending_clarification_target = ""
    session.pending_clarification_options = []


def contract_from_pending_clarification(
    *,
    clean_query: str,
    session: SessionContext,
    clarification_choice: dict[str, Any] | None = None,
) -> QueryContract | None:
    if session.pending_clarification_type != "ambiguity" or not session.pending_clarification_options:
        return None
    selected = option_from_clarification_choice(clarification_choice, session.pending_clarification_options)
    if selected is None:
        selected = select_pending_clarification_option(
            clean_query=clean_query,
            options=session.pending_clarification_options,
        )
    if selected is None:
        return None
    target = session.pending_clarification_target or str(selected.get("target", "") or "").strip()
    if not target:
        target = " ".join(extract_targets(clean_query)[:1])
    return contract_from_selected_clarification_option(
        clean_query=clean_query,
        target=target,
        selected=selected,
    )


def ambiguity_clarification_question(*, contract: QueryContract, session: SessionContext) -> str:
    ambiguity_options = ambiguity_options_from_notes(contract_notes(contract))
    if (
        not ambiguity_options
        and session.pending_clarification_type == "ambiguity"
        and session.pending_clarification_options
        and (
            not contract.targets
            or not session.pending_clarification_target
            or normalize_lookup_text(session.pending_clarification_target)
            in {normalize_lookup_text(target) for target in contract.targets}
        )
    ):
        ambiguity_options = list(session.pending_clarification_options)
    if not ambiguity_options:
        return ""
    target = contract.targets[0] if contract.targets else "这个缩写"
    ambiguity_options = normalize_clarification_options(
        ambiguity_options,
        contract=contract,
        target=target,
        kind="acronym_meaning",
        source="clarification_question",
    )
    lines = [f"`{target}` 在本地论文库里有多个可能含义，我不应该继续猜。你想问哪一个？"]
    for index, option in enumerate(ambiguity_options, start=1):
        display_label = str(option.get("display_label", "") or "").strip()
        base_label = str(option.get("label", "") or option.get("meaning", "") or target).strip()
        meaning = f"{display_label}：{base_label}" if display_label and base_label else (display_label or base_label)
        title = str(option.get("display_title", "") or option.get("title", "")).strip()
        year = str(option.get("year", "")).strip()
        suffix = f"（{year}）" if year else ""
        reason = str(option.get("display_reason", "") or "").strip()
        reason_suffix = f"：{reason}" if reason else ""
        lines.append(f"{index}. {meaning}，见《{title}》{suffix}{reason_suffix}")
    return "\n".join(lines)


def selected_clarification_paper_id(contract: QueryContract) -> str:
    selected_paper_id = contract_note_value(contract, prefix="selected_paper_id=")
    if selected_paper_id:
        return selected_paper_id.strip()
    payload = contract_note_json_value(contract, prefix="selected_ambiguity_option=")
    return str(payload.get("paper_id", "") or "").strip()


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
    notes = contract_notes_without_prefixes(contract, prefixes={"ambiguity_option="})
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
