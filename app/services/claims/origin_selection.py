from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract
from app.services.answers.evidence_presentation import safe_year


ORIGIN_INTRO_MARKER_RE = re.compile(
    r"\b(introduce|introduces|introduced|propose|proposes|proposed)\b",
    re.IGNORECASE,
)


def origin_paper_text(paper: CandidatePaper) -> str:
    return "\n".join(
        [
            paper.title,
            str(paper.metadata.get("aliases", "")),
            str(paper.metadata.get("paper_card_text", "")),
            str(paper.metadata.get("generated_summary", "")),
            str(paper.metadata.get("abstract_note", "")),
        ]
    )


def origin_target_aliases(targets: list[str]) -> list[str]:
    aliases: list[str] = []
    suffixes = [
        "架构",
        "模型",
        "方法",
        "算法",
        "数据集",
        "基准",
        "论文",
        " architecture",
        " model",
        " method",
        " algorithm",
        " dataset",
        " benchmark",
        " paper",
    ]
    for target in targets:
        raw = str(target or "").strip()
        if not raw:
            continue
        variants = [raw]
        compact = re.sub(r"[^A-Za-z0-9]", "", raw)
        spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", compact).strip()
        if spaced and spaced.lower() != raw.lower():
            variants.extend([spaced, spaced.upper()])
        lowered = raw.lower()
        for suffix in suffixes:
            if lowered.endswith(suffix):
                variants.append(raw[: len(raw) - len(suffix)].strip())
        for variant in variants:
            if variant and variant.lower() not in {item.lower() for item in aliases}:
                aliases.append(variant)
    return aliases


def origin_display_entity(*, targets: list[str], paper: CandidatePaper) -> str:
    fallback = str(targets[0] if targets else "").strip()
    aliases = origin_target_aliases(targets)
    text = origin_paper_text(paper)
    for alias in aliases:
        alias = str(alias or "").strip()
        if not alias:
            continue
        pattern = re.compile(rf"(?<![a-z0-9\-]){re.escape(alias)}(?![a-z0-9\-])", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return " ".join(match.group(0).split())
    return fallback


def origin_lookup_claim(*, contract: QueryContract, paper: CandidatePaper, evidence_ids: list[str]) -> Claim:
    return Claim(
        claim_type="origin",
        entity=origin_display_entity(targets=contract.targets, paper=paper),
        value=paper.title,
        structured_data={"year": paper.year, "paper_title": paper.title},
        evidence_ids=list(evidence_ids) or list(paper.doc_ids[:1]),
        paper_ids=[paper.paper_id],
        confidence=0.94,
    )


def pick_origin_paper(papers: list[CandidatePaper]) -> CandidatePaper | None:
    if not papers:
        return None
    ranked = sorted(papers, key=lambda item: (safe_year(item.year), -item.score))
    return ranked[0] if ranked else None


def select_origin_paper(
    *,
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    paper_documents: Iterable[Any],
    candidate_from_paper_id: Callable[[str], CandidatePaper | None],
    paper_identity_matches_targets: Callable[[CandidatePaper, list[str]], bool],
    target_matcher: Callable[[str, str], bool],
) -> CandidatePaper | None:
    if not papers:
        return None
    candidate_pool = list(papers)
    paper_by_id = {item.paper_id: item for item in candidate_pool}
    for paper in origin_candidates_from_corpus(
        contract=contract,
        paper_documents=paper_documents,
        candidate_from_paper_id=candidate_from_paper_id,
    ):
        if paper.paper_id not in paper_by_id:
            paper_by_id[paper.paper_id] = paper
            candidate_pool.append(paper)
    if contract.targets:
        identity_matched = [
            item
            for item in candidate_pool
            if paper_identity_matches_targets(item, contract.targets)
            or paper_has_origin_intro_support(paper=item, targets=contract.targets)
        ]
        if identity_matched:
            candidate_pool = identity_matched
    if not evidence:
        return pick_origin_paper_with_intro_support(contract=contract, papers=candidate_pool)
    target_aliases = origin_target_aliases(contract.targets)
    scored: list[tuple[float, float, CandidatePaper]] = []
    for paper in candidate_pool:
        support = [item for item in evidence if item.paper_id == paper.paper_id]
        if target_aliases:
            support = [
                item
                for item in support
                if any(
                    target_matcher(haystack, alias)
                    for alias in target_aliases
                    for haystack in [item.snippet, item.caption, item.title]
                    if haystack
                )
            ]
        score = 0.0
        intro_score = 0.0
        for item in support:
            score += float(item.score)
            snippet = item.snippet.lower()
            if ORIGIN_INTRO_MARKER_RE.search(snippet):
                score += 1.5
            if " is a " in snippet or " is an " in snippet:
                score += 0.8
            intro_score += origin_target_intro_score(item.snippet, target_aliases)
            score += origin_target_definition_score(item.snippet, target_aliases)
        if target_aliases:
            paper_text = origin_paper_text(paper)
            if any(target_matcher(paper_text, alias) for alias in target_aliases):
                score += 0.8
            intro_score += origin_target_intro_score(paper_text, target_aliases)
            score += origin_target_definition_score(paper_text, target_aliases)
        scored.append((intro_score, score, paper))
    scored.sort(key=lambda item: (-item[0], -item[1], safe_year(item[2].year), -item[2].score, item[2].title))
    if scored and scored[0][0] > 0:
        return scored[0][2]
    return pick_origin_paper_with_intro_support(contract=contract, papers=candidate_pool)


def origin_candidates_from_corpus(
    *,
    contract: QueryContract,
    paper_documents: Iterable[Any],
    candidate_from_paper_id: Callable[[str], CandidatePaper | None],
) -> list[CandidatePaper]:
    aliases = origin_target_aliases(contract.targets)
    if not aliases:
        return []
    candidates: list[tuple[float, CandidatePaper]] = []
    for doc in paper_documents:
        meta = dict(getattr(doc, "metadata", {}) or {})
        paper_id = str(meta.get("paper_id", "") or "").strip()
        if not paper_id:
            continue
        paper = candidate_from_paper_id(paper_id)
        if paper is None:
            continue
        text = "\n".join(
            [
                str(getattr(doc, "page_content", "") or ""),
                str(meta.get("aliases", "")),
                str(meta.get("abstract_note", "")),
                str(meta.get("generated_summary", "")),
            ]
        )
        score = origin_target_intro_score(text, aliases)
        if score <= 0:
            continue
        candidates.append((score, paper.model_copy(update={"score": max(float(paper.score), score)})))
    candidates.sort(key=lambda item: (-item[0], safe_year(item[1].year), item[1].title))
    return [paper for _, paper in candidates[:8]]


def pick_origin_paper_with_intro_support(
    *,
    contract: QueryContract,
    papers: list[CandidatePaper],
) -> CandidatePaper | None:
    aliases = origin_target_aliases(contract.targets)
    if not aliases:
        return pick_origin_paper(papers)
    scored = [
        (origin_target_intro_score(origin_paper_text(paper), aliases), paper)
        for paper in papers
    ]
    scored = [(score, paper) for score, paper in scored if score > 0]
    scored.sort(key=lambda item: (-item[0], safe_year(item[1].year), -item[1].score, item[1].title))
    return scored[0][1] if scored else None


def origin_target_intro_score(text: str, aliases: list[str]) -> float:
    if not text or not aliases:
        return 0.0
    compact = " ".join(str(text).split())
    if not compact:
        return 0.0
    origin_cue = (
        r"(?:we\s+|this\s+paper\s+|our\s+work\s+)?"
        r"(?:propose|proposes|proposed|introduce|introduces|introduced|present|presents|presented|"
        r"define|defines|defined|construct|constructs|constructed|create|creates|created|release|releases|released|"
        r"提出|引入|介绍|定义|构建|发布)"
    )
    allowed_previous = {
        "the",
        "a",
        "an",
        "as",
        "called",
        "named",
        "propose",
        "proposes",
        "proposed",
        "introduce",
        "introduces",
        "introduced",
        "present",
        "presents",
        "presented",
        "define",
        "defines",
        "defined",
        "construct",
        "constructs",
        "constructed",
        "create",
        "creates",
        "created",
        "release",
        "releases",
        "released",
    }
    score = 0.0
    sentences = re.split(r"(?<=[.!?。！？])\s+|[\n\r]+", compact)
    for alias in aliases:
        alias = str(alias or "").strip()
        if not alias:
            continue
        escaped = re.escape(alias.lower())
        pattern = re.compile(rf"(?<![a-z0-9\-]){escaped}(?![a-z0-9\-])", re.IGNORECASE)
        for sentence in sentences:
            lowered = sentence.lower()
            for match in pattern.finditer(lowered):
                before = lowered[max(0, match.start() - 180) : match.start()]
                after = lowered[match.end() : match.end() + 120]
                previous_words = re.findall(r"[a-z]+", before)
                previous_word = previous_words[-1] if previous_words else ""
                modifier_use = bool(previous_word and previous_word not in allowed_previous)
                cue_matches = list(re.finditer(origin_cue, before, flags=re.IGNORECASE))
                if cue_matches and match.start() - cue_matches[-1].end() <= 150:
                    score += 6.0 if not modifier_use else 1.0
                if re.search(
                    r"\b(is|was|has been)\s+(?:first\s+|originally\s+)?(?:introduced|proposed|presented|defined|constructed|created|released)\b",
                    after,
                ):
                    score += 5.0 if not modifier_use else 1.0
                if re.search(r"(提出|引入|介绍|定义|构建|发布)", before[-120:] + after[:80]):
                    score += 4.0 if not modifier_use else 0.8
    return score


def origin_target_definition_score(text: str, aliases: list[str]) -> float:
    if not text or not aliases:
        return 0.0
    compact = " ".join(str(text).split())
    if not compact:
        return 0.0
    score = 0.0
    score += origin_target_intro_score(compact, aliases)
    sentences = re.split(r"(?<=[.!?。！？])\s+|[\n\r]+", compact)
    for alias in aliases:
        alias = str(alias or "").strip()
        if not alias:
            continue
        escaped = re.escape(alias.lower())
        pattern = re.compile(rf"(?<![a-z0-9\-]){escaped}(?![a-z0-9\-])", re.IGNORECASE)
        for sentence in sentences:
            lowered = sentence.lower()
            for match in pattern.finditer(lowered):
                before = lowered[max(0, match.start() - 160) : match.start()]
                after = lowered[match.end() : match.end() + 120]
                if re.search(r"\b(is|was|are|refers to|denotes)\b.{0,50}\b(architecture|model|method|dataset|benchmark)\b", after):
                    score += 3.0
                if re.search(r"\b(first|original|initial)\b", before[-80:] + after[:80]):
                    score += 1.5
    return score


def paper_has_origin_intro_support(*, paper: CandidatePaper, targets: list[str]) -> bool:
    aliases = origin_target_aliases(targets)
    return bool(aliases and origin_target_intro_score(origin_paper_text(paper), aliases) > 0)
