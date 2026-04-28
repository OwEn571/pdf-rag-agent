from __future__ import annotations

import re

from app.domain.models import CandidatePaper


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
