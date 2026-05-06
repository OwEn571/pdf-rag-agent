from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services.contracts.normalization import normalize_lookup_text
from app.services.entities.definition_profiles import ENTITY_DEFINITION_MARKERS
from app.services.entities.supporting_paper_selector import is_noisy_entity_line
from app.services.intents.marker_matching import query_matches_any
from app.services.claims.paper_summary import paper_summary_text
from app.services.infra.prompt_safety import DOCUMENT_SAFETY_INSTRUCTION, wrap_untrusted_document_text

PaperDocLookupFn = Callable[[str], Any]


def entity_supporting_lines(evidence: list[EvidenceBlock], *, kind: str) -> list[str]:
    scored: list[tuple[float, str]] = []
    for item in evidence:
        snippet = " ".join(item.snippet.split())
        if is_noisy_entity_line(snippet):
            continue
        definition_score = float(item.metadata.get("definition_score", 0) or 0)
        mechanism_score = float(item.metadata.get("mechanism_score", 0) or 0)
        application_score = float(item.metadata.get("application_score", 0) or 0)
        score = 0.0
        if kind == "definition":
            score = definition_score
        elif kind == "mechanism":
            score = mechanism_score
        elif kind == "application":
            score = application_score
        if score <= 0:
            continue
        scored.append((score, snippet[:220]))
    lines: list[str] = []
    seen: set[str] = set()
    for _, line in sorted(scored, key=lambda item: (-item[0], item[1])):
        normalized = " ".join(line.lower().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        lines.append(line)
    return lines[:3]


def compose_entity_answer_markdown(
    *,
    contract: QueryContract,
    claims: list[Any],
    evidence: list[EvidenceBlock],
    citations: list[Any],
) -> str:
    claim = next((item for item in claims if item.claim_type == "entity_definition"), None)
    if claim is None:
        return ""
    target = contract.targets[0] if contract.targets else (claim.entity or "该对象")
    label = claim.value or "相关技术"
    structured = dict(claim.structured_data or {})
    local_evidence = [item for item in evidence if item.doc_id in claim.evidence_ids]
    if not local_evidence and claim.paper_ids:
        local_evidence = [item for item in evidence if item.paper_id in claim.paper_ids]
    if not local_evidence:
        local_evidence = evidence
    definition_lines = [str(item).strip() for item in list(structured.get("definition_lines", [])) if str(item).strip()]
    mechanism_lines = [str(item).strip() for item in list(structured.get("mechanism_lines", [])) if str(item).strip()]
    application_lines = [str(item).strip() for item in list(structured.get("application_lines", [])) if str(item).strip()]
    if not definition_lines:
        definition_lines = entity_supporting_lines(local_evidence, kind="definition")
    if not mechanism_lines:
        mechanism_lines = entity_supporting_lines(local_evidence, kind="mechanism")
    if not application_lines:
        application_lines = entity_supporting_lines(local_evidence, kind="application")

    requested_fields = {normalize_lookup_text(item) for item in contract.requested_fields if item}
    detail_requested = contract.continuation_mode == "followup" or bool(
        requested_fields
        & {
            "mechanism",
            "workflow",
            "objective",
            "reward_signal",
            "training_signal",
            "formula",
            "variable_explanation",
        }
    )
    paper_title = str(structured.get("paper_title", "")).strip()
    description = sanitize_entity_description(str(structured.get("description", "") or ""))
    if len(description) > 520:
        description = description[:517].rstrip() + "..."
    answer: list[str] = [f"### {target}：机制与流程" if detail_requested else f"### {target} 技术简介", ""]
    if description and not is_noisy_entity_line(description):
        answer.append(description)
    else:
        answer.append(
            entity_intro_sentence(
                target=target,
                label=label,
                paper_title=paper_title,
                definition_lines=definition_lines,
                mechanism_lines=mechanism_lines,
                application_lines=application_lines,
                evidence=local_evidence,
            )
        )

    if detail_requested:
        mechanism_bullets = entity_mechanism_bullets(
            mechanism_lines=mechanism_lines,
            evidence=local_evidence,
        )
        if mechanism_bullets:
            answer.extend(["", "核心机制："])
            answer.extend([f"- {line}" for line in mechanism_bullets[:4]])
        workflow_steps = entity_workflow_steps(evidence=local_evidence)
        if workflow_steps:
            answer.extend(["", "典型流程："])
            answer.extend([f"{index}. {step}" for index, step in enumerate(workflow_steps, start=1)])
        reward_bullets = entity_reward_bullets(evidence=local_evidence)
        if reward_bullets:
            answer.extend(["", "目标与奖励信号："])
            answer.extend([f"- {line}" for line in reward_bullets[:3]])
    else:
        summary_bullets = entity_summary_bullets(
            definition_lines=definition_lines,
            mechanism_lines=mechanism_lines,
            application_lines=application_lines,
        )
        if summary_bullets and not description:
            answer.extend(["", "核心要点："])
            answer.extend([f"- {line}" for line in summary_bullets[:4]])

    application_bullets = entity_clean_lines(application_lines, limit=2)
    if application_bullets and not description:
        heading = "当前语料里的应用：" if detail_requested else "应用场景："
        answer.extend(["", heading])
        answer.extend([f"- {line}" for line in application_bullets])

    if paper_title and not detail_requested:
        answer.extend(["", f"当前最直接的定义证据来自《{paper_title}》。"])
    if citations:
        anchor = citations[0]
        page_label = f"第 {anchor.page} 页" if anchor.page else "相关页"
        answer.extend(["", f"主要依据《{anchor.title}》{page_label} 的证据整理。"])
    return "\n".join(line for line in answer if line is not None).strip()


def sanitize_entity_description(text: str) -> str:
    compact_lines: list[str] = []
    for raw_line in str(text or "").replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^#{1,6}\s*", "", line).strip()
        line = re.sub(r"^[-*]\s+", "", line).strip()
        line = re.sub(r"^\d+\.\s+", "", line).strip()
        if not line:
            continue
        if re.fullmatch(r"(定义|目的|关键特性|证据|应用场景|核心要点|机制|流程)[:：]?", line):
            continue
        compact_lines.append(line)
    compact = " ".join(" ".join(compact_lines).split())
    compact = re.sub(r"\[([^\]]{0,80})$", r"\1", compact).strip()
    compact = re.sub(r"\s*#+\s*", " ", compact).strip()
    return compact


def entity_intro_sentence(
    *,
    target: str,
    label: str,
    paper_title: str,
    definition_lines: list[str],
    mechanism_lines: list[str],
    application_lines: list[str],
    evidence: list[EvidenceBlock],
) -> str:
    joined = " \n".join([*definition_lines, *mechanism_lines, *application_lines, *[item.snippet for item in evidence[:6]]]).lower()
    if "ppo" in joined and ("variant" in joined or "from ppo to grpo" in joined):
        lead = f"{target} 更接近一种基于 PPO 的 `{label}`。"
    elif query_matches_any(joined, "", ENTITY_DEFINITION_MARKERS["intro_dataset"]):
        lead = f"{target} 更接近一个 `{label}`。"
    elif query_matches_any(joined, "", ENTITY_DEFINITION_MARKERS["intro_algorithm"]):
        lead = f"{target} 更接近一种 `{label}`。"
    else:
        lead = f"{target} 可以定位为 `{label}`。"
    details: list[str] = []
    if query_matches_any(joined, "", ENTITY_DEFINITION_MARKERS["group_comparison"]):
        details.append("它会把同一问题的多个候选输出放在一组里比较，并利用组内相对 reward 计算 advantage")
    if query_matches_any(joined, "", ENTITY_DEFINITION_MARKERS["critic_free"]):
        details.append("这样可以不再依赖单独的 value model / critic")
    if query_matches_any(joined, "", ENTITY_DEFINITION_MARKERS["resource_saving"]):
        details.append("设计动机之一是降低 PPO 类方法的训练资源开销")
    if query_matches_any(joined, "", ENTITY_DEFINITION_MARKERS["reasoning_alignment"]):
        details.append("常见目标是提升推理或对齐表现")
    if details:
        return lead + " " + "；".join(details[:3]) + "。"
    if definition_lines:
        return lead + " 当前最直接的证据把它描述为：" + entity_clean_lines(definition_lines, limit=1)[0]
    if paper_title:
        return lead + f" 当前主要依据《{paper_title}》中的相关描述。"
    return lead


def entity_clean_lines(lines: list[str], *, limit: int) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for line in lines:
        compact = " ".join(str(line or "").split()).strip(" -")
        if not compact:
            continue
        if is_noisy_entity_line(compact):
            continue
        normalized = compact.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        if len(compact) > 220:
            compact = compact[:217].rstrip() + "..."
        cleaned.append(compact)
        if len(cleaned) >= limit:
            break
    return cleaned


def entity_mechanism_bullets(*, mechanism_lines: list[str], evidence: list[EvidenceBlock]) -> list[str]:
    cleaned = entity_clean_lines(mechanism_lines, limit=4)
    if cleaned:
        return cleaned
    return entity_focus_lines(
        evidence=evidence,
        keywords=["group", "relative", "advantage", "critic", "value model", "objective", "clip", "kl"],
        limit=4,
    )


def entity_workflow_steps(*, evidence: list[EvidenceBlock]) -> list[str]:
    joined = " \n".join(item.snippet for item in evidence[:8]).lower()
    steps: list[str] = []
    if query_matches_any(joined, "", ENTITY_DEFINITION_MARKERS["workflow_sampling"]):
        steps.append("先对同一个问题采样一组候选输出。")
    if query_matches_any(joined, "", ENTITY_DEFINITION_MARKERS["workflow_reward"]):
        steps.append("再根据组内 reward 或 baseline 信息，为每个输出构造训练信号。")
    if query_matches_any(joined, "", ENTITY_DEFINITION_MARKERS["workflow_advantage"]):
        steps.append("随后把组内 reward 做相对化或归一化，构造 advantage，而不是依赖单独的 value critic。")
    if query_matches_any(joined, "", ENTITY_DEFINITION_MARKERS["workflow_policy_update"]):
        steps.append("最后在 clipping / KL 约束下更新 policy model。")
    return steps[:4]


def entity_reward_bullets(*, evidence: list[EvidenceBlock]) -> list[str]:
    bullets: list[str] = []
    focus = entity_focus_lines(
        evidence=evidence,
        keywords=["reward", "objective", "gradient", "rule", "model", "kl", "clip", "advantage"],
        limit=4,
    )
    for line in focus:
        if line not in bullets:
            bullets.append(line)
    return bullets[:3]


def entity_summary_bullets(
    *,
    definition_lines: list[str],
    mechanism_lines: list[str],
    application_lines: list[str],
) -> list[str]:
    joined = " \n".join([*definition_lines, *mechanism_lines, *application_lines]).lower()
    bullets: list[str] = []
    if "ppo" in joined and "variant" in joined:
        bullets.append("它可以看作 PPO 的一个变体。")
    if query_matches_any(joined, "", ENTITY_DEFINITION_MARKERS["summary_group_comparison"]):
        bullets.append("它通过组内相对 reward / group scores 来估计 baseline 或 advantage。")
    if query_matches_any(joined, "", ENTITY_DEFINITION_MARKERS["summary_critic"]):
        bullets.append("它的关键区别是不再依赖单独的 value critic。")
    if query_matches_any(joined, "", ENTITY_DEFINITION_MARKERS["summary_resource"]):
        bullets.append("这样做的直接收益是减少训练资源和内存开销。")
    if query_matches_any(joined, "", ENTITY_DEFINITION_MARKERS["summary_reasoning"]):
        bullets.append("常见用途是提升推理或对齐表现。")
    if not bullets:
        bullets = entity_clean_lines(definition_lines, limit=2)
        for line in entity_clean_lines(mechanism_lines, limit=2):
            if line not in bullets:
                bullets.append(line)
    for line in entity_clean_lines(application_lines, limit=1):
        if line not in bullets:
            bullets.append(line)
    return bullets[:4]


def entity_focus_lines(*, evidence: list[EvidenceBlock], keywords: list[str], limit: int) -> list[str]:
    scored: list[tuple[float, str]] = []
    for item in evidence:
        compact = " ".join(item.snippet.split())
        if is_noisy_entity_line(compact):
            continue
        lowered = compact.lower()
        score = float(item.score)
        for token in keywords:
            if token in lowered:
                score += 1.0
        if score <= 0:
            continue
        if any(token in lowered for token in keywords):
            scored.append((score, compact))
    lines: list[str] = []
    seen: set[str] = set()
    for _, line in sorted(scored, key=lambda item: (-item[0], item[1])):
        normalized = line.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        if len(line) > 220:
            line = line[:217].rstrip() + "..."
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def compose_entity_description(
    *,
    clients: Any,
    paper_doc_lookup: PaperDocLookupFn,
    contract: QueryContract,
    target: str,
    label: str,
    paper: CandidatePaper,
    evidence: list[EvidenceBlock],
) -> str:
    summary = paper_summary_text(paper.paper_id, paper_doc_lookup=paper_doc_lookup)
    requested_fields = list(contract.requested_fields)
    prompt = (
        f"target={target}\n"
        f"entity_type={label}\n"
        f"continuation_mode={contract.continuation_mode}\n"
        f"requested_fields={json.dumps(requested_fields, ensure_ascii=False)}\n"
        f"paper_title={paper.title}\n"
        f"summary={summary}\n"
        f"evidence={json.dumps([wrap_untrusted_document_text(item.snippet[:240], doc_id=item.doc_id, title=item.title) for item in evidence[:4]], ensure_ascii=False)}"
    )
    llm_text = clients.invoke_text(
        system_prompt=(
            "你是论文实体解释器。请根据给定论文摘要、requested_fields 和证据，用一段简洁中文回答。"
            "不要输出 Markdown 标题、列表、链接或引用，不要使用 #、####、项目符号。"
            "控制在 2-4 句内。"
            "如果 continuation_mode 是 followup，或者 requested_fields 包含 mechanism/workflow/objective/reward_signal，"
            "优先解释它如何工作、优化什么、依赖什么奖励或训练信号，不要只重复泛泛定义。"
            "如果证据同时包含“某篇论文使用它”和“它本身的定义/机制”，优先解释技术本身，再补充应用场景。"
            "不要编造。"
            f"{DOCUMENT_SAFETY_INSTRUCTION}"
        ),
        human_prompt=prompt,
        fallback="",
    )
    if llm_text:
        return llm_text
    mechanism_lines = entity_supporting_lines(evidence, kind="mechanism")
    application_lines = entity_supporting_lines(evidence, kind="application")
    if any(field in {"mechanism", "workflow", "objective", "reward_signal"} for field in requested_fields):
        parts = [f"{target} 可以定位为 `{label}`。"]
        if mechanism_lines:
            parts.append("它的工作机制可以概括为：")
            parts.extend([f"- {line}" for line in mechanism_lines[:2]])
        if application_lines:
            parts.append("在当前语料里，它被应用在：")
            parts.extend([f"- {line}" for line in application_lines[:1]])
        return "\n".join(parts).strip()
    if summary:
        snippet = " ".join(summary.split())[:180]
        return f"{target} 可以定位为 `{label}`。相关论文《{paper.title}》提到：{snippet}"
    return f"{target} 可以定位为 `{label}`，相关论文是《{paper.title}》。"
