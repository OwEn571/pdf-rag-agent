from __future__ import annotations

import json
import logging
from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract
from app.services.answers.evidence_presentation import (
    build_figure_contexts,
    figure_fallback_summary,
)
from app.services.intents.figure import extract_figure_benchmarks, figure_signal_score
from app.services.retrieval.pdf_rendering import render_pdf_page_image_data_url
from app.services.claims.visual_helpers import (
    figure_conclusion_claim_from_vlm_payload,
    figure_conclusion_text_claim,
    figure_vlm_human_content,
    figure_vlm_system_prompt,
)


def solve_figure_claims(
    *,
    clients: Any,
    settings: Any,
    rendered_page_data_url_cache: dict[str, str],
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    logger: logging.Logger,
) -> list[Claim]:
    del papers
    figure_contexts = build_figure_contexts(evidence)
    if not figure_contexts:
        return []
    primary_context = figure_contexts[0]
    entity = contract.targets[0] if contract.targets else primary_context["title"]
    fallback_text = figure_fallback_text(figure_contexts=figure_contexts, evidence=evidence)
    payload = {"claims": [], "draft_answer": ""}
    if getattr(settings, "enable_figure_vlm", False):
        content = figure_vlm_human_content(
            contract=contract,
            figure_contexts=figure_contexts,
            render_page_image=lambda file_path, page: render_pdf_page_image_data_url(
                file_path=file_path,
                page=page,
                pdf_render_dpi=int(settings.pdf_render_dpi),
                timeout_seconds=float(settings.figure_vlm_timeout_seconds),
                cache=rendered_page_data_url_cache,
                logger=logger,
            ),
        )
        if any(block.get("type") == "image_url" for block in content):
            payload = clients.invoke_multimodal_json(
                system_prompt=figure_vlm_system_prompt(),
                human_content=content,
                fallback={"claims": [], "draft_answer": ""},
            )
    vlm_claim = figure_conclusion_claim_from_vlm_payload(
        payload,
        entity=entity,
        evidence_ids=primary_context["doc_ids"],
        paper_id=primary_context["paper_id"],
        fallback_text=fallback_text,
        signal_score=figure_signal_score,
    )
    if vlm_claim is not None:
        return [vlm_claim]
    text_summary = summarize_figure_text(clients=clients, contract=contract, fallback_text=fallback_text, evidence=evidence)
    if text_summary:
        return [
            figure_conclusion_text_claim(
                entity=entity,
                text=text_summary,
                figure_context=primary_context,
                mode="text_summary",
                confidence=0.82,
            )
        ]
    return [
        figure_conclusion_text_claim(
            entity=entity,
            text=fallback_text,
            figure_context=primary_context,
            mode="caption_fallback",
            confidence=0.74,
        )
    ]


def figure_fallback_text(*, figure_contexts: list[dict[str, Any]], evidence: list[EvidenceBlock]) -> str:
    fallback_text = figure_fallback_summary(figure_contexts)
    evidence_benchmarks = extract_figure_benchmarks("\n".join(item.snippet for item in evidence[:10]))
    if len(evidence_benchmarks) < 3:
        return fallback_text
    benchmark_suffix = " 图中提到的 benchmark 包括：" + "、".join(evidence_benchmarks) + "。"
    if benchmark_suffix in fallback_text:
        return fallback_text
    remaining = max(80, 700 - len(benchmark_suffix))
    return fallback_text[:remaining].rstrip() + benchmark_suffix


def summarize_figure_text(
    *,
    clients: Any,
    contract: QueryContract,
    fallback_text: str,
    evidence: list[EvidenceBlock],
) -> str:
    prompt = (
        f"query={contract.clean_query}\n"
        f"fallback={fallback_text}\n"
        f"evidence={json.dumps([item.snippet[:280] for item in evidence[:6]], ensure_ascii=False)}"
    )
    llm_text = clients.invoke_text(
        system_prompt=(
            "你是论文 figure 回答整理器。请基于图注和邻近文本，用简洁中文 Markdown 概括 figure 展示的内容。"
            "优先输出一个结论句，再补 2-4 条 bullet，说明比较对象、关键 benchmark、总体结论。"
            "不要逐字转抄原文，不要编造未出现的数值。"
        ),
        human_prompt=prompt,
        fallback="",
    )
    return llm_text.strip()
