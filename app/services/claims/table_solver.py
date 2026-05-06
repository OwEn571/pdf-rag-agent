from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract
from app.services.claims import metric_text as metric_helpers
from app.services.answers.evidence_presentation import evidence_ids_for_paper
from app.services.retrieval.pdf_rendering import render_pdf_page_image_data_url
from app.services.claims.visual_helpers import (
    table_metric_claim_from_vlm_payload,
    table_vlm_human_content,
    table_vlm_system_prompt,
)


def solve_table_claims(
    *,
    clients: Any,
    settings: Any,
    rendered_page_data_url_cache: dict[str, str],
    contract: QueryContract,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    paper_identity_matches_targets: Callable[[CandidatePaper, list[str]], bool],
    logger: logging.Logger,
) -> list[Claim]:
    ranked_blocks = metric_helpers.ranked_table_metric_blocks(
        contract=contract,
        papers=papers,
        evidence=evidence,
        token_weights=settings.solver_metric_token_weights,
        paper_target_matcher=paper_identity_matches_targets,
    )
    if not ranked_blocks:
        return []
    selected_paper, _, paper_ids = metric_helpers.metric_paper_selection(
        papers=papers,
        ranked_evidence=ranked_blocks,
    )
    if selected_paper is None:
        return []
    evidence_ids = [item.doc_id for item in ranked_blocks[:4]]
    vlm_claim = solve_table_with_vlm(
        clients=clients,
        settings=settings,
        rendered_page_data_url_cache=rendered_page_data_url_cache,
        contract=contract,
        ranked_blocks=ranked_blocks,
        evidence_ids=evidence_ids,
        paper_ids=paper_ids or [selected_paper.paper_id],
        selected_paper=selected_paper,
        logger=logger,
    )
    if vlm_claim is not None:
        return [vlm_claim]
    lines = metric_helpers.extract_metric_lines(ranked_blocks, token_weights=settings.solver_metric_token_weights)
    return [
        metric_helpers.text_table_metric_claim(
            entity=contract.targets[0] if contract.targets else selected_paper.title,
            metric_lines=lines,
            evidence_ids=evidence_ids,
            paper_ids=paper_ids or [selected_paper.paper_id],
            selected_paper=selected_paper,
        )
    ]


def solve_table_with_vlm(
    *,
    clients: Any,
    settings: Any,
    rendered_page_data_url_cache: dict[str, str],
    contract: QueryContract,
    ranked_blocks: list[EvidenceBlock],
    evidence_ids: list[str],
    paper_ids: list[str],
    selected_paper: CandidatePaper,
    logger: logging.Logger,
) -> Claim | None:
    if not getattr(settings, "enable_table_vlm", False):
        return None
    content = table_vlm_human_content(
        contract=contract,
        ranked_blocks=ranked_blocks,
        render_page_image=lambda file_path, page: render_pdf_page_image_data_url(
            file_path=file_path,
            page=page,
            pdf_render_dpi=int(settings.pdf_render_dpi),
            timeout_seconds=float(settings.figure_vlm_timeout_seconds),
            cache=rendered_page_data_url_cache,
            logger=logger,
        ),
    )
    if not any(block.get("type") == "image_url" for block in content):
        return None
    payload = clients.invoke_multimodal_json(
        system_prompt=table_vlm_system_prompt(),
        human_content=content,
        fallback={"claims": [], "draft_answer": ""},
    )
    return table_metric_claim_from_vlm_payload(
        payload,
        entity=contract.targets[0] if contract.targets else selected_paper.title,
        evidence_ids=evidence_ids,
        paper_ids=paper_ids,
    )
