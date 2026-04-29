from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.domain.models import Claim, EvidenceBlock, QueryContract
from app.services.confidence import coerce_confidence_value

PageImageRenderer = Callable[[str, int], str]


def table_vlm_system_prompt() -> str:
    return "你是论文表格视觉理解求解器。只输出 JSON。"


def table_vlm_human_content(
    *,
    contract: QueryContract,
    ranked_blocks: list[EvidenceBlock],
    render_page_image: PageImageRenderer,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "你是论文表格视觉理解求解器。请结合表格抽取文本和页面图片回答用户查询。"
                "只输出 JSON，字段为 claims 和 draft_answer。"
                "claims 中每项包含 claim, metric_lines, confidence；不要编造看不见的数值。"
                f"\nquery={contract.clean_query}"
            ),
        }
    ]
    rendered_pages: set[tuple[str, int]] = set()
    for idx, item in enumerate(ranked_blocks[:3], start=1):
        content.append(
            {
                "type": "text",
                "text": (
                    f"[table_context_{idx}] title={item.title} page={item.page} block_type={item.block_type}\n"
                    f"caption={item.caption}\ntable_text={item.snippet[:1200]}"
                ),
            }
        )
        if item.block_type not in {"table", "caption"}:
            continue
        page_key = (item.file_path, item.page)
        if page_key in rendered_pages:
            continue
        rendered_pages.add(page_key)
        image_url = render_page_image(item.file_path, item.page)
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
    return content


def figure_vlm_system_prompt() -> str:
    return "你是论文图像理解求解器。只输出 JSON。"


def figure_vlm_human_content(
    *,
    contract: QueryContract,
    figure_contexts: list[dict[str, Any]],
    render_page_image: PageImageRenderer,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "你是论文图像理解求解器。只输出 JSON，字段为 claims 和 draft_answer。"
                "claims 中每项包含 claim, confidence。不要编造看不见的数值。"
                f"\nquery={contract.clean_query}"
            ),
        }
    ]
    for idx, context in enumerate(figure_contexts, start=1):
        content.append(
            {
                "type": "text",
                "text": (
                    f"[figure_context_{idx}] title={context['title']} page={context['page']}\n"
                    f"caption={context['caption']}\nfigure_text={context['figure_text']}\npage_text={context['page_text']}"
                ),
            }
        )
        image_url = render_page_image(str(context["file_path"]), int(context["page"]))
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
    return content


def table_metric_claim_from_vlm_payload(
    payload: Any,
    *,
    entity: str,
    evidence_ids: list[str],
    paper_ids: list[str],
) -> Claim | None:
    raw_claim = _first_raw_claim(payload)
    claim_text = str(
        raw_claim.get("claim", "") or (payload.get("draft_answer", "") if isinstance(payload, dict) else "")
    ).strip()
    if not claim_text:
        return None
    raw_lines = raw_claim.get("metric_lines", [])
    metric_lines = [str(item).strip() for item in raw_lines if str(item).strip()] if isinstance(raw_lines, list) else []
    return Claim(
        claim_type="metric_value",
        entity=entity,
        value=claim_text,
        structured_data={"metric_lines": metric_lines or [claim_text], "mode": "vlm_table"},
        evidence_ids=evidence_ids,
        paper_ids=paper_ids,
        confidence=coerce_confidence_value(raw_claim.get("confidence", 0.84), default=0.82),
    )


def figure_conclusion_claim_from_vlm_payload(
    payload: Any,
    *,
    entity: str,
    evidence_ids: list[str],
    paper_id: str,
    fallback_text: str,
    signal_score: Callable[[str], float],
) -> Claim | None:
    raw_claim = _first_raw_claim(payload)
    if not raw_claim:
        return None
    claim_text = str(raw_claim.get("claim", "") or (payload.get("draft_answer", "") if isinstance(payload, dict) else "")).strip()
    if not claim_text:
        return None
    if signal_score(claim_text) < max(3, signal_score(fallback_text)):
        return None
    return Claim(
        claim_type="figure_conclusion",
        entity=entity,
        value=claim_text,
        structured_data={"mode": "vlm"},
        evidence_ids=evidence_ids,
        paper_ids=[paper_id],
        confidence=coerce_confidence_value(raw_claim.get("confidence", 0.82), default=0.82),
    )


def figure_conclusion_text_claim(
    *,
    entity: str,
    text: str,
    figure_context: dict[str, Any],
    mode: str,
    confidence: float,
) -> Claim:
    return Claim(
        claim_type="figure_conclusion",
        entity=entity,
        value=text,
        structured_data={"mode": mode},
        evidence_ids=list(figure_context["doc_ids"]),
        paper_ids=[str(figure_context["paper_id"])],
        confidence=confidence,
    )


def _first_raw_claim(payload: Any) -> dict[str, Any]:
    raw_claims = payload.get("claims", []) if isinstance(payload, dict) else []
    if isinstance(raw_claims, list) and raw_claims and isinstance(raw_claims[0], dict):
        return raw_claims[0]
    return {}
