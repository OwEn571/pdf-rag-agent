from __future__ import annotations

import logging
from typing import Any, Callable

from app.domain.models import CandidatePaper, DisambiguationJudgeDecision, EvidenceBlock, QueryContract, ResearchPlan, SessionContext
from app.services.agent.runtime_helpers import (
    EVIDENCE_RETRIEVAL_STAGE,
    excluded_focus_titles,
    refresh_selected_ambiguity_materials,
)
from app.services.agent.tool_events import record_agent_observation as record_agent_observation_event
from app.services.tools.registry_helpers import _trim_evidence_item
from app.services.clarification.intents import (
    acronym_evidence_from_corpus,
    acronym_options_from_evidence,
    disambiguation_judge_human_prompt,
    disambiguation_judge_option_payload,
    disambiguation_judge_system_prompt,
    evidence_disambiguation_options,
)
from app.services.contracts.conversation_memory import target_binding_from_memory
from app.services.intents.followup import is_negative_correction_query


logger = logging.getLogger(__name__)

EmitFn = Callable[[str, dict[str, Any]], None]


def _trim_candidate_model(payload: dict[str, Any]) -> dict[str, Any]:
    meta = payload.get("metadata")
    if isinstance(meta, dict):
        trimmed = {k: v for k, v in meta.items() if k != "vector"}
        for key in ("body_acronyms",):
            v = str(trimmed.get(key, ""))
            if len(v) > 240:
                trimmed[key] = v[:240] + "..."
        for key in ("paper_card_text", "generated_summary", "abstract_note"):
            v = str(trimmed.get(key, ""))
            if len(v) > 400:
                trimmed[key] = v[:400] + "..."
        authors = str(trimmed.get("authors", ""))
        if len(authors) > 200:
            first_three = [a.strip() for a in authors.split(",")[:3]]
            trimmed["authors"] = ", ".join(first_three) + " 等"
        payload["metadata"] = trimmed
    return payload


def disambiguation_options_from_evidence(
    *,
    contract: QueryContract,
    session: SessionContext,
    papers: list[CandidatePaper],
    evidence: list[EvidenceBlock],
    paper_lookup: Callable[[str], CandidatePaper | None],
    search_concept_evidence: Callable[..., list[EvidenceBlock]],
    evidence_limit_default: int,
    paper_documents: Callable[[], list[Any]],
    block_documents_for_paper: Callable[[str, int], list[Any]],
) -> list[dict[str, Any]]:
    target = str(contract.targets[0] or "").strip() if contract.targets else ""
    target_binding_exists = bool(target and target_binding_from_memory(session=session, target=target))
    return evidence_disambiguation_options(
        contract=contract,
        target_binding_exists=target_binding_exists,
        is_negative_correction=is_negative_correction_query(contract.clean_query),
        initial_options=lambda: acronym_options_from_evidence(
            target=target,
            papers=papers,
            evidence=evidence,
            paper_lookup=paper_lookup,
        ),
        broad_options=lambda: acronym_options_from_evidence(
            target=target,
            papers=papers,
            evidence=search_concept_evidence(
                query=target,
                contract=contract,
                limit=max(evidence_limit_default, 96),
            ),
            paper_lookup=paper_lookup,
        ),
        corpus_options=lambda: acronym_options_from_evidence(
            target=target,
            papers=papers,
            evidence=acronym_evidence_from_corpus(
                target=target,
                limit=160,
                paper_documents=paper_documents,
                block_documents_for_paper=block_documents_for_paper,
            ),
            paper_lookup=paper_lookup,
        ),
        excluded_titles=excluded_focus_titles(
            session=session,
            contract=contract,
            is_negative_correction_query=is_negative_correction_query,
        ),
    )


def _try_fast_auto_resolve(
    target: str, options: list[dict[str, Any]]
) -> DisambiguationJudgeDecision | None:
    """Skip the LLM judge when one option is clearly the original paper."""
    target_lower = target.lower()
    best_option_id = None
    best_paper_id = None
    best_confidence = 0.0
    for opt in options:
        title = str(opt.get("title", "") or "").lower()
        snippet = str(opt.get("snippet", "") or "").lower()
        paper_id = str(opt.get("paper_id", "") or "").strip()
        score = 0.0
        # Direct title match: "DPO" → "Direct Preference Optimization"
        if target_lower in title:
            score += 4.0
        # Snippet has propose/introduce + target
        if target_lower in snippet:
            score += 1.0
            if any(kw in snippet for kw in ("propose", "introduce", "present", "提出", "引入")):
                score += 2.0
        if score > best_confidence and paper_id:
            best_confidence = score
            best_option_id = str(opt.get("option_id", "") or "").strip()
            best_paper_id = paper_id
    # Require: target appears in paper title (e.g. "DPO" in "Direct Preference Optimization: ...")
    if best_confidence >= 4.0 and best_option_id and best_paper_id:
        return DisambiguationJudgeDecision(
            decision="auto_resolve",
            selected_option_id=best_option_id,
            selected_paper_id=best_paper_id,
            confidence=0.90,
            reason=f"Fast path: paper title directly contains target '{target}'.",
        )
    return None


def judge_disambiguation_options(
    *,
    contract: QueryContract,
    options: list[dict[str, Any]],
    clients: Any,
    paper_lookup: Callable[[str], CandidatePaper | None],
) -> DisambiguationJudgeDecision | None:
    if getattr(clients, "chat", None) is None or len(options) < 2:
        return None
    # Fast path: if one option has a paper title that directly contains the
    # target (e.g., "DPO" → "Direct Preference Optimization: ..."), auto-resolve
    # without an LLM call.
    target = str(contract.targets[0] or "").strip() if contract.targets else ""
    if target and len(options) >= 2:
        fast = _try_fast_auto_resolve(target, options)
        if fast is not None:
            return fast
    payload = clients.invoke_json(
        system_prompt=disambiguation_judge_system_prompt(),
        human_prompt=disambiguation_judge_human_prompt(
            contract=contract,
            candidate_options=[
                disambiguation_judge_option_payload(
                    option=option,
                    paper=paper_lookup(str(option.get("paper_id", "") or "").strip())
                    if str(option.get("paper_id", "") or "").strip()
                    else None,
                )
                for option in options[:8]
            ],
        ),
        fallback={},
    )
    if not isinstance(payload, dict):
        return None
    try:
        return DisambiguationJudgeDecision.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("disambiguation judge returned invalid payload: %s", exc)
        return None


def refresh_state_for_selected_ambiguity(
    *,
    state: dict[str, Any],
    selected: dict[str, Any],
    emit: EmitFn,
    execution_steps: list[dict[str, Any]],
    paper_lookup: Callable[[str], CandidatePaper | None],
    search_concept_evidence: Callable[..., list[EvidenceBlock]],
    expand_evidence: Callable[..., list[EvidenceBlock]],
) -> None:
    candidate_pool: list[CandidatePaper] = [
        *list(state.get("screened_papers") or []),
        *list(state.get("candidate_papers") or []),
    ]
    contract: QueryContract = state["contract"]
    plan: ResearchPlan = state["plan"]
    excluded_titles: set[str] = state["excluded_titles"]
    refresh = refresh_selected_ambiguity_materials(
        selected=selected,
        contract=contract,
        plan=plan,
        candidate_papers=candidate_pool,
        existing_evidence=list(state.get("evidence") or []),
        excluded_titles=excluded_titles,
        paper_lookup=paper_lookup,
        search_concept_evidence=search_concept_evidence,
        expand_evidence=expand_evidence,
    )
    if refresh is None:
        return
    if refresh.selected_papers:
        state["screened_papers"] = refresh.selected_papers
        emit("screened_papers", {"count": len(state["screened_papers"]), "items": [_trim_candidate_model(item.model_dump()) for item in state["screened_papers"]]})
    evidence = refresh.evidence
    if refresh.evidence_refreshed:
        record_agent_observation_event(
            emit=emit,
            execution_steps=execution_steps,
            tool="search_corpus",
            summary=f"auto_resolved_evidence={len(evidence)}",
            payload={
                "stage": EVIDENCE_RETRIEVAL_STAGE,
                "selected_paper_id": refresh.paper_id,
                "evidence_count": len(evidence),
            },
        )
    state["evidence"] = evidence
    emit("evidence", {"count": len(evidence), "items": [_trim_evidence_item(item.model_dump()) for item in evidence]})
