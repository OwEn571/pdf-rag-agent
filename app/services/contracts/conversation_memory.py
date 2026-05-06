from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from app.domain.models import QueryContract, SessionContext
from app.services.contracts.context import (
    contract_allows_active_context_override,
    contract_answer_slots,
    contract_has_note,
    contract_notes,
    contract_topic_state,
)
from app.services.contracts.normalization import normalize_lookup_text
from app.services.intents.followup import is_negative_correction_query
from app.services.planning.research import research_plan_context_from_contract


ConversationContextFn = Callable[[SessionContext], dict[str, Any]]


def target_binding_from_memory(*, session: SessionContext, target: str) -> dict[str, Any] | None:
    key = normalize_lookup_text(target)
    if not key:
        return None
    bindings = dict((session.working_memory or {}).get("target_bindings", {}) or {})
    binding = bindings.get(key)
    return dict(binding) if isinstance(binding, dict) else None


def active_memory_bindings(session: SessionContext) -> list[dict[str, Any]]:
    bindings = dict((session.working_memory or {}).get("target_bindings", {}) or {})
    selected: list[dict[str, Any]] = []
    for target in session.effective_active_research().targets:
        binding = bindings.get(normalize_lookup_text(target))
        if isinstance(binding, dict):
            selected.append(dict(binding))
    if len(selected) >= 2:
        return selected
    for binding in bindings.values():
        if isinstance(binding, dict) and binding not in selected:
            selected.append(dict(binding))
        if len(selected) >= 4:
            break
    return selected


def memory_binding_doc_ids(bindings: list[dict[str, Any]]) -> list[str]:
    doc_ids: list[str] = []
    for binding in bindings:
        for doc_id in list(binding.get("evidence_ids", []) or [])[:2]:
            if str(doc_id).strip():
                doc_ids.append(str(doc_id).strip())
        paper_id = str(binding.get("paper_id", "") or "").strip()
        if paper_id:
            doc_ids.append(f"paper::{paper_id}")
    return list(dict.fromkeys(doc_ids))


def llm_memory_followup_contract(
    *,
    clean_query: str,
    session: SessionContext,
    current_contract: QueryContract,
    clients: Any,
    conversation_context: ConversationContextFn,
) -> QueryContract | None:
    if getattr(clients, "chat", None) is None or not session.turns:
        return None
    payload = clients.invoke_json(
        system_prompt=(
            "你是论文研究 Agent 的会话记忆追问判别器。"
            "判断当前用户问题是否只是在追问上一轮工具输出本身，而不需要读取新的论文证据。"
            "只有当问题可以主要基于 conversation_context 回答时，返回 should_use_memory=true。"
            "典型 true：为什么这么推荐、推荐理由、上一轮排序依据、上一轮回答里的某个结论依据。"
            "典型 false：用户问某篇论文具体说了什么、核心结论、方法、实验结果、图表、公式、更多细节；"
            "这类问题虽然要用记忆解析指代，但必须交给后续研究工具检索正文证据。"
            "如果当前问题需要新的论文检索、外部动态信息、全新主题，返回 false。"
            "只输出 JSON：should_use_memory, reason, targets, requested_fields, answer_shape。"
        ),
        human_prompt=json.dumps(
            {
                "current_query": clean_query,
                "current_contract": current_contract.model_dump(),
                "conversation_context": conversation_context(session),
            },
            ensure_ascii=False,
        ),
        fallback={},
    )
    return memory_followup_contract_from_payload(
        payload=payload,
        clean_query=clean_query,
        current_contract=current_contract,
    )


def memory_followup_contract_from_payload(
    *,
    payload: Any,
    clean_query: str,
    current_contract: QueryContract,
) -> QueryContract | None:
    if not isinstance(payload, dict) or not bool(payload.get("should_use_memory")):
        return None
    raw_targets = payload.get("targets", [])
    targets = [str(item).strip() for item in raw_targets if str(item).strip()] if isinstance(raw_targets, list) else []
    raw_fields = payload.get("requested_fields", [])
    requested_fields = [str(item).strip() for item in raw_fields if str(item).strip()] if isinstance(raw_fields, list) else ["answer"]
    answer_shape = str(payload.get("answer_shape", current_contract.answer_shape) or "").strip().lower()
    if answer_shape not in {"bullets", "narrative", "table"}:
        answer_shape = "narrative"
    return QueryContract(
        clean_query=clean_query,
        interaction_mode="conversation",
        relation="memory_followup",
        targets=targets,
        requested_fields=requested_fields or ["answer"],
        required_modalities=[],
        answer_shape=answer_shape,
        precision_requirement="normal",
        continuation_mode="followup",
        notes=["agent_tool", "llm_memory_followup", str(payload.get("reason", ""))[:180]],
    )


def apply_conversation_memory_to_contract(
    *,
    contract: QueryContract,
    session: SessionContext,
    selected_clarification_paper_id: str = "",
) -> QueryContract:
    if contract.interaction_mode != "research" or not contract.targets:
        return contract
    target_bindings = {
        target: binding
        for target in contract.targets
        if (binding := target_binding_from_memory(session=session, target=target))
    }
    topic_state = contract_topic_state(contract)
    goals = set(research_plan_context_from_contract(contract).goals)
    if contract.relation == "origin_lookup" or "origin" in contract_answer_slots(contract) or goals & {"paper_title", "year"}:
        return contract
    allow_explicit_target_binding = bool(target_bindings) and topic_state != "switch"
    if "formula" in goals and topic_state != "continue":
        allow_explicit_target_binding = False
    if not contract_allows_active_context_override(contract) and not allow_explicit_target_binding:
        return contract
    if contract_has_note(contract, "exclude_previous_focus") or is_negative_correction_query(contract.clean_query):
        return contract
    if selected_clarification_paper_id:
        return contract
    notes = contract_notes(contract)
    for target in contract.targets:
        binding = target_bindings.get(target)
        if not binding:
            continue
        paper_id = str(binding.get("paper_id", "") or "").strip()
        title = str(binding.get("title", "") or "").strip()
        if not paper_id:
            continue
        notes = list(dict.fromkeys([*notes, "resolved_from_conversation_memory", f"selected_paper_id={paper_id}"]))
        if title:
            notes.append("memory_title=" + title)
        return contract.model_copy(update={"continuation_mode": "followup", "notes": notes})
    return contract
