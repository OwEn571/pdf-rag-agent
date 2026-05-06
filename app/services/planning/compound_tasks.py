from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from app.domain.models import Claim, QueryContract, SessionContext, VerificationReport
from app.services.contracts.conversation_memory import active_memory_bindings
from app.services.contracts.context import contract_notes
from app.services.contracts.normalization import normalize_contract_targets, normalize_lookup_text, normalize_modalities


TargetNormalizer = Callable[[list[str], list[str]], list[str]]
ConversationContextFn = Callable[..., dict[str, Any]]
HistoryMessagesFn = Callable[[SessionContext], list[dict[str, str]]]
SubtaskContractBuilder = Callable[[object, str, int], QueryContract | None]

ALLOWED_COMPOUND_SUBTASK_RELATIONS = {
    "library_status",
    "library_recommendation",
    "origin_lookup",
    "formula_lookup",
    "followup_research",
    "entity_definition",
    "topology_discovery",
    "topology_recommendation",
    "figure_question",
    "paper_summary_results",
    "metric_value_lookup",
    "concept_definition",
    "paper_recommendation",
    "general_question",
    "comparison_synthesis",
}


def default_compound_target_normalizer(targets: list[str], requested_fields: list[str]) -> list[str]:
    return normalize_contract_targets(
        targets=targets,
        requested_fields=requested_fields,
        canonicalize_targets=lambda values: values,
    )


def compound_decomposer_system_prompt() -> str:
    return (
        "你是论文研究 Agent 的任务分解器，不是最终回答器。"
        "你的任务是先判断当前用户消息是否需要拆成多个可执行步骤，并在需要时输出有序 QueryContract。"
        "检索论文只是工具，子任务应围绕用户真实意图组织，而不是套模板。"
        "你可以参考 available_tools，但你不调用工具；planner/executor 会基于你的子任务调用工具。"
        "如果问题只有一个任务，输出 is_compound=false 和空 subtasks。"
        "如果一句话中有多个需求，例如多个公式查询、总结+实验结果、查询+比较、数量+推荐，输出 is_compound=true。"
        "如果问题要求比较多个实体/方法/论文，必须先为每个实体/方法/论文建立独立的检索或解释子任务，"
        "再追加一个 comparison 综合子任务；不要让综合子任务自己猜测任何缺失对象的信息。"
        "但同一篇论文/同一实体的多个字段（例如“核心结论是什么，实验结果如何”）不是 compound；"
        "必须合并为一个 QueryContract，并把 requested_fields 写成多个字段。"
        "每个 subtask 字段为 clean_query, interaction_mode, intent_kind, continuation_mode, targets, answer_slots, "
        "requested_fields, required_modalities, answer_shape, precision_requirement, notes。"
        "不要输出 relation；用 answer_slots/requested_fields 表达子任务目标。"
        "可用 answer_slots 包括 library_status, library_recommendation, origin, formula, followup_research, "
        "entity_definition, topology_discovery, topology_recommendation, figure, paper_summary, metric_value, "
        "concept_definition, paper_recommendation, comparison, general_answer。"
        "interaction_mode 只能是 conversation 或 research。"
        "required_modalities 只能使用 page_text, paper_card, table, caption, figure。"
        "answer_shape 只能是 bullets, narrative, table。precision_requirement 只能是 exact, high, normal。"
        "公式查询使用 answer_slots=[formula] + requested_fields=[formula, variable_explanation] + required_modalities=[page_text, table]。"
        "库状态/库列表/库元信息问题使用 library_status，必须 interaction_mode=conversation，targets=[]，不要走 research 检索；"
        "例如按年份、作者、标签、分类、PDF 有无统计或筛选当前库内论文。"
        "库内默认推荐问题使用 library_recommendation，必须 interaction_mode=conversation，targets=[]。"
        "比较/综合使用 answer_slots=[comparison]，并且应放在其依赖的检索子任务之后。"
        "如果某个依赖子任务证据不足，最终综合只能说明证据不足，不能用“可能/推测”补全。"
        "targets 只能放实体本身，不要把“公式、结果、summary”等任务词拼进 target。"
        "只输出 JSON：is_compound, reason, subtasks。"
        "\n--- 示例 1（复合：比较两个方法） ---\n"
        "用户：比较 DPO 和 PPO 的目标函数\n"
        '输出：{"is_compound":true,"reason":"比较两个方法,先各自检索再综合","subtasks":['
        '{"clean_query":"DPO 目标函数是什么","interaction_mode":"research","targets":["DPO"],"answer_slots":["formula"],"requested_fields":["formula","variable_explanation"]},'
        '{"clean_query":"PPO 目标函数是什么","interaction_mode":"research","targets":["PPO"],"answer_slots":["formula"],"requested_fields":["formula","variable_explanation"]},'
        '{"clean_query":"比较 DPO 和 PPO 的目标函数","interaction_mode":"conversation","targets":["DPO","PPO"],"answer_slots":["comparison"]}]}\n'
        "\n--- 示例 2（复合：多篇论文总结） ---\n"
        "用户：总结 Attention Is All You Need 和 BERT 论文的核心结论\n"
        '输出：{"is_compound":true,"reason":"两篇不同论文各自总结","subtasks":['
        '{"clean_query":"Attention Is All You Need 核心结论","interaction_mode":"research","targets":["Attention Is All You Need"],"answer_slots":["paper_summary"],"requested_fields":["summary","results"]},'
        '{"clean_query":"BERT 核心结论","interaction_mode":"research","targets":["BERT"],"answer_slots":["paper_summary"],"requested_fields":["summary","results"]}]}\n'
        "\n--- 示例 3（复合：查询+比较） ---\n"
        "用户：RLHF 和 DPO 分别怎么训练的，哪个更好\n"
        '输出：{"is_compound":true,"reason":"两个方法的训练方式+比较","subtasks":['
        '{"clean_query":"RLHF 训练方法","interaction_mode":"research","targets":["RLHF"],"answer_slots":["general_answer"],"requested_fields":["definition","mechanism"]},'
        '{"clean_query":"DPO 训练方法","interaction_mode":"research","targets":["DPO"],"answer_slots":["general_answer"],"requested_fields":["definition","mechanism"]},'
        '{"clean_query":"比较 RLHF 和 DPO 训练方法优劣","interaction_mode":"conversation","targets":["RLHF","DPO"],"answer_slots":["comparison"]}]}\n'
        "\n--- 示例 4（不应拆：单论文多字段） ---\n"
        "用户：LoRA 论文的核心结论和实验结果是什么\n"
        '输出：{"is_compound":false,"reason":"同一篇论文的多个字段,合并为一个任务","subtasks":[]}\n'
    )


def llm_decompose_compound_query(
    *,
    clean_query: str,
    session: SessionContext,
    clients: Any,
    available_tools: list[dict[str, Any]],
    conversation_context: ConversationContextFn,
    history_messages: HistoryMessagesFn,
    target_normalizer: TargetNormalizer = default_compound_target_normalizer,
) -> list[QueryContract]:
    if getattr(clients, "chat", None) is None:
        return []
    system_prompt = compound_decomposer_system_prompt()
    human_payload = {
        "current_query": clean_query,
        "available_tools": available_tools,
        "conversation_context": conversation_context(session, max_chars=10000),
        "planner_instruction": "Always decide whether this turn should be decomposed; return false for a single task.",
    }
    # Prefer invoke_json: simpler prompt format (SystemMessage + HumanMessage)
    # produces more reliable JSON output from GPT-4o than invoke_json_messages.
    invoke_json = getattr(clients, "invoke_json", None)
    if callable(invoke_json):
        payload = invoke_json(
            system_prompt=system_prompt,
            human_prompt=json.dumps(human_payload, ensure_ascii=False),
            fallback={},
        )
    else:
        invoke_json_messages = getattr(clients, "invoke_json_messages", None)
        if not callable(invoke_json_messages):
            return []
        payload = invoke_json_messages(
            system_prompt=system_prompt,
            messages=[
                *history_messages(session),
                {"role": "user", "content": json.dumps(human_payload, ensure_ascii=False)},
            ],
            fallback={},
        )
    contracts = compound_contracts_from_decomposer_payload(
        payload=payload,
        fallback_query=clean_query,
        target_normalizer=target_normalizer,
    )
    # P0-6: Rule-based fallback when LLM decomposer returns no contracts
    if not contracts:
        contracts = _rule_based_compound_split(
            clean_query=clean_query,
            target_normalizer=target_normalizer,
        )
    return contracts


def _rule_based_compound_split(
    *,
    clean_query: str,
    target_normalizer: TargetNormalizer = default_compound_target_normalizer,
) -> list[QueryContract]:
    compound_connectors = re.compile(
        r"(?:和|与|以及|对比|比较|比|vs\.?|vs\.?\s|and|compare|versus)",
        re.IGNORECASE,
    )
    comparison_keywords = re.compile(
        r"(?:哪个更好|哪个更优|区别|差异|不同|优缺点|优劣|对比|比较|vs\.?|versus)",
        re.IGNORECASE,
    )
    target_candidates = re.findall(
        r"(?:[A-Z]{2,}(?:\s*[+-]?\s*[A-Za-z0-9]+)?|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
        clean_query,
    )
    cjk_targets = re.findall(r"《([^》]{2,80})》", clean_query)
    all_candidates = target_candidates + cjk_targets

    has_comparison = bool(comparison_keywords.search(clean_query))
    has_connector = bool(compound_connectors.search(clean_query))

    if not all_candidates or len(all_candidates) < 2:
        return []
    if not has_comparison and not has_connector:
        return []

    unique_targets = list(dict.fromkeys(all_candidates))[:3]
    if len(unique_targets) < 2:
        return []

    normalized = target_normalizer(unique_targets, [])
    if len(normalized) < 2:
        return []

    contracts: list[QueryContract] = []
    for target in normalized[:2]:
        contracts.append(QueryContract(
            clean_query=target + " 是什么",
            interaction_mode="research",
            relation="general_question",
            targets=[target],
            requested_fields=["summary", "results"],
            notes=["rule_based_compound_split"],
        ))
    if has_comparison:
        contracts.append(QueryContract(
            clean_query=clean_query,
            interaction_mode="conversation",
            relation="comparison_synthesis",
            targets=normalized[:2],
            answer_slots=["comparison"],
            notes=["rule_based_compound_split"],
        ))
    return contracts if len(contracts) >= 2 else []


def compound_contracts_from_decomposer_payload(
    *,
    payload: Any,
    fallback_query: str,
    target_normalizer: TargetNormalizer = default_compound_target_normalizer,
) -> list[QueryContract]:
    if not isinstance(payload, dict) or not bool(payload.get("is_compound")):
        return []
    raw_subtasks = payload.get("subtasks", [])
    if not isinstance(raw_subtasks, list):
        return []
    contracts: list[QueryContract] = []
    for index, item in enumerate(raw_subtasks[:5]):
        contract = compound_subtask_contract_from_payload(
            item,
            fallback_query=fallback_query,
            index=index,
            target_normalizer=target_normalizer,
        )
        if contract is not None:
            contracts.append(contract)
    return contracts if len(contracts) >= 2 else []


# Map of relation → human-readable label for compound task display.
# Some labels include inline target text from the contract.
_COMPOUND_TASK_LABELS: dict[str, str] = {
    "library_status": "查看论文库概览和文章预览",
    "library_recommendation": "从库内给出默认推荐",
}


def compound_task_label(contract: QueryContract) -> str:
    if contract.relation in _COMPOUND_TASK_LABELS:
        return _COMPOUND_TASK_LABELS[contract.relation]
    if contract.relation == "formula_lookup":
        target = contract.targets[0] if contract.targets else "目标对象"
        return f"查询 {target} 公式"
    if contract.relation == "comparison_synthesis":
        target_text = " 和 ".join(contract.targets) if contract.targets else "前面结果"
        return f"比较 {target_text}"
    return contract.clean_query


def compound_subtask_contract_from_payload(
    payload: object,
    *,
    fallback_query: str,
    index: int,
    target_normalizer: TargetNormalizer = default_compound_target_normalizer,
) -> QueryContract | None:
    if not isinstance(payload, dict):
        return None
    continuation_mode = str(payload.get("continuation_mode", "") or "").strip().lower()
    # Normalize common LLM hallucinations for continuation_mode
    if continuation_mode in {"context_continuation", "context_continue", "contextual"}:
        continuation_mode = "context_switch"
    if continuation_mode not in {"fresh", "followup", "context_switch"}:
        continuation_mode = "fresh" if index == 0 else "followup"
    clean_query = " ".join(str(payload.get("clean_query", "") or fallback_query).strip().split())
    raw_targets = payload.get("targets", [])
    targets = [str(item).strip() for item in raw_targets if str(item).strip()] if isinstance(raw_targets, list) else []
    raw_answer_slots = payload.get("answer_slots", [])
    if isinstance(raw_answer_slots, str):
        raw_answer_slots = [raw_answer_slots]
    answer_slots = [str(item).strip() for item in raw_answer_slots if str(item).strip()] if isinstance(raw_answer_slots, list) else []
    raw_requested_fields = payload.get("requested_fields", [])
    requested_fields = [str(item).strip() for item in raw_requested_fields if str(item).strip()] if isinstance(raw_requested_fields, list) else []
    targets = target_normalizer(targets, requested_fields)
    relation = str(payload.get("relation", "") or "").strip()
    if relation not in ALLOWED_COMPOUND_SUBTASK_RELATIONS:
        relation = compound_subtask_relation_from_slots(
            answer_slots=answer_slots,
            requested_fields=requested_fields,
            targets=targets,
        )
    if relation not in ALLOWED_COMPOUND_SUBTASK_RELATIONS:
        # P2-1: Don't silently discard — fall back to general_question with a note
        # so the upper layer can decide whether to ask for clarification
        relation = "general_question"
        relation_fallback = True
    else:
        relation_fallback = False
    interaction_mode = str(payload.get("interaction_mode", "") or "").strip().lower()
    if interaction_mode not in {"conversation", "research"}:
        interaction_mode = "conversation" if relation in {"library_status", "library_recommendation", "comparison_synthesis"} else "research"
    if relation in {"library_status", "library_recommendation", "comparison_synthesis"}:
        interaction_mode = "conversation"
    if relation in {"library_status", "library_recommendation"}:
        targets = []
        requested_fields = []
    raw_required_modalities = payload.get("required_modalities", [])
    required_modalities = normalize_modalities(
        [str(item).strip() for item in raw_required_modalities if str(item).strip()] if isinstance(raw_required_modalities, list) else [],
        relation=relation,
    )
    if relation == "formula_lookup":
        requested_fields = [*requested_fields, *[field for field in ["formula", "variable_explanation"] if field not in requested_fields]]
        required_modalities = [*required_modalities, *[modality for modality in ["page_text", "table"] if modality not in required_modalities]]
        interaction_mode = "research"
    if interaction_mode == "conversation":
        required_modalities = []
    elif not required_modalities:
        required_modalities = ["page_text", "paper_card"]
    if interaction_mode == "research" and not requested_fields:
        requested_fields = ["answer"]
    answer_shape = str(payload.get("answer_shape", "") or "").strip().lower()
    if answer_shape not in {"bullets", "narrative", "table"}:
        answer_shape = "table" if relation == "comparison_synthesis" else "narrative"
    precision_requirement = str(payload.get("precision_requirement", "") or "").strip().lower()
    if precision_requirement not in {"exact", "high", "normal"}:
        precision_requirement = "exact" if relation in {"formula_lookup", "metric_value_lookup"} else "high"
    raw_notes = payload.get("notes", [])
    notes = [str(item).strip() for item in raw_notes if str(item).strip()] if isinstance(raw_notes, list) else []
    base_notes = ["compound_subtask", *[f"answer_slot={slot}" for slot in answer_slots], f"subtask_{relation}"]
    if relation_fallback:
        base_notes.append("compound_relation_fallback")
    notes = list(dict.fromkeys([*notes, *base_notes]))
    return QueryContract(
        clean_query=clean_query,
        interaction_mode=interaction_mode,
        relation=relation,
        targets=targets,
        answer_slots=answer_slots,
        requested_fields=requested_fields,
        required_modalities=required_modalities,
        answer_shape=answer_shape,
        precision_requirement=precision_requirement,  # type: ignore[arg-type]
        continuation_mode=continuation_mode,  # type: ignore[arg-type]
        notes=notes,
    )


def compound_subtask_relation_from_slots(
    *,
    answer_slots: list[str],
    requested_fields: list[str],
    targets: list[str],
) -> str:
    slots = {"_".join(str(item or "").strip().lower().replace("-", "_").split()) for item in answer_slots}
    fields = {"_".join(str(item or "").strip().lower().replace("-", "_").split()) for item in requested_fields}
    tokens = slots | fields
    if "library_status" in tokens:
        return "library_status"
    if "library_recommendation" in tokens:
        return "library_recommendation"
    if "comparison" in tokens or "synthesis" in tokens:
        return "comparison_synthesis"
    if "origin" in tokens or {"paper_title", "year"} <= tokens:
        return "origin_lookup"
    if "formula" in tokens:
        return "formula_lookup"
    if "followup_research" in tokens or "followup_papers" in tokens:
        return "followup_research"
    if "figure" in tokens or "figure_conclusion" in tokens:
        return "figure_question"
    if "metric_value" in tokens:
        return "metric_value_lookup"
    if "paper_summary" in tokens or "summary" in tokens or "results" in tokens:
        return "paper_summary_results"
    if "paper_recommendation" in tokens or "recommended_papers" in tokens:
        return "paper_recommendation"
    if "topology_recommendation" in tokens or "best_topology" in tokens:
        return "topology_recommendation"
    if "topology_discovery" in tokens or "relevant_papers" in tokens:
        return "topology_discovery"
    if "entity_definition" in tokens or "entity_type" in tokens or ("definition" in tokens and targets):
        return "entity_definition"
    if "concept_definition" in tokens or "definition" in tokens:
        return "concept_definition"
    return "general_question"


def merge_redundant_field_subtasks(subcontracts: list[QueryContract]) -> list[QueryContract]:
    mergeable_relations = {
        "paper_summary_results",
        "metric_value_lookup",
        "entity_definition",
        "concept_definition",
        "formula_lookup",
        "figure_question",
        "general_question",
        "followup_research",
    }
    merged: list[QueryContract] = []
    by_key: dict[tuple[str, str, tuple[str, ...]], int] = {}
    precision_rank = {"normal": 0, "high": 1, "exact": 2}
    for contract in subcontracts:
        normalized_targets = tuple(normalize_lookup_text(target) for target in contract.targets if target)
        key = (contract.interaction_mode, contract.relation, normalized_targets)
        if contract.relation not in mergeable_relations or key not in by_key:
            by_key[key] = len(merged)
            merged.append(contract)
            continue
        existing_index = by_key[key]
        existing = merged[existing_index]
        requested_fields = list(dict.fromkeys([*existing.requested_fields, *contract.requested_fields]))
        required_modalities = list(dict.fromkeys([*existing.required_modalities, *contract.required_modalities]))
        notes = list(dict.fromkeys([*contract_notes(existing), *contract_notes(contract), "merged_same_target_fields"]))
        # P1-6: Choose the semantically more complete query instead of
        # concatenating with "；" which confuses downstream planners.
        clean_query = (
            contract.clean_query
            if len(contract.clean_query) > len(existing.clean_query)
            else existing.clean_query
        )
        precision = (
            contract.precision_requirement
            if precision_rank.get(contract.precision_requirement, 0) > precision_rank.get(existing.precision_requirement, 0)
            else existing.precision_requirement
        )
        merged[existing_index] = existing.model_copy(
            update={
                "clean_query": clean_query,
                "requested_fields": requested_fields or existing.requested_fields,
                "required_modalities": required_modalities or existing.required_modalities,
                "precision_requirement": precision,
                "notes": notes,
            }
        )
    return merged


def comparison_results_with_memory(
    *,
    subtask_results: list[dict[str, Any]],
    session: SessionContext,
    comparison_contract: QueryContract | None,
) -> list[dict[str, Any]]:
    augmented = list(subtask_results)
    present_targets = {
        normalize_lookup_text(target)
        for result in augmented
        if isinstance(result.get("contract"), QueryContract)
        for target in result["contract"].targets
        if str(target).strip()
    }
    requested_targets = list(comparison_contract.targets if comparison_contract is not None else [])
    if not requested_targets:
        requested_targets = list(
            dict.fromkeys(
                [
                    *session.effective_active_research().targets,
                    *[item.get("target", "") for item in active_memory_bindings(session)],
                ]
            )
        )
    bindings = dict((session.working_memory or {}).get("target_bindings", {}) or {})
    for target in requested_targets:
        clean_target = str(target or "").strip()
        key = normalize_lookup_text(clean_target)
        if not key or key in present_targets:
            continue
        binding = bindings.get(key)
        if not isinstance(binding, dict):
            continue
        relation = str(binding.get("relation", "") or "followup_research")
        requested_fields = [str(item) for item in list(binding.get("requested_fields", []) or []) if str(item)]
        contract = QueryContract(
            clean_query=str(binding.get("clean_query", "") or clean_target),
            relation=relation,
            targets=[str(binding.get("target", "") or clean_target)],
            requested_fields=requested_fields or ["answer"],
            required_modalities=[str(item) for item in list(binding.get("required_modalities", []) or []) if str(item)] or ["page_text"],
            continuation_mode="followup",
            notes=["restored_from_session_memory_for_comparison"],
        )
        augmented.append(
            {
                "contract": contract,
                "answer": str(binding.get("answer_preview", "") or ""),
                "citations": [],
                "claims": [],
                "evidence": [],
                "verification": VerificationReport(status="pass", recommended_action="memory_comparison_context"),
            }
        )
        present_targets.add(key)
    return augmented


def compose_compound_comparison_answer(
    *,
    query: str,
    subtask_results: list[dict[str, Any]],
    session: SessionContext,
    comparison_contract: QueryContract | None,
    clients: Any,
    clean_text: Callable[[str], str],
) -> str:
    comparable_results = comparison_results_with_memory(
        subtask_results=subtask_results,
        session=session,
        comparison_contract=comparison_contract,
    )
    comparable = [
        {
            "relation": result["contract"].relation if isinstance(result.get("contract"), QueryContract) else "",
            "targets": result["contract"].targets if isinstance(result.get("contract"), QueryContract) else [],
            "answer": str(result.get("answer", "")),
            "claims": [claim.model_dump() for claim in list(result.get("claims", []) or []) if isinstance(claim, Claim)],
            "evidence": [
                {
                    "title": item.get("title", "") if isinstance(item, dict) else getattr(item, "title", ""),
                    "page": item.get("page", "") if isinstance(item, dict) else getattr(item, "page", ""),
                    "snippet": (item.get("snippet", "") if isinstance(item, dict) else getattr(item, "snippet", ""))[:600],
                }
                for item in list(result.get("evidence", []) or [])[:4]
            ],
        }
        for result in comparable_results
    ]
    if getattr(clients, "chat", None) is not None:
        text = clients.invoke_text(
            system_prompt=(
                "你是论文研究助手的多子任务综合器。"
                "只基于输入的子任务答案、claims 和 evidence 做比较，不要引入外部记忆。"
                "必须从 evidence 里挑引用，如果 evidence 中缺少某个比较维度就直接说明证据不足。"
                "请用简洁中文 Markdown 输出：先给 1 句总览，再用表格比较目标函数/优化信号/是否需要 reward model/使用场景，最后给读法建议。"
                "如果某个子任务证据不足，要明确说证据不足，不要补公式。"
            ),
            human_prompt=json.dumps(
                {
                    "query": query,
                    "subtasks": comparable,
                },
                ensure_ascii=False,
            ),
            fallback="",
        ).strip()
        if text:
            return clean_text(text)
    rows: list[str] = []
    for result in comparable:
        targets = result.get("targets") or []
        target = str(targets[0]) if targets else "对象"
        answer = " ".join(str(result.get("answer", "")).split())
        rows.append(f"- **{target}**：{answer[:260] if answer else '当前证据不足。'}")
    return "基于前两个子任务的证据，可以先做保守比较：\n\n" + "\n".join(rows)


def compound_section_heading(*, contract: QueryContract, index: int) -> str:
    return f"## {index}. {compound_task_label(contract)}"


def compound_research_progress_markdown(*, contract: QueryContract, index: int) -> str:
    heading = compound_section_heading(contract=contract, index=index)
    if contract.relation == "formula_lookup":
        target = contract.targets[0] if contract.targets else "目标对象"
        return f"{heading}\n\n好的，我现在去查询 **{target}** 的公式。"
    return heading


def demote_markdown_headings(answer: str) -> str:
    return re.sub(r"^(#{1,5})\s+", lambda match: "#" + match.group(1) + " ", str(answer or "").strip(), flags=re.M)


def format_compound_section(*, contract: QueryContract, answer: str, index: int) -> str:
    normalized = demote_markdown_headings(str(answer or "").strip())
    return f"{compound_section_heading(contract=contract, index=index)}\n\n{normalized}".strip()


def compound_task_result_from_task_payload(
    task_result: dict[str, Any],
    *,
    fallback_contract: QueryContract,
) -> dict[str, Any]:
    contract = task_result.get("contract_obj")
    if not isinstance(contract, QueryContract):
        raw_contract = task_result.get("contract")
        if isinstance(raw_contract, dict):
            try:
                contract = QueryContract.model_validate(raw_contract)
            except Exception:  # noqa: BLE001
                contract = fallback_contract
        else:
            contract = fallback_contract
    verification = task_result.get("verification_obj")
    if not isinstance(verification, VerificationReport):
        raw_verification = task_result.get("verification")
        if isinstance(raw_verification, dict):
            try:
                verification = VerificationReport.model_validate(raw_verification)
            except Exception:  # noqa: BLE001
                verification = VerificationReport(status="pass", recommended_action="task_subagent")
        else:
            verification = VerificationReport(status="pass", recommended_action="task_subagent")
    return {
        "contract": contract,
        "answer": str(task_result.get("answer", "") or ""),
        "citations": list(task_result.get("citations", []) or []),
        "claims": list(task_result.get("claims", []) or []),
        "evidence": list(task_result.get("evidence", []) or []),
        "verification": verification,
    }
