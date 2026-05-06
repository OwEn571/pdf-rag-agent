from __future__ import annotations

from typing import Any, Callable

from app.domain.models import (
    AssistantCitation,
    AssistantResponse,
    QueryContract,
    SessionContext,
    SessionTurn,
    VerificationReport,
)
from app.services.agent.runtime_summary import build_runtime_summary
from app.services.agent.task import run_task_subagent
from app.services.agent.tools import agent_tool_manifest
from app.services.agent.tool_events import record_agent_observation as record_agent_observation_event
from app.services.clarification.intents import (
    clarification_options_from_contract_notes,
    clarification_tracking_key,
    clear_pending_clarification,
    contract_from_pending_clarification,
    remember_clarification_attempt,
    reset_clarification_tracking,
    store_pending_clarification,
)
from app.services.clarification.questions import build_agent_clarification_question
from app.services.contracts.context import contract_notes
from app.services.contracts.normalization import normalize_contract_targets
from app.services.planning.compound_tasks import (
    compose_compound_comparison_answer,
    compound_task_label,
    compound_task_result_from_task_payload,
    compound_research_progress_markdown,
    compound_section_heading,
    demote_markdown_headings,
    format_compound_section,
    llm_decompose_compound_query,
    merge_redundant_field_subtasks,
)
from app.services.answers.evidence_presentation import chunk_text, dedupe_citations
from app.services.memory.research import remember_compound_outcome
from app.services.contracts.session_context import (
    agent_session_conversation_context,
    make_active_research,
    session_llm_history_messages,
)

EmitFn = Callable[[str, dict[str, Any]], None]
PENDING_COMPOUND_PLAN_KEY = "pending_compound_plan"


def execute_compound_task_subagent(
    *,
    agent: Any,
    contract: QueryContract,
    session: SessionContext,
    emit: EmitFn,
    execution_steps: list[dict[str, Any]],
    # P1-4: prior results from earlier subtasks for evidence/paper reuse
    prior_results: list[dict[str, Any]] | None = None,
    compound_index: int = 1,
    compound_total: int = 1,
) -> dict[str, Any]:
    # P1-5: Exclude Task tool from subtask planner to prevent recursive
    # sub-sub-task spawning within compound execution.
    from app.services.agent.tools import all_agent_tool_names as _all_tool_names
    _subtool_names = sorted(_all_tool_names() - {"Task"})
    task_result = run_task_subagent(
        agent=agent,
        prompt=contract.clean_query,
        description=compound_task_label(contract),
        tools_allowed=_subtool_names,
        max_steps=8,
        session=session,
        max_web_results=3,
        emit=emit,
        execution_steps=execution_steps,
        contract=contract,
        prior_results=prior_results,
        compound_index=compound_index,
        compound_total=compound_total,
    )
    result = compound_task_result_from_task_payload(task_result, fallback_contract=contract)
    result_contract = result.get("contract")
    relation = result_contract.relation if isinstance(result_contract, QueryContract) else contract.relation
    record_agent_observation_event(
        emit=emit,
        execution_steps=execution_steps,
        tool="Task",
        summary=f"compound_subtask:{relation}",
        payload={
            "prompt": contract.clean_query,
            "relation": relation,
            "verification": task_result.get("verification", {}),
            "answer_chars": len(str(result.get("answer", "") or "")),
        },
    )
    return result


def run_compound_query_if_needed(
    *,
    agent: Any,
    query: str,
    session_id: str,
    session: SessionContext,
    clarification_choice: dict[str, Any] | None,
    emit: EmitFn,
    execution_steps: list[dict[str, Any]],
) -> dict[str, Any] | None:
    clean_query = " ".join(query.strip().split())

    def target_normalizer(targets: list[str], fields: list[str]) -> list[str]:
        return normalize_contract_targets(
            targets=targets,
            requested_fields=fields,
            canonicalize_targets=agent.retriever.canonicalize_targets,
        )

    resumed_plan = pending_compound_plan(session=session)
    if resumed_plan is not None:
        selected_contract = contract_from_pending_clarification(
            clean_query=clean_query,
            session=session,
            clarification_choice=clarification_choice,
        )
        if selected_contract is None:
            return None
        clean_query = str(resumed_plan.get("query", "") or clean_query)
        subcontracts = pending_compound_subcontracts(plan=resumed_plan, selected_contract=selected_contract)
    else:
        if clarification_choice is not None:
            return None
        subcontracts = llm_decompose_compound_query(
            clean_query=clean_query,
            session=session,
            clients=agent.clients,
            available_tools=list(agent_tool_manifest()),
            conversation_context=lambda current_session, *, max_chars=24000: agent_session_conversation_context(
                current_session,
                settings=agent.settings,
                max_chars=max_chars,
            ),
            history_messages=session_llm_history_messages,
            target_normalizer=target_normalizer,
        )
    subcontracts = merge_redundant_field_subtasks(subcontracts)
    if len(subcontracts) < 2:
        return None
    compound_targets = list(dict.fromkeys(target for item in subcontracts for target in item.targets))
    compound_requested_fields = list(dict.fromkeys(field for item in subcontracts for field in item.requested_fields))
    compound_required_modalities = list(dict.fromkeys(modality for item in subcontracts for modality in item.required_modalities))

    compound_contract = QueryContract(
        clean_query=clean_query,
        interaction_mode="conversation",
        relation="compound_query",
        targets=compound_targets,
        requested_fields=compound_requested_fields or ["subtasks"],
        required_modalities=compound_required_modalities,
        answer_shape="narrative",
        precision_requirement="normal",
        continuation_mode="fresh",
        notes=[
            "llm_compound_planning",
            *[f"subtask:{item.relation}" for item in subcontracts],
        ],
    )
    plan_summary = {
        "thought": "Split the user message into independent subtasks and answer each with the appropriate tool path.",
        "subtasks": [
            {
                "relation": item.relation,
                "clean_query": item.clean_query,
                "interaction_mode": item.interaction_mode,
            }
            for item in subcontracts
        ],
    }
    emit("contract", compound_contract.model_dump())
    emit("agent_plan", plan_summary)
    emit("plan", plan_summary)
    execution_steps.append({"node": "compound_planner", "summary": " + ".join(item.relation for item in subcontracts)})

    answer_parts: list[str] = []
    # P0-5: Load previously completed subtask results when resuming
    subtask_results: list[dict[str, Any]] = list(
        (resumed_plan or {}).get("subtask_results", []) if resumed_plan is not None else []
    )
    blocked_subtasks: list[dict[str, Any]] = []  # P0-5: collect instead of immediate return

    def publish(text: str) -> None:
        if not text:
            return
        answer_parts.append(text)
        for chunk in chunk_text(text, size=96):
            emit("answer_delta", {"text": chunk})

    for index, sub_contract in enumerate(subcontracts, start=1):
        emit(
            "compound_task",
            {
                "index": index,
                "relation": sub_contract.relation,
                "query": sub_contract.clean_query,
            },
        )
        execution_steps.append({"node": f"compound_task:{sub_contract.relation}", "summary": sub_contract.clean_query})
        if sub_contract.relation == "comparison_synthesis":
            comparison = compose_compound_comparison_answer(
                query=clean_query,
                subtask_results=subtask_results,
                session=session,
                comparison_contract=sub_contract,
                clients=agent.clients,
                clean_text=agent._clean_common_ocr_artifacts,
            )
            section = format_compound_section(contract=sub_contract, answer=comparison, index=index)
            publish("\n\n" + section)
            subtask_results.append({"contract": sub_contract, "answer": comparison, "citations": [], "claims": []})
            continue
        if sub_contract.interaction_mode == "conversation":
            heading = compound_section_heading(contract=sub_contract, index=index)
            publish("\n\n" + heading + "\n\n")
            subtask_result = execute_compound_task_subagent(
                agent=agent,
                contract=sub_contract,
                session=session,
                emit=emit,
                execution_steps=execution_steps,
                prior_results=list(subtask_results),  # P1-4
                compound_index=index,
                compound_total=len(subcontracts),
            )
            answer_parts.append(demote_markdown_headings(str(subtask_result.get("answer", "")).strip()))
            subtask_results.append(subtask_result)
            continue
        publish("\n\n" + compound_research_progress_markdown(contract=sub_contract, index=index) + "\n\n")
        subtask_result = execute_compound_task_subagent(
            agent=agent,
            contract=sub_contract,
            session=session,
            emit=emit,
            execution_steps=execution_steps,
            prior_results=list(subtask_results),  # P1-4
            compound_index=index,
            compound_total=len(subcontracts),
        )
        sub_answer = demote_markdown_headings(str(subtask_result.get("answer", "")).strip())
        publish(sub_answer)
        subtask_results.append(subtask_result)
        sub_verification = subtask_result.get("verification")
        sub_result_contract = subtask_result.get("contract")
        if isinstance(sub_verification, VerificationReport) and sub_verification.status == "clarify" and isinstance(
            sub_result_contract,
            QueryContract,
        ):
            # P0-5: Collect blocked subtask instead of immediate return
            blocked_subtasks.append({
                "index": index,
                "contract": sub_result_contract,
                "verification": sub_verification,
            })

    # P0-5: If any subtasks need clarification, return after collecting all
    if blocked_subtasks:
        _first_blocked = blocked_subtasks[0]
        store_pending_compound_plan(
            session=session,
            query=query,
            subcontracts=subcontracts,
            blocked_index=_first_blocked["index"],
            subtask_results=subtask_results,  # preserve completed work
        )
        return compound_clarification_response(
            agent=agent,
            query=query,
            session_id=session_id,
            session=session,
            compound_contract=compound_contract,
            blocked_contract=_first_blocked["contract"],
            plan_summary=plan_summary,
            answer="".join(answer_parts).strip(),
            verification=_first_blocked["verification"],
            execution_steps=execution_steps,
            subcontracts=subcontracts,
            blocked_index=_first_blocked["index"],
        )

    answer = "".join(answer_parts).strip()
    clear_pending_compound_plan(session)
    remember_compound_outcome(
        session=session,
        clean_query=clean_query,
        subtask_results=subtask_results,
        candidate_lookup=agent._candidate_from_paper_id,
    )
    session.last_relation = "compound_query"
    compound_titles = list(
        dict.fromkeys(
            citation.title
            for result in subtask_results
            for citation in list(result.get("citations", []) or [])
            if isinstance(citation, AssistantCitation) and citation.title
        )
    )
    active_research = make_active_research(
        relation="compound_query",
        targets=compound_targets,
        titles=compound_titles,
        requested_fields=compound_requested_fields or ["subtasks"],
        required_modalities=compound_required_modalities,
        answer_shape="narrative",
        precision_requirement="normal",
        clean_query=clean_query,
    )
    session.answered_titles = list(dict.fromkeys([*session.answered_titles, *active_research.titles]))
    clear_pending_clarification(session)
    reset_clarification_tracking(session)
    agent.sessions.commit_turn(
        session,
        SessionTurn.from_contract(
            query=query,
            answer=answer,
            contract=compound_contract,
            interaction_mode="conversation",
            titles=list(active_research.titles),
        ),
        active=active_research,
    )
    citations = dedupe_citations(
        [
            citation
            for result in subtask_results
            for citation in list(result.get("citations", []) or [])
            if isinstance(citation, AssistantCitation)
        ]
    )
    response = AssistantResponse(
        session_id=session_id,
        interaction_mode="conversation",
        answer=answer,
        citations=citations,
        query_contract=compound_contract.model_dump(),
        research_plan_summary=plan_summary,
        runtime_summary=build_runtime_summary(
            contract=compound_contract,
            active_research_context=session.active_research_context_payload(),
            tool_plan=plan_summary,
            execution_steps=execution_steps,
            verification_report={"status": "pass", "recommended_action": "compound_answer"},
            citations=citations,
        ),
        execution_steps=execution_steps,
        verification_report={"status": "pass", "recommended_action": "compound_answer"},
    )
    return response.model_dump()


def compound_clarification_response(
    *,
    agent: Any,
    query: str,
    session_id: str,
    session: SessionContext,
    compound_contract: QueryContract,
    blocked_contract: QueryContract,
    plan_summary: dict[str, Any],
    answer: str,
    verification: VerificationReport,
    execution_steps: list[dict[str, Any]],
    subcontracts: list[QueryContract],
    blocked_index: int,
) -> dict[str, Any]:
    clarification_options = clarification_options_from_contract_notes(blocked_contract)
    if clarification_options:
        store_pending_clarification(session=session, contract=blocked_contract, options=clarification_options)
        store_pending_compound_plan(
            session=session,
            query=query,
            subcontracts=subcontracts,
            blocked_index=blocked_index,
        )
    remember_clarification_attempt(
        session=session,
        key=clarification_tracking_key(
            contract=blocked_contract,
            verification=verification,
            options=clarification_options,
        ),
    )
    target_text = " / ".join(blocked_contract.targets) if blocked_contract.targets else "其中一个子任务"
    prefix = (
        f"在继续复合问题的后续检索和比较前，我需要先确认 `{target_text}` 的具体含义；"
        "否则后面的比较会建立在错误对象上。\n\n"
    )
    fallback_question = build_agent_clarification_question(
        contract=blocked_contract,
        session=session,
        clients=agent.clients,
        settings=agent.settings,
    )
    final_answer = prefix + answer if answer else fallback_question
    compound_notes = list(
        dict.fromkeys(
            [
                *contract_notes(compound_contract),
                "compound_blocked_by_clarification",
                f"blocked_subtask_relation={blocked_contract.relation}",
                *[f"blocked_target={target}" for target in blocked_contract.targets],
            ]
        )
    )
    response_contract = compound_contract.model_copy(
        update={
            "notes": compound_notes,
            "targets": list(dict.fromkeys([*compound_contract.targets, *blocked_contract.targets])),
        }
    )
    agent.sessions.commit_turn(
        session,
        SessionTurn.from_contract(
            query=query,
            answer=final_answer,
            contract=response_contract,
            interaction_mode="conversation",
        ),
    )
    return AssistantResponse(
        session_id=session_id,
        interaction_mode="conversation",
        answer=final_answer,
        citations=[],
        query_contract=response_contract.model_dump(),
        research_plan_summary=plan_summary,
        runtime_summary=build_runtime_summary(
            contract=response_contract,
            active_research_context=session.active_research_context_payload(),
            tool_plan=plan_summary,
            execution_steps=execution_steps,
            verification_report=verification.model_dump(),
            citations=[],
        ),
        execution_steps=execution_steps,
        verification_report=verification.model_dump(),
        needs_human=True,
        clarification_question=fallback_question,
        clarification_options=clarification_options,
    ).model_dump()


def pending_compound_plan(*, session: SessionContext) -> dict[str, Any] | None:
    payload = dict((session.working_memory or {}).get(PENDING_COMPOUND_PLAN_KEY, {}) or {})
    raw_subcontracts = payload.get("subcontracts", [])
    if not isinstance(raw_subcontracts, list) or not raw_subcontracts:
        return None
    return payload


def store_pending_compound_plan(
    *,
    session: SessionContext,
    query: str,
    subcontracts: list[QueryContract],
    blocked_index: int,
    subtask_results: list[dict[str, Any]] | None = None,
) -> None:
    memory = dict(session.working_memory or {})
    memory[PENDING_COMPOUND_PLAN_KEY] = {
        "query": query,
        "blocked_index": blocked_index,
        "subcontracts": [contract.model_dump() for contract in subcontracts],
        "subtask_results": subtask_results or [],
    }
    session.working_memory = memory


def clear_pending_compound_plan(session: SessionContext) -> None:
    memory = dict(session.working_memory or {})
    memory.pop(PENDING_COMPOUND_PLAN_KEY, None)
    session.working_memory = memory


def pending_compound_subcontracts(*, plan: dict[str, Any], selected_contract: QueryContract) -> list[QueryContract]:
    blocked_index = int(plan.get("blocked_index", 0) or 0)
    contracts: list[QueryContract] = []
    for index, raw_contract in enumerate(list(plan.get("subcontracts", []) or []), start=1):
        if index == blocked_index:
            contracts.append(selected_contract)
            continue
        try:
            contracts.append(QueryContract.model_validate(raw_contract))
        except Exception:  # noqa: BLE001
            continue
    return contracts
