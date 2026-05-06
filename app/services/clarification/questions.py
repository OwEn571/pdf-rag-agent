from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from app.domain.models import QueryContract, SessionContext
from app.services.clarification.intents import ambiguity_clarification_question
from app.services.contracts.context import contract_answer_slots
from app.services.intents.followup import is_negative_correction_query
from app.services.planning.research import research_plan_context_from_contract
from app.services.contracts.session_context import agent_session_conversation_context

ConversationContext = Callable[[SessionContext], dict[str, Any]]


def build_clarification_question(
    *,
    contract: QueryContract,
    session: SessionContext,
    clients: Any,
    conversation_context: ConversationContext,
) -> str:
    ambiguity_question = ambiguity_clarification_question(contract=contract, session=session)
    if ambiguity_question:
        return ambiguity_question
    requested = {str(item) for item in contract.requested_fields}
    query_lower = str(contract.clean_query or "").lower()
    targets = [str(item).strip() for item in contract.targets if str(item).strip()]
    if "formula" in requested and is_negative_correction_query(query_lower):
        target_text = " / ".join(targets) if targets else "当前目标"
        return (
            f"你说得对，上一条候选公式不能直接当作 `{target_text}` 的公式。"
            "我这边需要重新定位目标含义和对应论文证据；如果本地 PDF 里只有文字说明而没有公式，我会明确说未找到，而不是继续套用不匹配的公式。"
        )
    if "formula" in requested and targets:
        target_text = " / ".join(targets)
        return (
            f"我还不能确认 `{target_text}` 的目标函数或公式。"
            "请指定它对应的论文、方法全称或上下文；如果本地 PDF 里没有明确公式，我会直接说明未找到。"
        )
    if getattr(clients, "chat", None) is not None:
        response_text = clients.invoke_text(
            system_prompt=(
                "你是论文研究助手的研究澄清问题生成器。"
                "请根据当前 query_contract 和会话上下文，生成一句自然、具体、不生硬的中文回复。"
                "如果当前像是在质疑上一轮回答，就先承认需要重新核对，再给出 1-2 个具体追问方向。"
                "如果当前缺的是目标实体或论文来源，也要直接点明缺口，但不要只说“请明确你的问题”。"
            ),
            human_prompt=json.dumps(
                {
                    "query": contract.clean_query,
                    "interaction_mode": contract.interaction_mode,
                    "continuation_mode": contract.continuation_mode,
                    "targets": contract.targets,
                    "requested_fields": contract.requested_fields,
                    "required_modalities": contract.required_modalities,
                    "answer_slots": contract_answer_slots(contract),
                    "conversation_context": conversation_context(session),
                    "active_research_context": session.active_research_context_payload(),
                    "recent_turns": [
                        {
                            "query": turn.query,
                            "targets": turn.targets,
                            "requested_fields": turn.requested_fields,
                        }
                        for turn in session.turns[-2:]
                    ],
                },
                ensure_ascii=False,
            ),
            fallback="",
        ).strip()
        if response_text:
            return response_text
    if contract.continuation_mode == "followup" and not session.effective_active_research().targets:
        return "我需要确认你是在延续上一轮的哪篇论文或哪个主题。"
    goals = set(research_plan_context_from_contract(contract).goals)
    if contract.targets and goals & {"definition", "entity_type", "mechanism", "figure_conclusion", "answer", "general_answer"}:
        return f"当前语料里还没有稳定定位到与 `{contract.targets[0]}` 直接相关的证据。你可以指定论文、上下文，或换一种问法再试一次。"
    return "我需要更多上下文来确定你当前要继续的研究任务。"


def build_agent_clarification_question(
    *,
    contract: QueryContract,
    session: SessionContext,
    clients: Any,
    settings: Any,
) -> str:
    return build_clarification_question(
        contract=contract,
        session=session,
        clients=clients,
        conversation_context=lambda current_session: agent_session_conversation_context(
            current_session,
            settings=settings,
        ),
    )
