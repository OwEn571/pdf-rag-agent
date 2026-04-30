from __future__ import annotations

import json
from pathlib import Path
from types import MethodType

from langchain_core.documents import Document

from app.core.agent_settings import AgentSettings
from app.core.config import Settings
from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan, SessionContext, SessionTurn, VerificationReport
from app.services.agent import ResearchAssistantAgentV4
from app.services.clarification_intents import (
    contract_from_selected_clarification_option,
    contract_with_ambiguity_options,
)
from app.services.evidence_presentation import build_figure_contexts
from app.services.figure_intents import figure_signal_score
from app.services.library import LibraryBrowserService
from app.services.memory_followup_answers import compose_memory_followup_answer
from app.services.model_clients import ModelClients
from app.services.retrieval import DualIndexRetriever
from app.services.session_store import InMemorySessionStore
from app.services.web_evidence import build_web_research_claim, collect_web_evidence


class StubModelClients:
    def __init__(self) -> None:
        self.chat = object()
        self.vlm = None
        self.concept_definition_calls = 0
        self.entity_type_calls = 0
        self.entity_grounding_calls = 0
        self.followup_refine_calls = 0
        self.compound_decompose_calls = 0
        self.relationship_verifier_calls = 0

    def invoke_json_messages(self, *, system_prompt: str, messages: list[dict[str, str]], fallback: object) -> object:
        human_prompt = messages[-1]["content"] if messages else "{}"
        return self.invoke_json(system_prompt=system_prompt, human_prompt=human_prompt, fallback=fallback)

    def invoke_text(self, *, system_prompt: str, human_prompt: str, fallback: str = "") -> str:
        if "论文公式讲解器" in system_prompt:
            return (
                "## 怎么理解\n\n"
                "这个式子不是让模型死记公式，而是在偏好对里提高被选中回答的相对概率，"
                "同时压低被拒绝回答的相对概率。参考策略提供锚点，避免当前策略偏离太远。"
            )
        if "回答语言修正器" in system_prompt:
            return "好的，我会改成中文说明；公式符号和论文标题保留原样，变量含义用中文解释。"
        if "通用会话记忆问答工具" in system_prompt:
            payload = json.loads(human_prompt)
            query = str(payload.get("current_query", "") or "").strip()
            context = dict(payload.get("conversation_context", {}) or {})
            memory = dict(context.get("working_memory", {}) or {})
            tool_results = [item for item in list(memory.get("tool_results", []) or []) if isinstance(item, dict)]
            recommendation = next(
                (item for item in reversed(tool_results) if item.get("tool") == "get_library_recommendation"),
                {},
            )
            preview = str(recommendation.get("answer_preview", "") or "")
            if "推荐理由" in query or "为什么选择" in query:
                return (
                    "## 推荐理由\n\n"
                    "我延续上一轮库内推荐结果回答：首选它，是因为上一轮推荐工具按基础性、综述性、"
                    "和当前论文库主线相关度来排序，而《A Survey on LLM-as-a-Judge》覆盖评测这个基础入口。"
                    f"\n\n上一轮依据摘要：{preview[:160]}"
                )
            return "我根据上一轮工具结果继续回答这个追问。"
        if "会话记忆综合器" in system_prompt:
            payload = json.loads(human_prompt)
            context = dict(payload.get("conversation_context", {}) or {})
            memory = dict(context.get("working_memory", {}) or {})
            bindings = dict(memory.get("target_bindings", {}) or {})
            names = [str(item.get("target", "")) for item in bindings.values() if isinstance(item, dict)]
            return "## 对比\n\n" + "\n".join(f"- {name}：沿用上一轮工具结果。" for name in names[:2])
        if "对话回复整理器" in system_prompt:
            payload = json.loads(human_prompt)
            relation = payload.get("relation", "")
            if relation == "greeting":
                return "你好，我是论文研究助手 V4。"
            if relation == "self_identity":
                return "我是论文研究助手 V4，专注于论文检索、阅读和多轮研究追问。"
            if relation == "capability":
                return "## 我能做什么\n\n- 检索并推荐论文\n- 总结核心结论和实验结果\n- 解释术语、公式、图表和后续工作"
            if relation == "clarify_user_intent":
                return "你可以直接补充你想继续的主题、论文名或上一条里不清楚的点。"
            return "我们可以继续把问题说具体一点。"
        if "研究澄清问题生成器" in system_prompt:
            payload = json.loads(human_prompt)
            targets = list(payload.get("targets", []) or [])
            active = dict(payload.get("active_research_context", {}) or {})
            target = targets[0] if targets else (list(active.get("targets", []) or [""])[0] if active.get("targets") else "")
            if target:
                return f"我这边还没有稳定定位到与 `{target}` 直接对应的证据，我先重新核对来源。你也可以告诉我是想确认它的定义、起源，还是想核对上一条引用的论文。"
            return "我先重新核对一下当前线索。你也可以补一句你是想追问定义、来源，还是上一条回答里的哪一部分。"
        if "论文实体解释器" in system_prompt:
            fields: dict[str, str] = {}
            for line in human_prompt.splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                fields[key.strip()] = value.strip()
            target = fields.get("target", "该对象")
            label = fields.get("entity_type", "实体")
            summary = fields.get("summary", "")
            if summary:
                return f"{target} 是一个{label}。{summary[:120]}"
            return f"{target} 是一个{label}。"
        if "最终回答整理器" in system_prompt:
            payload = json.loads(human_prompt)
            relation = payload.get("relation", "")
            claims = payload.get("claims", [])
            first = claims[0] if claims else {}
            structured = dict(first.get("structured_data", {}) or {})
            if relation == "entity_definition":
                entity = str(first.get("entity", "") or (payload.get("targets") or ["该对象"])[0])
                label = str(first.get("value", "") or "相关实体")
                definition_lines = [str(item) for item in structured.get("definition_lines", []) if str(item)]
                mechanism_lines = [str(item) for item in structured.get("mechanism_lines", []) if str(item)]
                application_lines = [str(item) for item in structured.get("application_lines", []) if str(item)]
                lead = str(structured.get("description", "") or "").strip()
                if not lead:
                    lead = f"{entity} 是一个{label}。"
                    if definition_lines:
                        lead += f" {definition_lines[0]}"
                sections = ["## 回答", "", lead]
                if mechanism_lines:
                    sections.extend(["", "## 核心机制", "", *[f"- {line}" for line in mechanism_lines[:3]]])
                if application_lines:
                    sections.extend(["", "## 用途", "", *[f"- {line}" for line in application_lines[:2]]])
                return "\n".join(sections)
            if relation == "concept_definition":
                return str(first.get("value", ""))
            if relation == "origin_lookup":
                return (
                    "## 结论\n\n"
                    f"`{first.get('entity', '该架构')}` 最早由 {structured.get('year', '')} 年的《{first.get('value', '')}》提出。"
                )
            if relation == "formula_lookup":
                return f"## 核心公式\n\n```text\n{first.get('value', '')}\n```"
            if relation == "paper_recommendation":
                rows = structured.get("recommended_papers", [])
                body = "\n".join(f"- 《{row['title']}》({row['year']})：{row['reason']}" for row in rows)
                return f"## 推荐阅读\n\n{body}"
            if relation == "followup_research":
                rows = structured.get("followup_titles", [])
                body = "\n".join(f"- 《{row['title']}》({row['year']})" for row in rows)
                return f"## 后续工作\n\n{body}"
            if relation == "topology_discovery":
                terms = ", ".join(structured.get("topology_terms", []))
                return f"## 检索结果\n\n- 最相关论文：{first.get('value', '')}\n- 可识别拓扑：{terms}"
            if relation == "topology_recommendation":
                return first.get("value", "")
            if relation == "metric_value_lookup":
                lines = structured.get("metric_lines", [])
                body = "\n".join(f"- {line}" for line in lines[:3])
                return f"## 实验结果\n\n{body}"
            if relation == "paper_summary_results":
                query = str(payload.get("query", "") or "")
                context = dict(payload.get("conversation_context", {}) or {})
                turns = [item for item in list(context.get("turns", []) or []) if isinstance(item, dict)]
                previous_followup = next(
                    (
                        item
                        for item in reversed(turns)
                        if dict(item.get("query_contract", {}) or {}).get("relation") == "followup_research"
                    ),
                    {},
                )
                if (
                    "Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals" in query
                    and ("是吗" in query or "仔细看看" in query)
                    and previous_followup
                ):
                    return (
                        "## 判断\n\n"
                        "它相对 AlignX 更像“强相关延续候选”，不是当前证据下可以直接写成严格后续工作的论文。\n\n"
                        "## 为什么\n\n"
                        "- 上一轮问题是在追踪 AlignX 数据集的后续工作，上一轮回答已经把这篇论文放在候选延续线索里。\n"
                        "- 当前证据只能确认它讨论 behavioral signals 下的 preference inference / AlignXplore，主题接近 AlignX 的 personalized preference/user-level alignment。\n"
                        "- 但证据没有稳定显示它明确使用、继承、引用或评测 AlignX 数据集，所以应保守写成 related/strong candidate。"
                    )
                lines = structured.get("metric_lines", [])
                body = "\n".join(f"- {line}" for line in lines[:3])
                return f"## 核心结论\n\n- {first.get('value', '')}\n\n## 实验结果\n\n{body}"
            if relation == "figure_question":
                return str(first.get("value", ""))
            return str(first.get("value", ""))
        return fallback

    def stream_text(self, *, system_prompt: str, human_prompt: str, on_delta, fallback: str = "") -> str:
        text = self.invoke_text(system_prompt=system_prompt, human_prompt=human_prompt, fallback=fallback)
        if not text:
            return fallback
        split_at = max(1, len(text) // 2)
        on_delta(text[:split_at])
        on_delta(text[split_at:])
        return text

    def invoke_json(self, *, system_prompt: str, human_prompt: str, fallback: object) -> object:
        if "任务分解器" in system_prompt:
            self.compound_decompose_calls += 1
            payload = json.loads(human_prompt)
            query = str(payload.get("current_query", "") or "").strip()

            def subtask(
                *,
                clean_query: str,
                relation: str,
                interaction_mode: str = "research",
                targets: list[str] | None = None,
                requested_fields: list[str] | None = None,
                required_modalities: list[str] | None = None,
                answer_shape: str = "narrative",
                precision_requirement: str = "high",
                continuation_mode: str = "fresh",
            ) -> dict[str, object]:
                return {
                    "clean_query": clean_query,
                    "interaction_mode": interaction_mode,
                    "relation": relation,
                    "continuation_mode": continuation_mode,
                    "targets": targets or [],
                    "requested_fields": requested_fields or [],
                    "required_modalities": required_modalities or [],
                    "answer_shape": answer_shape,
                    "precision_requirement": precision_requirement,
                    "notes": [],
                }

            if "知识库中有多少论文" in query and "最值得一读" in query:
                return {
                    "is_compound": True,
                    "reason": "count plus recommendation",
                    "subtasks": [
                        subtask(
                            clean_query="知识库中有多少论文",
                            relation="library_status",
                            interaction_mode="conversation",
                            requested_fields=[],
                            required_modalities=[],
                            precision_requirement="exact",
                        ),
                        subtask(
                            clean_query="最值得一读的是哪篇",
                            relation="library_recommendation",
                            interaction_mode="conversation",
                            requested_fields=[],
                            required_modalities=[],
                            precision_requirement="normal",
                            continuation_mode="followup",
                        ),
                    ],
                }
            if "论文库里有哪些文章" in query and "值得" in query:
                return {
                    "is_compound": True,
                    "reason": "library list plus recommendation",
                    "subtasks": [
                        subtask(
                            clean_query="你的论文库里有哪些文章啊",
                            relation="library_status",
                            interaction_mode="research",
                            targets=["论文库"],
                            requested_fields=["answer"],
                            required_modalities=["page_text"],
                            precision_requirement="exact",
                        ),
                        subtask(
                            clean_query="哪个值得一看",
                            relation="library_recommendation",
                            interaction_mode="research",
                            targets=["论文库"],
                            requested_fields=["recommendation"],
                            required_modalities=["paper_card"],
                            precision_requirement="normal",
                            continuation_mode="followup",
                        ),
                    ],
                }
            if "DPO" in query and "PPO" in query and "公式" in query:
                subtasks = [
                    subtask(
                        clean_query="DPO 的公式是什么？",
                        relation="formula_lookup",
                        targets=["DPO"],
                        requested_fields=["formula", "variable_explanation"],
                        required_modalities=["page_text", "table"],
                        answer_shape="bullets",
                        precision_requirement="exact",
                    ),
                    subtask(
                        clean_query="PPO 的公式是什么？",
                        relation="formula_lookup",
                        targets=["PPO"],
                        requested_fields=["formula", "variable_explanation"],
                        required_modalities=["page_text", "table"],
                        answer_shape="bullets",
                        precision_requirement="exact",
                        continuation_mode="followup",
                    ),
                ]
                if "比较" in query or "对比" in query:
                    subtasks.append(
                        subtask(
                            clean_query="比较 DPO 和 PPO 的公式与优化思路。",
                            relation="comparison_synthesis",
                            interaction_mode="conversation",
                            targets=["DPO", "PPO"],
                            requested_fields=["comparison"],
                            required_modalities=[],
                            answer_shape="table",
                            precision_requirement="high",
                            continuation_mode="followup",
                        )
                    )
                return {"is_compound": True, "reason": "parallel formula subtasks", "subtasks": subtasks}
            if "PPO" in query and ("区别" in query or "比较" in query or "对比" in query):
                context = dict(payload.get("conversation_context", {}) or {})
                memory = dict(context.get("working_memory", {}) or {})
                bindings = dict(memory.get("target_bindings", {}) or {})
                previous_targets = [
                    str(item.get("target", "") or "").strip()
                    for item in bindings.values()
                    if isinstance(item, dict) and str(item.get("target", "") or "").strip()
                ]
                comparison_targets = list(dict.fromkeys([*(previous_targets or ["DPO"]), "PPO"]))
                return {
                    "is_compound": True,
                    "reason": "new formula plus comparison with previous context",
                    "subtasks": [
                        subtask(
                            clean_query="PPO 的公式是什么？",
                            relation="formula_lookup",
                            targets=["PPO"],
                            requested_fields=["formula", "variable_explanation"],
                            required_modalities=["page_text", "table"],
                            answer_shape="bullets",
                            precision_requirement="exact",
                            continuation_mode="followup",
                        ),
                        subtask(
                            clean_query="比较 DPO 和 PPO 的公式与优化思路。",
                            relation="comparison_synthesis",
                            interaction_mode="conversation",
                            targets=comparison_targets,
                            requested_fields=["comparison"],
                            required_modalities=[],
                            answer_shape="table",
                            precision_requirement="high",
                            continuation_mode="followup",
                        ),
                    ],
                }
            if "POPI" in query and "核心结论" in query and "实验结果" in query:
                return {
                    "is_compound": True,
                    "reason": "bad split from upstream model; agent should merge same-target fields",
                    "subtasks": [
                        subtask(
                            clean_query="POPI的核心结论是什么？",
                            relation="paper_summary_results",
                            targets=["POPI"],
                            requested_fields=["core_conclusion"],
                            required_modalities=["page_text", "paper_card"],
                            answer_shape="narrative",
                            precision_requirement="high",
                        ),
                        subtask(
                            clean_query="POPI的实验结果如何？",
                            relation="paper_summary_results",
                            targets=["POPI"],
                            requested_fields=["experiment_results"],
                            required_modalities=["page_text", "paper_card", "table", "caption"],
                            answer_shape="narrative",
                            precision_requirement="high",
                        ),
                    ],
                }
            return {"is_compound": False, "reason": "single task", "subtasks": []}
        if "会话记忆追问判别器" in system_prompt:
            payload = json.loads(human_prompt)
            query = str(payload.get("current_query", "") or "").strip()
            if "推荐理由" in query or "为什么选择" in query:
                return {
                    "should_use_memory": True,
                    "reason": "The user asks for the rationale behind the previous library recommendation.",
                    "targets": ["A Survey on LLM-as-a-Judge"],
                    "requested_fields": ["recommendation_reason", "previous_tool_basis"],
                    "answer_shape": "narrative",
                }
            return {"should_use_memory": False, "reason": "not a memory follow-up", "targets": [], "requested_fields": []}
        if "意图路由器" in system_prompt:
            payload = json.loads(human_prompt)
            query = str(payload.get("current_query", "") or payload.get("query", "")).strip()
            active = dict(payload.get("active_research_context", {}) or {})
            last_relation = str(payload.get("last_relation", "") or "")
            query_lower = query.lower()

            def contract(
                *,
                interaction_mode: str,
                relation: str,
                continuation_mode: str = "fresh",
                targets: list[str] | None = None,
                requested_fields: list[str] | None = None,
                required_modalities: list[str] | None = None,
                answer_shape: str = "narrative",
                precision_requirement: str = "high",
                notes: list[str] | None = None,
            ) -> dict[str, object]:
                return {
                    "interaction_mode": interaction_mode,
                    "relation": relation,
                    "continuation_mode": continuation_mode,
                    "targets": targets or [],
                    "requested_fields": requested_fields or [],
                    "required_modalities": required_modalities or [],
                    "answer_shape": answer_shape,
                    "precision_requirement": precision_requirement,
                    "notes": notes or [],
                }

            if query == "你好":
                return contract(interaction_mode="conversation", relation="greeting", precision_requirement="normal")
            if query == "你有什么功能":
                return contract(interaction_mode="conversation", relation="capability", precision_requirement="normal")
            if query == "你是谁":
                return contract(interaction_mode="conversation", relation="self_identity", precision_requirement="normal")
            if query == "何意味":
                return contract(interaction_mode="conversation", relation="clarify_user_intent", precision_requirement="normal")
            if "为什么选择" in query or "为什么推荐这篇" in query:
                return contract(
                    interaction_mode="conversation",
                    relation="memory_followup",
                    continuation_mode="followup",
                    targets=["A Survey on LLM-as-a-Judge"],
                    requested_fields=["recommendation_reason", "previous_tool_basis"],
                    required_modalities=[],
                    answer_shape="narrative",
                    precision_requirement="normal",
                    notes=["contextual_memory_answer"],
                )
            if "再推荐" in query or "换一篇" in query or "还有别的" in query:
                return contract(
                    interaction_mode="conversation",
                    relation="library_recommendation",
                    continuation_mode="followup",
                    targets=[],
                    requested_fields=[],
                    required_modalities=[],
                    answer_shape="bullets",
                    precision_requirement="normal",
                    notes=["avoid_recent_recommendations"],
                )
            if query == "好，那他具体说了啥":
                return contract(
                    interaction_mode="conversation",
                    relation="memory_followup",
                    continuation_mode="followup",
                    targets=["A Survey on LLM-as-a-Judge"],
                    requested_fields=["paper_content"],
                    required_modalities=[],
                    answer_shape="narrative",
                    precision_requirement="normal",
                    notes=["needs_contextual_refine"],
                )
            if query == "最早不是在这里吧":
                return contract(interaction_mode="conversation", relation="clarify_user_intent", precision_requirement="normal")
            if query == "不对吧":
                repair_relation = str(active.get("relation") or last_relation or "general_question")
                return contract(
                    interaction_mode="research",
                    relation=repair_relation,
                    continuation_mode="followup",
                    targets=list(active.get("targets", []) or []),
                    requested_fields=["entity_type", "supporting_paper"] if repair_relation == "entity_definition" else ["answer"],
                    required_modalities=["page_text", "paper_card"],
                    notes=["repair_previous_answer"],
                )
            if "另一个PBA" in query or "不是这个" in query:
                return contract(
                    interaction_mode="research",
                    relation="entity_definition",
                    continuation_mode="followup",
                    targets=["PBA"],
                    requested_fields=["definition", "mechanism"],
                    required_modalities=["page_text", "paper_card"],
                    notes=["repair_previous_answer", "exclude_previous_focus"],
                )
            if query == "PPO呢" and active.get("relation") == "formula_lookup":
                return contract(
                    interaction_mode="research",
                    relation="formula_lookup",
                    continuation_mode="followup",
                    targets=["PPO"],
                    requested_fields=["formula", "variable_explanation"],
                    required_modalities=["page_text", "table"],
                    answer_shape="bullets",
                    precision_requirement="exact",
                    notes=["inherit_previous_task"],
                )
            if query == "变量解释呢" and active.get("relation") == "formula_lookup":
                return contract(
                    interaction_mode="research",
                    relation="formula_lookup",
                    continuation_mode="followup",
                    targets=list(active.get("targets", []) or ["DPO"]),
                    requested_fields=["formula", "variable_explanation"],
                    required_modalities=["page_text", "table"],
                    answer_shape="bullets",
                    precision_requirement="exact",
                    notes=["inherit_previous_task"],
                )
            if "agent拓扑" in query or "拓扑" in query or "topology" in query_lower:
                wants_recommendation = (
                    "哪种最好" in query
                    or "比较好" in query
                    or "最应该" in query
                    or "应该使用" in query
                    or "怎么组织" in query
                    or "如何组织" in query
                    or "langgraph" in query_lower
                )
                relation = "topology_recommendation" if wants_recommendation else "topology_discovery"
                continuation_mode = "followup" if relation == "topology_recommendation" else "fresh"
                return contract(
                    interaction_mode="research",
                    relation=relation,
                    continuation_mode=continuation_mode,
                    targets=["agent topology"],
                    requested_fields=["best_topology", "langgraph_recommendation"] if relation == "topology_recommendation" else ["relevant_papers", "topology_types"],
                    required_modalities=["page_text"],
                    answer_shape="bullets",
                )
            if ("pdf" in query_lower and ("智能体" in query or "agent" in query_lower) and ("解析" in query or "问答" in query or "通信" in query or "框架" in query)):
                return contract(
                    interaction_mode="research",
                    relation="topology_recommendation",
                    continuation_mode="followup" if active.get("relation") else "fresh",
                    targets=["agent topology"],
                    requested_fields=["best_topology", "langgraph_recommendation"],
                    required_modalities=["page_text"],
                    answer_shape="bullets",
                    notes=["local_protected_pdf_agent_topology_design"],
                )
            if "看哪些论文" in query or "推荐一些论文" in query or "如何入门" in query:
                return contract(
                    interaction_mode="research",
                    relation="paper_recommendation",
                    continuation_mode="context_switch" if active.get("relation") else "fresh",
                    targets=["偏好学习"] if "偏好学习" in query else [],
                    requested_fields=["recommended_papers"],
                    required_modalities=["paper_card", "page_text"],
                    answer_shape="bullets",
                )
            if "核心结论" in query and "实验结果" in query:
                target = "POPI" if "POPI" in query else ""
                return contract(
                    interaction_mode="research",
                    relation="paper_summary_results",
                    targets=[target] if target else [],
                    requested_fields=["answer"],
                    required_modalities=["page_text", "paper_card"],
                    answer_shape="narrative",
                )
            if (
                "Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals" in query
                and "后续" in query
            ):
                return contract(
                    interaction_mode="research",
                    relation="followup_research",
                    continuation_mode="fresh",
                    targets=["Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals"],
                    requested_fields=["followup_papers", "relationship", "evidence"],
                    required_modalities=["paper_card", "page_text"],
                    answer_shape="bullets",
                    notes=["upstream_wrong_seed_direction"],
                )
            if "Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals" in query:
                return contract(
                    interaction_mode="research",
                    relation="paper_summary_results",
                    continuation_mode="followup" if active.get("relation") else "fresh",
                    targets=["Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals"],
                    requested_fields=["summary", "relation_to_previous_context"],
                    required_modalities=["page_text", "paper_card"],
                    answer_shape="narrative",
                    notes=["needs_history_context"],
                )
            if "figure1" in query_lower or "图1" in query:
                return contract(
                    interaction_mode="research",
                    relation="figure_question",
                    targets=["DeepSeek-R1"],
                    requested_fields=["figure_conclusion"],
                    required_modalities=["figure", "caption", "page_text"],
                    answer_shape="bullets",
                )
            if "表现如何" in query or "指标" in query:
                return contract(
                    interaction_mode="research",
                    relation="metric_value_lookup",
                    targets=["AlignX"] if "AlignX" in query else [],
                    requested_fields=["answer"],
                    required_modalities=["table", "caption", "page_text"],
                    answer_shape="narrative",
                    precision_requirement="exact",
                )
            if query == "PBA是什么":
                return contract(
                    interaction_mode="research",
                    relation="entity_definition",
                    targets=["PBA"],
                    requested_fields=["definition", "mechanism"],
                    required_modalities=["page_text", "paper_card"],
                    precision_requirement="high",
                )
            if "公式" in query or "objective" in query_lower or "loss" in query_lower:
                target = "DPO" if "DPO" in query else ""
                return contract(
                    interaction_mode="research",
                    relation="formula_lookup",
                    targets=[target] if target else [],
                    requested_fields=["formula", "variable_explanation"],
                    required_modalities=["page_text", "table"],
                    answer_shape="bullets",
                    precision_requirement="exact",
                )
            if query.startswith("什么是"):
                target = query.replace("什么是", "").strip()
                relation = "entity_definition" if target == "AlignX" else "concept_definition"
                fields = ["entity_type", "supporting_paper"] if relation == "entity_definition" else ["definition", "supporting_evidence"]
                return contract(
                    interaction_mode="research",
                    relation=relation,
                    targets=[target] if target else [],
                    requested_fields=fields,
                    required_modalities=["page_text", "paper_card"],
                )
            if query == "AlignX是什么":
                return contract(
                    interaction_mode="research",
                    relation="entity_definition",
                    targets=["AlignX"],
                    requested_fields=["entity_type", "supporting_paper"],
                    required_modalities=["page_text", "paper_card"],
                )
            return {}
        if "论文概念解释器" in system_prompt:
            self.concept_definition_calls += 1
            payload = json.loads(human_prompt)
            target = str(payload.get("target", "") or "").strip()
            if target == "PPO":
                return {
                    "expansion": "Proximal Policy Optimization",
                    "category": "optimization method",
                    "definition": "PPO 即 Proximal Policy Optimization，是一种常用于 RLHF 和 LLM 对齐的 on-policy 强化学习算法，用来基于奖励信号稳定更新策略。",
                    "supporting_doc_ids": ["block-ppo-openagi", "block-ppo-survey"],
                    "confidence": 0.91,
                }
            if target == "RAG":
                return {
                    "expansion": "Retrieval-Augmented Generation",
                    "category": "framework",
                    "definition": "RAG 即 Retrieval-Augmented Generation，是一种先检索相关信息、再结合生成模型组织回答的框架。",
                    "supporting_doc_ids": [],
                    "confidence": 0.83,
                }
            return {}
        if "研究追问合同修复器" in system_prompt:
            self.followup_refine_calls += 1
            payload = json.loads(human_prompt)
            query = str(payload.get("current_query", "") or "").strip()
            active = dict(payload.get("active_research_context", {}) or {})
            active_targets = [str(item).strip() for item in list(active.get("targets", []) or []) if str(item).strip()]
            if query == "最早不是在这里吧" and active_targets:
                return {
                    "relation": "origin_lookup",
                    "continuation_mode": "followup",
                    "targets": active_targets,
                    "requested_fields": ["paper_title", "year", "supporting_paper"],
                    "required_modalities": ["page_text", "paper_card"],
                    "answer_shape": "narrative",
                    "precision_requirement": "high",
                    "notes": ["challenge_previous_answer"],
                    "rewritten_query": f"{active_targets[0]} 最早是在哪篇论文里提出的？",
                }
            if query == "具体是什么样的呢" and active_targets:
                return {
                    "relation": "entity_definition",
                    "continuation_mode": "followup",
                    "targets": active_targets,
                    "requested_fields": ["mechanism", "workflow", "objective", "reward_signal"],
                    "required_modalities": ["page_text", "table"],
                    "answer_shape": "bullets",
                    "precision_requirement": "high",
                    "notes": ["followup_detail"],
                    "rewritten_query": f"{active_targets[0]} 的具体机制、工作流程和奖励/优化目标是什么？",
                }
            if query == "好，那他具体说了啥":
                target = active_targets[0] if active_targets else "A Survey on LLM-as-a-Judge"
                return {
                    "relation": "paper_summary_results",
                    "continuation_mode": "followup",
                    "targets": [target],
                    "requested_fields": ["summary", "key_findings", "method", "experiments"],
                    "required_modalities": ["page_text", "paper_card", "table", "caption"],
                    "answer_shape": "bullets",
                    "precision_requirement": "high",
                    "notes": ["resolved_from_conversation_memory", "requires_paper_reading"],
                    "rewritten_query": f"{target} 具体说了什么？核心结论、方法和实验结果是什么？",
                }
            return {}
        if "库内论文推荐重排器" in system_prompt:
            payload = json.loads(human_prompt)
            candidates = [item for item in list(payload.get("candidate_papers", []) or []) if isinstance(item, dict)]
            recent = {str(title) for title in list(payload.get("recently_recommended_titles", []) or [])}
            if recent:
                selected = [item for item in candidates if str(item.get("title", "")) not in recent][:5]
                return {
                    "criteria_note": "这次我会避开刚刚已经推荐过的论文，换几个不同方向。",
                    "recommendations": [
                        {"title": item.get("title", ""), "reason": f"换一个入口：{item.get('reason', '')}"}
                        for item in selected
                    ],
                }
            preferred_titles = [
                "A Survey on LLM-as-a-Judge",
                "Attention Is All You Need",
                "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
            ]
            selected = []
            for title in preferred_titles:
                selected.extend([item for item in candidates if item.get("title") == title])
            selected.extend([item for item in candidates if item not in selected])
            return {
                "criteria_note": "我会按当前问题、主题覆盖和论文类型综合挑选，而不是固定某一篇。",
                "recommendations": [
                    {"title": item.get("title", ""), "reason": str(item.get("reason", ""))}
                    for item in selected[:5]
                ],
            }
        if "论文实体 grounding 裁判器" in system_prompt:
            self.entity_grounding_calls += 1
            payload = json.loads(human_prompt)
            target = str(payload.get("target", "") or "").strip()
            candidates = list(payload.get("candidates", []) or [])
            fallback_match: dict[str, object] | None = None
            for item in candidates:
                paper_id = str(item.get("paper_id", "") or "").strip()
                snippets = " ".join(str(block.get("snippet", "") or "") for block in list(item.get("evidence", []) or []))
                evidence_doc_ids = [str(block.get("doc_id", "") or "").strip() for block in list(item.get("evidence", []) or [])]
                if target == "AlignX" and ("alignx is a" in snippets.lower() or "introduces alignx" in snippets.lower()):
                    return {
                        "paper_id": paper_id,
                        "evidence_doc_ids": [doc_id for doc_id in evidence_doc_ids if doc_id],
                        "relation_to_target": "direct_definition",
                        "confidence": 0.92,
                        "reason": "This candidate directly defines AlignX as a dataset and benchmark.",
                    }
                if target == "AlignX" and ("dataset" in snippets.lower() or "benchmark" in snippets.lower()) and fallback_match is None:
                    fallback_match = {
                        "paper_id": paper_id,
                        "evidence_doc_ids": [doc_id for doc_id in evidence_doc_ids if doc_id],
                        "relation_to_target": "direct_definition",
                        "confidence": 0.72,
                        "reason": "This candidate contains target-specific dataset evidence.",
                    }
            if fallback_match is not None:
                return fallback_match
            return {}
        if "论文实体类型判别器" in system_prompt:
            self.entity_type_calls += 1
            payload = json.loads(human_prompt)
            target = str(payload.get("target", "") or "").strip()
            evidence = " ".join(str(item) for item in list(payload.get("evidence", []) or [])).lower()
            if target == "AlignX" or "dataset" in evidence or "benchmark" in evidence:
                return {"entity_type": "偏好数据集", "confidence": 0.9, "rationale": "The evidence directly describes a dataset."}
            if target == "GRPO" or "ppo" in evidence or "advantage" in evidence:
                return {"entity_type": "强化学习算法", "confidence": 0.92, "rationale": "The evidence describes an optimization algorithm."}
            return {}
        if "论文关系验证器" in system_prompt:
            self.relationship_verifier_calls += 1
            payload = json.loads(human_prompt)
            evidence = [item for item in list(payload.get("relationship_evidence", []) or []) if isinstance(item, dict)]
            candidate_text = " ".join(
                str(item.get("snippet", ""))
                for item in evidence
                if str(item.get("role", "")) == "candidate"
            ).lower()
            evidence_ids = [str(item.get("doc_id", "")) for item in evidence if str(item.get("doc_id", ""))]
            if "we evaluate on alignx" in candidate_text or "uses alignx" in candidate_text:
                return {
                    "classification": "direct_use_or_evaluation",
                    "strict_followup": True,
                    "relation_type": "直接使用/评测 AlignX",
                    "relationship_strength": "direct",
                    "reason": "候选论文证据明确说明使用或评测 AlignX。",
                    "confidence": 0.86,
                    "evidence_ids": evidence_ids[:4],
                }
            return {
                "classification": "related_continuation",
                "strict_followup": False,
                "relation_type": "强相关延续候选",
                "relationship_strength": "strong_related",
                "reason": "候选和种子都围绕 personalized preference / preference inference，但候选证据未明确写出使用、继承、引用或评测 AlignX。",
                "confidence": 0.66,
                "evidence_ids": evidence_ids[:4],
            }
        if "topology 证据分析器" in system_prompt:
            return {
                "overall_best": "",
                "engineering_best": "",
                "rationale": "需要结合任务和调度成本选择。",
                "summary": "当前证据显示需要结合具体任务和工程约束来选择合适的 topology。",
            }
        return fallback

    def invoke_multimodal_json(self, *, system_prompt: str, human_content: list[dict[str, object]], fallback: object) -> object:
        return fallback


class CaptureComposerClients(StubModelClients):
    def __init__(self) -> None:
        super().__init__()
        self.last_system_prompt = ""

    def invoke_text(self, *, system_prompt: str, human_prompt: str, fallback: str = "") -> str:
        self.last_system_prompt = system_prompt
        return "## 回答\n\n按论文分节回答。"


def _write_jsonl(path: Path, docs: list[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps({"page_content": doc.page_content, "metadata": doc.metadata}, ensure_ascii=False) + "\n")


def _build_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        paper_store_path=tmp_path / "data" / "v4_papers.jsonl",
        block_store_path=tmp_path / "data" / "v4_blocks.jsonl",
        ingestion_state_path=tmp_path / "data" / "v4_ingestion_state.json",
        eval_cases_path=tmp_path / "evals" / "cases_test_md.yaml",
        openai_api_key="",
    )


def _build_agent(tmp_path: Path) -> tuple[ResearchAssistantAgentV4, DualIndexRetriever]:
    settings = _build_settings(tmp_path)
    paper_docs = [
        Document(
            page_content="title: Attention Is All You Need\naliases: AIAYN | Transformer\nabstract_or_summary: Transformer is introduced in this paper.",
            metadata={
                "doc_id": "paper::AIAYN",
                "paper_id": "AIAYN",
                "title": "Attention Is All You Need",
                "authors": "Vaswani et al.",
                "year": "2017",
                "tags": "transformer||attention",
                "file_path": "/tmp/aiayn.pdf",
                "aliases": "Attention Is All You Need||AIAYN||Transformer",
                "generated_summary": "Transformer is introduced in this paper.",
                "abstract_note": "",
            },
        ),
        Document(
            page_content="title: Scaling Large Language Model-based Multi-Agent Collaboration\naliases: topology\nabstract_or_summary: This paper studies chain, tree, mesh, DAG and irregular multi-agent topology.",
            metadata={
                "doc_id": "paper::TOPO",
                "paper_id": "TOPO",
                "title": "Scaling Large Language Model-based Multi-Agent Collaboration",
                "authors": "Zhang et al.",
                "year": "2025",
                "tags": "agent||topology",
                "file_path": "/tmp/topology.pdf",
                "aliases": "Scaling Large Language Model-based Multi-Agent Collaboration||topology",
                "generated_summary": "This paper studies chain, tree, mesh, DAG and irregular multi-agent topology.",
                "abstract_note": "",
            },
        ),
        Document(
            page_content=(
                "title: A Survey on LLM-as-a-Judge\n"
                "aliases: LLM-as-a-Judge survey\n"
                "abstract_or_summary: This survey reviews LLM-as-a-Judge evaluation, covering reliability, bias, consistency, benchmarks, and practical evaluation protocols."
            ),
            metadata={
                "doc_id": "paper::JUDGE",
                "paper_id": "JUDGE",
                "title": "A Survey on LLM-as-a-Judge",
                "authors": "Chen et al.",
                "year": "2025",
                "tags": "survey||evaluation||llm-as-a-judge",
                "file_path": "/tmp/llm-as-a-judge.pdf",
                "aliases": "A Survey on LLM-as-a-Judge||LLM-as-a-Judge survey",
                "generated_summary": "This survey reviews LLM-as-a-Judge evaluation, covering reliability, bias, consistency, benchmarks, and practical evaluation protocols.",
                "abstract_note": "",
            },
        ),
        Document(
            page_content=(
                "title: Direct Preference Optimization: Your Language Model is Secretly a Reward Model\n"
                "aliases: DPO\n"
                "abstract_or_summary: Existing RLHF methods often rely on proximal policy optimization (PPO), "
                "while this paper proposes DPO as a simpler alternative."
            ),
            metadata={
                "doc_id": "paper::DPO",
                "paper_id": "DPO",
                "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
                "authors": "Rafailov et al.",
                "year": "2024",
                "tags": "alignment||preference",
                "file_path": "/tmp/dpo.pdf",
                "aliases": "Direct Preference Optimization||DPO",
                "generated_summary": "Existing RLHF methods often rely on proximal policy optimization (PPO), while this paper proposes DPO as a simpler alternative.",
                "abstract_note": "",
            },
        ),
        Document(
            page_content=(
                "title: From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment\n"
                "aliases: AlignX\n"
                "abstract_or_summary: This paper introduces AlignX, a large-scale dataset and benchmark with more than 1.3 million personalized preference examples."
            ),
            metadata={
                "doc_id": "paper::ALIGNX",
                "paper_id": "ALIGNX",
                "title": "From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
                "authors": "Li et al.",
                "year": "2025",
                "tags": "alignx||personalization",
                "file_path": "/tmp/alignx.pdf",
                "aliases": "From 1,000,000 Users to Every User||AlignX",
                "generated_summary": "This paper introduces AlignX, a large-scale dataset and benchmark with more than 1.3 million personalized preference examples.",
                "abstract_note": "",
            },
        ),
        Document(
            page_content=(
                "title: Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals\n"
                "aliases: AlignXplore\n"
                "abstract_or_summary: This paper proposes AlignXplore, a model for preference inference from behavioral signals."
            ),
            metadata={
                "doc_id": "paper::ALIGNXPLORE",
                "paper_id": "ALIGNXPLORE",
                "title": "Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals",
                "authors": "Wu et al.",
                "year": "2025",
                "tags": "alignxplore||personalization",
                "file_path": "/tmp/alignxplore.pdf",
                "aliases": "Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals||AlignXplore",
                "generated_summary": "This paper proposes AlignXplore, a model for preference inference from behavioral signals.",
                "abstract_note": "",
            },
        ),
        Document(
            page_content=(
                "title: CURP: Codebook-based Continuous User Representation for Personalized Generation with LLMs\n"
                "aliases: CURP\n"
                "abstract_or_summary: CURP uses Prototype Codebook Construction and Prototype Behavior Aligning for personalized generation."
            ),
            metadata={
                "doc_id": "paper::CURP",
                "paper_id": "CURP",
                "title": "CURP: Codebook-based Continuous User Representation for Personalized Generation with LLMs",
                "authors": "Wang et al.",
                "year": "2026",
                "tags": "personalization",
                "file_path": "/tmp/curp.pdf",
                "aliases": "CURP",
                "generated_summary": "CURP uses Prototype Codebook Construction and Prototype Behavior Aligning for personalized generation.",
                "abstract_note": "",
            },
        ),
    ]
    block_docs = [
        Document(
            page_content="Table 4 reports ALIGNXPERT_PBA win rate and accuracy against p-soups and llama baselines.",
            metadata={
                "doc_id": "block-alignx-table",
                "paper_id": "ALIGNX",
                "title": "From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
                "authors": "Li et al.",
                "year": "2025",
                "tags": "alignx||personalization",
                "file_path": "/tmp/alignx.pdf",
                "page": 7,
                "block_type": "table",
                "caption": "Table 4",
                "bbox": "",
                "formula_hint": 0,
            },
        ),
        Document(
            page_content="AlignX is a large-scale dataset and benchmark containing over 1.3 million personalized preference examples for user-level alignment.",
            metadata={
                "doc_id": "block-alignx-definition",
                "paper_id": "ALIGNX",
                "title": "From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
                "authors": "Li et al.",
                "year": "2025",
                "tags": "alignx||personalization",
                "file_path": "/tmp/alignx.pdf",
                "page": 1,
                "block_type": "page_text",
                "caption": "",
                "bbox": "",
                "formula_hint": 0,
            },
        ),
        Document(
            page_content=(
                "Preference-bridged alignment (PBA) introduces a latent preference direction variable as an explicit proxy of user personas "
                "for preference-guided generation in AlignX."
            ),
            metadata={
                "doc_id": "block-alignx-pba-definition",
                "paper_id": "ALIGNX",
                "title": "From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
                "authors": "Li et al.",
                "year": "2025",
                "tags": "alignx||personalization",
                "file_path": "/tmp/alignx.pdf",
                "page": 6,
                "block_type": "page_text",
                "caption": "",
                "bbox": "",
                "formula_hint": 1,
            },
        ),
        Document(
            page_content=(
                "The CURP training procedure has two stages: Prototype Codebook Construction (PCC) and Prototype Behavior Aligning (PBA), "
                "where PBA maps user representations into the frozen LLM space."
            ),
            metadata={
                "doc_id": "block-curp-pba-definition",
                "paper_id": "CURP",
                "title": "CURP: Codebook-based Continuous User Representation for Personalized Generation with LLMs",
                "authors": "Wang et al.",
                "year": "2026",
                "tags": "personalization",
                "file_path": "/tmp/curp.pdf",
                "page": 3,
                "block_type": "page_text",
                "caption": "",
                "bbox": "",
                "formula_hint": 0,
            },
        ),
        Document(
            page_content="AlignXplore is a preference inference model that builds explicit reasoning chains from behavioral signals.",
            metadata={
                "doc_id": "block-alignxplore-definition",
                "paper_id": "ALIGNXPLORE",
                "title": "Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals",
                "authors": "Wu et al.",
                "year": "2025",
                "tags": "alignxplore||personalization",
                "file_path": "/tmp/alignxplore.pdf",
                "page": 1,
                "block_type": "page_text",
                "caption": "",
                "bbox": "",
                "formula_hint": 0,
            },
        ),
        Document(
            page_content="Figure 1 | Benchmark performance of DeepSeek-R1 on AIME, Codeforces, GPQA, MATH-500, MMLU and SWE-bench.",
            metadata={
                "doc_id": "block-deepseek-figure",
                "paper_id": "DEEPSEEK",
                "title": "DeepSeek-R1",
                "authors": "DeepSeek",
                "year": "2025",
                "tags": "deepseek||reasoning",
                "file_path": "/tmp/deepseek.pdf",
                "page": 3,
                "block_type": "caption",
                "caption": "Figure 1",
                "bbox": "",
                "formula_hint": 0,
            },
        ),
        Document(
            page_content=(
                "At its core, RLHF deploys reinforcement learning algorithms, notably "
                "Proximal Policy Optimization (PPO), to tailor LLMs to feedback via a reward model."
            ),
            metadata={
                "doc_id": "block-ppo-openagi",
                "paper_id": "OPENAGI",
                "title": "OpenAGI: When LLM Meets Domain Experts",
                "authors": "Ge et al.",
                "year": "2023",
                "tags": "agent||rlhf",
                "file_path": "/tmp/openagi.pdf",
                "page": 4,
                "block_type": "page_text",
                "caption": "",
                "bbox": "",
                "formula_hint": 1,
            },
        ),
        Document(
            page_content=(
                "RL for LLMs leverages on-policy methods such as proximal policy optimization (PPO) "
                "and GRPO to improve alignment and instruction following."
            ),
            metadata={
                "doc_id": "block-ppo-survey",
                "paper_id": "SURVEY",
                "title": "The Landscape of Agentic Reinforcement Learning for LLMs: A Survey",
                "authors": "Zhang et al.",
                "year": "2025",
                "tags": "survey||rl",
                "file_path": "/tmp/agentic-rl-survey.pdf",
                "page": 4,
                "block_type": "page_text",
                "caption": "",
                "bbox": "",
                "formula_hint": 1,
            },
        ),
        Document(
            page_content="The paper compares chain, tree, mesh, DAG and irregular random topologies.",
            metadata={
                "doc_id": "block-topology-page",
                "paper_id": "TOPO",
                "title": "Scaling Large Language Model-based Multi-Agent Collaboration",
                "authors": "Zhang et al.",
                "year": "2025",
                "tags": "agent||topology",
                "file_path": "/tmp/topology.pdf",
                "page": 5,
                "block_type": "page_text",
                "caption": "",
                "bbox": "",
                "formula_hint": 0,
            },
        ),
        Document(
            page_content=(
                "A Survey on LLM-as-a-Judge organizes the field around evaluation reliability, judge bias, "
                "agreement with human preferences, benchmark design, and practical protocols for using LLMs as evaluators."
            ),
            metadata={
                "doc_id": "block-judge-summary",
                "paper_id": "JUDGE",
                "title": "A Survey on LLM-as-a-Judge",
                "authors": "Chen et al.",
                "year": "2025",
                "tags": "survey||evaluation||llm-as-a-judge",
                "file_path": "/tmp/llm-as-a-judge.pdf",
                "page": 1,
                "block_type": "page_text",
                "caption": "",
                "bbox": "",
                "formula_hint": 0,
            },
        ),
        Document(
            page_content="Table 2 compares LLM-as-a-Judge benchmarks by judged task, human agreement, bias control, and evaluation protocol.",
            metadata={
                "doc_id": "block-judge-table",
                "paper_id": "JUDGE",
                "title": "A Survey on LLM-as-a-Judge",
                "authors": "Chen et al.",
                "year": "2025",
                "tags": "survey||evaluation||llm-as-a-judge",
                "file_path": "/tmp/llm-as-a-judge.pdf",
                "page": 6,
                "block_type": "table",
                "caption": "Table 2",
                "bbox": "",
                "formula_hint": 0,
            },
        ),
    ]
    _write_jsonl(settings.paper_store_path, paper_docs)
    _write_jsonl(settings.block_store_path, block_docs)
    retriever = DualIndexRetriever(settings)
    agent = ResearchAssistantAgentV4(
        settings=settings,
        retriever=retriever,
        clients=StubModelClients(),
        sessions=InMemorySessionStore(max_turns=6),
    )
    return agent, retriever


def test_query_contract_routes_conversation_and_capability(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = SessionContext(session_id="demo")

    greeting = agent._extract_query_contract(query="你好", session=session, mode="auto")
    capability = agent._extract_query_contract(query="你有什么功能", session=session, mode="auto")
    identity = agent._extract_query_contract(query="你是谁", session=session, mode="auto")
    library_status = agent._extract_query_contract(query="你一共有多少论文？", session=session, mode="auto")

    assert greeting.interaction_mode == "conversation"
    assert capability.interaction_mode == "conversation"
    assert capability.relation == "capability"
    assert identity.relation == "self_identity"
    assert library_status.interaction_mode == "conversation"
    assert library_status.relation == "library_status"


def test_library_status_counts_total_paper_documents_not_recall_top_k(tmp_path: Path) -> None:
    agent, retriever = _build_agent(tmp_path)
    fixture_total = len(retriever.paper_documents())

    result = agent.chat(query="我是问你一共有多少论文？")
    nodes = [step["node"] for step in result["execution_steps"]]

    assert result["interaction_mode"] == "conversation"
    assert result["query_contract"]["relation"] == "library_status"
    assert f"{fixture_total} 篇论文" in result["answer"]
    assert "top-k" in result["answer"]
    assert "agent_tool:search_papers" not in nodes


def test_library_status_compound_count_and_recommendation_does_not_clarify(tmp_path: Path) -> None:
    agent, retriever = _build_agent(tmp_path)
    fixture_total = len(retriever.paper_documents())

    result = agent.chat(query="知识库中有多少论文?最值得一读的是哪篇?")

    assert result["interaction_mode"] == "conversation"
    assert result["query_contract"]["relation"] == "compound_query"
    assert "subtask:library_status" in result["query_contract"]["notes"]
    assert "subtask:library_recommendation" in result["query_contract"]["notes"]
    assert result["needs_human"] is False
    assert f"{fixture_total} 篇论文" in result["answer"]
    assert "我会优先看这几篇" in result["answer"]


def test_recommendation_rationale_followup_uses_generic_memory_tool(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    events: list[dict[str, object]] = []

    agent.chat(query="知识库中有多少论文?最值得一读的是哪篇?", session_id="recommendation-memory")
    result, _ = agent._run(
        query="为什么选择A Survey on LLM-as-a-Judge",
        session_id="recommendation-memory",
        mode="auto",
        use_web_search=False,
        max_web_results=3,
        event_callback=events.append,
    )
    nodes = [step["node"] for step in result["execution_steps"]]

    assert result["interaction_mode"] == "conversation"
    assert result["query_contract"]["relation"] == "memory_followup"
    assert "contextual_memory_answer" in result["query_contract"]["notes"]
    assert "agent_tool:read_memory" in nodes
    assert "agent_tool:compose" in nodes
    assert any(item["event"] == "tool_call" and item["data"]["tool"] == "read_memory" for item in events)
    assert "推荐理由" in result["answer"]
    assert "接着看" not in result["answer"]
    assert "Qwen2.5-VL" not in result["answer"]


def test_contextual_paper_content_followup_uses_memory_to_route_then_reads_paper(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    events: list[dict[str, object]] = []

    agent.chat(query="知识库中有多少论文?最值得一读的是哪篇?", session_id="recommendation-paper-read")
    agent.chat(query="为什么推荐这篇？", session_id="recommendation-paper-read")
    result, _ = agent._run(
        query="好，那他具体说了啥",
        session_id="recommendation-paper-read",
        mode="auto",
        use_web_search=False,
        max_web_results=3,
        event_callback=events.append,
    )
    nodes = [step["node"] for step in result["execution_steps"]]

    assert result["interaction_mode"] == "research"
    assert result["query_contract"]["relation"] == "paper_summary_results"
    assert result["query_contract"]["targets"] == ["A Survey on LLM-as-a-Judge"]
    assert "resolved_from_conversation_memory" in result["query_contract"]["notes"]
    assert "agent_tool:search_corpus" in nodes
    assert "agent_tool:read_memory" in nodes
    assert result["citations"]


def test_repeated_library_recommendation_avoids_recent_primary(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    first = agent.chat(query="知识库中有多少论文?最值得一读的是哪篇?", session_id="recommendation-diversity")
    second = agent.chat(query="再推荐一篇别的", session_id="recommendation-diversity")

    assert first["query_contract"]["relation"] == "compound_query"
    assert second["query_contract"]["relation"] == "library_recommendation"
    assert "avoid_recent_recommendations" in second["query_contract"]["notes"]
    assert "A Survey on LLM-as-a-Judge" in first["answer"]
    assert "A Survey on LLM-as-a-Judge" not in second["answer"]
    assert "避开刚刚已经推荐过的论文" in second["answer"]


def test_worth_reading_library_query_prioritizes_recommendation_over_inventory(tmp_path: Path) -> None:
    agent, retriever = _build_agent(tmp_path)
    fixture_total = len(retriever.paper_documents())
    events: list[dict[str, object]] = []

    result, _ = agent._run(
        query="你的论文库中有哪些论文值得一看呢",
        session_id="worth-reading-tool-loop",
        mode="auto",
        use_web_search=False,
        max_web_results=3,
        event_callback=events.append,
    )
    nodes = [step["node"] for step in result["execution_steps"]]

    assert result["interaction_mode"] == "conversation"
    assert result["query_contract"]["relation"] == "library_recommendation"
    assert f"{fixture_total} 篇" in result["answer"]
    assert "我会优先看这几篇" in result["answer"]
    assert "当前论文库" not in result["answer"]
    assert "文章预览" not in result["answer"]
    assert "agent_loop" in nodes
    assert "agent_tool:compose" in nodes
    assert any(item["event"] == "tool_call" and item["data"]["tool"] == "compose" for item in events)


def test_greeting_uses_conversation_tool_instead_of_direct_answer(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    events: list[dict[str, object]] = []

    result, _ = agent._run(
        query="你好",
        session_id="greeting-tool-loop",
        mode="auto",
        use_web_search=False,
        max_web_results=3,
        event_callback=events.append,
    )
    nodes = [step["node"] for step in result["execution_steps"]]

    assert result["query_contract"]["relation"] == "greeting"
    assert "agent_loop" in nodes
    assert "agent_tool:compose" in nodes
    assert any(item["event"] == "tool_call" and item["data"]["tool"] == "compose" for item in events)


def test_library_list_plus_recommendation_stays_self_knowledge_and_streams_sections(tmp_path: Path) -> None:
    agent, retriever = _build_agent(tmp_path)
    fixture_total = len(retriever.paper_documents())
    events: list[dict[str, object]] = []

    result, _ = agent._run(
        query="你的论文库里有哪些文章啊，哪个值得一看",
        session_id="library-list-recommend",
        mode="auto",
        use_web_search=False,
        max_web_results=3,
        event_callback=events.append,
    )
    nodes = [step["node"] for step in result["execution_steps"]]
    deltas = [item for item in events if item.get("event") == "answer_delta"]

    assert result["query_contract"]["relation"] == "compound_query"
    assert result["query_contract"]["targets"] == []
    assert f"{fixture_total} 篇论文" in result["answer"]
    assert "文章预览" in result["answer"]
    assert "默认推荐" in result["answer"]
    assert "agent_tool:search_papers" not in nodes
    assert len(deltas) >= 4


def test_citation_count_followup_uses_web_lookup_instead_of_default_recommendation(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    class StubCitationSearch:
        is_configured = True

        def search(
            self,
            *,
            query: str,
            max_results: int = 5,
            topic: str = "general",
            include_domains: list[str] | None = None,
        ) -> list[EvidenceBlock]:
            if "Attention Is All You Need" in query:
                title = "Attention Is All You Need | Semantic Scholar"
                count = "120,000 citations"
                url = "https://www.semanticscholar.org/paper/attention"
            elif "From 1,000,000 Users" in query:
                title = "From 1,000,000 Users to Every User | Semantic Scholar"
                count = "42 citations"
                url = "https://www.semanticscholar.org/paper/alignx"
            else:
                title = "Direct Preference Optimization | Semantic Scholar"
                count = "3,200 citations"
                url = "https://www.semanticscholar.org/paper/dpo"
            return [
                EvidenceBlock(
                    doc_id="web::" + title.split("|", 1)[0].strip().lower().replace(" ", "-"),
                    paper_id="web",
                    title=title,
                    file_path=url,
                    page=0,
                    block_type="web",
                    caption=url,
                    snippet=f"This paper has {count}.",
                    metadata={"source": "tavily", "query": query, "include_domains": include_domains or []},
                )
            ]

    agent.web_search = StubCitationSearch()
    session_id = "citation-followup"
    first = agent.chat(query="知识库中有多少论文?最值得一读的是哪篇?", session_id=session_id)
    assert first["query_contract"]["relation"] == "compound_query"

    events: list[dict[str, object]] = []
    result, emitted = agent._run(
        query="按引用数呢",
        session_id=session_id,
        mode="auto",
        use_web_search=False,
        max_web_results=3,
        event_callback=events.append,
    )
    nodes = [step["node"] for step in result["execution_steps"]]

    assert result["query_contract"]["relation"] == "library_citation_ranking"
    assert "agent_tool:web_search" in nodes
    assert "120,000" in result["answer"]
    assert "我会先读" not in result["answer"]
    assert result["citations"]
    assert any(item["event"] == "answer_delta" for item in emitted)
    assert any(item["event"] == "tool_call" and item["data"]["tool"] == "web_search" for item in events)


def test_citation_count_followup_without_web_refuses_local_heuristic_ranking(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session_id = "citation-no-web"
    agent.chat(query="知识库中有多少论文?最值得一读的是哪篇?", session_id=session_id)

    result = agent.chat(query="按引用数呢", session_id=session_id)

    assert result["query_contract"]["relation"] == "library_citation_ranking"
    assert result["verification_report"]["recommended_action"] == "citation_count_not_found_in_web_snippets"
    assert "不能只靠本地 PDF 摘要推断" in result["answer"]
    assert "我会先读" not in result["answer"]


def test_query_contract_routes_concept_definition_without_whitespace(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = SessionContext(session_id="demo")

    contract = agent._extract_query_contract(query="什么是PPO", session=session, mode="auto")

    assert contract.relation == "concept_definition"
    assert contract.targets == ["PPO"]


def test_paper_summary_results_always_requests_table_evidence(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = SessionContext(session_id="demo")

    contract = agent._extract_query_contract(query="POPI的核心结论是什么，实验结果如何？", session=session, mode="auto")

    assert contract.relation == "paper_summary_results"
    assert "page_text" in contract.required_modalities
    assert "table" in contract.required_modalities
    assert "caption" in contract.required_modalities


def test_same_paper_summary_fields_are_not_rendered_as_compound_steps(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    result = agent.chat(query="POPI的核心结论是什么，实验结果如何？", session_id="popi-single-summary")
    nodes = [step["node"] for step in result["execution_steps"]]

    assert result["query_contract"]["relation"] == "paper_summary_results"
    assert result["interaction_mode"] == "research"
    assert "compound_planner" not in nodes
    assert not any(str(step["node"]).startswith("compound_task:") for step in result["execution_steps"])


def test_research_plan_uses_larger_evidence_budget_for_results_queries(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="POPI的核心结论是什么，实验结果如何？",
        relation="paper_summary_results",
        targets=["POPI"],
    )

    plan = agent._build_research_plan(contract)

    assert plan.paper_limit >= 8
    assert plan.evidence_limit >= 36


def test_followup_contract_inherits_topology_context(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = SessionContext(session_id="demo", last_relation="topology_discovery")
    session.set_active_research(
        relation="topology_discovery",
        targets=["agent topology"],
        titles=[],
        requested_fields=["topology"],
        required_modalities=["page_text"],
        answer_shape="narrative",
        precision_requirement="high",
        clean_query="agent topology",
    )

    contract = agent._extract_query_contract(
        query="这些不同的拓扑结构哪种最好呢？如果我要组织LangGraph节点，你觉得哪种比较好？",
        session=session,
        mode="auto",
    )

    assert contract.relation == "topology_recommendation"
    assert contract.continuation_mode == "followup"
    assert contract.targets == ["agent topology"]


def test_title_anchor_prefers_attention_is_all_you_need(tmp_path: Path) -> None:
    _, retriever = _build_agent(tmp_path)

    anchors = retriever.title_anchor(["Transformer"])

    assert anchors
    assert anchors[0].metadata["title"] == "Attention Is All You Need"


def test_topology_search_prefers_scaling_multi_agent_collaboration(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = SessionContext(session_id="demo")
    contract = agent._extract_query_contract(query="论文中有没有研究agent拓扑结构的", session=session, mode="auto")
    papers = agent.retriever.search_papers(query=contract.clean_query, contract=contract, limit=4)

    assert papers
    assert papers[0].title == "Scaling Large Language Model-based Multi-Agent Collaboration"


def test_pdf_agent_topology_recommendation_answers_directly(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    result = agent.chat(
        query="如果我想要做一个pdf-agent，内部是multi-agent系统，你觉得我最应该使用什么拓扑结构组织呢",
        session_id="pdf-agent-topology",
    )

    assert result["query_contract"]["relation"] == "topology_recommendation"
    assert result["verification_report"]["status"] == "pass"
    assert "证据不足" not in result["answer"]
    assert "DAG" in result["answer"]
    assert "PDF-Agent" in result["answer"] or "PDF-RAG" in result["answer"]
    assert "## 组织建议" in result["answer"]
    assert "The evidence" not in result["answer"]
    assert "impossible to determine" not in result["answer"].lower()


def test_pdf_agent_topology_clarification_reply_routes_to_recommendation(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    result = agent.chat(
        query="需要解析pdf文档内容，希望交互式问答，智能体频繁通信交流问题，论文框架的话你自己找找看",
        session_id="pdf-agent-topology-spec",
    )

    assert result["query_contract"]["relation"] == "topology_recommendation"
    assert result["verification_report"]["status"] == "pass"
    assert "DAG" in result["answer"]
    assert "## 组织建议" in result["answer"]
    assert "Based on the evidence" not in result["answer"]
    assert "The provided evidence" not in result["answer"]


def test_topology_recommendation_composer_suppresses_negative_english_rationale(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="PDF-Agent 应该用什么 multi-agent topology？",
        relation="topology_recommendation",
        targets=["agent topology"],
        requested_fields=["best_topology", "langgraph_recommendation"],
    )
    evidence = [
        EvidenceBlock(
            doc_id="topology-evidence",
            paper_id="TOPO",
            title="Scaling Large Language Model-based Multi-Agent Collaboration",
            file_path="/tmp/topology.pdf",
            page=5,
            block_type="page_text",
            snippet="The paper compares chain, tree, mesh, DAG and irregular random topologies.",
        )
    ]
    claims = [
        Claim(
            claim_type="topology_recommendation",
            entity="agent topology",
            value="The evidence does not address or evaluate the listed topology terms, making it impossible to determine the best topology.",
            structured_data={
                "topology_terms": ["dag", "chain", "tree", "mesh"],
                "rationale": "The provided evidence does not contain any direct analysis, comparison, or evaluation.",
            },
            evidence_ids=["topology-evidence"],
            paper_ids=["TOPO"],
        )
    ]

    answer = agent._compose_topology_recommendation_answer(contract=contract, claims=claims, evidence=evidence)

    assert "DAG" in answer
    assert "The evidence" not in answer
    assert "does not contain" not in answer
    assert "impossible to determine" not in answer.lower()


def test_table_solver_returns_metric_claim(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="AlignX的PBA表现如何？",
        relation="metric_value_lookup",
        targets=["AlignX"],
        required_modalities=["table", "caption", "page_text"],
    )
    papers = [
        CandidatePaper(
            paper_id="ALIGNX",
            title="From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
            year="2025",
        )
    ]
    evidence = [
        EvidenceBlock(
            doc_id="block-alignx-table",
            paper_id="ALIGNX",
            title=papers[0].title,
            file_path="/tmp/alignx.pdf",
            page=7,
            block_type="table",
            caption="Table 4",
            snippet="Table 4 reports ALIGNXPERT_PBA win rate and accuracy against p-soups and llama baselines.",
        )
    ]

    claims = agent._solve_table(contract=contract, papers=papers, evidence=evidence)

    assert claims
    assert claims[0].claim_type == "metric_value"
    assert "win rate" in claims[0].structured_data["metric_lines"][0].lower()


def test_table_solver_uses_vlm_when_page_image_is_available(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    calls: list[list[dict[str, object]]] = []

    def fake_multimodal_json(*, system_prompt: str, human_content: list[dict[str, object]], fallback: object) -> object:
        calls.append(human_content)
        return {
            "claims": [
                {
                    "claim": "表格显示 AlignX 的 PBA win rate 最高。",
                    "metric_lines": ["AlignX PBA win rate is highest"],
                    "confidence": 0.87,
                }
            ],
            "draft_answer": "",
        }

    agent.clients.invoke_multimodal_json = fake_multimodal_json  # type: ignore[method-assign]
    agent._render_page_image_data_url = lambda *, file_path, page: "data:image/png;base64,ZmFrZQ=="  # type: ignore[method-assign]
    contract = QueryContract(
        clean_query="AlignX的PBA表现如何？",
        relation="metric_value_lookup",
        targets=["AlignX"],
        required_modalities=["table", "caption", "page_text"],
    )
    papers = [
        CandidatePaper(
            paper_id="ALIGNX",
            title="From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
            year="2025",
        )
    ]
    evidence = [
        EvidenceBlock(
            doc_id="block-alignx-table-vlm",
            paper_id="ALIGNX",
            title=papers[0].title,
            file_path="/tmp/alignx.pdf",
            page=7,
            block_type="table",
            caption="Table 4",
            snippet="Table 4 reports ALIGNXPERT_PBA win rate and accuracy against baselines.",
        )
    ]

    claims = agent._solve_table(contract=contract, papers=papers, evidence=evidence)

    assert calls
    assert claims
    assert claims[0].structured_data["mode"] == "vlm_table"
    assert claims[0].value == "表格显示 AlignX 的 PBA win rate 最高。"
    assert any(block.get("type") == "image_url" for block in calls[0])


def test_table_solver_grounds_primary_paper_in_metric_evidence(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="AlignX的PBA表现如何？",
        relation="metric_value_lookup",
        targets=["AlignX", "PBA"],
        required_modalities=["table", "caption", "page_text"],
    )
    papers = [
        CandidatePaper(
            paper_id="NOISE",
            title="Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals",
            year="2025",
            score=1.0,
        ),
        CandidatePaper(
            paper_id="ALIGNX",
            title="From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
            year="2025",
            score=0.8,
            metadata={"aliases": "AlignX"},
        ),
    ]
    evidence = [
        EvidenceBlock(
            doc_id="noise-table",
            paper_id="NOISE",
            title=papers[0].title,
            file_path="/tmp/noise.pdf",
            page=4,
            block_type="table",
            caption="Table 2",
            snippet="Table 2 reports preference inference accuracy on unrelated baselines.",
            score=5.0,
        ),
        EvidenceBlock(
            doc_id="alignx-pba-table",
            paper_id="ALIGNX",
            title=papers[1].title,
            file_path="/tmp/alignx.pdf",
            page=8,
            block_type="table",
            caption="Table 4",
            snippet="ALIGNXPERTPBA win rate and accuracy against p-soups and Llama baselines.",
            score=2.0,
        ),
    ]

    claims = agent._solve_table(contract=contract, papers=papers, evidence=evidence)

    assert claims
    assert claims[0].paper_ids[0] == "ALIGNX"
    assert claims[0].evidence_ids[0] == "alignx-pba-table"


def test_figure_solver_fallback_uses_caption_when_vlm_disabled(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="Deepseek R1 的 figure1 展示了什么问题？",
        relation="figure_question",
        targets=["DeepSeek-R1"],
        required_modalities=["figure", "caption", "page_text"],
    )
    papers = [CandidatePaper(paper_id="DEEPSEEK", title="DeepSeek-R1", year="2025")]
    evidence = [
        EvidenceBlock(
            doc_id="block-deepseek-figure",
            paper_id="DEEPSEEK",
            title="DeepSeek-R1",
            file_path="/not-found/deepseek.pdf",
            page=3,
            block_type="caption",
            caption="Figure 1",
            snippet="Figure 1 | Benchmark performance of DeepSeek-R1 on AIME, Codeforces, GPQA, MATH-500, MMLU and SWE-bench.",
        )
    ]

    claims = agent._solve_figure(contract=contract, papers=papers, evidence=evidence)

    assert claims
    assert claims[0].structured_data["mode"] == "caption_fallback"
    assert "benchmark performance" in claims[0].value.lower()


def test_build_figure_contexts_prefers_explicit_figure_page_over_summary_page(tmp_path: Path) -> None:
    evidence = [
        EvidenceBlock(
            doc_id="page-summary-1",
            paper_id="DEEPSEEK",
            title="DeepSeek-R1",
            file_path="/tmp/deepseek.pdf",
            page=4,
            block_type="page_text",
            snippet="Summary of evaluation results. AIME 2024 79.8, MATH-500 97.3, GPQA 71.5, MMLU 90.8.",
        ),
        EvidenceBlock(
            doc_id="page-summary-2",
            paper_id="DEEPSEEK",
            title="DeepSeek-R1",
            file_path="/tmp/deepseek.pdf",
            page=4,
            block_type="page_text",
            snippet="Reasoning tasks summary with Codeforces, GPQA, and MATH-500.",
        ),
        EvidenceBlock(
            doc_id="page-figure-1",
            paper_id="DEEPSEEK",
            title="DeepSeek-R1",
            file_path="/tmp/deepseek.pdf",
            page=1,
            block_type="page_text",
            snippet="Figure 1 | Benchmark performance of DeepSeek-R1 on AIME 2024, Codeforces, GPQA Diamond, MATH-500, MMLU and SWE-bench.",
        ),
    ]

    contexts = build_figure_contexts(evidence)

    assert contexts
    assert contexts[0]["page"] == 1


def test_normalize_contract_targets_drops_structural_figure_reference(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    targets = agent._normalize_contract_targets(
        targets=["Deepseek R1", "figure1"],
        requested_fields=["problem_addressed"],
    )

    assert targets == ["Deepseek R1"]


def test_figure_verifier_rejects_wrong_primary_paper(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="Deepseek R1 的 figure1 展示了什么问题？",
        relation="figure_question",
        targets=["DeepSeek-R1"],
        required_modalities=["figure", "caption", "page_text"],
    )
    plan = ResearchPlan(required_claims=["figure_conclusion"])
    claims = [
        Claim(
            claim_type="figure_conclusion",
            entity="DeepSeek-R1",
            value="图1比较了多个 benchmark。",
            evidence_ids=["block-wrong-paper"],
            paper_ids=["WRONG"],
        )
    ]
    papers = [
        CandidatePaper(
            paper_id="WRONG",
            title="Unrelated Paper",
            year="2025",
            metadata={"aliases": "Some Other Model"},
        )
    ]
    evidence = [
        EvidenceBlock(
            doc_id="block-wrong-paper",
            paper_id="WRONG",
            title="Unrelated Paper",
            file_path="/tmp/unrelated.pdf",
            page=4,
            block_type="page_text",
            snippet="This page happens to mention DeepSeek-R1 but is not the DeepSeek-R1 paper.",
        )
    ]

    verification = agent._verify_claims(
        contract=contract,
        plan=plan,
        claims=claims,
        papers=papers,
        evidence=evidence,
    )

    assert verification.status == "clarify"
    assert verification.recommended_action == "clarify_target"


def test_followup_citations_can_resolve_paper_card_doc_ids(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    citations = agent._citations_from_doc_ids(["paper::TOPO"], evidence=[])

    assert len(citations) == 1
    assert citations[0].title == "Scaling Large Language Model-based Multi-Agent Collaboration"


def test_figure_confidence_accepts_string_labels(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    assert agent._coerce_confidence("high") > agent._coerce_confidence("medium")


def test_concept_definition_answer_uses_definition_evidence(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    result = agent.chat(query="什么是PPO")

    assert result["query_contract"]["relation"] == "concept_definition"
    assert "Proximal Policy Optimization" in result["answer"]
    assert ("强化学习" in result["answer"]) or ("on-policy" in result["answer"])
    assert result["citations"]
    assert agent.clients.concept_definition_calls >= 1


def test_unknown_concept_returns_clarification_instead_of_random_summary(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    result = agent.chat(query="什么是NeverSeenMethod")

    assert result["query_contract"]["relation"] in {"concept_definition", "entity_definition"}
    assert result["needs_human"] is True
    assert "没有稳定定位" in result["answer"]
    assert agent._coerce_confidence("low") < agent._coerce_confidence("medium")


def test_ambiguous_acronym_asks_human_instead_of_guessing(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    result = agent.chat(query="PBA是什么")

    assert result["needs_human"] is True
    assert result["verification_report"]["recommended_action"] == "clarify_ambiguous_entity"
    assert "Preference-bridged alignment" in result["answer"]
    assert "Prototype Behavior Aligning" in result["answer"]


def test_correction_excludes_previous_focus_before_answering(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = agent.sessions.get("pba-correction")
    session.set_active_research(
        relation="entity_definition",
        targets=["PBA"],
        titles=["CURP: Codebook-based Continuous User Representation for Personalized Generation with LLMs"],
        requested_fields=["definition", "mechanism", "role_in_context"],
        required_modalities=["page_text", "paper_card"],
        answer_shape="narrative",
        precision_requirement="high",
        clean_query="PBA是什么",
    )
    session.turns.append(
        SessionTurn(
            query="PBA是什么",
            answer="PBA is Prototype Behavior Aligning.",
            relation="entity_definition",
            targets=["PBA"],
            titles=["CURP: Codebook-based Continuous User Representation for Personalized Generation with LLMs"],
        )
    )
    agent.sessions.upsert(session)

    result = agent.chat(query="我是说另一个PBA，不是这个", session_id="pba-correction")

    assert result["needs_human"] is False
    assert "CURP" not in result["answer"]
    assert result["citations"]
    assert result["citations"][0]["title"] == "From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment"


def test_concept_verifier_retries_before_clarify_when_initial_recall_is_misaligned(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(clean_query="什么是PPO", relation="concept_definition", targets=["PPO"], required_modalities=["page_text"])
    plan = ResearchPlan(required_claims=["definition"])
    papers = [
        CandidatePaper(
            paper_id="OTHER",
            title="Unrelated Alignment Note",
            year="2025",
            score=0.6,
            metadata={"paper_card_text": "This paper discusses alignment at a high level."},
        )
    ]
    evidence = [
        EvidenceBlock(
            doc_id="other-page",
            paper_id="OTHER",
            title="Unrelated Alignment Note",
            file_path="/tmp/other.pdf",
            page=3,
            block_type="page_text",
            snippet="This section gives a broad overview of alignment methods without defining the target acronym.",
            score=0.7,
        )
    ]
    claims = [
        Claim(
            claim_type="concept_definition",
            entity="PPO",
            value="PPO 是一种优化方法。",
            evidence_ids=["other-page"],
            paper_ids=["OTHER"],
        )
    ]

    report = agent._verify_claims(contract=contract, plan=plan, claims=claims, papers=papers, evidence=evidence)

    assert report.status == "retry"
    assert report.recommended_action == "retry_definition"


def test_short_vague_query_routes_to_conversation_clarification(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = SessionContext(session_id="demo")

    contract = agent._extract_query_contract(query="何意味", session=session, mode="auto")

    assert contract.interaction_mode == "conversation"
    assert contract.relation == "clarify_user_intent"


def test_alignx_what_is_routes_to_entity_definition_and_avoids_alignxplore(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    result = agent.chat(query="AlignX是什么")

    assert result["query_contract"]["relation"] == "entity_definition"
    assert "数据集" in result["answer"] or "benchmark" in result["answer"].lower()
    assert "强化学习算法" not in result["answer"]
    assert "### AlignX 技术简介" not in result["answer"]
    assert result["citations"]
    assert result["citations"][0]["title"] == "From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment"


def test_entity_definition_solver_prefers_supporting_paper_over_top_ranked_candidate(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="GRPO是什么技术",
        relation="entity_definition",
        targets=["GRPO"],
        required_modalities=["page_text"],
    )
    papers = [
        CandidatePaper(
            paper_id="CURP",
            title="CURP: Codebook-based Continuous User Representation for Personalized Generation with LLMs",
            year="2026",
            score=1.0,
            doc_ids=["paper::CURP"],
            metadata={"paper_card_text": "CURP is a benchmark for personalized generation."},
        ),
        CandidatePaper(
            paper_id="USERLM",
            title="UserLM-R1: Modeling Human Reasoning in User Language Models with Multi-Reward Reinforcement Learning",
            year="2026",
            score=0.6,
            doc_ids=["paper::USERLM"],
            metadata={"paper_card_text": "UserLM-R1 uses reinforcement learning for reasoning."},
        ),
    ]
    evidence = [
        EvidenceBlock(
            doc_id="curp-page-1",
            paper_id="CURP",
            title=papers[0].title,
            file_path="/tmp/curp.pdf",
            page=1,
            block_type="page_text",
            snippet="CURP is a benchmark for personalized generation.",
        ),
        EvidenceBlock(
            doc_id="userlm-page-5",
            paper_id="USERLM",
            title=papers[1].title,
            file_path="/tmp/userlm.pdf",
            page=5,
            block_type="page_text",
            snippet="During the reinforcement learning stage, we employ the GRPO algorithm with rule-based rewards as guiding signals.",
        ),
    ]

    claims = agent._solve_text(contract=contract, papers=papers, evidence=evidence, session=SessionContext(session_id="grpo-demo"))

    assert claims
    claim = claims[0]
    assert claim.paper_ids == ["USERLM"]
    assert claim.evidence_ids == ["userlm-page-5"]
    assert ("算法" in claim.value) or ("方法" in claim.value)
    assert "数据集" not in claim.value


def test_entity_definition_verifier_retries_when_claim_points_to_wrong_paper(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="GRPO是什么技术",
        relation="entity_definition",
        targets=["GRPO"],
        required_modalities=["page_text"],
    )
    plan = ResearchPlan(required_claims=["entity_type"])
    papers = [
        CandidatePaper(paper_id="CURP", title="CURP", year="2026", score=1.0),
        CandidatePaper(paper_id="USERLM", title="UserLM-R1", year="2026", score=0.6),
    ]
    evidence = [
        EvidenceBlock(
            doc_id="curp-page-1",
            paper_id="CURP",
            title="CURP",
            file_path="/tmp/curp.pdf",
            page=1,
            block_type="page_text",
            snippet="CURP is a benchmark for personalized generation.",
        ),
        EvidenceBlock(
            doc_id="userlm-page-5",
            paper_id="USERLM",
            title="UserLM-R1",
            file_path="/tmp/userlm.pdf",
            page=5,
            block_type="page_text",
            snippet="During the reinforcement learning stage, we employ the GRPO algorithm with rule-based rewards as guiding signals.",
        ),
    ]
    claims = [
        Claim(
            claim_type="entity_definition",
            entity="GRPO",
            value="数据集/benchmark",
            structured_data={"paper_title": "CURP", "description": "GRPO 是一个数据集。"},
            evidence_ids=["curp-page-1"],
            paper_ids=["CURP"],
        )
    ]

    report = agent._verify_claims(contract=contract, plan=plan, claims=claims, papers=papers, evidence=evidence)

    assert report.status == "retry"
    assert report.recommended_action == "retry_entity"


def test_entity_definition_verifier_accepts_algorithm_entity_with_supporting_evidence(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="GRPO是什么技术",
        relation="entity_definition",
        targets=["GRPO"],
        required_modalities=["page_text"],
    )
    plan = ResearchPlan(required_claims=["entity_type"])
    papers = [
        CandidatePaper(paper_id="USERLM", title="UserLM-R1", year="2026", score=0.6),
    ]
    evidence = [
        EvidenceBlock(
            doc_id="userlm-page-5",
            paper_id="USERLM",
            title="UserLM-R1",
            file_path="/tmp/userlm.pdf",
            page=5,
            block_type="page_text",
            snippet="During the reinforcement learning stage, we employ the GRPO algorithm with rule-based rewards as guiding signals.",
        ),
    ]
    claims = [
        Claim(
            claim_type="entity_definition",
            entity="GRPO",
            value="优化算法/训练方法",
            structured_data={"paper_title": "UserLM-R1", "description": "GRPO 是一种强化学习算法。"},
            evidence_ids=["userlm-page-5"],
            paper_ids=["USERLM"],
        )
    ]

    report = agent._verify_claims(contract=contract, plan=plan, claims=claims, papers=papers, evidence=evidence)

    assert report.status == "pass"


def test_followup_entity_contract_refines_vague_detail_query(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = SessionContext(session_id="detail-demo")
    session.set_active_research(
        relation="entity_definition",
        targets=["GRPO"],
        titles=[],
        requested_fields=["definition", "applications", "key_features"],
        required_modalities=["page_text"],
        answer_shape="narrative",
        precision_requirement="high",
        clean_query="GRPO是什么技术？",
    )
    contract = QueryContract(
        clean_query="具体是什么样的呢",
        relation="entity_definition",
        interaction_mode="research",
        targets=["GRPO"],
        requested_fields=["definition", "applications", "key_features"],
        required_modalities=["page_text"],
        continuation_mode="followup",
    )

    refined = agent._refine_followup_contract(contract=contract, session=session)

    assert refined.requested_fields == ["mechanism", "workflow", "objective", "reward_signal"]
    assert "GRPO" in refined.clean_query
    assert refined.answer_shape == "bullets"
    assert refined.precision_requirement == "high"


def test_entity_supporting_paper_prefers_definition_source_over_usage_source(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="GRPO是什么技术",
        relation="entity_definition",
        targets=["GRPO"],
        required_modalities=["page_text"],
    )
    papers = [
        CandidatePaper(
            paper_id="SURVEY",
            title="The Landscape of Agentic Reinforcement Learning for LLMs: A Survey",
            year="2025",
            score=0.4,
            metadata={"paper_card_text": "A survey of RL algorithms for LLMs."},
        ),
        CandidatePaper(
            paper_id="USERLM",
            title="UserLM-R1: Modeling Human Reasoning in User Language Models with Multi-Reward Reinforcement Learning",
            year="2026",
            score=0.6,
            metadata={"paper_card_text": "UserLM-R1 uses GRPO during reinforcement learning."},
        ),
    ]
    evidence = [
        EvidenceBlock(
            doc_id="survey-def",
            paper_id="SURVEY",
            title=papers[0].title,
            file_path="/tmp/survey.pdf",
            page=11,
            block_type="page_text",
            snippet="Group Relative Policy Optimization (GRPO) introduces a lightweight evaluation paradigm and uses relative rewards within a group to compute advantages.",
            metadata={"definition_score": 3.0, "mechanism_score": 2.0},
        ),
        EvidenceBlock(
            doc_id="survey-mech",
            paper_id="SURVEY",
            title=papers[0].title,
            file_path="/tmp/survey.pdf",
            page=12,
            block_type="page_text",
            snippet="GRPO uses group-based relative reward to eliminate value estimates and improve sample efficiency.",
            metadata={"definition_score": 0.0, "mechanism_score": 2.5},
        ),
        EvidenceBlock(
            doc_id="userlm-use",
            paper_id="USERLM",
            title=papers[1].title,
            file_path="/tmp/userlm.pdf",
            page=5,
            block_type="page_text",
            snippet="During the reinforcement learning stage, we employ the GRPO algorithm with rule-based and rubric-based rewards as guiding signals.",
            metadata={"definition_score": 0.0, "mechanism_score": 1.0, "application_score": 0.8},
        ),
    ]

    paper, supporting = agent._select_entity_supporting_paper(contract=contract, papers=papers, evidence=evidence)

    assert paper is not None
    assert paper.paper_id == "SURVEY"
    assert supporting
    assert supporting[0].paper_id == "SURVEY"


def test_claim_focus_titles_follow_claim_papers_not_candidate_rank(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    papers = [
        CandidatePaper(paper_id="CURP", title="CURP", year="2026", score=1.0),
        CandidatePaper(paper_id="USERLM", title="UserLM-R1", year="2026", score=0.6),
    ]
    claims = [
        Claim(
            claim_type="entity_definition",
            entity="GRPO",
            value="优化算法/训练方法",
            evidence_ids=["userlm-page-5"],
            paper_ids=["USERLM"],
        )
    ]

    titles = agent._claim_focus_titles(claims=claims, papers=papers)

    assert titles == ["UserLM-R1"]


def test_entity_answer_composer_uses_grounded_detail_sections_for_followup(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="GRPO具体是什么样的呢",
        relation="entity_definition",
        targets=["GRPO"],
        requested_fields=["mechanism", "workflow", "objective", "reward_signal"],
        required_modalities=["page_text"],
        continuation_mode="followup",
        answer_shape="bullets",
        precision_requirement="high",
    )
    claim = Claim(
        claim_type="entity_definition",
        entity="GRPO",
        value="优化算法/训练方法",
        structured_data={
            "paper_title": "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            "definition_lines": [
                "Group Relative Policy Optimization (GRPO) is a variant of Proximal Policy Optimization (PPO)."
            ],
            "mechanism_lines": [
                "GRPO foregoes the value model, instead estimating the baseline from group scores.",
                "GRPO uses relative rewards within a group to compute advantages.",
            ],
            "application_lines": [
                "GRPO is used to improve mathematical reasoning while reducing training resources."
            ],
        },
        evidence_ids=["grpo-def", "grpo-fig", "grpo-steps"],
        paper_ids=["DEEPMATH"],
    )
    evidence = [
        EvidenceBlock(
            doc_id="grpo-def",
            paper_id="DEEPMATH",
            title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            file_path="/tmp/deepseekmath.pdf",
            page=13,
            block_type="page_text",
            snippet="GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources.",
            score=4.5,
        ),
        EvidenceBlock(
            doc_id="grpo-fig",
            paper_id="DEEPMATH",
            title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            file_path="/tmp/deepseekmath.pdf",
            page=14,
            block_type="page_text",
            snippet="For each question, a group of outputs are sampled, rewards are computed, and the policy model is updated by maximizing the GRPO objective.",
            score=4.2,
        ),
        EvidenceBlock(
            doc_id="grpo-steps",
            paper_id="DEEPMATH",
            title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            file_path="/tmp/deepseekmath.pdf",
            page=20,
            block_type="page_text",
            snippet="GRPO uniquely adjusts its gradient coefficient based on the reward value provided by the reward model.",
            score=3.8,
        ),
    ]

    answer, citations = agent._compose_answer(
        contract=contract,
        claims=[claim],
        evidence=evidence,
        papers=[],
        verification=VerificationReport(status="pass"),
    )

    assert "核心机制" in answer
    assert "组内相对 reward" in answer or "group scores" in answer
    assert "value model / critic" in answer or "value model" in answer
    assert citations


def test_infer_entity_type_prefers_algorithm_signals_over_benchmark_words(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(clean_query="GRPO是什么技术", relation="entity_definition", targets=["GRPO"])
    paper = CandidatePaper(
        paper_id="DEEPMATH",
        title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
        year="2024",
        metadata={"paper_card_text": "GRPO is evaluated on math benchmarks."},
    )
    evidence = [
        EvidenceBlock(
            doc_id="grpo-type",
            paper_id="DEEPMATH",
            title=paper.title,
            file_path="/tmp/deepseekmath.pdf",
            page=13,
            block_type="page_text",
            snippet="GRPO is a variant of PPO that uses relative rewards within a group to compute advantages and removes the need for a value critic.",
        )
    ]

    label = agent._infer_entity_type(contract=contract, papers=[paper], evidence=evidence)

    assert label == "优化算法/训练方法"


def test_infer_entity_type_uses_llm_classifier_and_canonicalizes_label(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(clean_query="AlignX是什么", relation="entity_definition", targets=["AlignX"])
    paper = CandidatePaper(
        paper_id="ALIGNX",
        title="From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
        year="2025",
        metadata={"paper_card_text": "AlignX is introduced for user-level alignment and evaluation."},
    )
    evidence = [
        EvidenceBlock(
            doc_id="alignx-type",
            paper_id="ALIGNX",
            title=paper.title,
            file_path="/tmp/alignx.pdf",
            page=1,
            block_type="page_text",
            snippet="AlignX is a large-scale dataset and benchmark containing personalized preference triplets.",
        )
    ]

    label = agent._infer_entity_type(contract=contract, papers=[paper], evidence=evidence)

    assert label == "数据集/benchmark"
    assert agent.clients.entity_type_calls >= 1


def test_entity_supporting_paper_uses_llm_grounding_for_direct_definition(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(clean_query="AlignX是什么", relation="entity_definition", targets=["AlignX"])
    papers = [
        CandidatePaper(
            paper_id="TRANSFER",
            title="Text as a Universal Interface for Transferable Personalization",
            year="2026",
            score=1.0,
            metadata={"paper_card_text": "This paper uses AlignX as one of several datasets."},
        ),
        CandidatePaper(
            paper_id="ALIGNX",
            title="From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
            year="2025",
            score=0.8,
            metadata={"paper_card_text": "This paper introduces AlignX, a dataset and benchmark."},
        ),
    ]
    evidence = [
        EvidenceBlock(
            doc_id="transfer-mention",
            paper_id="TRANSFER",
            title=papers[0].title,
            file_path="/tmp/transfer.pdf",
            page=6,
            block_type="page_text",
            snippet="We evaluate on datasets with varied history formats, from structured preference triplets (AlignX) to raw dialogues.",
        ),
        EvidenceBlock(
            doc_id="alignx-def",
            paper_id="ALIGNX",
            title=papers[1].title,
            file_path="/tmp/alignx.pdf",
            page=1,
            block_type="page_text",
            snippet="AlignX is a large-scale dataset and benchmark with more than 1.3 million personalized preference examples.",
        ),
    ]

    paper, supporting = agent._select_entity_supporting_paper(contract=contract, papers=papers, evidence=evidence)

    assert paper is not None
    assert paper.paper_id == "ALIGNX"
    assert supporting
    assert supporting[0].paper_id == "ALIGNX"
    assert agent.clients.entity_grounding_calls >= 1


def test_paper_identity_match_rejects_glued_superstring_alias(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    paper = CandidatePaper(
        paper_id="ALIGNXPLORE",
        title="Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals",
        metadata={"aliases": "AlignXplore"},
    )

    assert agent._paper_identity_matches_targets(paper=paper, targets=["AlignX"]) is False


def test_origin_solver_prefers_exact_entity_paper_over_superstring_alias(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="AlignX最早是由哪篇论文提出的？",
        relation="origin_lookup",
        targets=["AlignX"],
        required_modalities=["page_text", "paper_card"],
    )
    papers = [
        CandidatePaper(
            paper_id="ALIGNXPLORE",
            title="Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals",
            year="2025",
            score=0.95,
            metadata={"aliases": "AlignXplore", "paper_card_text": "ALIGNXPLORE is trained with reinforcement learning."},
        ),
        CandidatePaper(
            paper_id="ALIGNX",
            title="From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
            year="2025",
            score=0.8,
            metadata={"aliases": "AlignX", "paper_card_text": "This paper introduces AlignX, a dataset and benchmark."},
        ),
    ]
    evidence = [
        EvidenceBlock(
            doc_id="alignxplore-page-3",
            paper_id="ALIGNXPLORE",
            title=papers[0].title,
            file_path="/tmp/alignxplore.pdf",
            page=3,
            block_type="page_text",
            snippet="ALIGNXPLORE improves preference inference with reinforcement learning and reasoning chains.",
            score=4.2,
        ),
        EvidenceBlock(
            doc_id="alignx-page-1",
            paper_id="ALIGNX",
            title=papers[1].title,
            file_path="/tmp/alignx.pdf",
            page=1,
            block_type="page_text",
            snippet="We introduce AlignX, a large-scale dataset and benchmark for user-level alignment.",
            score=3.4,
        ),
    ]

    claims = agent._solve_text(contract=contract, papers=papers, evidence=evidence, session=SessionContext(session_id="origin-alignx"))

    assert claims
    assert claims[0].paper_ids == ["ALIGNX"]


def test_paper_summary_prefers_exact_entity_paper_over_superstring_alias(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="AlignX中主要结论是什么？用什么数据支持？",
        relation="paper_summary_results",
        targets=["AlignX"],
        answer_slots=["paper_summary"],
        requested_fields=["summary", "results", "evidence"],
        required_modalities=["page_text", "paper_card", "table", "caption"],
    )
    papers = [
        CandidatePaper(
            paper_id="ALIGNXPLORE",
            title="Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals",
            year="2025",
            score=0.95,
            metadata={
                "aliases": "AlignXplore",
                "paper_card_text": "This paper proposes AlignXplore, a model for preference inference from behavioral signals.",
            },
        ),
        CandidatePaper(
            paper_id="ALIGNX",
            title="From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
            year="2025",
            score=0.8,
            metadata={
                "aliases": "AlignX",
                "paper_card_text": "This paper introduces AlignX, a large-scale dataset and benchmark.",
            },
        ),
    ]
    evidence = [
        EvidenceBlock(
            doc_id="alignxplore-table",
            paper_id="ALIGNXPLORE",
            title=papers[0].title,
            file_path="/tmp/alignxplore.pdf",
            page=8,
            block_type="table",
            snippet="ALIGNXPLORE-7B improves preference inference accuracy.",
            score=4.2,
        ),
        EvidenceBlock(
            doc_id="alignx-table",
            paper_id="ALIGNX",
            title=papers[1].title,
            file_path="/tmp/alignx.pdf",
            page=7,
            block_type="table",
            snippet="ALIGNXPERT_PBA achieves the best alignment accuracy on AlignX benchmarks.",
            score=3.4,
        ),
    ]

    claims = agent._solve_text(contract=contract, papers=papers, evidence=evidence, session=SessionContext(session_id="summary-alignx"))

    assert claims
    assert claims[0].paper_ids == ["ALIGNX"]


def test_origin_solver_requires_intro_cue_not_benchmark_usage(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="ALignX是哪篇论文提出的？",
        relation="origin_lookup",
        targets=["ALignX"],
        requested_fields=["paper_title", "year", "evidence"],
        required_modalities=["paper_card", "page_text"],
        precision_requirement="exact",
    )
    papers = [
        CandidatePaper(
            paper_id="TEXTUI",
            title="Text as a Universal Interface for Transferable Personalization",
            year="2026",
            score=1.8,
            metadata={
                "aliases": "Text as a Universal Interface for Transferable Personalization",
                "paper_card_text": "Experiments on nine benchmarks include AlignX and P-SOUPS.",
            },
        ),
        CandidatePaper(
            paper_id="ALIGNX",
            title="From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
            year="2025",
            score=0.4,
            doc_ids=["paper::ALIGNX"],
            metadata={
                "aliases": "From 1,000,000 Users to Every User",
                "paper_card_text": "Building upon this foundation, we introduce AlignX, a large-scale dataset of over 1.3 million personalized preference examples.",
            },
        ),
    ]
    evidence = [
        EvidenceBlock(
            doc_id="textui-page-6",
            paper_id="TEXTUI",
            title=papers[0].title,
            file_path="/tmp/textui.pdf",
            page=6,
            block_type="page_text",
            snippet="Our model is evaluated on AlignX and P-SOUPS benchmarks.",
            score=8.0,
        ),
        EvidenceBlock(
            doc_id="alignx-page-1",
            paper_id="ALIGNX",
            title=papers[1].title,
            file_path="/tmp/alignx.pdf",
            page=1,
            block_type="page_text",
            snippet="Building upon this foundation, we introduce AlignX, a large-scale dataset of over 1.3 million personalized preference examples.",
            score=2.0,
        ),
    ]

    claims = agent._solve_text(contract=contract, papers=papers, evidence=evidence, session=SessionContext(session_id="origin-alignx-usage"))

    assert claims
    assert claims[0].paper_ids == ["ALIGNX"]


def test_origin_verifier_rejects_usage_without_intro_cue(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="AlignX是哪篇论文提出的？",
        relation="origin_lookup",
        targets=["AlignX"],
        requested_fields=["paper_title", "year", "evidence"],
    )
    paper = CandidatePaper(
        paper_id="TEXTUI",
        title="Text as a Universal Interface for Transferable Personalization",
        year="2026",
        metadata={"paper_card_text": "Experiments use AlignX as one benchmark."},
    )
    evidence = [
        EvidenceBlock(
            doc_id="textui-page-6",
            paper_id="TEXTUI",
            title=paper.title,
            file_path="/tmp/textui.pdf",
            page=6,
            block_type="page_text",
            snippet="Our model is evaluated on AlignX and P-SOUPS benchmarks.",
            score=8.0,
        )
    ]
    claim = Claim(
        claim_type="origin",
        entity="AlignX",
        value=paper.title,
        structured_data={"paper_title": paper.title, "year": "2026"},
        evidence_ids=["textui-page-6"],
        paper_ids=["TEXTUI"],
        confidence=0.9,
    )

    report = agent._verify_claims(
        contract=contract,
        plan=ResearchPlan(required_claims=["paper_title", "year"]),
        claims=[claim],
        papers=[paper],
        evidence=evidence,
    )

    assert report.status == "retry"
    assert "origin_evidence" in report.missing_fields


def test_formula_screening_prefers_exact_target_paper_over_higher_scored_noise(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    candidates = [
        CandidatePaper(
            paper_id="NOISE",
            title="Unveiling Inference Scaling for Difference-Aware User Modeling in LLM Personalization",
            year="2025",
            score=1.0,
            metadata={"aliases": "Difference-Aware User Modeling"},
        ),
        CandidatePaper(
            paper_id="DPO",
            title="Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
            year="2024",
            score=0.7,
            metadata={"aliases": "Direct Preference Optimization||DPO"},
        ),
    ]

    screened = agent._prefer_identity_matching_papers(candidates=candidates, targets=["DPO"])

    assert [item.paper_id for item in screened] == ["DPO"]


def test_entity_clean_lines_skips_formula_noise(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    cleaned = agent._entity_clean_lines(
        [
            "GRPO foregoes the value model and uses group scores to estimate the baseline.",
            " 𝜋𝜃 (𝑜𝑡 |𝑞, 𝑜<𝑡) 𝜋𝜃𝑜𝑙𝑑 (𝑜𝑡 |𝑞, 𝑜<𝑡) , 1 − 𝜀, 1 + 𝜀  𝐴𝑡  . (15)",
        ],
        limit=3,
    )

    assert cleaned == ["GRPO foregoes the value model and uses group scores to estimate the baseline."]


def test_prune_entity_supporting_evidence_drops_formula_heavy_blocks(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    evidence = [
        EvidenceBlock(
            doc_id="clean-grpo",
            paper_id="DEEPMATH",
            title="DeepSeekMath",
            file_path="/tmp/deepseekmath.pdf",
            page=13,
            block_type="page_text",
            snippet="GRPO uses the average reward of multiple sampled outputs as the baseline.",
        ),
        EvidenceBlock(
            doc_id="formula-grpo",
            paper_id="DEEPMATH",
            title="DeepSeekMath",
            file_path="/tmp/deepseekmath.pdf",
            page=29,
            block_type="page_text",
            snippet=" 𝜋𝜃 (𝑜𝑡 |𝑞, 𝑜<𝑡) 𝜋𝜃𝑜𝑙𝑑 (𝑜𝑡 |𝑞, 𝑜<𝑡) , 1 − 𝜀, 1 + 𝜀  𝐴𝑡  . (15)",
        ),
    ]

    pruned = agent._prune_entity_supporting_evidence(evidence)

    assert [item.doc_id for item in pruned] == ["clean-grpo"]


def test_focused_snippet_centers_on_target_instead_of_page_prefix(tmp_path: Path) -> None:
    _, retriever = _build_agent(tmp_path)
    text = (
        "This page starts with unrelated background and several benchmark descriptions. "
        "More prefix text that should not dominate the snippet. "
        "Group Relative Policy Optimization (GRPO) is a variant of PPO that uses group-relative rewards to compute advantages "
        "and removes the need for a separate value critic."
    )

    snippet = retriever._focused_snippet(text=text, targets=["GRPO"], query="GRPO是什么技术")

    assert "GRPO" in snippet
    assert "variant of PPO" in snippet
    assert "This page starts with unrelated background" not in snippet


def test_formula_heavy_text_is_filtered_for_entity_answers(tmp_path: Path) -> None:
    _, retriever = _build_agent(tmp_path)

    assert retriever._looks_formula_heavy(" 𝜋𝜃 (𝑜𝑡 |𝑞, 𝑜<𝑡) 𝜋𝜃𝑜𝑙𝑑 (𝑜𝑡 |𝑞, 𝑜<𝑡) , 1 − 𝜀, 1 + 𝜀  𝐴𝑡  . (15)")
    assert not retriever._looks_formula_heavy(
        "Group Relative Policy Optimization (GRPO) is a variant of PPO that uses relative rewards within a group."
    )


def test_entity_answer_composer_scopes_followup_details_to_claim_evidence(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="GRPO是什么技术",
        relation="entity_definition",
        targets=["GRPO"],
        requested_fields=["definition", "applications"],
        continuation_mode="fresh",
    )
    claim = Claim(
        claim_type="entity_definition",
        entity="GRPO",
        value="优化算法/训练方法",
        structured_data={
            "paper_title": "DeepSeekMath",
            "definition_lines": [
                "Group Relative Policy Optimization (GRPO) is a variant of PPO that foregoes the critic model."
            ],
            "mechanism_lines": [
                "GRPO uses group scores to estimate the baseline."
            ],
            "application_lines": [],
        },
        evidence_ids=["deepseek-grpo"],
        paper_ids=["DEEPMATH"],
    )
    evidence = [
        EvidenceBlock(
            doc_id="deepseek-grpo",
            paper_id="DEEPMATH",
            title="DeepSeekMath",
            file_path="/tmp/deepseekmath.pdf",
            page=2,
            block_type="page_text",
            snippet="Group Relative Policy Optimization (GRPO) is a variant of PPO that foregoes the critic model and improves mathematical reasoning.",
            score=4.0,
            metadata={"definition_score": 3.2, "mechanism_score": 1.0},
        ),
        EvidenceBlock(
            doc_id="other-paper",
            paper_id="OTH",
            title="Another Paper",
            file_path="/tmp/other.pdf",
            page=5,
            block_type="page_text",
            snippet="We employ GRPO as the RL framework to improve model performance in reasoning.",
            score=5.0,
            metadata={"application_score": 0.8},
        ),
    ]

    answer, _ = agent._compose_answer(
        contract=contract,
        claims=[claim],
        evidence=evidence,
        papers=[],
        verification=VerificationReport(status="pass"),
    )

    assert "Another Paper" not in answer
    assert "We employ GRPO as the RL framework" not in answer


def test_followup_solver_uses_seed_paper_then_excludes_it_from_followups(tmp_path: Path) -> None:
    agent, retriever = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="AlignX数据集有后续工作吗？",
        relation="followup_research",
        targets=["AlignX"],
        required_modalities=["paper_card", "page_text"],
    )
    papers = []
    for paper_id in ["ALIGNX", "ALIGNXPLORE"]:
        doc = retriever.paper_doc_by_id(paper_id)
        assert doc is not None
        meta = dict(doc.metadata or {})
        papers.append(
            CandidatePaper(
                paper_id=paper_id,
                title=str(meta.get("title", "")),
                year=str(meta.get("year", "")),
                score=1.0,
                doc_ids=[str(meta.get("doc_id", ""))],
                metadata=meta,
            )
        )

    claims = agent._solve_followup_research(
        contract=contract,
        papers=papers,
        session=SessionContext(session_id="followup-demo"),
    )

    assert claims
    claim = claims[0]
    assert claim.claim_type == "followup_research"
    assert claim.structured_data["seed_papers"][0]["paper_id"] == "ALIGNX"
    followup_ids = [item["paper_id"] for item in claim.structured_data["followup_titles"]]
    assert "ALIGNX" not in followup_ids
    assert "ALIGNXPLORE" in followup_ids


def test_followup_answer_composer_lists_structured_candidates(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    claim = Claim(
        claim_type="followup_research",
        entity="AlignX",
        structured_data={
            "seed_papers": [
                {
                    "title": "From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
                    "year": "2025",
                    "paper_id": "ALIGNX",
                }
            ],
            "followup_titles": [
                {"title": "POPI: Personalizing LLMs via Optimized Preference Inference", "year": "2026", "relation_type": "后续工作", "reason": "preference inference"},
                {"title": "Text as a Universal Interface for Transferable Personalization", "year": "2026", "relation_type": "扩展工作", "reason": "transferable personalization"},
                {"title": "PersonaDual: Balancing Personalization and Objectivity via Adaptive Reasoning", "year": "2026", "relation_type": "相关延续", "reason": "personalized alignment"},
            ],
        },
        evidence_ids=[],
        paper_ids=["ALIGNX", "POPI", "TEXT", "PERSONADUAL"],
    )

    answer, _ = agent._compose_answer(
        contract=QueryContract(clean_query="AlignX数据集有后续工作吗？", relation="followup_research", targets=["AlignX"]),
        claims=[claim],
        evidence=[],
        papers=[],
        verification=VerificationReport(status="pass"),
    )

    assert "后续工作" in answer
    assert "AlignX" in answer
    assert "POPI" in answer
    assert "Text as a Universal Interface" in answer
    assert "PersonaDual" in answer


def test_candidate_validation_followup_uses_previous_followup_history_in_answer(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session_id = "alignx-followup-history"
    clients = agent.clients

    first = agent.chat(query="AlignX数据集有后续工作吗？", session_id=session_id)
    first_decompose_calls = clients.compound_decompose_calls
    first_refine_calls = clients.followup_refine_calls
    second = agent.chat(
        query="那仔细看看先，Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals是吗",
        session_id=session_id,
    )

    assert first["query_contract"]["relation"] == "followup_research"
    assert second["query_contract"]["relation"] == "followup_research"
    assert "AlignX" in second["answer"]
    assert "强相关延续候选" in second["answer"] or "严格后续：否" in second["answer"]
    assert "严格后续：否" in second["answer"] or "不能写成严格后续工作" in second["answer"]
    assert "证据范围" in second["answer"]
    assert clients.relationship_verifier_calls >= 1
    citation_doc_ids = {item["doc_id"] for item in second["citations"]}
    assert "block-alignxplore-definition" in citation_doc_ids
    assert "block-alignx-definition" in citation_doc_ids
    assert clients.compound_decompose_calls == first_decompose_calls
    assert clients.followup_refine_calls == first_refine_calls


def test_explicit_followup_direction_uses_seed_after_de_particle(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = SessionContext(session_id="followup-direction")

    contract = agent._extract_query_contract(
        query="Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals是ALignX的后续工作吗",
        session=session,
        mode="auto",
    )

    assert contract.relation == "followup_research"
    assert contract.targets == ["AlignX"]
    assert "followup_direction_resolved" in contract.notes
    assert any(
        note.startswith("candidate_title=Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals")
        for note in contract.notes
    )


def test_explicit_followup_direction_answer_keeps_alignx_as_seed(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    result = agent.chat(
        query="Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals是ALignX的后续工作吗",
        session_id="followup-direction-answer",
    )

    assert result["query_contract"]["relation"] == "followup_research"
    assert result["query_contract"]["targets"] == ["AlignX"]
    assert "种子论文是《From 1,000,000 Users to Every User" in result["answer"]
    assert "《Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals》" in result["answer"]
    assert "验证分类：related_continuation" in result["answer"]
    assert "围绕 Extended Inductive Reasoning" not in result["answer"]


def test_strict_followup_check_inherits_previous_seed_and_candidate(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session_id = "strict-followup-context"

    first = agent.chat(
        query="Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals真是AlignX后续吗，确认一下",
        session_id=session_id,
    )
    second = agent.chat(query="就是让你仔细看看是不是严格后续工作嘛", session_id=session_id)

    assert first["query_contract"]["targets"] == ["AlignX"]
    assert second["query_contract"]["relation"] == "followup_research"
    assert second["query_contract"]["targets"] == ["AlignX"]
    assert "inherited_followup_relationship" in second["query_contract"]["notes"]
    assert "candidate_title=Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals" in second["query_contract"]["notes"]
    assert "让你仔细看看是不是严格" not in second["answer"]
    assert "严格后续：否" in second["answer"] or "严格后续：否/证据不足" in second["answer"]
    assert "验证分类：related_continuation" in second["answer"]


def test_followup_answer_groups_relationship_strength_and_hides_raw_english_reason(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    claim = Claim(
        claim_type="followup_research",
        entity="AlignX",
        structured_data={
            "seed_papers": [
                {
                    "title": "From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
                    "year": "2025",
                    "paper_id": "ALIGNX",
                }
            ],
            "followup_titles": [
                {
                    "title": "DirectUse: Evaluating Personalized Alignment on AlignX",
                    "year": "2026",
                    "relation_type": "直接使用/评测证据",
                    "reason": "This paper evaluates on AlignX and directly builds on the dataset.",
                    "relationship_strength": "direct",
                },
                {
                    "title": "POPI: Personalizing LLMs via Optimized Preference Inference",
                    "year": "2026",
                    "relation_type": "强相关延续候选",
                    "reason": "This paper studies preference inference and personalized alignment.",
                    "relationship_strength": "strong_related",
                },
                {
                    "title": "PersonaDual: Balancing Personalization and Objectivity via Adaptive Reasoning",
                    "year": "2026",
                    "relation_type": "同主题待确认候选",
                    "reason": "This paper is in the same broad personalization area.",
                    "relationship_strength": "weak_related",
                },
            ],
        },
        evidence_ids=[],
        paper_ids=["ALIGNX", "DIRECT", "POPI", "PERSONADUAL"],
    )

    answer, _ = agent._compose_answer(
        contract=QueryContract(clean_query="AlignX数据集有后续工作吗？", relation="followup_research", targets=["AlignX"]),
        claims=[claim],
        evidence=[],
        papers=[],
        verification=VerificationReport(status="pass"),
    )

    assert "## 直接后续/使用证据" in answer
    assert "## 强相关延续候选" in answer
    assert "## 同主题但待确认" in answer
    assert "This paper" not in answer
    assert "读法建议" in answer


def test_web_search_fallback_uses_web_evidence_when_enabled(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    class StubWebSearch:
        is_configured = True

        def search(
            self,
            *,
            query: str,
            max_results: int = 5,
            topic: str = "general",
            include_domains: list[str] | None = None,
        ) -> list[EvidenceBlock]:
            return [
                EvidenceBlock(
                    doc_id="web::rag-news",
                    paper_id="web::rag-news",
                    title="Recent RAG Paper",
                    file_path="https://example.com/rag",
                    page=0,
                    block_type="web",
                    caption="https://example.com/rag",
                    snippet="A recent paper reports a new retrieval-augmented generation method.",
                    metadata={"source": "tavily", "query": query, "topic": topic, "include_domains": include_domains or []},
                )
            ]

    agent.web_search = StubWebSearch()
    contract = QueryContract(
        clean_query="最近RAG有什么新论文？",
        relation="general_question",
        allow_web_search=True,
    )
    evidence = collect_web_evidence(
        web_search=agent.web_search,
        contract=contract,
        use_web_search=True,
        max_web_results=3,
    )
    claim = build_web_research_claim(contract=contract, web_evidence=evidence)
    report = agent._verify_claims(
        contract=contract,
        plan=ResearchPlan(required_claims=["answer"]),
        claims=[claim],
        papers=[],
        evidence=evidence,
    )
    answer, citations = agent._compose_answer(
        contract=contract,
        claims=[claim],
        evidence=evidence,
        papers=[],
        verification=report,
    )

    assert evidence
    assert evidence[0].block_type == "web"
    assert "arxiv.org" in evidence[0].metadata["include_domains"]
    assert report.status == "pass"
    assert "Recent RAG Paper" in answer
    assert citations and citations[0].file_path == "https://example.com/rag"


def test_latest_query_auto_enables_external_search(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(clean_query="最新的多模态RAG论文有哪些？", relation="paper_recommendation")

    assert agent._should_use_web_search(use_web_search=False, contract=contract)


def test_correction_followup_reuses_active_target_for_repair(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = agent.sessions.get("demo")
    session.set_active_research(
        relation="entity_definition",
        targets=["AlignX"],
        titles=[],
        requested_fields=["entity_type", "supporting_paper"],
        required_modalities=["page_text", "paper_card"],
        answer_shape="narrative",
        precision_requirement="high",
        clean_query="AlignX是什么",
    )
    session.last_relation = "entity_definition"

    contract = agent._extract_query_contract(query="不对吧", session=session, mode="auto")

    assert contract.relation == "entity_definition"
    assert contract.targets == ["AlignX"]
    assert contract.continuation_mode == "followup"


def test_contextual_origin_challenge_is_recovered_as_research_followup(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = agent.sessions.get("origin-demo")
    session.set_active_research(
        relation="entity_definition",
        targets=["AlignX"],
        titles=["From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment"],
        requested_fields=["entity_type", "supporting_paper"],
        required_modalities=["page_text", "paper_card"],
        answer_shape="narrative",
        precision_requirement="high",
        clean_query="AlignX是什么",
    )

    contract = agent._extract_query_contract(query="最早不是在这里吧", session=session, mode="auto")

    assert contract.relation == "origin_lookup"
    assert contract.continuation_mode == "followup"
    assert contract.targets == ["AlignX"]
    assert "提出" in contract.clean_query or "最早" in contract.clean_query
    assert "challenge_previous_answer" in contract.notes
    assert agent.clients.followup_refine_calls >= 1


def test_explicit_first_proposed_query_does_not_bind_stale_memory_paper(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = agent.sessions.get("origin-explicit-demo")
    session.set_active_research(
        relation="paper_summary_results",
        targets=["AlignX"],
        titles=["Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals"],
        requested_fields=["summary", "results"],
        required_modalities=["page_text", "paper_card"],
        answer_shape="narrative",
        precision_requirement="high",
        clean_query="AlignX中主要结论是什么？用什么数据支持？",
    )
    session.working_memory = {
        "target_bindings": {
            "alignx": {
                "target": "AlignX",
                "paper_id": "ALIGNXPLORE",
                "title": "Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals",
            }
        }
    }

    contract = agent._extract_query_contract(query="不是这一篇，是第一个提出AlignX的", session=session, mode="auto")

    assert contract.relation == "origin_lookup"
    assert contract.targets == ["AlignX"]
    assert "origin" in contract.answer_slots
    assert "selected_paper_id=ALIGNXPLORE" not in contract.notes


def test_first_paper_origin_phrase_routes_to_origin_lookup(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = agent.sessions.get("origin-first-paper-demo")

    contract = agent._extract_query_contract(query="我问的是AlignX的第一篇论文", session=session, mode="auto")

    assert contract.relation == "origin_lookup"
    assert contract.targets == ["AlignX"]


def test_explicit_summary_query_keeps_named_target(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = agent.sessions.get("summary-explicit-demo")

    contract = agent._extract_query_contract(query="AlignX中主要结论是什么？用什么数据支持？", session=session, mode="auto")

    assert contract.relation == "paper_summary_results"
    assert contract.targets == ["AlignX"]
    assert "paper_summary" in contract.answer_slots


def test_followup_short_query_inherits_formula_task_from_session_memory(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    first = agent.chat(query="DPO的公式是什么？", session_id="memory-demo")
    second = agent.chat(query="PPO呢", session_id="memory-demo")

    assert first["query_contract"]["relation"] == "formula_lookup"
    assert second["query_contract"]["relation"] == "formula_lookup"
    assert second["query_contract"]["continuation_mode"] == "followup"
    assert second["query_contract"]["targets"] == ["PPO"]


def test_formula_variable_descriptions_wrap_bare_latex() -> None:
    answer = ResearchAssistantAgentV4._compose_formula_answer(
        claims=[
            Claim(
                claim_type="formula",
                entity="DPO",
                value=r"r(x,y)=\beta\log\frac{\pi_{\theta}(y|x)}{\pi_{\mathrm{ref}}(y|x)}",
                structured_data={
                    "formula_format": "latex",
                    "variables": [
                        {
                            "symbol": r"r(x,y)",
                            "description": "隐式定义的奖励函数",
                        },
                        {
                            "symbol": r"\beta",
                            "description": r"控制与参考策略 \pi_{\mathrm{ref}} 偏离程度的参数",
                        },
                    ],
                },
            )
        ]
    )

    assert "$r(x,y)$" in answer
    assert r"$\pi_{\mathrm{ref}}$" in answer
    assert r"\pi_{\mathrm{ref}} 偏离" not in answer


def test_compound_formula_comparison_streams_subtasks_without_plan_in_answer(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    events: list[dict[str, object]] = []

    result, _ = agent._run(
        query="DPO公式是什么，PPO呢，顺便比较一下",
        session_id="compound-formula-demo",
        mode="auto",
        use_web_search=False,
        max_web_results=3,
        event_callback=events.append,
    )
    nodes = [step["node"] for step in result["execution_steps"]]
    deltas = [item for item in events if item.get("event") == "answer_delta"]

    assert result["query_contract"]["relation"] == "compound_query"
    assert "subtask:formula_lookup" in result["query_contract"]["notes"]
    assert "subtask:comparison_synthesis" in result["query_contract"]["notes"]
    assert nodes.count("compound_task:formula_lookup") == 2
    assert "compound_task:comparison_synthesis" in nodes
    assert len(deltas) >= 4
    assert "## 计划" not in result["answer"]
    assert "查询 DPO 公式" in result["answer"]
    assert "查询 PPO 公式" in result["answer"]
    assert "比较 DPO 和 PPO" in result["answer"]


def test_compound_followup_comparison_uses_previous_formula_memory(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    events: list[dict[str, object]] = []

    first = agent.chat(query="DPO的公式是什么？", session_id="compare-memory")
    result, _ = agent._run(
        query="那PPO呢，两者有什么区别",
        session_id="compare-memory",
        mode="auto",
        use_web_search=False,
        max_web_results=3,
        event_callback=events.append,
    )
    deltas = [item for item in events if item.get("event") == "answer_delta"]

    assert first["query_contract"]["relation"] == "formula_lookup"
    assert result["query_contract"]["relation"] == "compound_query"
    assert "DPO" in result["answer"]
    assert "PPO" in result["answer"]
    assert "证据不足" not in result["answer"]
    assert len(deltas) >= 3


def test_compound_query_stops_for_subtask_clarification(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    calls: list[str] = []

    def fake_execute(self, *, contract: QueryContract, session: SessionContext, emit, execution_steps):
        calls.append(contract.relation)
        blocked = contract_with_ambiguity_options(
            contract=contract,
            options=[
                {
                    "paper_id": "PPO-1",
                    "title": "Training language models to follow instructions with human feedback",
                    "year": "2022",
                    "meaning": "Proximal Policy Optimization",
                    "snippet": "reinforcement learning via proximal policy optimization (PPO)",
                },
                {
                    "paper_id": "PPO-2",
                    "title": "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
                    "year": "2024",
                    "meaning": "variant of Proximal Policy Optimization",
                    "snippet": "variant of Proximal Policy Optimization",
                },
            ],
        )
        verification = VerificationReport(
            status="clarify",
            missing_fields=["disambiguation"],
            recommended_action="clarify_ambiguous_entity",
        )
        return {
            "contract": blocked,
            "answer": self._clarification_question(blocked, session),
            "citations": [],
            "claims": [],
            "evidence": [],
            "verification": verification,
        }

    agent._execute_compound_task_subagent = MethodType(fake_execute, agent)

    result, _ = agent._run(
        query="PPO和DPO有什么区别",
        session_id="compound-clarify-stop",
        mode="auto",
        use_web_search=False,
        max_web_results=3,
        event_callback=lambda _: None,
    )
    nodes = [step["node"] for step in result["execution_steps"]]
    session = agent.sessions.get("compound-clarify-stop")

    assert result["needs_human"] is True
    assert result["verification_report"]["status"] == "clarify"
    assert result["verification_report"]["recommended_action"] == "clarify_ambiguous_entity"
    assert result["clarification_options"]
    assert "compound_task:comparison_synthesis" not in nodes
    assert calls == ["formula_lookup"]
    assert "在继续复合问题" in result["answer"]
    assert session.pending_clarification_type == "ambiguity"


def test_compound_parallel_formula_query_without_compare_splits_targets(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    events: list[dict[str, object]] = []

    result, _ = agent._run(
        query="DPO的公式是什么？PPO公式又是什么",
        session_id="compound-formula-no-compare",
        mode="auto",
        use_web_search=False,
        max_web_results=3,
        event_callback=events.append,
    )
    nodes = [step["node"] for step in result["execution_steps"]]
    agent_steps = [item for item in events if item.get("event") == "agent_step"]

    assert result["query_contract"]["relation"] == "compound_query"
    assert result["query_contract"]["notes"].count("subtask:formula_lookup") == 2
    assert "subtask:comparison_synthesis" not in result["query_contract"]["notes"]
    assert nodes.count("compound_task:formula_lookup") == 2
    assert "查询 DPO 公式" in result["answer"]
    assert "查询 PPO 公式" in result["answer"]
    assert "好的，我现在去查询 **DPO** 的公式。" in result["answer"]
    assert "好的，我现在去查询 **PPO** 的公式。" in result["answer"]
    assert agent_steps


def test_compound_formula_query_persists_working_memory_for_followups(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    first = agent.chat(query="DPO公式和PPO公式分别都是什么？", session_id="compound-memory")
    second = agent.chat(query="你觉得两者的区别是什么", session_id="compound-memory")

    session = agent.sessions.get("compound-memory")
    bindings = session.working_memory.get("target_bindings", {})
    assert first["query_contract"]["relation"] == "compound_query"
    assert "dpo" in bindings
    assert "ppo" in bindings
    assert second["query_contract"]["relation"] == "memory_synthesis"
    assert "DPO" in second["answer"]
    assert "PPO" in second["answer"]


def test_reward_model_followup_uses_conversation_memory_without_acronym_clarification(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    agent.chat(query="DPO公式和PPO公式分别都是什么？", session_id="reward-memory")
    result = agent.chat(query="你帮我查查DPO 是否需要 reward model", session_id="reward-memory")

    assert result["needs_human"] is False
    assert result["query_contract"]["relation"] == "general_question"
    assert "resolved_from_conversation_memory" in result["query_contract"]["notes"]
    assert any(str(note).startswith("selected_paper_id=") for note in result["query_contract"]["notes"])
    assert result["citations"]


def test_formula_answer_uses_rendered_math_without_raw_alias_labels() -> None:
    answer = ResearchAssistantAgentV4._compose_formula_answer(
        claims=[
            Claim(
                claim_type="formula",
                entity="DPO",
                value="L_{DPO}(\\pi_\\theta; \\pi_{ref})",
                structured_data={"formula_format": "latex", "terms": ["pi_theta", "pi_ref", "beta", "log_sigma"]},
                evidence_ids=[],
            )
        ]
    )

    assert "$\\pi_\\theta$：当前策略" in answer
    assert "/ pi_theta" not in answer
    assert "/ pi_ref" not in answer
    assert "/ beta" not in answer


def test_compact_unicode_formula_is_normalized_to_markdown_latex() -> None:
    formula = (
        "∇θLDPO(πθ; πref) = -∇θE(x, yw, yl) ~ D[logσ("
        "βlogπθ(yw|x)/πref(yw|x))]"
    )

    normalized = ResearchAssistantAgentV4._normalize_extracted_formula_text(formula)
    symbol = ResearchAssistantAgentV4._normalize_formula_variable_symbol("∇θLDPO")

    assert r"\nabla_{\theta}\mathcal{L}_{\mathrm{DPO}}" in normalized
    assert r"\pi_{\theta}" in normalized
    assert r"\pi_{\mathrm{ref}}" in normalized
    assert r"\log \sigma" in normalized
    assert r"\beta \log \pi_{\theta}" in normalized
    assert r"y_w" in normalized
    assert r"y_l" in normalized
    assert symbol == r"\nabla_{\theta}\mathcal{L}_{\mathrm{DPO}}"


def test_formula_interpretation_followup_uses_memory_not_retrieval(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = agent.sessions.get("formula-interpret")
    formula_contract = QueryContract(
        clean_query="DPO 的公式是什么？",
        relation="formula_lookup",
        targets=["DPO"],
        answer_slots=["formula"],
        requested_fields=["formula", "variable_explanation"],
        required_modalities=["page_text", "table"],
        precision_requirement="exact",
    )
    session.set_active_research(
        relation="formula_lookup",
        targets=["DPO"],
        titles=["Direct Preference Optimization: Your Language Model is Secretly a Reward Model"],
        requested_fields=["formula", "variable_explanation"],
        required_modalities=["page_text", "table"],
        answer_shape="bullets",
        precision_requirement="exact",
        clean_query="DPO 的公式是什么？",
    )
    session.turns.append(
        SessionTurn.from_contract(
            query="DPO 的公式是什么？",
            answer="## 核心公式\n\n$$L_{DPO}(\\pi_\\theta;\\pi_{ref})$$",
            contract=formula_contract,
            titles=["Direct Preference Optimization: Your Language Model is Secretly a Reward Model"],
        )
    )

    contract = agent._extract_query_contract(query="怎么理解这个公式？", session=session, mode="auto")
    answer = compose_memory_followup_answer(
        query="怎么理解这个公式？",
        session=session,
        contract=contract,
        clients=agent.clients,
        conversation_context=agent._session_conversation_context,
        clean_text=agent._clean_common_ocr_artifacts,
    )

    assert contract.interaction_mode == "conversation"
    assert contract.relation == "memory_followup"
    assert contract.requested_fields == ["formula_interpretation"]
    assert "提高被选中回答" in answer


def test_negative_formula_correction_keeps_active_paper_scope(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = agent.sessions.get("formula-correction")
    session.set_active_research(
        relation="formula_lookup",
        targets=["DPO"],
        titles=["Direct Preference Optimization: Your Language Model is Secretly a Reward Model"],
        requested_fields=["formula", "variable_explanation"],
        required_modalities=["page_text", "table"],
        answer_shape="bullets",
        precision_requirement="exact",
        clean_query="DPO 的公式是什么？",
    )

    contract = agent._extract_query_contract(query="我觉得不是这个公式哦", session=session, mode="auto")

    assert contract.relation == "formula_lookup"
    assert contract.targets == ["DPO"]
    assert contract.continuation_mode == "followup"
    assert "formula_answer_correction" in contract.notes
    assert "prefer_scalar_objective" in contract.notes
    assert any(note == "selected_paper_id=DPO" for note in contract.notes)
    assert "目标函数" in contract.clean_query


def test_language_preference_followup_does_not_search_random_papers(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = agent.sessions.get("language-pref")
    session.turns.append(
        SessionTurn(
            query="DPO 的公式是什么？",
            answer="变量解释里有 English sentence.",
            relation="formula_lookup",
            clean_query="DPO 的公式是什么？",
            targets=["DPO"],
            requested_fields=["formula", "variable_explanation"],
        )
    )

    contract = agent._extract_query_contract(query="你怎么回答还中英文混杂，我要中文", session=session, mode="auto")
    answer = compose_memory_followup_answer(
        query="你怎么回答还中英文混杂，我要中文",
        session=session,
        contract=contract,
        clients=agent.clients,
        conversation_context=agent._session_conversation_context,
        clean_text=agent._clean_common_ocr_artifacts,
    )

    assert contract.interaction_mode == "conversation"
    assert contract.relation == "memory_followup"
    assert contract.requested_fields == ["answer_language_preference"]
    assert "中文说明" in answer


def test_formula_claim_preserves_structured_variables_from_llm(tmp_path: Path) -> None:
    class FormulaSchemaClients(StubModelClients):
        def invoke_json(self, *, system_prompt: str, human_prompt: str, fallback: object) -> object:
            if "论文公式抽取器" in system_prompt:
                return {
                    "formula_latex": r"L_{PBA} = -\log \sigma(\beta \Delta)",
                    "formula_format": "latex",
                    "variables": [
                        {"symbol": "πϕ", "description": "显式偏好方向条件下的生成策略。"},
                        {"symbol": "P̃", "description": "从用户画像聚合得到的偏好方向向量。"},
                    ],
                    "evidence_ids": ["schema-pba-formula"],
                    "confidence": 0.94,
                }
            return super().invoke_json(system_prompt=system_prompt, human_prompt=human_prompt, fallback=fallback)

    agent, _ = _build_agent(tmp_path)
    agent.clients = FormulaSchemaClients()
    contract = QueryContract(
        clean_query="PBA 的公式是什么？",
        relation="formula_lookup",
        targets=["PBA"],
        answer_slots=["formula"],
        requested_fields=["formula", "variable_explanation"],
        required_modalities=["page_text", "table"],
        precision_requirement="exact",
    )
    paper = CandidatePaper(
        paper_id="ALIGNX",
        title="From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
        year="2025",
        score=1.0,
        metadata={"paper_card_text": "Preference-bridged alignment (PBA) defines a PBA objective."},
    )
    evidence = [
        EvidenceBlock(
            doc_id="schema-pba-formula",
            paper_id="ALIGNX",
            title=paper.title,
            file_path="/tmp/alignx.pdf",
            page=6,
            block_type="page_text",
            snippet="Preference-bridged alignment (PBA) objective: L_{PBA} = - log sigma(beta Delta).",
            metadata={"formula_hint": 1},
        )
    ]

    claims = agent._solve_formula(contract=contract, papers=[paper], evidence=evidence)

    assert claims
    structured = dict(claims[0].structured_data)
    assert structured["formula_latex"] == r"L_{\mathrm{PBA}}= -\log \sigma(\beta \Delta)"
    assert structured["paper_id"] == "ALIGNX"
    assert structured["evidence_ids"] == ["schema-pba-formula"]
    assert structured["variables"][0]["symbol"] == r"\pi_{\phi}"
    assert structured["variables"][1]["symbol"] == r"\tilde{P}"
    answer = ResearchAssistantAgentV4._compose_formula_answer(claims=claims)
    assert "显式偏好方向条件下的生成策略" in answer
    assert "从用户画像聚合得到的偏好方向向量" in answer
    assert "PBA 中条件化在显式偏好方向上的生成策略" not in answer


def test_llm_research_composer_prompt_requires_paper_sections_for_multi_paper_claims(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    clients = CaptureComposerClients()
    agent.clients = clients  # type: ignore[assignment]

    answer = agent._compose_research_answer_markdown(
        contract=QueryContract(
            clean_query="PBA 在不同论文中的公式分别是什么？",
            relation="formula_lookup",
            targets=["PBA"],
            requested_fields=["formula"],
            required_modalities=["page_text"],
        ),
        claims=[
            Claim(claim_type="formula", entity="PBA", value="L_A", paper_ids=["PAPER_A"], evidence_ids=["a"]),
            Claim(claim_type="formula", entity="PBA", value="L_B", paper_ids=["PAPER_B"], evidence_ids=["b"]),
        ],
        evidence=[],
        papers=[
            CandidatePaper(paper_id="PAPER_A", title="Paper A", year="2025"),
            CandidatePaper(paper_id="PAPER_B", title="Paper B", year="2026"),
        ],
        citations=[],
        verification=VerificationReport(status="pass"),
    )

    assert "按论文分节" in answer
    assert "claims 来自不同 paper_id" in clients.last_system_prompt
    assert "禁止把不同论文的定义揉成同一句话" in clients.last_system_prompt


def test_pba_formula_retry_does_not_render_ppo_clip_formula(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    result = agent.chat(query="PBA公式是什么", session_id="pba-formula-guard")

    assert result["verification_report"]["status"] in {"retry", "clarify"}
    if result["verification_report"]["status"] == "retry":
        assert "target_aligned_formula" in result["verification_report"]["missing_fields"] or "formula" in result["verification_report"]["missing_fields"]
        assert "不能确认公式" in result["answer"]
    else:
        assert result["needs_human"] is True
        assert "多个可能含义" in result["answer"]
    assert "L^{\\mathrm{CLIP}}" not in result["answer"]
    assert "clipped surrogate" not in result["answer"].lower()


def test_formula_correction_does_not_select_pending_pba_ambiguity(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = agent.sessions.get("pba-pending-correction")
    session.pending_clarification_type = "ambiguity"
    session.pending_clarification_target = "PBA"
    session.pending_clarification_options = [
        {"index": 0, "paper_id": "ALIGNX", "meaning": "PBA", "title": "AlignX"},
        {"index": 1, "paper_id": "CURP", "meaning": "Prototype Behavior Aligning", "title": "CURP"},
    ]
    agent.sessions.upsert(session)

    result = agent.chat(query="你这是PBA的公式吗？这不是PPO的公式", session_id="pba-pending-correction")

    assert "resolved_human_choice" not in result["query_contract"]["notes"]
    assert result["query_contract"]["relation"] == "formula_lookup"
    assert result["verification_report"]["status"] == "retry"
    assert "L^{\\mathrm{CLIP}}" not in result["answer"]


def test_formula_location_followup_binds_active_target_to_named_paper(tmp_path: Path) -> None:
    agent, retriever = _build_agent(tmp_path)
    retriever._block_docs.append(
        Document(
            page_content=(
                "Preference-bridged alignment. PBA introduces a latent variable P̃ as an explicit proxy of P. "
                "The final optimization objective becomes: "
                "LPBA = - log σ β(log πϕ(yw|x,P̃)/πref(yw|x,P̃) - log πϕ(yl|x,P̃)/πref(yl|x,P̃))."
            ),
            metadata={
                "doc_id": "block-alignx-pba-formula",
                "paper_id": "ALIGNX",
                "title": "From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
                "authors": "Li et al.",
                "year": "2025",
                "tags": "alignx||personalization",
                "file_path": "/tmp/alignx.pdf",
                "page": 6,
                "block_type": "page_text",
                "caption": "",
                "bbox": "",
                "formula_hint": 1,
            },
        )
    )
    retriever._rebuild_lookup_indexes()
    session_id = "pba-formula-location-followup"

    first = agent.chat(query="PBA的公式是什么", session_id=session_id)
    second = agent.chat(query="有PBA，就在AlignX的论文里", session_id=session_id)

    assert first["query_contract"]["relation"] == "formula_lookup"
    assert second["query_contract"]["relation"] == "formula_lookup"
    assert second["query_contract"]["targets"] == ["PBA"]
    assert "selected_paper_id=ALIGNX" in second["query_contract"]["notes"]
    assert second["verification_report"]["status"] == "pass"
    assert "L_{\\mathrm{PBA}}" in second["answer"]
    assert "\\tilde{P}" in second["answer"]
    assert "PersonaDual" not in second["answer"]
    assert "L^{\\mathrm{CLIP}}" not in second["answer"]


def test_formula_title_correction_keeps_previous_formula_goal(tmp_path: Path) -> None:
    agent, retriever = _build_agent(tmp_path)
    retriever._block_docs.append(
        Document(
            page_content=(
                "Preference-bridged alignment. PBA models preference directions explicitly. "
                "LPBA = - log σ β(log πϕ(yw|x,P̃)/πref(yw|x,P̃) - log πϕ(yl|x,P̃)/πref(yl|x,P̃))."
            ),
            metadata={
                "doc_id": "block-alignx-pba-formula-title",
                "paper_id": "ALIGNX",
                "title": "From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
                "authors": "Li et al.",
                "year": "2025",
                "tags": "alignx||personalization",
                "file_path": "/tmp/alignx.pdf",
                "page": 6,
                "block_type": "page_text",
                "caption": "",
                "bbox": "",
                "formula_hint": 1,
            },
        )
    )
    retriever._rebuild_lookup_indexes()
    session_id = "pba-title-correction"

    agent.chat(query="PBA的公式是什么", session_id=session_id)
    result = agent.chat(
        query="在From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment中啊",
        session_id=session_id,
    )

    assert result["query_contract"]["relation"] == "formula_lookup"
    assert result["query_contract"]["targets"] == ["PBA"]
    assert "selected_paper_id=ALIGNX" in result["query_contract"]["notes"]
    assert result["verification_report"]["status"] == "pass"
    assert "L_{\\mathrm{PBA}}" in result["answer"]
    assert "User-level 的定义" not in result["answer"]


def test_formula_query_uses_active_paper_context_when_user_signals_followup(tmp_path: Path) -> None:
    agent, retriever = _build_agent(tmp_path)
    retriever._block_docs.append(
        Document(
            page_content=(
                "Preference-bridged alignment. PBA introduces a latent variable P̃. "
                "The final optimization objective is LPBA = - log σ β(log πϕ(yw|x,P̃)/πref(yw|x,P̃) "
                "- log πϕ(yl|x,P̃)/πref(yl|x,P̃))."
            ),
            metadata={
                "doc_id": "block-alignx-pba-formula-active-context",
                "paper_id": "ALIGNX",
                "title": "From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
                "authors": "Li et al.",
                "year": "2025",
                "tags": "alignx||personalization",
                "file_path": "/tmp/alignx.pdf",
                "page": 6,
                "block_type": "page_text",
                "caption": "",
                "bbox": "",
                "formula_hint": 1,
            },
        )
    )
    retriever._rebuild_lookup_indexes()
    session = agent.sessions.get("pba-active-alignx-context")
    session.set_active_research(
        relation="entity_definition",
        targets=["AlignX"],
        titles=["From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment"],
        requested_fields=["definition"],
        required_modalities=["page_text", "paper_card"],
        answer_shape="narrative",
        precision_requirement="high",
        clean_query="AlignX是什么",
    )
    agent.sessions.upsert(session)

    result = agent.chat(query="那PBA的公式是什么", session_id="pba-active-alignx-context")

    assert result["query_contract"]["relation"] == "formula_lookup"
    assert result["query_contract"]["targets"] == ["PBA"]
    assert "selected_paper_id=ALIGNX" in result["query_contract"]["notes"]
    assert result["verification_report"]["status"] == "pass"
    assert "L_{\\mathrm{PBA}}" in result["answer"]


def test_contextual_method_result_query_binds_to_active_paper(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = agent.sessions.get("alignx-method-results-followup")
    session.set_active_research(
        relation="paper_summary_results",
        targets=["AlignX"],
        titles=["From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment"],
        requested_fields=["summary", "results", "evidence"],
        required_modalities=["page_text", "paper_card", "table", "caption"],
        answer_shape="narrative",
        precision_requirement="high",
        clean_query="完整名字是 From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
    )
    agent.sessions.upsert(session)

    contract = agent._extract_query_contract(
        query="这篇论文中PBA和ICA的具体效果如何呢",
        session=session,
        mode="auto",
    )

    assert contract.relation == "metric_value_lookup"
    assert contract.targets == ["PBA", "ICA"]
    assert "selected_paper_id=ALIGNX" in contract.notes
    assert any(note.startswith("memory_title=From 1,000,000 Users") for note in contract.notes)
    assert "PersonaDual" not in contract.clean_query


def test_extract_query_contract_prefers_llm_tool_router_when_available(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = agent.sessions.get("llm-tool-router")

    def plan_messages(self: object, **_: object) -> dict[str, object]:
        return {
            "actions": ["need_corpus_search"],
            "tool_call_args": [
                {
                    "name": "need_corpus_search",
                    "args": {
                        "query": "PBA 准确率多少",
                        "targets": ["PBA"],
                        "confidence": 0.91,
                        "rationale": "table-backed metric question",
                    },
                }
            ],
        }

    agent.clients.invoke_tool_plan_messages = MethodType(plan_messages, agent.clients)

    contract = agent._extract_query_contract(query="PBA 准确率多少", session=session, mode="auto")

    assert contract.relation == "metric_value_lookup"
    assert contract.targets == ["PBA"]
    assert "llm_tool_router" in contract.notes
    assert "intent_confidence=0.91" in contract.notes
    assert "local_protected_explicit_target_metric" not in contract.notes


def test_extract_query_contract_falls_back_when_llm_tool_router_misses(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = agent.sessions.get("llm-tool-router-miss")

    def bad_plan_messages(self: object, **_: object) -> dict[str, object]:
        return {"actions": ["not_a_router_tool"], "tool_call_args": []}

    agent.clients.invoke_tool_plan_messages = MethodType(bad_plan_messages, agent.clients)

    contract = agent._extract_query_contract(query="PBA 准确率多少", session=session, mode="auto")

    assert contract.relation == "metric_value_lookup"
    assert contract.targets == ["PBA"]
    assert "llm_tool_router" not in contract.notes
    assert "local_protected_explicit_target_metric" in contract.notes


def test_extract_query_contract_can_disable_legacy_router_fallback(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = agent.sessions.get("llm-tool-router-disabled-fallback")

    def bad_plan_messages(self: object, **_: object) -> dict[str, object]:
        return {"actions": ["not_a_router_tool"], "tool_call_args": []}

    agent.clients.invoke_tool_plan_messages = MethodType(bad_plan_messages, agent.clients)
    agent.agent_settings = AgentSettings(legacy_intent_fallback_enabled=False)

    contract = agent._extract_query_contract(query="PBA 准确率多少", session=session, mode="auto")

    assert contract.interaction_mode == "conversation"
    assert contract.relation == "clarify_user_intent"
    assert contract.targets == []
    assert "legacy_intent_fallback_disabled" in contract.notes
    assert "local_protected_explicit_target_metric" not in contract.notes


def test_metric_definition_followup_reuses_active_metric_context(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = agent.sessions.get("metric-definition-followup")
    session.set_active_research(
        relation="metric_value_lookup",
        targets=["ICA", "PBA"],
        titles=["From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment"],
        requested_fields=["metric_value", "setting", "evidence"],
        required_modalities=["table", "caption", "page_text"],
        answer_shape="table",
        precision_requirement="exact",
        clean_query="ICA、PBA 方法在各数据集上的准确度是多少？",
    )
    agent.sessions.upsert(session)

    contract = agent._extract_query_contract(
        query="这个准确度是怎么定义的？",
        session=session,
        mode="auto",
    )

    assert contract.relation == "metric_value_lookup"
    assert contract.interaction_mode == "research"
    assert contract.continuation_mode == "followup"
    assert contract.targets == ["ICA", "PBA"]
    assert contract.requested_fields == ["metric_value", "metric_definition", "setting", "evidence"]
    assert contract.required_modalities == ["table", "caption", "page_text"]
    assert "metric_definition_followup" in contract.notes


def test_paper_scope_correction_reuses_previous_targets_in_named_paper(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    persona_title = "PersonaDual: Balancing Personalization and Objectivity via Adaptive Reasoning"
    session = agent.sessions.get("alignx-scope-correction")
    session.set_active_research(
        relation="metric_value_lookup",
        targets=["PBA", "ICA"],
        titles=[persona_title],
        requested_fields=["metric_value", "setting", "evidence"],
        required_modalities=["table", "caption", "page_text"],
        answer_shape="narrative",
        precision_requirement="exact",
        clean_query="PBA和ICA方法的结果分别如何？",
    )
    agent.sessions.upsert(session)

    contract = agent._extract_query_contract(
        query="我问的就是AlignX最初的论文中的",
        session=session,
        mode="auto",
    )

    assert contract.relation == "metric_value_lookup"
    assert contract.targets == ["PBA", "ICA"]
    assert "selected_paper_id=ALIGNX" in contract.notes
    assert any(note.startswith("memory_title=From 1,000,000 Users") for note in contract.notes)
    assert "PersonaDual" not in contract.clean_query


def test_fresh_formula_query_does_not_bind_stale_active_or_memory_paper(tmp_path: Path) -> None:
    agent, retriever = _build_agent(tmp_path)
    persona_title = "PersonaDual: Balancing Personalization and Objectivity via Adaptive Reasoning"
    retriever._paper_docs.append(
        Document(
            page_content=(
                "title: PersonaDual: Balancing Personalization and Objectivity via Adaptive Reasoning\n"
                "aliases: PersonaDual\n"
                "abstract_or_summary: PersonaDual discusses personalized alignment and mentions PBA as related context."
            ),
            metadata={
                "doc_id": "paper::PERSONADUAL",
                "paper_id": "PERSONADUAL",
                "title": persona_title,
                "authors": "Zhao et al.",
                "year": "2026",
                "tags": "personalization||alignment",
                "file_path": "/tmp/personadual.pdf",
                "aliases": "PersonaDual",
                "generated_summary": "PersonaDual discusses personalized alignment and mentions PBA as related context.",
                "abstract_note": "",
            },
        )
    )
    retriever._block_docs.append(
        Document(
            page_content="PersonaDual compares against PBA and includes a related objective discussion near formula-like notation.",
            metadata={
                "doc_id": "block-personadual-pba",
                "paper_id": "PERSONADUAL",
                "title": persona_title,
                "authors": "Zhao et al.",
                "year": "2026",
                "tags": "personalization||alignment",
                "file_path": "/tmp/personadual.pdf",
                "page": 4,
                "block_type": "page_text",
                "caption": "",
                "bbox": "",
                "formula_hint": 1,
            },
        )
    )
    retriever._rebuild_lookup_indexes()
    session = agent.sessions.get("pba-stale-personadual")
    session.set_active_research(
        relation="entity_definition",
        targets=["PersonaDual"],
        titles=[persona_title],
        requested_fields=["definition"],
        required_modalities=["page_text", "paper_card"],
        answer_shape="narrative",
        precision_requirement="high",
        clean_query="PersonaDual 是什么",
    )
    session.working_memory = {
        "target_bindings": {
            "pba": {
                "target": "PBA",
                "paper_id": "PERSONADUAL",
                "title": persona_title,
                "evidence_ids": ["block-personadual-pba"],
            }
        }
    }
    agent.sessions.upsert(session)

    result = agent.chat(query="PBA的公式是什么", session_id="pba-stale-personadual")

    notes = result["query_contract"]["notes"]
    assert "selected_paper_id=PERSONADUAL" not in notes
    assert "formula_contextual_paper_binding" not in notes
    assert "resolved_from_conversation_memory" not in notes
    assert "限定在论文《PersonaDual" not in result["query_contract"]["clean_query"]


def test_repeated_bare_formula_query_does_not_bind_stale_personadual_context(tmp_path: Path) -> None:
    agent, retriever = _build_agent(tmp_path)
    persona_title = "PersonaDual: Balancing Personalization and Objectivity via Adaptive Reasoning"
    retriever._paper_docs.append(
        Document(
            page_content=(
                "title: PersonaDual: Balancing Personalization and Objectivity via Adaptive Reasoning\n"
                "aliases: PersonaDual\n"
                "abstract_or_summary: PersonaDual mentions PBA in a personalized alignment comparison."
            ),
            metadata={
                "doc_id": "paper::PERSONADUAL",
                "paper_id": "PERSONADUAL",
                "title": persona_title,
                "authors": "Zhao et al.",
                "year": "2026",
                "tags": "personalization||alignment",
                "file_path": "/tmp/personadual.pdf",
                "aliases": "PersonaDual",
                "generated_summary": "PersonaDual mentions PBA in a personalized alignment comparison.",
                "abstract_note": "",
            },
        )
    )
    retriever._block_docs.append(
        Document(
            page_content="PersonaDual mentions PBA near an unrelated formula discussion.",
            metadata={
                "doc_id": "block-personadual-pba-repeat",
                "paper_id": "PERSONADUAL",
                "title": persona_title,
                "authors": "Zhao et al.",
                "year": "2026",
                "tags": "personalization||alignment",
                "file_path": "/tmp/personadual.pdf",
                "page": 4,
                "block_type": "page_text",
                "caption": "",
                "bbox": "",
                "formula_hint": 1,
            },
        )
    )
    retriever._rebuild_lookup_indexes()
    session = agent.sessions.get("pba-repeat-stale-personadual")
    session.set_active_research(
        relation="formula_lookup",
        targets=["PBA"],
        titles=[persona_title],
        requested_fields=["formula", "variable_explanation"],
        required_modalities=["page_text", "table"],
        answer_shape="bullets",
        precision_requirement="exact",
        clean_query=f"PBA 的公式是什么？限定在论文《{persona_title}》中查找。",
    )
    session.turns.append(
        SessionTurn(
            query="PBA公式是什么？",
            answer="不能确认公式",
            relation="formula_lookup",
            interaction_mode="research",
            targets=["PBA"],
            titles=[persona_title],
            requested_fields=["formula", "variable_explanation"],
            required_modalities=["page_text", "table"],
            answer_shape="bullets",
            precision_requirement="exact",
        )
    )
    session.working_memory = {
        "target_bindings": {
            "pba": {
                "target": "PBA",
                "paper_id": "PERSONADUAL",
                "title": persona_title,
                "evidence_ids": ["block-personadual-pba-repeat"],
            }
        }
    }
    agent.sessions.upsert(session)

    result = agent.chat(query="PBA公式是什么？", session_id="pba-repeat-stale-personadual")

    notes = result["query_contract"]["notes"]
    assert "selected_paper_id=PERSONADUAL" not in notes
    assert "formula_contextual_paper_binding" not in notes
    assert "resolved_from_conversation_memory" not in notes
    assert "限定在论文《PersonaDual" not in result["query_contract"]["clean_query"]
    assert "我已经限定在《PersonaDual" not in result["answer"]


def test_schema_claim_solver_is_not_used_for_formula_or_followup(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    formula_contract = QueryContract(
        clean_query="PBA公式是什么",
        relation="formula_lookup",
        targets=["PBA"],
        requested_fields=["formula", "variable_explanation"],
        required_modalities=["page_text", "table"],
    )
    followup_contract = QueryContract(
        clean_query="AlignX数据集有后续工作吗？",
        relation="followup_research",
        targets=["AlignX"],
        requested_fields=["followup_papers", "relationship", "evidence"],
        required_modalities=["paper_card", "page_text"],
    )

    assert not agent._should_use_schema_claim_solver(contract=formula_contract, plan=agent._build_research_plan(formula_contract))
    assert not agent._should_use_schema_claim_solver(contract=followup_contract, plan=agent._build_research_plan(followup_contract))


def test_paper_recommendation_query_does_not_fall_into_topology_followup(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    agent.chat(query="POPI的核心结论是什么，实验结果如何？", session_id="pref-learning-demo")
    result = agent.chat(query="如果我想学习偏好学习，我应该看哪些论文", session_id="pref-learning-demo")

    assert result["query_contract"]["relation"] == "paper_recommendation"
    assert result["query_contract"]["targets"] == ["偏好学习"]
    assert "推荐阅读" in result["answer"]


def test_conversation_turn_does_not_clear_active_research_memory(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    agent.chat(query="DPO的公式是什么？", session_id="memory-demo-2")
    agent.chat(query="你是谁", session_id="memory-demo-2")
    contract = agent._extract_query_contract(
        query="变量解释呢",
        session=agent.sessions.get("memory-demo-2"),
        mode="auto",
    )

    assert contract.relation == "formula_lookup"
    assert contract.continuation_mode == "followup"
    assert contract.targets == ["DPO"]


def test_figure_signal_score_detects_benchmark_rich_caption(tmp_path: Path) -> None:
    score = figure_signal_score(
        "Figure 1 | Benchmark performance on AIME, Codeforces, GPQA, MATH-500, MMLU and SWE-bench."
    )

    assert score >= 6


def test_verifier_requests_retry_when_metric_claim_missing(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="AlignX的PBA表现如何？",
        relation="metric_value_lookup",
        targets=["AlignX"],
    )
    plan = ResearchPlan(
        paper_recall_mode="anchor_first",
        paper_limit=4,
        evidence_limit=8,
        solver_sequence=["table_solver", "text_solver"],
        required_claims=["metric_value"],
        retry_budget=1,
    )

    report = agent._verify_claims(contract=contract, plan=plan, claims=[], papers=[], evidence=[])

    assert report.status == "retry"
    assert "metric_value" in report.missing_fields


def test_research_chat_runs_through_agent_tool_loop(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    result = agent.chat(query="AlignX是什么")
    nodes = [step["node"] for step in result["execution_steps"]]

    assert "agent_tool:search_corpus" in nodes
    assert "agent_tool:compose" in nodes
    assert "agent_reflection" in nodes
    assert "broad_paper_recall" not in nodes
    assert result["needs_human"] is False


def test_agent_loop_detects_ambiguity_before_final_answer(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    result = agent.chat(query="PBA是什么")
    nodes = [step["node"] for step in result["execution_steps"]]

    assert result["needs_human"] is True
    assert "agent_tool:ask_human" in nodes
    assert result["verification_report"]["recommended_action"] == "clarify_ambiguous_entity"
    assert len(result["clarification_options"]) >= 2
    first_option = result["clarification_options"][0]
    assert first_option["schema_version"] == "clarification_option.v1"
    assert first_option["option_id"]
    assert first_option["kind"] == "acronym_meaning"
    assert first_option["label"]
    assert first_option["target"] == "PBA"
    assert first_option["source_relation"]
    assert isinstance(first_option["source_requested_fields"], list)


def test_llm_disambiguation_judge_auto_resolves_high_confidence_formula_candidate(tmp_path: Path) -> None:
    class AutoJudgeClients(StubModelClients):
        def invoke_json(self, *, system_prompt: str, human_prompt: str, fallback: object) -> object:
            if "候选消歧裁判器" in system_prompt:
                payload = json.loads(human_prompt)
                candidates = [item for item in list(payload.get("candidate_options", []) or []) if isinstance(item, dict)]
                selected = next(
                    item
                    for item in candidates
                    if str(item.get("title", "")).startswith("Direct Preference Optimization:")
                )
                return {
                    "decision": "auto_resolve",
                    "selected_option_id": selected["option_id"],
                    "selected_paper_id": selected["paper_id"],
                    "confidence": 0.91,
                    "reason": "The query asks for the target paper's core formula, and this candidate title directly matches the method paper while other candidates only mention or use it.",
                    "rejected_options": [
                        {
                            "option_id": item.get("option_id", ""),
                            "reason": "This candidate only mentions or applies the target in context.",
                        }
                        for item in candidates
                        if item.get("option_id") != selected.get("option_id")
                    ],
                }
            if "论文公式抽取器" in system_prompt:
                payload = json.loads(human_prompt)
                evidence = [item for item in list(payload.get("evidence", []) or []) if isinstance(item, dict)]
                evidence_ids = [str(item.get("doc_id", "")) for item in evidence if str(item.get("doc_id", ""))]
                return {
                    "formula_latex": r"\log \sigma \left(\beta \log \frac{\pi_{\theta}(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta \log \frac{\pi_{\theta}(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)",
                    "formula_format": "latex",
                    "variables": [
                        {"symbol": r"\pi_{\theta}", "description": "当前待优化语言模型策略。"},
                        {"symbol": r"\pi_{\mathrm{ref}}", "description": "参考策略。"},
                        {"symbol": r"y_w", "description": "偏好回答。"},
                        {"symbol": r"y_l", "description": "非偏好回答。"},
                        {"symbol": r"\beta", "description": "控制偏离参考策略强度的参数。"},
                    ],
                    "evidence_ids": evidence_ids[:1],
                    "confidence": 0.94,
                }
            return super().invoke_json(system_prompt=system_prompt, human_prompt=human_prompt, fallback=fallback)

    agent, retriever = _build_agent(tmp_path)
    agent.clients = AutoJudgeClients()
    retriever._paper_docs.extend(
        [
            Document(
                page_content="title: Tulu 3: Pushing Frontiers in Open Language Model Post-Training\naliases: Tulu 3\nabstract_or_summary: Tulu 3 uses supervised finetuning and Direct Preference Optimization (DPO).",
                metadata={
                    "doc_id": "paper::TULU3",
                    "paper_id": "TULU3",
                    "title": "Tulu 3: Pushing Frontiers in Open Language Model Post-Training",
                    "authors": "Ai2",
                    "year": "2025",
                    "tags": "post-training",
                    "file_path": "/tmp/tulu3.pdf",
                    "aliases": "Tulu 3",
                    "generated_summary": "Tulu 3 uses Direct Preference Optimization during post-training.",
                    "abstract_note": "",
                },
            ),
            Document(
                page_content="title: ComPO: Community Preferences for Language Model Personalization\naliases: ComPO\nabstract_or_summary: ComPO adopts direct preference optimization (DPO) for personalized preference learning.",
                metadata={
                    "doc_id": "paper::COMPO",
                    "paper_id": "COMPO",
                    "title": "ComPO: Community Preferences for Language Model Personalization",
                    "authors": "Zhou et al.",
                    "year": "2024",
                    "tags": "personalization",
                    "file_path": "/tmp/compo.pdf",
                    "aliases": "ComPO",
                    "generated_summary": "ComPO adopts direct preference optimization as a training component.",
                    "abstract_note": "",
                },
            ),
        ]
    )
    retriever._block_docs.extend(
        [
            Document(
                page_content="Our main contribution is Direct Preference Optimization (DPO). DPO can optimize a policy with a binary cross entropy objective.",
                metadata={
                    "doc_id": "block-dpo-intro",
                    "paper_id": "DPO",
                    "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
                    "authors": "Rafailov et al.",
                    "year": "2024",
                    "tags": "alignment||preference",
                    "file_path": "/tmp/dpo.pdf",
                    "page": 1,
                    "block_type": "page_text",
                    "caption": "",
                    "bbox": "",
                    "formula_hint": 0,
                },
            ),
            Document(
                page_content=(
                    "log sigma beta log pi_theta(y_w|x)/pi_ref(y_w|x) - beta log pi_theta(y_l|x)/pi_ref(y_l|x). "
                    "This way, we fit an implicit reward using an alternative parameterization."
                ),
                metadata={
                    "doc_id": "block-dpo-formula",
                    "paper_id": "DPO",
                    "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
                    "authors": "Rafailov et al.",
                    "year": "2024",
                    "tags": "alignment||preference",
                    "file_path": "/tmp/dpo.pdf",
                    "page": 4,
                    "block_type": "page_text",
                    "caption": "",
                    "bbox": "",
                    "formula_hint": 1,
                },
            ),
            Document(
                page_content="The training algorithms include supervised finetuning and Direct Preference Optimization (DPO).",
                metadata={
                    "doc_id": "block-tulu3-dpo",
                    "paper_id": "TULU3",
                    "title": "Tulu 3: Pushing Frontiers in Open Language Model Post-Training",
                    "authors": "Ai2",
                    "year": "2025",
                    "tags": "post-training",
                    "file_path": "/tmp/tulu3.pdf",
                    "page": 5,
                    "block_type": "page_text",
                    "caption": "",
                    "bbox": "",
                    "formula_hint": 0,
                },
            ),
            Document(
                page_content="Rafailov et al. proposed direct preference optimization (DPO), which ComPO adopts for stable personalization.",
                metadata={
                    "doc_id": "block-compo-dpo",
                    "paper_id": "COMPO",
                    "title": "ComPO: Community Preferences for Language Model Personalization",
                    "authors": "Zhou et al.",
                    "year": "2024",
                    "tags": "personalization",
                    "file_path": "/tmp/compo.pdf",
                    "page": 3,
                    "block_type": "page_text",
                    "caption": "",
                    "bbox": "",
                    "formula_hint": 0,
                },
            ),
        ]
    )
    retriever._rebuild_lookup_indexes()

    result = agent.chat(query="帮我看看 DPO 这篇论文的核心公式", session_id="dpo-auto-judge")

    assert result["needs_human"] is False
    assert "auto_resolved_by_llm_judge" in result["query_contract"]["notes"]
    assert "selected_paper_id=DPO" in result["query_contract"]["notes"]
    assert result["verification_report"]["status"] == "pass"
    assert "我按最匹配的候选《Direct Preference Optimization: Your Language Model is Secretly a Reward Model》来回答。" in result["answer"]
    assert result["citations"][0]["paper_id"] == "DPO"


def test_llm_disambiguation_judge_recommends_but_keeps_human_gate(tmp_path: Path) -> None:
    class RecommendJudgeClients(StubModelClients):
        def invoke_json(self, *, system_prompt: str, human_prompt: str, fallback: object) -> object:
            if "候选消歧裁判器" in system_prompt:
                payload = json.loads(human_prompt)
                candidates = [item for item in list(payload.get("candidate_options", []) or []) if isinstance(item, dict)]
                selected = candidates[1]
                return {
                    "decision": "ask_human",
                    "selected_option_id": selected["option_id"],
                    "selected_paper_id": selected["paper_id"],
                    "confidence": 0.72,
                    "reason": "This candidate looks more direct, but the query is still under-specified.",
                    "rejected_options": [],
                }
            return super().invoke_json(system_prompt=system_prompt, human_prompt=human_prompt, fallback=fallback)

    agent, _ = _build_agent(tmp_path)
    agent.clients = RecommendJudgeClients()

    result = agent.chat(query="PBA是什么", session_id="pba-recommend-judge")

    assert result["needs_human"] is True
    assert result["verification_report"]["status"] == "clarify"
    assert result["clarification_options"][0]["judge_recommended"] is True
    assert result["clarification_options"][0]["disambiguation_confidence"] == 0.72
    assert "推荐候选" in result["answer"]


def test_formula_clarification_choice_preserves_answer_slots() -> None:
    contract = contract_from_selected_clarification_option(
        clean_query="选择第二个",
        target="XYZ",
        selected={
            "option_id": "formula-choice",
            "target": "XYZ",
            "meaning": "Example Method",
            "title": "Example Method: A Paper",
            "paper_id": "EXAMPLE",
            "source_relation": "formula_lookup",
            "source_requested_fields": ["formula", "variable_explanation", "source"],
            "source_answer_slots": ["formula"],
        },
    )

    assert contract.relation == "formula_lookup"
    assert contract.answer_slots == ["formula"]
    assert "answer_slot=formula" in contract.notes


def test_human_choice_resolves_pending_ambiguity_by_button_payload(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    first = agent.chat(query="PBA是什么", session_id="pba-choice")
    second = agent.chat(
        query="2. preference-bridged alignment",
        session_id="pba-choice",
        clarification_choice=first["clarification_options"][1],
    )

    assert first["needs_human"] is True
    assert second["needs_human"] is False
    assert "CURP" not in second["answer"]
    assert second["citations"]
    assert second["citations"][0]["title"] == "From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment"


def test_human_choice_resolves_pending_ambiguity_by_option_id(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    first = agent.chat(query="PBA是什么", session_id="pba-choice-option-id")
    selected = first["clarification_options"][1]
    second = agent.chat(
        query="选择第二个",
        session_id="pba-choice-option-id",
        clarification_choice={"option_id": selected["option_id"]},
    )

    assert first["needs_human"] is True
    assert second["needs_human"] is False
    assert second["citations"]
    assert second["citations"][0]["title"] == "From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment"


def test_human_choice_resolves_pending_ambiguity_from_text(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    agent.chat(query="PBA是什么", session_id="pba-choice-text")
    result = agent.chat(query="我说第二个", session_id="pba-choice-text")

    assert result["needs_human"] is False
    assert "CURP" not in result["answer"]
    assert result["citations"][0]["title"] == "From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment"


def test_repeated_clarification_hits_limit_and_answers_best_effort(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    first = agent.chat(query="PBA是什么", session_id="pba-clarify-limit")
    second = agent.chat(query="PBA是什么", session_id="pba-clarify-limit")

    assert first["needs_human"] is True
    assert second["needs_human"] is False
    assert "clarification_limit_reached" in second["query_contract"]["notes"]


def test_final_answer_composer_streams_llm_deltas(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="AlignX是什么",
        relation="entity_definition",
        targets=["AlignX"],
        requested_fields=["definition"],
        required_modalities=["page_text", "paper_card"],
    )
    papers = [
        CandidatePaper(
            paper_id="ALIGNX",
            title="From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
            year="2025",
            doc_ids=["paper::ALIGNX"],
        )
    ]
    evidence = [
        EvidenceBlock(
            doc_id="block-alignx-definition",
            paper_id="ALIGNX",
            title="From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
            file_path="/tmp/alignx.pdf",
            page=1,
            block_type="page_text",
            snippet="AlignX is a large-scale dataset and benchmark containing over 1.3 million personalized preference examples.",
            metadata={"authors": "Li et al.", "year": "2025", "tags": "alignx||personalization"},
        )
    ]
    claims = [
        Claim(
            claim_type="entity_definition",
            entity="AlignX",
            value="偏好数据集",
            structured_data={"definition_lines": ["AlignX is a large-scale dataset and benchmark."]},
            evidence_ids=["block-alignx-definition"],
            paper_ids=["ALIGNX"],
        )
    ]
    chunks: list[str] = []

    answer, citations = agent._compose_answer(
        contract=contract,
        claims=claims,
        evidence=evidence,
        papers=papers,
        verification=VerificationReport(status="pass"),
        stream_callback=chunks.append,
    )

    assert chunks
    assert "".join(chunks).strip() == answer
    assert citations


def test_final_answer_composer_forwards_optional_logprobs(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    contract = QueryContract(
        clean_query="AlignX是什么",
        relation="entity_definition",
        targets=["AlignX"],
        requested_fields=["definition"],
        required_modalities=["page_text", "paper_card"],
    )
    papers = [
        CandidatePaper(
            paper_id="ALIGNX",
            title="From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
            year="2025",
            doc_ids=["paper::ALIGNX"],
        )
    ]
    evidence = [
        EvidenceBlock(
            doc_id="block-alignx-definition",
            paper_id="ALIGNX",
            title="From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment",
            file_path="/tmp/alignx.pdf",
            page=1,
            block_type="page_text",
            snippet="AlignX is a large-scale dataset and benchmark.",
            metadata={"authors": "Li et al.", "year": "2025"},
        )
    ]
    claims = [
        Claim(
            claim_type="entity_definition",
            entity="AlignX",
            value="偏好数据集",
            structured_data={"definition_lines": ["AlignX is a large-scale dataset and benchmark."]},
            evidence_ids=["block-alignx-definition"],
            paper_ids=["ALIGNX"],
        )
    ]
    chunks: list[str] = []
    logprobs: list[float] = []
    captured: dict[str, object] = {}

    def stream_text(
        *,
        system_prompt: str,
        human_prompt: str,
        on_delta,
        on_logprobs=None,
        request_logprobs: bool = False,
        fallback: str = "",
    ) -> str:
        captured["request_logprobs"] = request_logprobs
        captured["has_logprob_callback"] = on_logprobs is not None
        on_delta("## 定义\n\nAlignX 是偏好数据集。")
        if on_logprobs is not None:
            on_logprobs([-0.1, -0.2])
        return "## 定义\n\nAlignX 是偏好数据集。"

    agent.clients.stream_text = stream_text

    answer, citations = agent._compose_answer(
        contract=contract,
        claims=claims,
        evidence=evidence,
        papers=papers,
        verification=VerificationReport(status="pass"),
        stream_callback=chunks.append,
        logprob_callback=logprobs.extend,
        request_logprobs=True,
    )

    assert captured == {"request_logprobs": True, "has_logprob_callback": True}
    assert logprobs == [-0.1, -0.2]
    assert "".join(chunks).strip() == answer
    assert citations


def test_correction_agent_reflects_previous_answer_before_search(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)
    session = agent.sessions.get("pba-agent-loop")
    session.set_active_research(
        relation="entity_definition",
        targets=["PBA"],
        titles=["CURP: Codebook-based Continuous User Representation for Personalized Generation with LLMs"],
        requested_fields=["definition", "mechanism", "role_in_context"],
        required_modalities=["page_text", "paper_card"],
        answer_shape="narrative",
        precision_requirement="high",
        clean_query="PBA是什么",
    )
    agent.sessions.upsert(session)

    result = agent.chat(query="我是说另一个PBA，不是这个", session_id="pba-agent-loop")
    nodes = [step["node"] for step in result["execution_steps"]]

    assert nodes.index("agent_tool:read_memory") < nodes.index("agent_tool:search_corpus")
    assert "CURP" not in result["answer"]


def test_library_browser_lists_papers_and_previews_citation(tmp_path: Path) -> None:
    agent, retriever = _build_agent(tmp_path)
    service = LibraryBrowserService(settings=agent.settings, retriever=retriever)

    library = service.list_library()
    paper_preview = service.paper_preview("ALIGNX")
    citation_preview = service.citation_preview(doc_id="block-alignx-pba-definition")

    assert library["total_papers"] >= 1
    assert any(category["papers"] for category in library["categories"])
    assert paper_preview is not None
    assert paper_preview["paper"]["title"].startswith("From 1,000,000 Users")
    assert citation_preview is not None
    assert citation_preview["paper_id"] == "ALIGNX"
    assert citation_preview["page"] == 6
