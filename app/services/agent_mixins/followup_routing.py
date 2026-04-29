from __future__ import annotations

import json

from app.domain.models import QueryContract, SessionContext
from app.services.research_planning import research_plan_goals


class FollowupRoutingMixin:
    def _refine_followup_contract(self, *, contract: QueryContract, session: SessionContext) -> QueryContract:
        contract = self._llm_refine_contextual_contract(contract=contract, session=session)
        if contract.continuation_mode != "followup":
            return contract
        active = session.effective_active_research()
        query = contract.clean_query
        targets = list(contract.targets)
        target = targets[0] if targets else (active.targets[0] if active.targets else "")
        requested_fields = list(contract.requested_fields)
        required_modalities = list(contract.required_modalities)
        answer_shape = contract.answer_shape
        precision_requirement = contract.precision_requirement
        notes = list(contract.notes)
        goals = research_plan_goals(contract)

        if goals & {"entity_type", "role_in_context"}:
            refined_fields, resolved_query = self._infer_entity_followup_focus(
                query=query,
                target=target,
                current_fields=requested_fields,
                previous_fields=active.requested_fields,
            )
            if refined_fields != requested_fields:
                requested_fields = refined_fields
                if "followup_detail" not in notes:
                    notes.append("followup_detail")
            if resolved_query and resolved_query != query:
                query = resolved_query
            if any(field in {"formula", "objective", "variable_explanation"} for field in requested_fields):
                for modality in ["table", "page_text"]:
                    if modality not in required_modalities:
                        required_modalities.append(modality)
            else:
                if "page_text" not in required_modalities:
                    required_modalities.append("page_text")
            if any(field in {"mechanism", "workflow", "objective", "reward_signal"} for field in requested_fields):
                answer_shape = "bullets"
                precision_requirement = "high"

        if goals & {"definition", "mechanism", "examples"} and "entity_type" not in goals:
            if self._is_vague_followup_query(query) and target:
                requested_fields = ["definition", "mechanism", "examples"]
                query = f"{target} 的具体含义、工作方式和典型用法是什么？"
                if "followup_detail" not in notes:
                    notes.append("followup_detail")
                answer_shape = "bullets"
                precision_requirement = "high"

        return contract.model_copy(
            update={
                "clean_query": query,
                "targets": targets or ([target] if target else []),
                "requested_fields": requested_fields,
                "required_modalities": required_modalities,
                "answer_shape": answer_shape,
                "precision_requirement": precision_requirement,
                "notes": notes,
            }
        )

    def _llm_refine_contextual_contract(self, *, contract: QueryContract, session: SessionContext) -> QueryContract:
        active = session.effective_active_research()
        if self.clients.chat is None or not active.relation:
            return contract
        needs_contextual_refine = "needs_contextual_refine" in contract.notes
        if contract.relation not in {"clarify_user_intent", "correction_without_context"} and not needs_contextual_refine:
            return contract
        if contract.continuation_mode != "followup" and not self._looks_like_contextual_followup_query(contract.clean_query):
            return contract
        system_prompt = (
                "你是论文研究助手的研究追问合同修复器。"
                "你的任务是基于【当前路由结果】和【当前活跃研究上下文】，判断用户这句话是否其实是在延续上一轮研究任务，"
                "并在需要时把它修正成更合适的 QueryContract。"
                "请只输出 JSON，字段为 relation, continuation_mode, targets, requested_fields, required_modalities, "
                "answer_shape, precision_requirement, notes, rewritten_query。"
                "你会收到 conversation_context，它包含保留下来的完整多轮对话、上一轮回答、pending clarification 和工作记忆；"
                "如果当前问题明显引用了历史里的目标、选择或比较对象，要继承这些含义，而不是重新猜。"
                "优先处理这几类情况："
                "1. 用户在质疑上一轮答案，例如“最早不是在这里吧”“你确定吗”“这个来源不对吧”；"
                "2. 用户用很短的追问继续上一轮，例如“具体呢”“那变量呢”“出处呢”；"
                "3. 用户没有重复实体名，但明显在沿用上一轮研究目标。"
                "如果用户沿用上一轮推荐/提到的论文并追问“具体说了啥”“讲了什么”“核心结论”“实验结果”“方法细节”，"
                "必须把它改成 research + paper_summary_results，并把 targets 设为 conversation_context 中对应的论文标题；"
                "记忆只用于解析指代，不能直接用 memory_followup 回答论文正文内容。"
                "如果用户是在追问某技术/术语最早由哪篇论文提出，优先改成 origin_lookup。"
                "如果用户是在纠正或核验上一轮 entity_definition 的来源，也优先考虑 origin_lookup 或带有 supporting_paper 的研究合同，"
                "不要简单退回澄清。"
                "如果当前路由已经合理，也可以原样返回。"
        )
        human_payload = {
            "current_query": contract.clean_query,
            "current_contract": contract.model_dump(),
            "conversation_context": self._session_conversation_context(session, max_chars=12000),
            "active_research_context": session.active_research_context_payload(),
            "recent_turns": [
                {
                    "query": turn.query,
                    "relation": turn.relation,
                    "targets": turn.targets,
                    "requested_fields": turn.requested_fields,
                }
                for turn in session.turns[-3:]
            ],
        }
        invoke_json_messages = getattr(self.clients, "invoke_json_messages", None)
        if callable(invoke_json_messages):
            payload = invoke_json_messages(
                system_prompt=system_prompt,
                messages=[
                    *self._session_llm_history_messages(session),
                    {"role": "user", "content": json.dumps(human_payload, ensure_ascii=False)},
                ],
                fallback={},
            )
        else:
            payload = self.clients.invoke_json(
                system_prompt=system_prompt,
                human_prompt=json.dumps(human_payload, ensure_ascii=False),
                fallback={},
            )
        if not isinstance(payload, dict) or not payload:
            return contract
        allowed_relations = {
            "greeting",
            "self_identity",
            "capability",
            "library_status",
            "library_recommendation",
            "memory_followup",
            "memory_synthesis",
            "library_citation_ranking",
            "clarify_user_intent",
            "correction_without_context",
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
        }
        conversation_relations = {
            "greeting",
            "self_identity",
            "capability",
            "library_status",
            "library_recommendation",
            "memory_followup",
            "memory_synthesis",
            "library_citation_ranking",
            "clarify_user_intent",
            "correction_without_context",
        }
        relation = str(payload.get("relation", contract.relation)).strip()
        if relation not in allowed_relations:
            relation = contract.relation
        continuation_mode = str(payload.get("continuation_mode", contract.continuation_mode)).strip().lower()
        if continuation_mode not in {"fresh", "followup", "context_switch"}:
            continuation_mode = contract.continuation_mode
        raw_targets = payload.get("targets", contract.targets)
        if isinstance(raw_targets, list):
            targets = [str(item).strip() for item in raw_targets if str(item).strip()]
        elif isinstance(raw_targets, str) and raw_targets.strip():
            targets = [raw_targets.strip()]
        else:
            targets = list(contract.targets)
        if not targets and continuation_mode == "followup":
            targets = list(active.targets)
        raw_requested_fields = payload.get("requested_fields", contract.requested_fields)
        requested_fields = (
            [str(item).strip() for item in raw_requested_fields if str(item).strip()]
            if isinstance(raw_requested_fields, list)
            else list(contract.requested_fields)
        )
        if relation not in conversation_relations and not requested_fields:
            requested_fields = list(contract.requested_fields) or list(active.requested_fields) or ["answer"]
        targets = self._normalize_contract_targets(targets=targets, requested_fields=requested_fields)
        raw_required_modalities = payload.get("required_modalities", contract.required_modalities)
        required_modalities = self._normalize_modalities(
            [str(item).strip() for item in raw_required_modalities if str(item).strip()]
            if isinstance(raw_required_modalities, list)
            else list(contract.required_modalities),
            relation=relation,
        )
        if relation not in conversation_relations and not required_modalities:
            required_modalities = list(contract.required_modalities) or list(active.required_modalities) or ["page_text", "paper_card"]
        answer_shape = str(payload.get("answer_shape", contract.answer_shape)).strip().lower()
        if answer_shape not in {"bullets", "narrative", "table"}:
            answer_shape = contract.answer_shape
        precision_requirement = str(payload.get("precision_requirement", contract.precision_requirement)).strip().lower()
        if precision_requirement not in {"exact", "high", "normal"}:
            precision_requirement = contract.precision_requirement
        rewritten_query = str(payload.get("rewritten_query", "") or "").strip() or contract.clean_query
        raw_notes = payload.get("notes", [])
        notes = list(contract.notes)
        if isinstance(raw_notes, list):
            for item in raw_notes:
                note = str(item).strip()
                if note and note not in notes:
                    notes.append(note)
        if payload:
            if "llm_context_refined" not in notes:
                notes.append("llm_context_refined")
        return contract.model_copy(
            update={
                "clean_query": rewritten_query,
                "interaction_mode": "conversation" if relation in conversation_relations else "research",
                "relation": relation,
                "continuation_mode": continuation_mode,
                "targets": targets,
                "requested_fields": requested_fields if relation not in conversation_relations else [],
                "required_modalities": required_modalities if relation not in conversation_relations else [],
                "answer_shape": answer_shape,
                "precision_requirement": precision_requirement,
                "notes": notes,
            }
        )

    def _infer_entity_followup_focus(
        self,
        *,
        query: str,
        target: str,
        current_fields: list[str],
        previous_fields: list[str],
    ) -> tuple[list[str], str]:
        normalized_query = self._normalize_lookup_text(query)
        generic_fields = {"definition", "applications", "key_features", "answer", "summary"}
        current_keys = {self._normalize_lookup_text(item) for item in current_fields if item}
        previous_keys = {self._normalize_lookup_text(item) for item in previous_fields if item}
        formula_cues = {"公式", "objective", "loss", "advantage", "变量", "推导"}
        detail_cues = {"具体", "细节", "详细", "原理", "机制", "怎么", "如何", "工作", "流程", "样子", "what is it like"}
        if any(cue in normalized_query for cue in formula_cues):
            resolved_query = f"{target} 的目标函数、关键公式和变量含义是什么？" if target else query
            return ["formula", "objective", "variable_explanation"], resolved_query
        if any(cue in normalized_query for cue in detail_cues) or (
            self._is_vague_followup_query(query) and current_keys <= generic_fields and previous_keys <= generic_fields
        ):
            resolved_query = f"{target} 的具体机制、工作流程和奖励/优化目标是什么？" if target else query
            return ["mechanism", "workflow", "objective", "reward_signal"], resolved_query
        return current_fields, query

    @staticmethod
    def _is_vague_followup_query(query: str) -> bool:
        normalized = " ".join(str(query or "").lower().split())
        if not normalized:
            return False
        vague_patterns = [
            "具体是什么样",
            "具体是怎样",
            "具体呢",
            "详细一点",
            "展开讲讲",
            "具体说说",
            "再具体一点",
            "怎么工作的",
            "如何工作的",
            "工作原理",
        ]
        if any(token in normalized for token in vague_patterns):
            return True
        return len(normalized) <= 12 and any(token in normalized for token in ["具体", "详细", "怎么", "如何", "原理", "机制"])

    @staticmethod
    def _looks_like_contextual_followup_query(query: str) -> bool:
        normalized = " ".join(str(query or "").lower().split())
        if not normalized:
            return False
        challenge_tokens = [
            "最早",
            "起源",
            "最初",
            "首次",
            "第一个",
            "第一篇",
            "提出",
            "出处",
            "来源",
            "证据",
            "不对",
            "不是",
            "确定",
            "真的吗",
        ]
        if any(token in normalized for token in challenge_tokens):
            return True
        if len(normalized) <= 18 and any(token in normalized for token in ["这里", "这个", "那篇", "上一条", "上一轮"]):
            return True
        return False
