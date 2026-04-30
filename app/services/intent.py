from __future__ import annotations

import json
import re
from typing import Any, Callable, Literal, cast

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.domain.models import QueryContract, SessionContext
from app.services.intent_marker_matching import MarkerProfile, query_matches_any
from app.services.intent_contract_adapter import (
    research_profile_slots,
    research_relation_from_slots,
    research_requirements_from_slots,
)
from app.services.library_intents import (
    is_citation_query,
    is_library_status_query,
    is_scoped_library_recommendation_query,
)
from app.services.memory_intents import (
    contains_ordinal_reference,
    is_memory_comparison_query,
    is_pdf_agent_topology_design_query,
    is_short_followup,
    looks_like_memory_reference,
    looks_like_recent_tool_result_reference,
)
from app.services.research_intents import (
    looks_like_metric_value_query,
    looks_like_origin_lookup_query,
    looks_like_summary_results_query,
    normalized_query_needs_external_search,
    research_answer_slots,
)
from app.services.query_shaping import fallback_query_targets, fallback_target_aliases

ConversationContextFn = Callable[[SessionContext], dict[str, Any]]
ConversationMessagesFn = Callable[[SessionContext], list[dict[str, str]]]
NormalizeTargetsFn = Callable[[list[str], list[str]], list[str]]


IntentKind = Literal["smalltalk", "meta_library", "research", "memory_op"]
TopicState = Literal["continue", "switch", "new"]
AnswerSlot = Literal[
    "greeting",
    "self_identity",
    "capability",
    "clarify",
    "library_status",
    "library_recommendation",
    "citation_ranking",
    "previous_rationale",
    "comparison",
    "definition",
    "entity_definition",
    "concept_definition",
    "formula",
    "origin",
    "followup_research",
    "paper_summary",
    "metric_value",
    "figure",
    "paper_recommendation",
    "topology_discovery",
    "topology_recommendation",
    "training_component",
    "general_answer",
]

INTENT_ROUTER_MARKERS: dict[str, MarkerProfile] = {
    "contextual_origin_refine": ("最早", "起源", "最初", "首次", "出处", "来源", "不对", "不是", "确定"),
    "strong_origin_refine": ("最早", "起源"),
    "self_identity": ("你是谁", "who are you", "你的身份"),
    "capability": ("你能做什么", "有什么功能", "capability", "abilities"),
    "previous_rationale": ("为什么选择", "为什么推荐", "推荐理由", "排序依据"),
}

class Intent(BaseModel):
    intent_kind: IntentKind = "research"
    topic_state: TopicState = "new"
    active_topic: str = ""
    needs_local_corpus: bool = True
    needs_web: bool = False
    refers_previous_turn: bool = False
    target_entities: list[str] = Field(default_factory=list)
    target_aliases: list[str] = Field(default_factory=list)
    user_goal: str = ""
    answer_slots: list[AnswerSlot] = Field(default_factory=lambda: ["general_answer"])
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    ambiguous_slots: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("target_entities", "target_aliases", "ambiguous_slots", "notes", mode="before")
    @classmethod
    def _normalize_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("answer_slots", mode="before")
    @classmethod
    def _normalize_slots(cls, value: Any) -> list[str]:
        if value is None:
            return ["general_answer"]
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return ["general_answer"]
        slots = [str(item).strip() for item in value if str(item).strip()]
        return slots or ["general_answer"]

    @model_validator(mode="after")
    def _sync_topic_state_compatibility(self) -> "Intent":
        if self.refers_previous_turn and self.topic_state == "new":
            self.topic_state = "continue"
        if self.topic_state == "continue":
            self.refers_previous_turn = True
        elif self.topic_state in {"switch", "new"}:
            self.refers_previous_turn = False
        return self


class IntentRecognizer:
    """Structured intent layer.

    The model is asked for orthogonal dimensions and slots, not a 22-way
    relation. QueryContract is still produced as a compatibility adapter for
    the existing retriever / solver / composer stack.
    """

    def __init__(
        self,
        *,
        clients: Any,
        conversation_context: ConversationContextFn,
        normalize_targets: NormalizeTargetsFn,
        conversation_messages: ConversationMessagesFn | None = None,
    ) -> None:
        self.clients = clients
        self.conversation_context = conversation_context
        self.conversation_messages = conversation_messages or (lambda session: [])
        self.normalize_targets = normalize_targets

    def contract_for_query(
        self,
        *,
        clean_query: str,
        session: SessionContext,
        extracted_targets: list[str],
    ) -> QueryContract:
        intent = self.recognize(
            clean_query=clean_query,
            session=session,
            extracted_targets=extracted_targets,
        )
        return self.to_query_contract(
            intent=intent,
            clean_query=clean_query,
            session=session,
            extracted_targets=extracted_targets,
        )

    def recognize(
        self,
        *,
        clean_query: str,
        session: SessionContext,
        extracted_targets: list[str],
    ) -> Intent:
        protected = self._protected_local_intent(clean_query, session=session)
        if protected is not None:
            return protected
        llm_intent = self._llm_intent(
            clean_query=clean_query,
            session=session,
            extracted_targets=extracted_targets,
        )
        if llm_intent is not None:
            return llm_intent
        return self._fallback_intent(
            clean_query=clean_query,
            session=session,
            extracted_targets=extracted_targets,
        )

    def to_query_contract(
        self,
        *,
        intent: Intent,
        clean_query: str,
        session: SessionContext,
        extracted_targets: list[str],
    ) -> QueryContract:
        slots = list(dict.fromkeys(intent.answer_slots or ["general_answer"]))
        requested_fields: list[str] = []
        required_modalities: list[str] = []
        relation = "general_question"
        interaction_mode: Literal["conversation", "research"] = "research"
        answer_shape = "narrative"
        precision_requirement: Literal["exact", "high", "normal"] = "high"
        target_aliases = list(dict.fromkeys(intent.target_aliases or []))
        targets = self.normalize_targets(
            list(intent.target_entities or extracted_targets),
            requested_fields,
        )
        active = session.effective_active_research()
        if not targets and intent.topic_state == "continue":
            targets = self.normalize_targets(list(active.targets), requested_fields)
        continuation_mode: Literal["fresh", "followup", "context_switch"] = {
            "continue": "followup",
            "switch": "context_switch",
            "new": "fresh",
        }.get(intent.topic_state, "fresh")  # type: ignore[assignment]
        notes = list(
            dict.fromkeys(
                [
                    "structured_intent",
                    f"intent_kind={intent.intent_kind}",
                    f"intent_confidence={intent.confidence:.2f}",
                    f"topic_state={intent.topic_state}",
                    *(["active_topic=" + intent.active_topic] if intent.active_topic else []),
                    *[f"answer_slot={slot}" for slot in slots],
                    *[f"target_alias={alias}" for alias in target_aliases],
                    *[f"ambiguous_slot={slot}" for slot in intent.ambiguous_slots],
                    *intent.notes,
                ]
            )
        )
        if intent.needs_web:
            notes.append("requires_external_tool")

        if intent.intent_kind == "smalltalk":
            interaction_mode = "conversation"
            relation = self._smalltalk_relation(slots)
            requested_fields = []
            required_modalities = []
            answer_shape = "bullets" if relation == "capability" else "narrative"
            precision_requirement = "normal"
            continuation_mode = "fresh"
            targets = []
        elif intent.intent_kind == "meta_library":
            interaction_mode = "conversation"
            requested_fields = []
            required_modalities = []
            precision_requirement = "exact" if "library_status" in slots else "normal"
            answer_shape = "table" if "citation_ranking" in slots else "bullets"
            if "citation_ranking" in slots:
                relation = "library_citation_ranking"
                requested_fields = ["citation_count_ranking"]
                continuation_mode = "followup" if session.turns else "fresh"
                intent.needs_web = True
                notes.extend(["external_metric", "citation_count_requires_web"])
            elif "library_recommendation" in slots:
                relation = "library_recommendation"
            else:
                relation = "library_status"
        elif intent.intent_kind == "memory_op" and not intent.needs_local_corpus:
            interaction_mode = "conversation"
            required_modalities = []
            precision_requirement = "normal"
            if "comparison" in slots:
                relation = "memory_synthesis"
                requested_fields = ["comparison", "synthesis"]
                answer_shape = "table"
            else:
                relation = "memory_followup"
                requested_fields = ["previous_tool_basis", "answer"]
                answer_shape = "narrative"
            continuation_mode = "followup"
        else:
            relation = self._research_relation(slots=slots, clean_query=clean_query, targets=targets)
            requested_fields, required_modalities, answer_shape, precision_requirement = self._research_requirements(
                slots=slots,
                targets=targets,
                clean_query=clean_query,
            )
            if intent.intent_kind == "memory_op":
                continuation_mode = "followup"
                notes.append("memory_resolved_research")

        needs_clarification = intent.intent_kind != "smalltalk" and (
            intent.confidence < 0.6 or bool(intent.ambiguous_slots)
        )
        if needs_clarification:
            clarification_notes = ["intent_needs_clarification", *intent.ambiguous_slots]
            if intent.confidence < 0.6:
                clarification_notes.append("low_intent_confidence")
            return QueryContract(
                clean_query=clean_query,
                interaction_mode="conversation",
                relation="clarify_user_intent",
                targets=targets,
                answer_slots=slots,
                requested_fields=[],
                required_modalities=[],
                answer_shape="narrative",
                precision_requirement="normal",
                continuation_mode="fresh",
                allow_web_search=False,
                notes=list(dict.fromkeys([*notes, *clarification_notes])),
            )

        return QueryContract(
            clean_query=clean_query,
            interaction_mode=interaction_mode,
            relation=relation,
            targets=targets,
            answer_slots=slots,
            requested_fields=requested_fields,
            required_modalities=required_modalities,
            answer_shape=answer_shape,
            precision_requirement=precision_requirement,
            continuation_mode=continuation_mode,
            allow_web_search=bool(intent.needs_web),
            notes=list(dict.fromkeys(notes)),
        )

    def _llm_intent(
        self,
        *,
        clean_query: str,
        session: SessionContext,
        extracted_targets: list[str],
    ) -> Intent | None:
        if getattr(self.clients, "chat", None) is None:
            return None
        payload = {
            "active_research_context": session.active_research_context_payload(),
            "last_relation": session.last_relation,
            "extracted_targets": extracted_targets,
            "schema": Intent.model_json_schema(),
        }
        context_json = json.dumps(payload, ensure_ascii=False)
        system_prompt = (
            "你是论文研究助手的结构化意图路由器。"
            "不要做 22 个 relation 单选；只输出一个 JSON Intent。"
            "Intent 由正交维度组成：intent_kind、needs_local_corpus、needs_web、"
            "topic_state、active_topic、refers_previous_turn、target_entities、target_aliases、"
            "user_goal、answer_slots、confidence、ambiguous_slots、notes。"
            "intent_kind 只能是 smalltalk/meta_library/research/memory_op。"
            "topic_state 只能是 continue/switch/new：continue 表示明确延续上一轮；"
            "switch 表示用户从活跃话题切到另一个话题；new 表示没有依赖上一轮的全新问题。"
            "topic_state 默认应为 new；不要因为存在 active_research_context 就默认 continue。"
            "只有当当前消息明确包含指代词（如 那、这、它、上一轮、刚才、上面那篇、this paper）"
            "或是对前一工具结果的序数追问（如 第三篇是啥、第二个、the third one），"
            "或是对前一论文/方法名的省略性追问（如 实验结果呢、公式呢、变量解释呢）时，才设为 continue。"
            "如果用户提到了与活跃 targets 完全不同的新实体名，必须设为 switch。"
            "只有 continue 才应继承 active_research_context 的 targets 或论文。"
            "active_topic 用一句话概括当前用户真正要问的话题。"
            "target_aliases 放目标的别名、缩写展开、公式下标写法，例如 PBA / Preference-Bridged Alignment / L_PBA。"
            "answer_slots 是槽位列表，不是执行计划；可以为空或多个。"
            "smalltalk 用于你好、你是谁、你能做什么、用户表达不清需要澄清。"
            "meta_library 用于本地论文库元信息问题，包括论文库数量、是否存在某类/某年/某作者/某标签/某分类论文、"
            "列出或统计库内论文、库内推荐、库内引用数排序。"
            "这类问题必须基于本地索引查询，不要用现实日期常识替代库内数据。"
            "research 用于论文正文、公式、图表、实验、概念、方法、推荐阅读等需要本地语料的问题。"
            "memory_op 用于追问上一轮结果、比较已有结果；如果追问论文正文细节，"
            "仍然设置 needs_local_corpus=true，让后续工具读取语料。"
            "如果用户问“第一个/第一篇/最初/首次提出 X 的论文”，这是 origin 槽位；"
            "不要因为上一轮正在讨论某篇后续论文就把目标改成那篇论文。"
            "confidence 低于 0.6 或 ambiguous_slots 非空表示应优先澄清。"
            "你会收到真实的最近多轮 user/assistant messages；请优先用这些自然对话解析指代。"
            "以下非语言上下文只用于补充 session 状态，不是用户新问题：\n"
            f"{context_json}\n"
            "只输出 JSON，不要回答用户。"
        )
        invoke_json_messages = getattr(self.clients, "invoke_json_messages", None)
        try:
            if callable(invoke_json_messages):
                raw = invoke_json_messages(
                    system_prompt=system_prompt,
                    messages=[
                        *self.conversation_messages(session),
                        {"role": "user", "content": clean_query},
                    ],
                    fallback={},
                )
            else:
                raw = self.clients.invoke_json(
                    system_prompt=system_prompt,
                    human_prompt=(
                        "用户原文：\n"
                        f"{clean_query}\n\n"
                        "非语言上下文 JSON：\n"
                        f"{context_json}"
                    ),
                    fallback={},
                )
        except Exception:  # noqa: BLE001
            if not callable(invoke_json_messages):
                return None
            try:
                raw = invoke_json_messages(
                    system_prompt=system_prompt,
                    messages=[
                        *self.conversation_messages(session),
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "current_query": clean_query,
                                    "conversation_context": self.conversation_context(session),
                                    **payload,
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    fallback={},
                )
            except Exception:  # noqa: BLE001
                return None
        if not isinstance(raw, dict) or not raw:
            return None
        if "intent_kind" not in raw and "relation" in raw:
            return self._intent_from_legacy_contract_payload(raw, clean_query=clean_query)
        intent = self._intent_from_payload(raw)
        if intent is not None:
            return intent
        return self._intent_from_legacy_contract_payload(raw, clean_query=clean_query)

    @staticmethod
    def _intent_from_payload(payload: dict[str, Any]) -> Intent | None:
        try:
            return Intent.model_validate(payload)
        except ValidationError:
            return None

    @staticmethod
    def _intent_from_legacy_contract_payload(payload: dict[str, Any], *, clean_query: str = "") -> Intent | None:
        relation = str(payload.get("relation", "") or "").strip()
        interaction_mode = str(payload.get("interaction_mode", "") or "").strip()
        if not relation or interaction_mode not in {"conversation", "research"}:
            return None
        targets = payload.get("targets", [])
        if not isinstance(targets, list):
            targets = []
        notes = payload.get("notes", [])
        if not isinstance(notes, list):
            notes = []
        requested_fields = payload.get("requested_fields", [])
        if not isinstance(requested_fields, list):
            requested_fields = []
        query_for_goal = clean_query or str(payload.get("clean_query", "") or "")
        slots = {
            "greeting": ["greeting"],
            "self_identity": ["self_identity"],
            "capability": ["capability"],
            "clarify_user_intent": ["clarify"],
            "library_status": ["library_status"],
            "library_recommendation": ["library_recommendation"],
            "library_citation_ranking": ["citation_ranking"],
            "memory_followup": ["previous_rationale"],
            "memory_synthesis": ["comparison"],
            "origin_lookup": ["origin"],
            "formula_lookup": ["formula"],
            "followup_research": ["followup_research"],
            "entity_definition": ["entity_definition"],
            "topology_discovery": ["topology_discovery"],
            "topology_recommendation": ["topology_recommendation"],
            "figure_question": ["figure"],
            "paper_summary_results": ["paper_summary"],
            "metric_value_lookup": ["metric_value"],
            "concept_definition": ["concept_definition"],
            "paper_recommendation": ["paper_recommendation"],
            "general_question": ["general_answer"],
        }.get(relation, ["general_answer"])
        needs_local_corpus = interaction_mode == "research"
        continuation_mode = str(payload.get("continuation_mode", "") or "").strip()
        topic_state: TopicState = "continue" if continuation_mode == "followup" else "switch" if continuation_mode == "context_switch" else "new"
        refers_previous_turn = topic_state == "continue"
        origin_like_goal = looks_like_origin_lookup_query(query_for_goal)
        has_explicit_origin_target = bool(targets) or bool(fallback_query_targets(query_for_goal))
        if origin_like_goal and relation in {
            "memory_followup",
            "clarify_user_intent",
            "correction_without_context",
            "general_question",
        } and has_explicit_origin_target:
            slots = ["origin"]
            needs_local_corpus = True
            refers_previous_turn = False
            topic_state = "new"
            notes = [*notes, "local_origin_lookup_override"]
        if relation == "memory_followup" and (
            any(str(field) in {"paper_content", "method", "experiments", "summary", "key_findings"} for field in requested_fields)
            or "needs_contextual_refine" in {str(item) for item in notes}
        ):
            slots = ["paper_summary"]
            needs_local_corpus = True
            refers_previous_turn = True
        if relation in {"clarify_user_intent", "correction_without_context"} and query_matches_any(
            query_for_goal,
            query_for_goal,
            INTENT_ROUTER_MARKERS["contextual_origin_refine"],
        ):
            has_strong_origin_refine = query_matches_any(
                query_for_goal,
                query_for_goal,
                INTENT_ROUTER_MARKERS["strong_origin_refine"],
            )
            slots = (
                ["origin"]
                if origin_like_goal or has_strong_origin_refine
                else ["general_answer"]
            )
            needs_local_corpus = True
            refers_previous_turn = not (origin_like_goal and has_explicit_origin_target)
            notes = [*notes, "needs_contextual_refine"] if refers_previous_turn else notes
        if interaction_mode == "conversation" and relation.startswith("library"):
            intent_kind: IntentKind = "meta_library"
        elif interaction_mode == "conversation" and relation.startswith("memory"):
            intent_kind = "memory_op"
        elif interaction_mode == "conversation" and not needs_local_corpus:
            intent_kind = "smalltalk"
        else:
            intent_kind = "memory_op" if relation in {"clarify_user_intent", "correction_without_context", "memory_followup"} else "research"
        return Intent(
            intent_kind=intent_kind,
            topic_state=topic_state,
            active_topic=query_for_goal,
            needs_local_corpus=needs_local_corpus,
            needs_web=bool(payload.get("allow_web_search")) or "citation_ranking" in slots,
            refers_previous_turn=refers_previous_turn,
            target_entities=[str(item).strip() for item in targets if str(item).strip()],
            target_aliases=[],
            user_goal=query_for_goal,
            answer_slots=slots,  # type: ignore[arg-type]
            confidence=0.82,
            notes=["legacy_router_payload", *[str(item) for item in notes]],
        )

    @staticmethod
    def _smalltalk_relation(slots: list[str]) -> str:
        if "self_identity" in slots:
            return "self_identity"
        if "capability" in slots:
            return "capability"
        if "clarify" in slots:
            return "clarify_user_intent"
        return "greeting"

    @staticmethod
    def _research_relation(*, slots: list[str], clean_query: str, targets: list[str]) -> str:
        return research_relation_from_slots(slots=slots, clean_query=clean_query, targets=targets)

    @staticmethod
    def _research_requirements(
        *,
        slots: list[str],
        targets: list[str],
        clean_query: str,
    ) -> tuple[list[str], list[str], str, Literal["exact", "high", "normal"]]:
        return research_requirements_from_slots(slots=slots, targets=targets, clean_query=clean_query)

    @staticmethod
    def _research_profile_slots(*, slots: list[str], clean_query: str, targets: list[str]) -> list[str]:
        return research_profile_slots(slots=slots, clean_query=clean_query, targets=targets)

    def _protected_local_intent(self, query: str, *, session: SessionContext | None = None) -> Intent | None:
        normalized = self._compact_text(query)
        lowered = " ".join(str(query or "").strip().lower().split())
        if not normalized:
            return Intent(
                intent_kind="smalltalk",
                topic_state="new",
                needs_local_corpus=False,
                target_entities=[],
                user_goal="用户没有输入有效内容，需要澄清。",
                answer_slots=["clarify"],
                confidence=0.95,
                notes=["local_protected_empty_query"],
            )
        greetings = {
            "你好",
            "您好",
            "你好吗",
            "嗨",
            "嗨嗨",
            "哈喽",
            "哈啰",
            "hello",
            "hi",
            "hey",
            "yo",
            "在吗",
            "早上好",
            "下午好",
            "晚上好",
        }
        punctuation_stripped = re.sub(r"[\s,.!?。！？~～]+", "", lowered)
        if punctuation_stripped in greetings:
            return Intent(
                intent_kind="smalltalk",
                topic_state="new",
                needs_local_corpus=False,
                target_entities=[],
                user_goal="回应用户寒暄。",
                answer_slots=["greeting"],
                confidence=0.99,
                notes=["local_protected_greeting"],
            )
        if query_matches_any(lowered, "", INTENT_ROUTER_MARKERS["self_identity"]):
            return Intent(
                intent_kind="smalltalk",
                topic_state="new",
                needs_local_corpus=False,
                target_entities=[],
                user_goal="介绍助手身份。",
                answer_slots=["self_identity"],
                confidence=0.96,
                notes=["local_protected_self_identity"],
            )
        if query_matches_any(lowered, "", INTENT_ROUTER_MARKERS["capability"]):
            return Intent(
                intent_kind="smalltalk",
                topic_state="new",
                needs_local_corpus=False,
                target_entities=[],
                user_goal="介绍助手能力范围。",
                answer_slots=["capability"],
                confidence=0.96,
                notes=["local_protected_capability"],
            )
        if looks_like_origin_lookup_query(query):
            targets = fallback_query_targets(query)
            if targets:
                return Intent(
                    intent_kind="research",
                    topic_state="new",
                    active_topic=f"查找 {targets[0]} 最早提出论文",
                    needs_local_corpus=True,
                    target_entities=targets,
                    target_aliases=fallback_target_aliases(targets),
                    user_goal=query,
                    answer_slots=["origin"],
                    confidence=0.92,
                    notes=["local_protected_origin_lookup"],
                )
        if self._looks_like_recent_tool_result_reference(query, session=session):
            return Intent(
                intent_kind="memory_op",
                topic_state="continue",
                needs_local_corpus=False,
                refers_previous_turn=True,
                target_entities=[],
                user_goal="回答用户对上一轮工具结果中某一项的追问。",
                answer_slots=["previous_rationale"],
                confidence=0.9,
                notes=["local_protected_tool_result_reference"],
            )
        explicit_targets = fallback_query_targets(query)
        if explicit_targets:
            if looks_like_metric_value_query(query):
                return Intent(
                    intent_kind="research",
                    topic_state="new",
                    active_topic=query,
                    needs_local_corpus=True,
                    target_entities=explicit_targets,
                    target_aliases=fallback_target_aliases(explicit_targets),
                    user_goal=query,
                    answer_slots=["metric_value"],
                    confidence=0.86,
                    notes=["local_protected_explicit_target_metric"],
                )
            if looks_like_summary_results_query(query):
                return Intent(
                    intent_kind="research",
                    topic_state="new",
                    active_topic=query,
                    needs_local_corpus=True,
                    target_entities=explicit_targets,
                    target_aliases=fallback_target_aliases(explicit_targets),
                    user_goal=query,
                    answer_slots=["paper_summary"],
                    confidence=0.86,
                    notes=["local_protected_explicit_target_summary"],
                )
        if normalized in {"何意味", "什么意思", "啥意思"}:
            return Intent(
                intent_kind="smalltalk",
                topic_state="new",
                needs_local_corpus=False,
                target_entities=[],
                user_goal="用户表达过短，需要补充想问的对象。",
                answer_slots=["clarify"],
                confidence=0.88,
                ambiguous_slots=["missing_target"],
                notes=["local_protected_short_clarification"],
            )
        if self._is_pdf_agent_topology_design_query(lowered=lowered, compact=normalized):
            return Intent(
                intent_kind="research",
                topic_state="continue" if session is not None and session.turns else "new",
                active_topic="PDF-Agent multi-agent topology design",
                needs_local_corpus=True,
                target_entities=["agent topology"],
                target_aliases=["DAG", "chain", "tree", "mesh", "irregular/random"],
                user_goal="为 PDF-Agent 的 multi-agent 内部协作选择拓扑结构。",
                answer_slots=["topology_recommendation"],
                confidence=0.9,
                notes=["local_protected_pdf_agent_topology_design"],
            )
        return None

    def _fallback_intent(
        self,
        *,
        clean_query: str,
        session: SessionContext,
        extracted_targets: list[str],
    ) -> Intent:
        lowered = " ".join(clean_query.lower().split())
        compact = self._compact_text(clean_query)
        targets = list(extracted_targets)
        active = session.effective_active_research()
        refers_previous = self._looks_like_memory_reference(clean_query) or (
            bool(active.targets) and self._is_short_followup(clean_query)
        )
        if is_library_status_query(clean_query):
            return Intent(
                intent_kind="meta_library",
                topic_state="new",
                needs_local_corpus=False,
                target_entities=[],
                user_goal="读取本地论文库状态。",
                answer_slots=["library_status"],
                confidence=0.9,
                notes=["local_intent_library_status"],
            )
        if is_scoped_library_recommendation_query(clean_query):
            return Intent(
                intent_kind="meta_library",
                topic_state="continue" if session.turns else "new",
                needs_local_corpus=False,
                target_entities=[],
                user_goal="基于本地论文库给出阅读推荐。",
                answer_slots=["library_recommendation"],
                confidence=0.86,
                notes=["local_intent_library_recommendation"],
            )
        if is_citation_query(clean_query):
            return Intent(
                intent_kind="meta_library",
                topic_state="continue" if session.turns else "new",
                needs_local_corpus=False,
                needs_web=True,
                refers_previous_turn=bool(session.turns),
                target_entities=[],
                user_goal="对库内或上一轮候选按引用数排序。",
                answer_slots=["citation_ranking"],
                confidence=0.82,
                notes=["local_intent_citation_ranking"],
            )
        if self._is_memory_comparison_query(lowered) and session.turns:
            return Intent(
                intent_kind="memory_op",
                topic_state="continue",
                needs_local_corpus=False,
                refers_previous_turn=True,
                target_entities=list(active.targets),
                user_goal="基于上一轮结果做比较或综合。",
                answer_slots=["comparison"],
                confidence=0.82,
                notes=["local_intent_memory_synthesis"],
            )
        if query_matches_any(lowered, "", INTENT_ROUTER_MARKERS["previous_rationale"]) and session.turns:
            return Intent(
                intent_kind="memory_op",
                topic_state="continue",
                needs_local_corpus=False,
                refers_previous_turn=True,
                target_entities=targets or list(active.targets),
                user_goal="解释上一轮工具输出的依据。",
                answer_slots=["previous_rationale"],
                confidence=0.84,
                notes=["local_intent_memory_followup"],
            )

        slots = self._research_slots(clean_query=clean_query, lowered=lowered, compact=compact, session=session)
        if not targets and refers_previous:
            targets = list(active.targets)
        return Intent(
            intent_kind="memory_op" if refers_previous else "research",
            topic_state="continue" if refers_previous else "new",
            active_topic=clean_query,
            needs_local_corpus=True,
            needs_web=self._needs_web(lowered, compact),
            refers_previous_turn=refers_previous,
            target_entities=targets,
            target_aliases=fallback_target_aliases(targets),
            user_goal=clean_query,
            answer_slots=slots,
            confidence=0.74,
            notes=["local_intent_fallback"],
        )

    @staticmethod
    def _compact_text(query: str) -> str:
        return re.sub(r"[\s,.!?。！？~～，、；;：:]+", "", str(query or "").strip().lower())

    @staticmethod
    def _is_pdf_agent_topology_design_query(*, lowered: str, compact: str) -> bool:
        return is_pdf_agent_topology_design_query(lowered=lowered, compact=compact)

    @staticmethod
    def _is_memory_comparison_query(lowered: str) -> bool:
        return is_memory_comparison_query(lowered)

    @staticmethod
    def _looks_like_memory_reference(query: str) -> bool:
        return looks_like_memory_reference(query)

    @staticmethod
    def _is_short_followup(query: str) -> bool:
        return is_short_followup(query)

    @staticmethod
    def _looks_like_recent_tool_result_reference(query: str, *, session: SessionContext | None) -> bool:
        return looks_like_recent_tool_result_reference(query, session=session)

    @staticmethod
    def _contains_ordinal_reference(query: str) -> bool:
        return contains_ordinal_reference(query)

    @staticmethod
    def _needs_web(lowered: str, compact: str) -> bool:
        return normalized_query_needs_external_search(lowered, compact, include_router_extras=True)

    @staticmethod
    def _research_slots(*, clean_query: str, lowered: str, compact: str, session: SessionContext) -> list[AnswerSlot]:
        return cast(
            list[AnswerSlot],
            research_answer_slots(
                clean_query=clean_query,
                lowered=lowered,
                compact=compact,
                active_relation=session.effective_active_research().relation,
            ),
        )
