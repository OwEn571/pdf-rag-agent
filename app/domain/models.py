from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SessionTurn(BaseModel):
    query: str
    answer: str
    relation: str = ""
    interaction_mode: Literal["conversation", "research"] = "research"
    clean_query: str = ""
    targets: list[str] = Field(default_factory=list)
    answer_slots: list[str] = Field(default_factory=list)
    titles: list[str] = Field(default_factory=list)
    requested_fields: list[str] = Field(default_factory=list)
    required_modalities: list[str] = Field(default_factory=list)
    answer_shape: str = ""
    precision_requirement: Literal["exact", "high", "normal"] = "normal"

    @classmethod
    def from_contract(
        cls,
        *,
        query: str,
        answer: str,
        contract: Any,
        titles: list[str] | None = None,
        targets: list[str] | None = None,
        requested_fields: list[str] | None = None,
        required_modalities: list[str] | None = None,
        interaction_mode: Literal["conversation", "research"] | None = None,
        clean_query: str | None = None,
    ) -> "SessionTurn":
        return cls(
            query=query,
            answer=answer,
            relation=str(getattr(contract, "relation", "") or ""),
            interaction_mode=interaction_mode or getattr(contract, "interaction_mode", "research"),
            clean_query=clean_query if clean_query is not None else str(getattr(contract, "clean_query", "") or ""),
            targets=list(targets) if targets is not None else list(getattr(contract, "targets", []) or []),
            answer_slots=list(getattr(contract, "answer_slots", []) or []),
            titles=list(titles or []),
            requested_fields=list(requested_fields)
            if requested_fields is not None
            else list(getattr(contract, "requested_fields", []) or []),
            required_modalities=list(required_modalities)
            if required_modalities is not None
            else list(getattr(contract, "required_modalities", []) or []),
            answer_shape=str(getattr(contract, "answer_shape", "") or ""),
            precision_requirement=getattr(contract, "precision_requirement", "normal"),
        )


class ActiveResearch(BaseModel):
    relation: str = ""
    targets: list[str] = Field(default_factory=list)
    titles: list[str] = Field(default_factory=list)
    requested_fields: list[str] = Field(default_factory=list)
    required_modalities: list[str] = Field(default_factory=list)
    answer_shape: str = ""
    precision_requirement: Literal["exact", "high", "normal"] = "normal"
    clean_query: str = ""
    last_topic_signature: str = ""

    def has_content(self) -> bool:
        return bool(
            self.relation
            or self.targets
            or self.titles
            or self.requested_fields
            or self.required_modalities
            or self.answer_shape
            or self.clean_query
        )

    def topic_signature(self) -> str:
        parts = [
            self.relation,
            " ".join(self.targets),
            " ".join(self.titles),
            " ".join(self.requested_fields),
            " ".join(self.required_modalities),
            self.answer_shape,
            self.precision_requirement,
            self.clean_query,
        ]
        return " ".join(" ".join(parts).lower().split())[:400]

    def context_payload(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "targets": list(self.targets),
            "titles": list(self.titles),
            "active_titles": list(self.titles),
            "requested_fields": list(self.requested_fields),
            "required_modalities": list(self.required_modalities),
            "answer_shape": self.answer_shape,
            "precision_requirement": self.precision_requirement,
            "clean_query": self.clean_query,
            "last_topic_signature": self.last_topic_signature or self.topic_signature(),
            "has_content": self.has_content(),
        }


class SessionContext(BaseModel):
    session_id: str
    active_task_id: str = ""
    continuation_mode: Literal["fresh", "followup", "context_switch"] = "fresh"
    active_research: ActiveResearch = Field(default_factory=ActiveResearch)
    active_targets: list[str] = Field(default_factory=list)
    active_titles: list[str] = Field(default_factory=list)
    answered_titles: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    summary: str = ""
    last_relation: str = ""
    active_research_relation: str = ""
    active_requested_fields: list[str] = Field(default_factory=list)
    active_required_modalities: list[str] = Field(default_factory=list)
    active_answer_shape: str = ""
    active_precision_requirement: Literal["exact", "high", "normal"] = "normal"
    active_clean_query: str = ""
    pending_clarification_type: str = ""
    pending_clarification_target: str = ""
    pending_clarification_options: list[dict[str, Any]] = Field(default_factory=list)
    last_clarification_key: str = ""
    clarification_attempts: int = 0
    working_memory: dict[str, Any] = Field(default_factory=dict)
    turns: list[SessionTurn] = Field(default_factory=list)

    @model_validator(mode="after")
    def sync_active_research_compatibility(self) -> "SessionContext":
        legacy_has_content = bool(
            self.active_research_relation
            or self.active_targets
            or self.active_titles
            or self.active_requested_fields
            or self.active_required_modalities
            or self.active_answer_shape
            or self.active_clean_query
        )
        if legacy_has_content and not self.active_research.has_content():
            self.active_research = ActiveResearch(
                relation=self.active_research_relation,
                targets=list(self.active_targets),
                titles=list(self.active_titles),
                requested_fields=list(self.active_requested_fields),
                required_modalities=list(self.active_required_modalities),
                answer_shape=self.active_answer_shape,
                precision_requirement=self.active_precision_requirement,
                clean_query=self.active_clean_query,
            )
        if self.active_research.has_content():
            if not self.active_research.last_topic_signature:
                self.active_research.last_topic_signature = self.active_research.topic_signature()
            if not legacy_has_content:
                self._sync_legacy_active_research_fields()
        return self

    def set_active_research(
        self,
        *,
        relation: str,
        targets: list[str],
        titles: list[str],
        requested_fields: list[str],
        required_modalities: list[str],
        answer_shape: str,
        precision_requirement: Literal["exact", "high", "normal"],
        clean_query: str,
        last_topic_signature: str = "",
    ) -> None:
        self.active_research = ActiveResearch(
            relation=relation,
            targets=list(targets),
            titles=list(titles),
            requested_fields=list(requested_fields),
            required_modalities=list(required_modalities),
            answer_shape=answer_shape,
            precision_requirement=precision_requirement,
            clean_query=clean_query,
            last_topic_signature=last_topic_signature,
        )
        if not self.active_research.last_topic_signature:
            self.active_research.last_topic_signature = self.active_research.topic_signature()
        self._sync_legacy_active_research_fields()

    def _sync_legacy_active_research_fields(self) -> None:
        self.active_research_relation = self.active_research.relation
        self.active_targets = list(self.active_research.targets)
        self.active_titles = list(self.active_research.titles)
        self.active_requested_fields = list(self.active_research.requested_fields)
        self.active_required_modalities = list(self.active_research.required_modalities)
        self.active_answer_shape = self.active_research.answer_shape
        self.active_precision_requirement = self.active_research.precision_requirement
        self.active_clean_query = self.active_research.clean_query

    def normalize_active_research(self) -> ActiveResearch:
        """Promote whichever active-context representation is freshest to the canonical model."""

        active = self.effective_active_research()
        self.active_research = ActiveResearch(
            relation=active.relation,
            targets=list(active.targets),
            titles=list(active.titles),
            requested_fields=list(active.requested_fields),
            required_modalities=list(active.required_modalities),
            answer_shape=active.answer_shape,
            precision_requirement=active.precision_requirement,
            clean_query=active.clean_query,
            last_topic_signature=active.last_topic_signature,
        )
        if self.active_research.has_content() and not self.active_research.last_topic_signature:
            self.active_research.last_topic_signature = self.active_research.topic_signature()
        self._sync_legacy_active_research_fields()
        return self.active_research

    def effective_active_research(self) -> ActiveResearch:
        legacy_has_content = bool(
            self.active_research_relation
            or self.active_targets
            or self.active_titles
            or self.active_requested_fields
            or self.active_required_modalities
            or self.active_answer_shape
            or self.active_clean_query
        )
        if legacy_has_content:
            legacy_active = ActiveResearch(
                relation=self.active_research_relation,
                targets=list(self.active_targets),
                titles=list(self.active_titles),
                requested_fields=list(self.active_requested_fields),
                required_modalities=list(self.active_required_modalities),
                answer_shape=self.active_answer_shape,
                precision_requirement=self.active_precision_requirement,
                clean_query=self.active_clean_query,
            )
            if not self.active_research.has_content():
                return legacy_active
            if legacy_active.topic_signature() != self.active_research.topic_signature():
                return legacy_active
        return self.active_research

    def active_research_context_payload(self) -> dict[str, Any]:
        active = self.effective_active_research()
        return active.context_payload()


class QueryContract(BaseModel):
    clean_query: str
    interaction_mode: Literal["conversation", "research"] = "research"
    relation: str = "general_question"
    targets: list[str] = Field(default_factory=list)
    answer_slots: list[str] = Field(default_factory=list)
    requested_fields: list[str] = Field(default_factory=lambda: ["answer"])
    required_modalities: list[str] = Field(default_factory=lambda: ["page_text"])
    answer_shape: str = "narrative"
    precision_requirement: Literal["exact", "high", "normal"] = "normal"
    continuation_mode: Literal["fresh", "followup", "context_switch"] = "fresh"
    allow_web_search: bool = False
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def sync_answer_slot_notes(self) -> "QueryContract":
        slots = [str(item).strip() for item in list(self.answer_slots or []) if str(item).strip()]
        notes = [str(item).strip() for item in list(self.notes or []) if str(item).strip()]
        note_slots = [
            note.split("=", 1)[1].strip()
            for note in notes
            if note.startswith("answer_slot=") and "=" in note and note.split("=", 1)[1].strip()
        ]
        combined_slots = list(dict.fromkeys([*slots, *note_slots]))
        self.answer_slots = combined_slots
        existing_note_slots = {slot for slot in note_slots}
        if combined_slots:
            notes = [
                *notes,
                *[f"answer_slot={slot}" for slot in combined_slots if slot not in existing_note_slots],
            ]
            self.notes = list(dict.fromkeys(notes))
        return self


class ResearchPlan(BaseModel):
    paper_recall_mode: Literal["anchor_first", "broad", "broad_then_anchor"] = "broad"
    paper_limit: int = 6
    evidence_limit: int = 14
    solver_sequence: list[str] = Field(default_factory=list)
    required_claims: list[str] = Field(default_factory=list)
    retry_budget: int = 1


class CandidatePaper(BaseModel):
    paper_id: str
    title: str
    year: str = ""
    score: float = 0.0
    match_reason: str = ""
    anchor_terms: list[str] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DisambiguationRejectedOption(BaseModel):
    option_id: str = ""
    reason: str = ""


class DisambiguationJudgeDecision(BaseModel):
    decision: Literal["auto_resolve", "ask_human"] = "ask_human"
    selected_option_id: str | None = None
    selected_paper_id: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    rejected_options: list[DisambiguationRejectedOption] = Field(default_factory=list)


class EvidenceBlock(BaseModel):
    doc_id: str
    paper_id: str
    title: str
    file_path: str
    page: int
    block_type: str
    caption: str = ""
    bbox: str = ""
    snippet: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    claim_type: str
    entity: str = ""
    value: str = ""
    structured_data: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.6
    required: bool = True


class VerificationReport(BaseModel):
    status: Literal["pass", "retry", "clarify"] = "pass"
    missing_fields: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictory_claims: list[str] = Field(default_factory=list)
    recommended_action: str = ""
    original_status: str = ""  # P0-8: set to "best_effort" when clarif limit reached, prevents target_binding pollution


class AssistantCitation(BaseModel):
    doc_id: str = ""
    paper_id: str = ""
    title: str
    authors: str = ""
    year: str = ""
    tags: list[str] = Field(default_factory=list)
    file_path: str
    page: int
    block_type: str = ""
    caption: str = ""
    snippet: str


class AssistantResponse(BaseModel):
    session_id: str
    interaction_mode: str
    answer: str
    citations: list[AssistantCitation] = Field(default_factory=list)
    query_contract: dict[str, Any] = Field(default_factory=dict)
    research_plan_summary: dict[str, Any] = Field(default_factory=dict)
    runtime_summary: dict[str, Any] = Field(default_factory=dict)
    execution_steps: list[dict[str, Any]] = Field(default_factory=list)
    verification_report: dict[str, Any] = Field(default_factory=dict)
    needs_human: bool = False
    clarification_question: str = ""
    clarification_options: list[dict[str, Any]] = Field(default_factory=list)
