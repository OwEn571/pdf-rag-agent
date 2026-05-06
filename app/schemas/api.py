from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str | None = None
    use_web_search: bool = False
    max_web_results: int = Field(default=3, ge=1, le=10)
    clarification_choice: dict[str, Any] | None = None


class AgentCitation(BaseModel):
    doc_id: str = ""
    paper_id: str = ""
    title: str
    authors: str
    year: str
    tags: list[str]
    file_path: str
    page: int
    block_type: str = ""
    caption: str = ""
    snippet: str


class AgentChatResponse(BaseModel):
    session_id: str
    interaction_mode: str
    answer: str
    citations: list[AgentCitation] = Field(default_factory=list)
    query_contract: dict[str, Any] = Field(default_factory=dict)
    research_plan_summary: dict[str, Any] = Field(default_factory=dict)
    runtime_summary: dict[str, Any] = Field(default_factory=dict)
    execution_steps: list[dict[str, Any]] = Field(default_factory=list)
    verification_report: dict[str, Any] = Field(default_factory=dict)
    needs_human: bool = False
    clarification_question: str = ""
    clarification_options: list[dict[str, Any]] = Field(default_factory=list)


class IngestRequest(BaseModel):
    force_rebuild: bool = True
    max_papers: int | None = Field(default=None, ge=1)


class IngestResponse(BaseModel):
    message: str
    paper_records: int = 0
    papers_indexed: int = 0
    papers_missing_pdf: int = 0
    block_docs: int = 0
    paper_docs: int = 0
    vectors_upserted: int = 0
    papers_with_generated_summary: int = 0


class ToolProposalTransitionRequest(BaseModel):
    next_status: str
    code_sha256: str
    reviewer: str
    note: str = ""
    sandbox_report: dict[str, Any] | None = None


class ToolProposalSandboxRequest(BaseModel):
    args: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=2.0, gt=0, le=30.0)
    memory_limit_mb: int = Field(default=256, ge=64, le=2048)


class HealthResponse(BaseModel):
    status: str = "ok"
    runtime_profile: str = "structured-intent-react-loop"
    runtime_summary_supported: bool = True
    canonical_tools: list[str] = Field(
        default_factory=lambda: ["read_memory", "search_corpus", "web_search", "query_library_metadata", "compose", "ask_human"]
    )


class LibraryPaper(BaseModel):
    paper_id: str
    title: str
    authors: str = ""
    year: str = ""
    tags: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    file_path: str = ""
    preview: str = ""


class LibraryCategory(BaseModel):
    name: str
    count: int = 0
    papers: list[LibraryPaper] = Field(default_factory=list)


class LibraryResponse(BaseModel):
    categories: list[LibraryCategory] = Field(default_factory=list)
    total_papers: int = 0


class PaperPreviewResponse(BaseModel):
    paper: LibraryPaper
    snippets: list[dict[str, Any]] = Field(default_factory=list)


class CitationPreviewResponse(BaseModel):
    paper_id: str = ""
    doc_id: str = ""
    title: str = ""
    authors: str = ""
    year: str = ""
    file_path: str = ""
    page: int = 0
    block_type: str = ""
    caption: str = ""
    snippet: str = ""
