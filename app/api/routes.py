from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from app.core.config import Settings, get_settings
from app.core.deps import get_agent, get_ingestion_service, get_library_service, get_retriever
from app.core.security import require_admin_access, require_pdf_access
from app.schemas.api import (
    AgentChatRequest,
    AgentChatResponse,
    AgentCitation,
    CitationPreviewResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    LibraryResponse,
    PaperPreviewResponse,
    ToolProposalSandboxRequest,
    ToolProposalTransitionRequest,
)
from app.services.agent import ResearchAssistantAgentV4
from app.services.agent.events import normalize_agent_event
from app.services.retrieval.indexing import V4IngestionService
from app.services.library import LibraryBrowserService
from app.services.tools.proposals import (
    find_tool_proposal_path,
    list_tool_proposals,
    load_tool_proposal,
    run_tool_proposal_sandbox,
    transition_tool_proposal_status,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _format_sse(event: str, data: object) -> str:
    if event == "heartbeat":
        return "event: heartbeat\ndata: {}\n\n"
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_error_events(exc: Exception) -> list[tuple[str, dict[str, object]]]:
    message = str(exc)
    return [
        ("error", normalize_agent_event("error", {"message": message})),
        ("final", normalize_agent_event("final", {"answer": "", "error": message})),
    ]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/library", response_model=LibraryResponse)
def library(
    library_service: LibraryBrowserService = Depends(get_library_service),
) -> LibraryResponse:
    return LibraryResponse(**library_service.list_library())


@router.get("/library/papers/{paper_id}/preview", response_model=PaperPreviewResponse)
def paper_preview(
    paper_id: str,
    library_service: LibraryBrowserService = Depends(get_library_service),
) -> PaperPreviewResponse:
    payload = library_service.paper_preview(paper_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="paper not found")
    return PaperPreviewResponse(**payload)


@router.get("/library/papers/{paper_id}/pdf")
def paper_pdf(
    paper_id: str,
    _: None = Depends(require_pdf_access),
    library_service: LibraryBrowserService = Depends(get_library_service),
) -> FileResponse:
    path = library_service.pdf_path(paper_id)
    if path is None:
        raise HTTPException(status_code=404, detail="pdf not found")
    return FileResponse(path, media_type="application/pdf", filename=path.name, content_disposition_type="inline")


@router.get("/citations/preview", response_model=CitationPreviewResponse)
def citation_preview(
    doc_id: str = Query(default=""),
    paper_id: str = Query(default=""),
    library_service: LibraryBrowserService = Depends(get_library_service),
) -> CitationPreviewResponse:
    payload = library_service.citation_preview(doc_id=doc_id, paper_id=paper_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="citation evidence not found")
    return CitationPreviewResponse(**payload)


@router.post("/ingest/rebuild", response_model=IngestResponse)
def ingest_rebuild(
    payload: IngestRequest,
    _: None = Depends(require_admin_access),
    ingestion_service: V4IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    try:
        stats = ingestion_service.rebuild(max_papers=payload.max_papers, force_rebuild=payload.force_rebuild)
        get_retriever().refresh()
    except Exception as exc:  # noqa: BLE001
        logger.exception("v4 ingest rebuild failed")
        raise HTTPException(status_code=500, detail="ingest rebuild failed") from exc
    return IngestResponse(message="v4 ingestion completed", **stats.to_dict())


@router.get("/admin/tools/proposals")
def admin_list_tool_proposals(
    include_code: bool = Query(default=False),
    _: None = Depends(require_admin_access),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    return {"items": list_tool_proposals(data_dir=settings.data_dir, include_code=include_code)}


@router.get("/admin/tools/proposals/{proposal_id}")
def admin_get_tool_proposal(
    proposal_id: str,
    include_code: bool = Query(default=True),
    _: None = Depends(require_admin_access),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    try:
        return load_tool_proposal(data_dir=settings.data_dir, proposal_id=proposal_id, include_code=include_code)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/admin/tools/proposals/{proposal_id}/sandbox")
def admin_run_tool_proposal_sandbox(
    proposal_id: str,
    payload: ToolProposalSandboxRequest,
    _: None = Depends(require_admin_access),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    try:
        proposal_path = find_tool_proposal_path(data_dir=settings.data_dir, proposal_id=proposal_id)
        return run_tool_proposal_sandbox(
            proposal_path=proposal_path,
            args=payload.args,
            timeout_seconds=payload.timeout_seconds,
            memory_limit_mb=payload.memory_limit_mb,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/admin/tools/proposals/{proposal_id}/status")
def admin_transition_tool_proposal_status(
    proposal_id: str,
    payload: ToolProposalTransitionRequest,
    _: None = Depends(require_admin_access),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    try:
        proposal_path = find_tool_proposal_path(data_dir=settings.data_dir, proposal_id=proposal_id)
        return transition_tool_proposal_status(
            proposal_path=proposal_path,
            next_status=payload.next_status,
            code_sha256=payload.code_sha256,
            reviewer=payload.reviewer,
            note=payload.note,
            sandbox_report=payload.sandbox_report,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat_v4(
    payload: AgentChatRequest,
    agent: ResearchAssistantAgentV4 = Depends(get_agent),
) -> AgentChatResponse:
    try:
        result = await agent.achat(
            query=payload.query,
            session_id=payload.session_id,
            use_web_search=payload.use_web_search,
            max_web_results=payload.max_web_results,
            clarification_choice=payload.clarification_choice,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("v4 chat failed")
        raise HTTPException(status_code=500, detail="chat failed") from exc
    citation_models = [AgentCitation(**item) for item in result.get("citations", [])]
    return AgentChatResponse(
        session_id=str(result.get("session_id", "")),
        interaction_mode=str(result.get("interaction_mode", "")),
        answer=str(result.get("answer", "")),
        citations=citation_models,
        query_contract=dict(result.get("query_contract", {})),
        research_plan_summary=dict(result.get("research_plan_summary", {})),
        runtime_summary=dict(result.get("runtime_summary", {})),
        execution_steps=list(result.get("execution_steps", [])),
        verification_report=dict(result.get("verification_report", {})),
        needs_human=bool(result.get("needs_human", False)),
        clarification_question=str(result.get("clarification_question", "")),
        clarification_options=list(result.get("clarification_options", [])),
    )


@router.post("/chat/stream")
async def agent_chat_v4_stream(
    payload: AgentChatRequest,
    agent: ResearchAssistantAgentV4 = Depends(get_agent),
) -> StreamingResponse:
    async def event_stream() -> object:
        try:
            async for item in agent.astream_chat_events(
                query=payload.query,
                session_id=payload.session_id,
                use_web_search=payload.use_web_search,
                max_web_results=payload.max_web_results,
                clarification_choice=payload.clarification_choice,
            ):
                yield _format_sse(str(item.get("event", "message")), item.get("data", {}))
        except Exception as exc:  # noqa: BLE001
            logger.exception("v4 stream failed")
            for event, data in _stream_error_events(exc):
                yield _format_sse(event, data)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
