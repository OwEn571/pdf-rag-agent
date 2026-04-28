from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

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
)
from app.services.agent import ResearchAssistantAgentV4
from app.services.indexing import V4IngestionService
from app.services.library import LibraryBrowserService

logger = logging.getLogger(__name__)
router = APIRouter()


def _format_sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/v4/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/v4/library", response_model=LibraryResponse)
def library(
    library_service: LibraryBrowserService = Depends(get_library_service),
) -> LibraryResponse:
    return LibraryResponse(**library_service.list_library())


@router.get("/v4/library/papers/{paper_id}/preview", response_model=PaperPreviewResponse)
def paper_preview(
    paper_id: str,
    library_service: LibraryBrowserService = Depends(get_library_service),
) -> PaperPreviewResponse:
    payload = library_service.paper_preview(paper_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="paper not found")
    return PaperPreviewResponse(**payload)


@router.get("/v4/library/papers/{paper_id}/pdf")
def paper_pdf(
    paper_id: str,
    _: None = Depends(require_pdf_access),
    library_service: LibraryBrowserService = Depends(get_library_service),
) -> FileResponse:
    path = library_service.pdf_path(paper_id)
    if path is None:
        raise HTTPException(status_code=404, detail="pdf not found")
    return FileResponse(path, media_type="application/pdf", filename=path.name, content_disposition_type="inline")


@router.get("/v4/citations/preview", response_model=CitationPreviewResponse)
def citation_preview(
    doc_id: str = Query(default=""),
    paper_id: str = Query(default=""),
    library_service: LibraryBrowserService = Depends(get_library_service),
) -> CitationPreviewResponse:
    payload = library_service.citation_preview(doc_id=doc_id, paper_id=paper_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="citation evidence not found")
    return CitationPreviewResponse(**payload)


@router.post("/v4/ingest/rebuild", response_model=IngestResponse)
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


@router.post("/v4/chat", response_model=AgentChatResponse)
async def agent_chat_v4(
    payload: AgentChatRequest,
    agent: ResearchAssistantAgentV4 = Depends(get_agent),
) -> AgentChatResponse:
    try:
        result = await agent.achat(
            query=payload.query,
            session_id=payload.session_id,
            mode=payload.mode,
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


@router.post("/v4/chat/stream")
async def agent_chat_v4_stream(
    payload: AgentChatRequest,
    agent: ResearchAssistantAgentV4 = Depends(get_agent),
) -> StreamingResponse:
    async def event_stream() -> object:
        try:
            async for item in agent.astream_chat_events(
                query=payload.query,
                session_id=payload.session_id,
                mode=payload.mode,
                use_web_search=payload.use_web_search,
                max_web_results=payload.max_web_results,
                clarification_choice=payload.clarification_choice,
            ):
                yield _format_sse(str(item.get("event", "message")), item.get("data", {}))
        except Exception as exc:  # noqa: BLE001
            logger.exception("v4 stream failed")
            yield _format_sse("error", {"message": str(exc)})
            yield _format_sse("final", {"answer": "", "error": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
