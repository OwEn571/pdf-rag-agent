from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except Exception:  # noqa: BLE001
    Instrumentator = None

from app.api.routes import router
from app.core.config import get_settings
from app.core.deps import close_cached_resources
from app.core.logging import setup_logging

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

settings = get_settings()
setup_logging(settings.log_level)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await close_cached_resources()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Tool-calling Zotero paper research agent",
    lifespan=lifespan,
)

if settings.cors_allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allow_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

if Instrumentator is not None:
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
    ).instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")
