from __future__ import annotations

from functools import lru_cache
import logging
from typing import Any

from app.core.config import Settings, get_settings
from app.services.agent import ResearchAssistantAgentV4
from app.services.indexing import V4IngestionService
from app.services.library import LibraryBrowserService
from app.services.model_clients import ModelClients
from app.services.retrieval import DualIndexRetriever
from app.services.session_store import SQLiteSessionStore, SessionStore

logger = logging.getLogger(__name__)


@lru_cache
def get_model_clients() -> ModelClients:
    return ModelClients(get_settings())


@lru_cache
def get_retriever() -> DualIndexRetriever:
    return DualIndexRetriever(get_settings())


@lru_cache
def get_sessions() -> SessionStore:
    settings = get_settings()
    return SQLiteSessionStore(settings.session_store_path, max_turns=settings.agent_history_max_turns)


@lru_cache
def get_ingestion_service() -> V4IngestionService:
    return V4IngestionService(get_settings(), clients=get_model_clients())


@lru_cache
def get_library_service() -> LibraryBrowserService:
    return LibraryBrowserService(settings=get_settings(), retriever=get_retriever())


@lru_cache
def get_agent() -> ResearchAssistantAgentV4:
    return ResearchAssistantAgentV4(
        settings=get_settings(),
        retriever=get_retriever(),
        clients=get_model_clients(),
        sessions=get_sessions(),
    )


def get_settings_dep() -> Settings:
    return get_settings()


async def close_cached_resources() -> None:
    resources: list[Any] = []
    for factory in (get_model_clients, get_retriever):
        try:
            resources.append(factory())
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to resolve cached resource for shutdown: %s", exc)
    seen: set[int] = set()
    for resource in resources:
        if id(resource) in seen:
            continue
        seen.add(id(resource))
        aclose = getattr(resource, "aclose", None)
        close = getattr(resource, "close", None)
        try:
            if callable(aclose):
                await aclose()
            elif callable(close):
                close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to close cached resource %s: %s", type(resource).__name__, exc)
    get_agent.cache_clear()
    get_library_service.cache_clear()
    get_ingestion_service.cache_clear()
    get_sessions.cache_clear()
    get_retriever.cache_clear()
    get_model_clients.cache_clear()
