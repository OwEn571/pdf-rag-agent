from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core.agent_settings import AgentSettings
from app.core.config import Settings
from app.services.agent.chat_runtime import run_agent_chat_turn
from app.services.infra.model_clients import ModelClients
from app.services.agent.events import normalize_agent_event
from app.services.agent.planner import AgentPlanner
from app.services.agent.runtime import AgentRuntime
from app.services.intents.followup import (
    is_negative_correction_query,
)
from app.services.answers.evidence_presentation import (
    chunk_text,
)
from app.services.intents.router import LLMIntentRouter
from app.services.tools.dynamic_context import load_agent_dynamic_tool_manifests
from app.services.retrieval.web_search import TavilyWebSearchClient
from app.services.agent_mixins import (
    AnswerComposerMixin,
    ClaimVerifierMixin,
    EntityDefinitionMixin,
    FollowupRoutingMixin,
    SolverPipelineMixin,
)
from app.services.retrieval import DualIndexRetriever
from app.services.memory.session_store import SessionStore
from app.services.contracts.session_context import (
    agent_session_conversation_context,
    session_llm_history_messages,
)

logger = logging.getLogger(__name__)


def _trim_final_payload_for_sse(payload: dict[str, Any]) -> dict[str, Any]:
    """Trim heavy fields in final SSE payload to prevent browser OOM."""
    trimmed = dict(payload)
    citations = trimmed.get("citations")
    if isinstance(citations, list):
        trimmed_citations = []
        for c in citations:
            if not isinstance(c, dict):
                trimmed_citations.append(c)
                continue
            tc = dict(c)
            # Trim snippet
            snippet = str(tc.get("snippet", ""))
            if len(snippet) > 280:
                tc["snippet"] = snippet[:280] + "..."
            # Trim authors
            authors = str(tc.get("authors", ""))
            if len(authors) > 180:
                first_three = [a.strip() for a in authors.split(",")[:3]]
                tc["authors"] = ", ".join(first_three) + " 等"
            # Remove file_path (not needed for display)
            tc.pop("file_path", None)
            trimmed_citations.append(tc)
        trimmed["citations"] = trimmed_citations
    # Trim runtime_summary (deeply nested, keep top-level only)
    runtime = trimmed.get("runtime_summary")
    if isinstance(runtime, dict):
        trimmed["runtime_summary"] = {
            "intent": runtime.get("intent", {}),
            "grounding": runtime.get("grounding", {}),
        }
    return trimmed


class ResearchAssistantAgentV4(
    FollowupRoutingMixin,
    AnswerComposerMixin,
    EntityDefinitionMixin,
    SolverPipelineMixin,
    ClaimVerifierMixin,
):
    def __init__(
        self,
        *,
        settings: Settings,
        retriever: DualIndexRetriever,
        clients: ModelClients,
        sessions: SessionStore,
        web_search: TavilyWebSearchClient | None = None,
    ) -> None:
        self.settings = settings
        self.agent_settings = AgentSettings.from_settings(settings)
        self.retriever = retriever
        self.clients = clients
        self.sessions = sessions
        self.web_search = web_search or TavilyWebSearchClient(settings)
        self.dynamic_tool_manifests = load_agent_dynamic_tool_manifests(
            settings=self.settings,
            agent_settings=self.agent_settings,
            logger=logger,
        )
        self.planner = AgentPlanner(
            clients=self.clients,
            conversation_context=lambda session, *, max_chars=24000: agent_session_conversation_context(
                session,
                settings=self.settings,
                max_chars=max_chars,
            ),
            conversation_messages=lambda session: session_llm_history_messages(
                session,
                max_turns=6,
                answer_limit=900,
            ),
            is_negative_correction_query=is_negative_correction_query,
            confidence_floor=self.agent_settings.confidence_floor,
            dynamic_tool_manifest=lambda: list(self.dynamic_tool_manifests),
        )
        self.llm_intent_router = LLMIntentRouter(
            clients=self.clients,
            conversation_context=lambda session, *, max_chars=12000: agent_session_conversation_context(
                session,
                settings=self.settings,
                max_chars=max_chars,
            ),
            conversation_messages=lambda session: session_llm_history_messages(
                session,
                max_turns=6,
                answer_limit=900,
            ),
        )
        self.runtime = AgentRuntime(agent=self)
        self._rendered_page_data_url_cache: dict[tuple[str, int], str] = {}
        self._redis = None
        try:
            from redis import Redis
            self._redis = Redis.from_url(self.settings.redis_url, socket_timeout=3)
            self._redis.ping()
        except Exception:
            self._redis = None

    def dynamic_tool_names(self) -> set[str]:
        if not self.agent_settings.dynamic_tools_enabled:
            return set()
        return {
            str(item.get("name") or "").strip()
            for item in list(self.dynamic_tool_manifests or [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        }

    def _cache_key(self, query: str) -> str:
        return "rag:cache:" + " ".join(query.strip().lower().split())[:200]

    def chat(
        self,
        *,
        query: str,
        session_id: str | None = None,
        use_web_search: bool = False,
        max_web_results: int = 3,
        clarification_choice: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import json as _json
        # Redis cache: return cached answer for identical queries
        if self._redis and not use_web_search and not clarification_choice:
            key = self._cache_key(query)
            try:
                raw = self._redis.get(key)
                if raw:
                    return _json.loads(raw)
            except Exception:
                pass
        result, _ = run_agent_chat_turn(
            agent=self,
            query=query,
            session_id=session_id,
            use_web_search=use_web_search,
            max_web_results=max_web_results,
            clarification_choice=clarification_choice,
        )
        # Store in Redis cache
        if self._redis and not use_web_search and not clarification_choice:
            ttl = int(getattr(self.settings, "redis_cache_ttl_seconds", 300))
            try:
                self._redis.setex(self._cache_key(query), ttl, _json.dumps(result, ensure_ascii=False))
            except Exception:
                pass
        return result

    async def achat(
        self,
        *,
        query: str,
        session_id: str | None = None,
        use_web_search: bool = False,
        max_web_results: int = 3,
        clarification_choice: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self.chat,
            query=query,
            session_id=session_id,
            use_web_search=use_web_search,
            max_web_results=max_web_results,
            clarification_choice=clarification_choice,
        )

    async def astream_chat_events(
        self,
        *,
        query: str,
        session_id: str | None = None,
        use_web_search: bool = False,
        max_web_results: int = 3,
        clarification_choice: dict[str, Any] | None = None,
    ) -> Any:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()
        _answer_buffer: str = ""
        _answer_streamed_live: bool = False

        def emit_event(item: dict[str, Any]) -> None:
            nonlocal _answer_buffer, _answer_streamed_live
            data = item.get("data", {})
            if isinstance(data, dict) and item.get("event") == "answer_delta":
                text = str(data.get("text", ""))
                if text:
                    _answer_buffer += text
                    _answer_streamed_live = True
                return  # suppress raw answer_delta; drip-feed after worker done
            loop.call_soon_threadsafe(queue.put_nowait, item)

        worker = asyncio.create_task(
            asyncio.to_thread(
                run_agent_chat_turn,
                agent=self,
                query=query,
                session_id=session_id,
                use_web_search=use_web_search,
                max_web_results=max_web_results,
                clarification_choice=clarification_choice,
                event_callback=emit_event,
            )
        )
        worker.add_done_callback(lambda _task: loop.call_soon_threadsafe(queue.put_nowait, sentinel))

        # Independent heartbeat task to keep SSE connection alive through proxies
        _worker_done = False

        async def _heartbeat(interval: float = 10.0) -> None:
            nonlocal _worker_done
            while not _worker_done:
                await asyncio.sleep(interval)
                if not _worker_done:
                    loop.call_soon_threadsafe(queue.put_nowait, {"event": "heartbeat", "data": ""})

        _heartbeat_task = asyncio.create_task(_heartbeat())

        while True:
            item = await queue.get()
            if item is sentinel:
                _worker_done = True
                _heartbeat_task.cancel()
                break
            yield item

        try:
            result, emitted_events = await asyncio.wait_for(worker, timeout=110.0)
        except asyncio.TimeoutError:
            result, emitted_events = {"answer": "请求超时，请简化问题重试。"}, []
        except Exception as exc:
            import logging
            _logger = logging.getLogger(__name__)
            _logger.exception("agent worker failed: %s", exc)
            answer = _answer_buffer or str(exc)
            yield {"event": "error", "data": normalize_agent_event("error", {"message": str(exc)})}
            yield {
                "event": "final",
                "data": normalize_agent_event(
                    "final", {"answer": answer, "error": str(exc)}
                ),
            }
            return

        final_payload = {k: v for k, v in result.items() if k != "answer"}
        # Trim heavy payload fields for lightweight SSE delivery
        final_payload = _trim_final_payload_for_sse(final_payload)
        answer_text = _answer_buffer or str(result.get("answer", ""))
        # Drip-feed answer in small chunks for true streaming feel
        chunk_size = 10 if _answer_streamed_live else 28
        for chunk in chunk_text(answer_text, size=chunk_size):
            yield {"event": "answer_delta", "data": {"text": chunk}}
            if _answer_streamed_live:
                await asyncio.sleep(0.015)  # ~15ms between chunks for visual streaming
        yield {
            "event": "final",
            "data": normalize_agent_event("final", final_payload | {"answer": answer_text}),
        }
