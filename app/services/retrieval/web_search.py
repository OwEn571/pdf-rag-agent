from __future__ import annotations

import hashlib
import logging
from typing import Any

import httpx

from app.core.config import Settings
from app.domain.models import EvidenceBlock

logger = logging.getLogger(__name__)


class TavilyWebSearchClient:
    endpoint = "https://api.tavily.com/search"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.tavily_api_key)

    def search(
        self,
        *,
        query: str,
        max_results: int = 5,
        topic: str = "general",
        include_domains: list[str] | None = None,
    ) -> list[EvidenceBlock]:
        if not self.is_configured:
            return []
        max_results = max(1, min(int(max_results or 5), 10))
        payload: dict[str, Any] = {
            "query": query,
            "topic": topic if topic in {"general", "news", "finance"} else "general",
            "search_depth": self._search_depth(),
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "max_results": max_results,
        }
        if include_domains:
            payload["include_domains"] = include_domains[:20]
        try:
            response = httpx.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.settings.tavily_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.settings.tavily_timeout_seconds,
                follow_redirects=True,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("tavily search failed: %s", exc)
            return []
        results = data.get("results", [])
        if not isinstance(results, list):
            return []
        evidence: list[EvidenceBlock] = []
        for rank, item in enumerate(results[:max_results], start=1):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "") or "").strip()
            title = str(item.get("title", "") or url or "Web result").strip()
            content = str(item.get("content", "") or item.get("raw_content", "") or "").strip()
            if not url and not content:
                continue
            score = self._coerce_score(item.get("score")) or (1.0 / rank)
            doc_id = "web::" + hashlib.sha1(f"{url}\n{title}".encode("utf-8")).hexdigest()[:16]
            evidence.append(
                EvidenceBlock(
                    doc_id=doc_id,
                    paper_id=doc_id,
                    title=title,
                    file_path=url,
                    page=0,
                    block_type="web",
                    caption=url,
                    snippet=content[:1600] or title,
                    score=score,
                    metadata={
                        "source": "tavily",
                        "url": url,
                        "rank": rank,
                        "query": query,
                        "topic": payload["topic"],
                    },
                )
            )
        return evidence

    def _search_depth(self) -> str:
        value = str(self.settings.tavily_search_depth or "basic").strip().lower()
        return value if value in {"basic", "advanced", "fast", "ultra-fast"} else "basic"

    @staticmethod
    def _coerce_score(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
