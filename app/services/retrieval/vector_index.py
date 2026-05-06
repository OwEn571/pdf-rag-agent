from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from pymilvus import MilvusClient, connections

from app.core.config import Settings

logger = logging.getLogger(__name__)


class CompatibleMilvus(Milvus):
    def _init(  # type: ignore[override]
        self,
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        partition_names: list[str] | None = None,
        replica_number: int = 1,
        timeout: float | None = None,
    ) -> None:
        if not connections.has_connection(self.alias):
            connections.connect(alias=self.alias, **self._connection_args)
        super()._init(
            embeddings=embeddings,
            metadatas=metadatas,
            partition_names=partition_names,
            replica_number=replica_number,
            timeout=timeout,
        )


class CollectionVectorIndex:
    def __init__(self, settings: Settings, *, collection_name: str, embedding_model: str | None = None) -> None:
        self.settings = settings
        self.collection_name = collection_name
        self.embedding_model = embedding_model or settings.embedding_model
        self._embeddings: OpenAIEmbeddings | None = None
        self._http_client: httpx.Client | None = None
        self._dense_search_disabled = False
        self._dense_search_disabled_until: float = 0.0  # P2-8: timestamp for auto-recovery

    @property
    def http_client(self) -> httpx.Client:
        if self._http_client is None:
            timeout = max(10.0, float(self.settings.embedding_request_timeout_seconds))
            self._http_client = httpx.Client(
                trust_env=False,
                timeout=httpx.Timeout(timeout, connect=min(20.0, timeout), read=timeout, write=timeout, pool=timeout),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._http_client

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        if self._embeddings is None:
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY 未配置，无法做 embedding。")
            embedding_url = str(getattr(self.settings, "embedding_base_url", "") or self.settings.openai_base_url).strip()
            if not embedding_url:
                embedding_url = self.settings.openai_base_url
            embedding_key = str(getattr(self.settings, "embedding_api_key", "") or "").strip()
            if not embedding_key:
                embedding_key = self.settings.openai_api_key
            self._embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=embedding_key,
                base_url=embedding_url,
                http_client=self.http_client,
            )
        return self._embeddings

    def upsert_documents(
        self,
        docs: list[Document],
        *,
        force_rebuild: bool = False,
        batch_size: int = 128,
        doc_ids: list[str] | None = None,
    ) -> int:
        if not docs:
            return 0
        client = MilvusClient(uri=self.settings.milvus_uri)
        if force_rebuild:
            self._drop_collection_if_exists()
        total = len(docs)
        started = time.perf_counter()
        batch_size = max(1, batch_size)
        start_offset = 0
        if not force_rebuild:
            start_offset = min(total, self._existing_row_count(client))
            if start_offset:
                logger.info(
                    "milvus resume collection=%s start_offset=%s total=%s",
                    self.collection_name,
                    start_offset,
                    total,
                )
        if start_offset >= total:
            return total
        vector_store = self._create_vector_store()
        for start in range(start_offset, total, batch_size):
            end = min(total, start + batch_size)
            batch = docs[start:end]
            batch_ids = doc_ids[start:end] if doc_ids else None
            vector_store = self._add_documents_with_retry(
                vector_store=vector_store,
                batch=batch,
                batch_ids=batch_ids,
                start=start,
                end=end,
                total=total,
            )
            logger.info(
                "milvus upsert progress collection=%s done=%s/%s elapsed=%.2fs",
                self.collection_name,
                end,
                total,
                time.perf_counter() - started,
            )
        return total

    def search_documents(self, query: str, *, limit: int, filter_expr: str | None = None) -> list[Document]:
        query = str(query or "").strip()
        # P2-8: Auto-recover dense search after TTL expires
        if self._dense_search_disabled and self._dense_search_disabled_until > 0:
            import time as _time
            if _time.time() > self._dense_search_disabled_until:
                self._dense_search_disabled = False
                self._dense_search_disabled_until = 0.0
                logger.info("milvus dense search re-enabled for collection=%s after TTL expiry", self.collection_name)
        if not query or not self.settings.openai_api_key or self._dense_search_disabled:
            return []
        client = MilvusClient(uri=self.settings.milvus_uri)
        if not client.has_collection(collection_name=self.collection_name):
            return []
        vector = self.embeddings.embed_query(query)
        try:
            rows = client.search(
                collection_name=self.collection_name,
                data=[vector],
                limit=max(1, limit),
                filter=filter_expr or "",
                output_fields=["*"],
            )
        except Exception as exc:  # noqa: BLE001
            if "vector dimension mismatch" in str(exc).lower():
                # P2-8: Use timestamp-based flag for automatic recovery
                import time as _time
                self._dense_search_disabled_until = _time.time() + 600  # retry every 10 min
                self._dense_search_disabled = True
                logger.warning(
                    "milvus dense search disabled for collection=%s due to embedding dimension mismatch model=%s (retry in 10min)",
                    self.collection_name,
                    self.embedding_model,
                )
            logger.warning("milvus search failed collection=%s err=%s", self.collection_name, exc)
            return []
        hits = rows[0] if rows else []
        docs: list[Document] = []
        for hit in hits:
            payload: dict[str, Any] = {}
            if isinstance(hit, dict):
                entity = hit.get("entity")
                if isinstance(entity, dict):
                    payload.update(entity)
                payload.update({k: v for k, v in hit.items() if k not in {"entity", "vector", "embedding"}})
            text = str(payload.pop("text", "") or "")
            if not text:
                continue
            score = payload.pop("distance", payload.pop("score", 0.0))
            payload["dense_score"] = float(score or 0.0)
            docs.append(Document(page_content=text, metadata=payload))
        return docs

    def close(self) -> None:
        self._reset_embedding_client()

    def _create_vector_store(self) -> Milvus:
        return CompatibleMilvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args={"uri": self.settings.milvus_uri},
            drop_old=False,
            auto_id=False,
        )

    def _add_documents_with_retry(
        self,
        *,
        vector_store: Milvus,
        batch: list[Document],
        batch_ids: list[str] | None,
        start: int,
        end: int,
        total: int,
    ) -> Milvus:
        attempts = max(1, int(self.settings.embedding_batch_retry_attempts))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                vector_store.add_documents(batch, ids=batch_ids)
                return vector_store
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "milvus batch add failed collection=%s batch=%s-%s/%s attempt=%s/%s err=%s",
                    self.collection_name,
                    start,
                    end,
                    total,
                    attempt,
                    attempts,
                    exc,
                )
                self._reset_embedding_client()
                vector_store = self._create_vector_store()
                time.sleep(min(8.0, float(attempt)))
        raise RuntimeError(f"failed to upsert batch {start}-{end} into {self.collection_name}: {last_error}")

    def _existing_row_count(self, client: MilvusClient) -> int:
        if not client.has_collection(collection_name=self.collection_name):
            return 0
        try:
            stats = client.get_collection_stats(collection_name=self.collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to read collection stats collection=%s err=%s", self.collection_name, exc)
            return 0
        try:
            return int(stats.get("row_count", 0) or 0)
        except (TypeError, ValueError, AttributeError):
            return 0

    def _reset_embedding_client(self) -> None:
        if self._http_client is not None:
            try:
                self._http_client.close()
            except Exception:  # noqa: BLE001
                pass
        self._http_client = None
        self._embeddings = None

    def _drop_collection_if_exists(self) -> None:
        client = MilvusClient(uri=self.settings.milvus_uri)
        if client.has_collection(collection_name=self.collection_name):
            client.drop_collection(collection_name=self.collection_name)
