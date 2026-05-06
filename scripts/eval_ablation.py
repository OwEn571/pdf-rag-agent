"""Ablation benchmark: 3 retrieval strategies × 2 summary conditions × 2 QE conditions.

Evaluates the impact of:
  - LLM summary in paper_card (with vs without abstract_or_summary)
  - Query Enhancement (LLM-extracted targets vs raw query)

across 3 retrieval strategies:
  - Pure Dense (Milvus vector search / brute-force cosine)
  - Pure BM25 (jieba tokenizer)
  - BM25 + Dense RRF (equal weights)

Total: 12 configurations on 159 queries from eval_queries_v3.json.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

if str(PROJECT_ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.retrieval.core import DualIndexRetriever, _cjk_aware_tokenize
from app.services.planning.query_shaping import extract_targets

# ── Data structures ──────────────────────────────────────────────────


@dataclass
class TestQuery:
    query: str
    ground_truth_ids: list[str]
    query_type: str = ""
    difficulty: str = ""


@dataclass
class EvalResult:
    config_name: str
    total_queries: int = 0
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    precision_at_5: float = 0.0
    recall_at_5: float = 0.0
    avg_latency_ms: float = 0.0
    per_query: list[dict[str, Any]] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────


def strip_abstract_or_summary(page_content: str) -> str:
    """Remove the abstract_or_summary: section from a paper_card."""
    pattern = r'\n(abstract_or_summary|abstract_note|generated_summary):\s*\n?.*?(?=\n(?:top_evidence_hints|body_acronyms|authors|year|tags|aliases|title):|\Z)'
    result = re.sub(pattern, '', page_content, flags=re.DOTALL)
    # Also handle the field name itself if it appears
    result = re.sub(r'\n(?:abstract_or_summary|abstract_note|generated_summary):\s*', '', result)
    # Clean up double newlines
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def load_queries(path: Path) -> list[TestQuery]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    queries: list[TestQuery] = []
    for item in raw:
        q = str(item.get("query", "")).strip()
        gt_ids = [str(pid).strip() for pid in list(item.get("gt_ids", []))]
        if not q:
            continue
        queries.append(TestQuery(
            query=q,
            ground_truth_ids=gt_ids,
            query_type=str(item.get("type", "")),
            difficulty=str(item.get("difficulty", "")),
        ))
    return queries


def compute_metrics(retrieved_ids: list[str], gt_ids: list[str]) -> dict[str, float]:
    gt_set = set(gt_ids)
    metrics: dict[str, float] = {}

    for k in [1, 3, 5]:
        top_k = set(retrieved_ids[:k])
        metrics[f"hit@{k}"] = 1.0 if top_k & gt_set else 0.0

    rr = 0.0
    for rank, pid in enumerate(retrieved_ids, start=1):
        if pid in gt_set:
            rr = 1.0 / rank
            break
    metrics["rr"] = rr

    dcg = 0.0
    for rank, pid in enumerate(retrieved_ids[:5], start=1):
        relevance = 1.0 if pid in gt_set else 0.0
        dcg += relevance / math.log2(rank + 1)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt_set), 5)))
    metrics["ndcg@5"] = dcg / max(ideal_dcg, 1.0)

    relevant_in_top5 = sum(1 for pid in retrieved_ids[:5] if pid in gt_set)
    metrics["precision@5"] = relevant_in_top5 / max(1, len(retrieved_ids[:5]))
    metrics["recall@5"] = relevant_in_top5 / max(1, len(gt_set))

    return metrics


def avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


# ── LLM Query Enhancement ────────────────────────────────────────────


def build_llm_enhanced_queries(
    queries: list[TestQuery],
    retriever: DualIndexRetriever,
    settings: Any,
) -> dict[int, str]:
    """Pre-compute LLM-enhanced queries for all test queries.

    Uses the LLM to extract method names, acronyms, and key technical terms,
    then canonicalizes against paper titles. Returns a dict mapping query
    index to enhanced query text.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=30.0,
    )

    prompt_template = """You are a target term extractor for an academic paper retrieval system.
Given a user query about ML/AI papers, extract key technical terms that can be used to find the relevant paper.

Rules:
- Extract method names (e.g. LoRA, PPO, DPO, CLIP, BERT, GPT)
- Extract full paper titles or key phrases from titles if you know them
- Extract technical concepts (e.g. "reinforcement learning", "chain of thought")
- For Chinese queries, also output English equivalents you know from ML literature
- Output ONLY a JSON array of strings, nothing else
- If no clear technical terms, output an empty array []

Examples:
Query: "LoRA是什么？" → ["LoRA", "Low-Rank Adaptation"]
Query: "Transformer最早由哪篇论文提出" → ["Transformer", "Attention Is All You Need"]
Query: "残差网络的核心思想是什么？" → ["ResNet", "Deep Residual Learning", "residual network"]
Query: "DPO为什么能替代RLHF中的PPO？" → ["DPO", "Direct Preference Optimization", "RLHF", "PPO"]

Query: {query}

Output:"""

    enhanced: dict[int, str] = {}
    batch_size = 10  # Process in small batches to avoid rate limiting

    for batch_start in range(0, len(queries), batch_size):
        batch_end = min(batch_start + batch_size, len(queries))
        batch = queries[batch_start:batch_end]

        for i, tq in enumerate(batch):
            query_idx = batch_start + i
            query = tq.query

            # Start with regex targets
            regex_targets = extract_targets(query)
            canonical_regex = retriever.canonicalize_targets(regex_targets)

            # Only call LLM if regex didn't find good targets
            # (regex typically works for English acronym queries but not Chinese descriptive ones)
            has_good_regex = len(canonical_regex) >= 1

            llm_targets: list[str] = []
            if not has_good_regex or any('一' <= ch <= '鿿' for ch in query):
                # Chinese or descriptive queries benefit from LLM extraction
                for attempt in range(3):
                    try:
                        resp = client.chat.completions.create(
                            model=settings.chat_model or "gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt_template.format(query=query)}],
                            temperature=0.1,
                            max_tokens=200,
                        )
                        raw = resp.choices[0].message.content or "[]"
                        # Parse JSON array
                        json_match = re.search(r'\[.*?\]', raw, re.DOTALL)
                        if json_match:
                            llm_targets = json.loads(json_match.group())
                        break
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                        else:
                            print(f"  LLM QE failed for query {query_idx}: {e}")

                if llm_targets:
                    # Avoid duplicates with regex targets
                    regex_lower = {t.lower() for t in regex_targets}
                    llm_targets = [t for t in llm_targets if t.lower() not in regex_lower]

            # Combine and canonicalize
            all_targets = list(dict.fromkeys(regex_targets + llm_targets))
            canonical = retriever.canonicalize_targets(all_targets)

            # Build enhanced query
            target_text = " ".join(canonical).strip()
            if target_text and target_text.lower() not in query.lower():
                enhanced[query_idx] = f"{target_text} {query}"
            else:
                enhanced[query_idx] = query

        print(f"  QE pre-computation: {batch_end}/{len(queries)} queries done")

    return enhanced


# ── Retriever wrappers ────────────────────────────────────────────────


class DenseSearcher:
    """Dense search, optionally with brute-force on stripped docs."""

    def __init__(self, retriever: DualIndexRetriever, settings: Any,
                 use_summary: bool = True):
        self.retriever = retriever
        self.settings = settings
        self.use_summary = use_summary
        self._stripped_embeddings: np.ndarray | None = None
        self._stripped_paper_ids: list[str] | None = None
        self._embedding_model = None

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from langchain_openai import OpenAIEmbeddings
            import httpx
            settings = self.settings
            embedding_url = str(getattr(settings, "embedding_base_url", "") or settings.openai_base_url).strip()
            embedding_key = str(getattr(settings, "embedding_api_key", "") or "").strip() or settings.openai_api_key
            timeout = max(10.0, float(getattr(settings, "embedding_request_timeout_seconds", 30)))
            http_client = httpx.Client(
                trust_env=False,
                timeout=httpx.Timeout(timeout, connect=min(20.0, timeout), read=timeout, write=timeout, pool=timeout),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
            self._embedding_model = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=embedding_key,
                base_url=embedding_url,
                http_client=http_client,
            )
        return self._embedding_model

    def search(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        if self.use_summary:
            # Use Milvus directly (paper cards have summaries in their embeddings)
            docs = self.retriever._paper_dense.search_documents(query, limit=limit)
            allowed = [d for d in docs if self.retriever._is_allowed_library_doc(d)]
            return [
                {
                    "paper_id": str((d.metadata or {}).get("paper_id", "")),
                    "title": str((d.metadata or {}).get("title", "")),
                    "year": str((d.metadata or {}).get("year", "")),
                    "score": float((d.metadata or {}).get("dense_score", 0)),
                }
                for d in allowed
            ]
        else:
            # Brute-force cosine similarity on stripped paper docs
            return self._brute_force_search(query, limit)

    def _brute_force_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        if self._stripped_embeddings is None:
            self._build_stripped_index()

        model = self._get_embedding_model()
        query_vec = np.array(model.embed_query(query), dtype=np.float32)

        # Cosine similarity
        norms = np.linalg.norm(self._stripped_embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        similarities = np.dot(self._stripped_embeddings, query_vec) / (norms * query_norm + 1e-10)

        top_indices = np.argsort(-similarities)[:limit]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            paper_id = self._stripped_paper_ids[idx]
            doc = self.retriever._paper_docs_by_id.get(paper_id)
            title = str((doc.metadata or {}).get("title", "")) if doc else ""
            year = str((doc.metadata or {}).get("year", "")) if doc else ""
            results.append({
                "paper_id": paper_id,
                "title": title,
                "year": year,
                "score": float(similarities[idx]),
            })
        return results

    def _build_stripped_index(self):
        """Pre-compute embeddings for all paper docs with summaries stripped."""
        print("  Building stripped-doc dense index (no summary)...")
        model = self._get_embedding_model()
        paper_ids: list[str] = []
        texts: list[str] = []

        for doc in self.retriever._paper_docs:
            pid = str((doc.metadata or {}).get("paper_id", ""))
            if not pid:
                continue
            stripped_text = strip_abstract_or_summary(doc.page_content)
            paper_ids.append(pid)
            texts.append(stripped_text)

        # Embed in batches
        embeddings: list[np.ndarray] = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            for attempt in range(3):
                try:
                    vecs = model.embed_documents(batch)
                    embeddings.extend([np.array(v, dtype=np.float32) for v in vecs])
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        raise RuntimeError(f"Failed to embed batch: {e}")

        self._stripped_embeddings = np.stack(embeddings)
        self._stripped_paper_ids = paper_ids
        print(f"  Stripped index built: {len(paper_ids)} papers")


class BM25Searcher:
    """BM25 search with jieba tokenizer."""

    def __init__(self, retriever: DualIndexRetriever, use_summary: bool = True):
        self.retriever = retriever
        self.use_summary = use_summary
        self._bm25 = None
        self._build_index()

    def _build_index(self):
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document

        if self.use_summary:
            docs = list(self.retriever._paper_docs)
        else:
            docs = []
            for doc in self.retriever._paper_docs:
                stripped = Document(
                    page_content=strip_abstract_or_summary(doc.page_content),
                    metadata=doc.metadata,
                )
                docs.append(stripped)

        # Rebuild BM25 with our tokenizer
        self._bm25 = BM25Retriever.from_documents(docs, preprocess_func=_cjk_aware_tokenize)
        self._bm25.k = 12

    def search(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        if self._bm25 is None:
            return []
        docs = self._bm25.invoke(query)[:limit]
        return [
            {
                "paper_id": str((d.metadata or {}).get("paper_id", "")),
                "title": str((d.metadata or {}).get("title", "")),
                "year": str((d.metadata or {}).get("year", "")),
                "score": 0.0,
            }
            for d in docs
        ]


class HybridSearcher:
    """BM25 + Dense RRF with equal weights."""

    def __init__(self, retriever: DualIndexRetriever, settings: Any,
                 use_summary: bool = True):
        self.dense_searcher = DenseSearcher(retriever, settings, use_summary=use_summary)
        self.bm25_searcher = BM25Searcher(retriever, use_summary=use_summary)
        self.retriever = retriever

    def search(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        bm25_results = self.bm25_searcher.search(query, limit=12)
        dense_results = self.dense_searcher.search(query, limit=12)

        # RRF fusion
        merged: dict[str, tuple[dict[str, Any], float]] = {}
        for rank, r in enumerate(bm25_results, start=1):
            pid = r["paper_id"]
            prev, prev_score = merged.get(pid, (r, 0.0))
            merged[pid] = (prev, prev_score + 1.0 / (60 + rank))
        for rank, r in enumerate(dense_results, start=1):
            pid = r["paper_id"]
            prev, prev_score = merged.get(pid, (r, 0.0))
            merged[pid] = (prev, prev_score + 1.0 / (60 + rank))

        ranked = sorted(merged.values(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in ranked[:limit]]


# ── Main benchmark ───────────────────────────────────────────────────


def run_benchmark() -> None:
    settings = get_settings()
    retriever = DualIndexRetriever(settings)
    print(f"Loaded {len(retriever._paper_docs)} papers, {len(retriever._block_docs)} blocks")

    # Load queries from v3
    queries_path = settings.data_dir / "eval_queries_v3.json"
    queries = load_queries(queries_path)
    print(f"Loaded {len(queries)} queries from {queries_path}")
    closed = [q for q in queries if q.ground_truth_ids]
    print(f"Closed queries (with ground truth): {len(closed)}")
    queries = closed

    # Pre-compute LLM-enhanced queries
    print("\nPre-computing LLM-enhanced queries...")
    enhanced_queries = build_llm_enhanced_queries(queries, retriever, settings)
    enhanced_count = sum(1 for i, eq in enhanced_queries.items() if eq != queries[i].query)
    print(f"  Enhanced {enhanced_count}/{len(queries)} queries with LLM targets")

    # Define configurations
    configs: list[tuple[str, dict[str, Any]]] = []

    for use_summary in [True, False]:
        sum_label = "+Summary" if use_summary else "-Summary"
        for use_qe in [True, False]:
            qe_label = "+QE" if use_qe else "-QE"
            suffix = f" ({sum_label}, {qe_label})"

            configs.append((f"Pure Dense{suffix}", {
                "type": "dense",
                "use_summary": use_summary,
                "use_qe": use_qe,
            }))
            configs.append((f"Pure BM25{suffix}", {
                "type": "bm25",
                "use_summary": use_summary,
                "use_qe": use_qe,
            }))
            configs.append((f"BM25+Dense RRF{suffix}", {
                "type": "hybrid",
                "use_summary": use_summary,
                "use_qe": use_qe,
            }))

    print(f"\nRunning {len(configs)} configurations on {len(queries)} queries...\n")

    # Pre-build searchers (some are expensive to create)
    searcher_cache: dict[str, Any] = {}

    def get_searcher(config: dict[str, Any]) -> Any:
        key = f"{config['type']}_{config['use_summary']}"
        if key not in searcher_cache:
            if config["type"] == "dense":
                searcher_cache[key] = DenseSearcher(retriever, settings, use_summary=config["use_summary"])
            elif config["type"] == "bm25":
                searcher_cache[key] = BM25Searcher(retriever, use_summary=config["use_summary"])
            elif config["type"] == "hybrid":
                searcher_cache[key] = HybridSearcher(retriever, settings, use_summary=config["use_summary"])
        return searcher_cache[key]

    all_results: dict[str, EvalResult] = {}

    for config_name, config in configs:
        print(f"  Running: {config_name} ...", end=" ", flush=True)
        searcher = get_searcher(config)

        eval_result = EvalResult(config_name=config_name, total_queries=len(queries))
        all_metrics: dict[str, list[float]] = defaultdict(list)
        latencies: list[float] = []

        for i, tq in enumerate(queries):
            # Determine query text
            if config["use_qe"]:
                search_query = enhanced_queries.get(i, tq.query)
            else:
                search_query = tq.query

            start = time.perf_counter()
            retrieved = searcher.search(search_query, limit=6)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

            retrieved_ids = [r["paper_id"] for r in retrieved]
            m = compute_metrics(retrieved_ids, tq.ground_truth_ids)
            for k, v in m.items():
                all_metrics[k].append(v)

            eval_result.per_query.append({
                "query": tq.query,
                "ground_truth": tq.ground_truth_ids,
                "retrieved": [
                    {"paper_id": r["paper_id"], "title": r["title"][:60]}
                    for r in retrieved[:5]
                ],
                "metrics": m,
                "latency_ms": round(elapsed_ms, 1),
            })

        eval_result.hit_at_1 = avg(all_metrics["hit@1"])
        eval_result.hit_at_3 = avg(all_metrics["hit@3"])
        eval_result.hit_at_5 = avg(all_metrics["hit@5"])
        eval_result.mrr = avg(all_metrics["rr"])
        eval_result.ndcg_at_5 = avg(all_metrics["ndcg@5"])
        eval_result.precision_at_5 = avg(all_metrics["precision@5"])
        eval_result.recall_at_5 = avg(all_metrics["recall@5"])
        eval_result.avg_latency_ms = avg(latencies)

        all_results[config_name] = eval_result
        print(f"Hit@1={eval_result.hit_at_1:.4f}, MRR={eval_result.mrr:.4f}, Lat={eval_result.avg_latency_ms:.0f}ms")

    # ── Print results matrix ──────────────────────────────────────────
    print("\n" + "=" * 120)
    print("ABLATION BENCHMARK RESULTS — 12 Configurations")
    print(f"Queries: {len(queries)}  |  Embedding: {settings.embedding_model}")
    print("=" * 120)

    # Group by strategy and condition for matrix display
    strategies = ["Pure Dense", "Pure BM25", "BM25+Dense RRF"]
    conditions = ["+Summary, +QE", "+Summary, -QE", "-Summary, +QE", "-Summary, -QE"]

    header = f"{'Strategy':<22} {'Condition':<18} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'MRR':>7} {'NDCG@5':>7} {'P@5':>7} {'R@5':>7} {'Lat(ms)':>8}"
    print(header)
    print("-" * 120)

    matrix: dict[str, dict[str, EvalResult]] = {}
    for strategy in strategies:
        matrix[strategy] = {}
        for cond in conditions:
            name = f"{strategy} ({cond})"
            matrix[strategy][cond] = all_results.get(name)

    for strategy in strategies:
        for j, cond in enumerate(conditions):
            r = matrix[strategy][cond]
            if r is None:
                continue
            display_strategy = strategy if j == 0 else ""
            row = f"{display_strategy:<22} {cond:<18} {r.hit_at_1:7.4f} {r.hit_at_3:7.4f} {r.hit_at_5:7.4f} {r.mrr:7.4f} {r.ndcg_at_5:7.4f} {r.precision_at_5:7.4f} {r.recall_at_5:7.4f} {r.avg_latency_ms:7.0f}"
            print(row)
        if strategy != strategies[-1]:
            print("-" * 120)

    # ── Ablation analysis ─────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("ABLATION ANALYSIS")
    print("=" * 120)

    # Summary impact
    print("\n── Summary Ablation (averaged across strategies and QE) ──")
    for metric_name, metric_key in [("Hit@1", "hit_at_1"), ("MRR", "mrr"), ("NDCG@5", "ndcg_at_5")]:
        with_sum = avg([getattr(all_results.get(f"{s} (+Summary, +QE)", None), metric_key, 0) or 0 for s in strategies] +
                       [getattr(all_results.get(f"{s} (+Summary, -QE)", None), metric_key, 0) or 0 for s in strategies])
        without_sum = avg([getattr(all_results.get(f"{s} (-Summary, +QE)", None), metric_key, 0) or 0 for s in strategies] +
                          [getattr(all_results.get(f"{s} (-Summary, -QE)", None), metric_key, 0) or 0 for s in strategies])
        delta = with_sum - without_sum
        print(f"  {metric_name}: +Summary={with_sum:.4f}  -Summary={without_sum:.4f}  Δ={delta:+.4f} ({delta/without_sum*100:+.1f}%)" if without_sum else f"  {metric_name}: +Summary={with_sum:.4f}  -Summary={without_sum:.4f}")

    # QE impact
    print("\n── QE Ablation (averaged across strategies and summary) ──")
    for metric_name, metric_key in [("Hit@1", "hit_at_1"), ("MRR", "mrr"), ("NDCG@5", "ndcg_at_5")]:
        with_qe = avg([getattr(all_results.get(f"{s} (+Summary, +QE)", None), metric_key, 0) or 0 for s in strategies] +
                      [getattr(all_results.get(f"{s} (-Summary, +QE)", None), metric_key, 0) or 0 for s in strategies])
        without_qe = avg([getattr(all_results.get(f"{s} (+Summary, -QE)", None), metric_key, 0) or 0 for s in strategies] +
                         [getattr(all_results.get(f"{s} (-Summary, -QE)", None), metric_key, 0) or 0 for s in strategies])
        delta = with_qe - without_qe
        print(f"  {metric_name}: +QE={with_qe:.4f}  -QE={without_qe:.4f}  Δ={delta:+.4f} ({delta/without_qe*100:+.1f}%)" if without_qe else f"  {metric_name}: +QE={with_qe:.4f}  -QE={without_qe:.4f}")

    # Per-strategy breakdown
    print("\n── Per-Strategy Summary Impact (Hit@1) ──")
    for strategy in strategies:
        with_sum = avg([
            getattr(all_results.get(f"{strategy} (+Summary, +QE)", None), "hit_at_1", 0) or 0,
            getattr(all_results.get(f"{strategy} (+Summary, -QE)", None), "hit_at_1", 0) or 0,
        ])
        without_sum = avg([
            getattr(all_results.get(f"{strategy} (-Summary, +QE)", None), "hit_at_1", 0) or 0,
            getattr(all_results.get(f"{strategy} (-Summary, -QE)", None), "hit_at_1", 0) or 0,
        ])
        delta = with_sum - without_sum
        print(f"  {strategy:<22}: +Summary={with_sum:.4f}  -Summary={without_sum:.4f}  Δ={delta:+.4f}")

    print("\n── Per-Strategy QE Impact (Hit@1) ──")
    for strategy in strategies:
        with_qe = avg([
            getattr(all_results.get(f"{strategy} (+Summary, +QE)", None), "hit_at_1", 0) or 0,
            getattr(all_results.get(f"{strategy} (-Summary, +QE)", None), "hit_at_1", 0) or 0,
        ])
        without_qe = avg([
            getattr(all_results.get(f"{strategy} (+Summary, -QE)", None), "hit_at_1", 0) or 0,
            getattr(all_results.get(f"{strategy} (-Summary, -QE)", None), "hit_at_1", 0) or 0,
        ])
        delta = with_qe - without_qe
        print(f"  {strategy:<22}: +QE={with_qe:.4f}  -QE={without_qe:.4f}  Δ={delta:+.4f}")

    # ── Save results ──────────────────────────────────────────────────
    output_path = settings.data_dir / "eval_ablation_results.json"
    payload: dict[str, Any] = {}
    for name, r in all_results.items():
        payload[name] = {
            "config": name,
            "total_queries": r.total_queries,
            "hit_at_1": round(r.hit_at_1, 4),
            "hit_at_3": round(r.hit_at_3, 4),
            "hit_at_5": round(r.hit_at_5, 4),
            "mrr": round(r.mrr, 4),
            "ndcg_at_5": round(r.ndcg_at_5, 4),
            "precision_at_5": round(r.precision_at_5, 4),
            "recall_at_5": round(r.recall_at_5, 4),
            "avg_latency_ms": round(r.avg_latency_ms, 1),
            "per_query": r.per_query[:5],
        }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {output_path}")

    # ── Difficulty-stratified breakdown ───────────────────────────────
    print("\n" + "=" * 120)
    print("DIFFICULTY-STRATIFIED BREAKDOWN")
    print("=" * 120)

    for diff in ["easy", "medium", "hard"]:
        diff_queries = [(i, q) for i, q in enumerate(queries) if q.difficulty == diff]
        if not diff_queries:
            continue
        print(f"\n── {diff} ({len(diff_queries)} queries) ──")
        print(f"{'Config':<45} {'Hit@1':>7} {'MRR':>7}")

        # Only show key configs for brevity
        key_configs = [
            "Pure Dense (+Summary, -QE)",
            "Pure Dense (-Summary, -QE)",
            "Pure BM25 (+Summary, -QE)",
            "Pure BM25 (-Summary, -QE)",
            "BM25+Dense RRF (+Summary, -QE)",
            "BM25+Dense RRF (-Summary, -QE)",
            "Pure Dense (+Summary, +QE)",
            "Pure Dense (-Summary, +QE)",
        ]
        for config_name in key_configs:
            r = all_results.get(config_name)
            if r is None:
                continue
            # Recompute for this subset
            diff_metrics: dict[str, list[float]] = defaultdict(list)
            for qi, _ in diff_queries:
                pq = r.per_query[qi]
                for k, v in pq["metrics"].items():
                    diff_metrics[k].append(v)
            hit1 = avg(diff_metrics.get("hit@1", []))
            mrr_val = avg(diff_metrics.get("rr", []))
            print(f"  {config_name:<45} {hit1:7.4f} {mrr_val:7.4f}")

    retriever.close()
    print("\nDone.")


if __name__ == "__main__":
    run_benchmark()
