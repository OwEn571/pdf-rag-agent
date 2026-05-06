"""Ablation benchmark V2 — uses the SAME code path as the production agent.

Key fix from V1: QE is now tested via paper_query_text(contract), matching
run_agent_paper_search() in runtime_helpers.py.  This is the REAL mechanism
by which Router-extracted targets enter the search query.

Conditions:
  +QE = paper_query_text(contract)  ← what the agent actually does
  -QE = contract.clean_query        ← true ablation (targets ignored)

3 strategies × 2 summary × 2 QE = 12 configs, same as before but correct.
"""

from __future__ import annotations

import json
import math
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
from app.domain.models import QueryContract, SessionContext
from app.services.retrieval.core import DualIndexRetriever, _cjk_aware_tokenize
from app.services.planning.query_shaping import paper_query_text, extract_targets
from app.services.intents.router import LLMIntentRouter
from app.services.infra.model_clients import ModelClients


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


# ── Helpers ──────────────────────────────────────────────────────────

def strip_abstract_or_summary(page_content: str) -> str:
    result = re.sub(
        r'\n(?:abstract_or_summary|abstract_note|generated_summary):\s*\n?.*?(?=\n(?:top_evidence_hints|body_acronyms|authors|year|tags|aliases|title):|\Z)',
        '', page_content, flags=re.DOTALL,
    )
    result = re.sub(r'\n(?:abstract_or_summary|abstract_note|generated_summary):\s*', '', result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def load_queries(path: Path) -> list[TestQuery]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    queries = []
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
    m = {}
    for k in [1, 3, 5]:
        m[f"hit@{k}"] = 1.0 if set(retrieved_ids[:k]) & gt_set else 0.0
    rr = 0.0
    for rank, pid in enumerate(retrieved_ids, start=1):
        if pid in gt_set:
            rr = 1.0 / rank
            break
    m["rr"] = rr
    dcg = sum((1.0 if pid in gt_set else 0.0) / math.log2(rank + 2)
              for rank, pid in enumerate(retrieved_ids[:5]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt_set), 5)))
    m["ndcg@5"] = dcg / max(idcg, 1.0)
    rel5 = sum(1 for pid in retrieved_ids[:5] if pid in gt_set)
    m["precision@5"] = rel5 / max(1, len(retrieved_ids[:5]))
    m["recall@5"] = rel5 / max(1, len(gt_set))
    return m


def avg(vals):
    return sum(vals) / len(vals) if vals else 0.0


# ── Build contracts with Router (same as production agent) ───────────

def build_contracts(
    queries: list[TestQuery],
    clients: ModelClients,
) -> list[QueryContract]:
    """Run Router on each query to get targets, matching the agent's code path."""
    router = LLMIntentRouter(
        clients=clients,
        conversation_context=lambda session, max_chars=12000: "",
        conversation_messages=lambda session: [],
    )
    contracts = []
    for i, tq in enumerate(queries):
        session = SessionContext(session_id=f"bench_{i}")
        try:
            decision = router.route(query=tq.query, session=session)
        except Exception:
            # Router failed — use regex-only fallback
            contracts.append(QueryContract(
                clean_query=tq.query,
                interaction_mode="research",
                relation="general_question",
                targets=extract_targets(tq.query),
                answer_slots=["general_answer"],
                requested_fields=["title", "authors", "year"],
                required_modalities=["page_text", "paper_card"],
                answer_shape="narrative",
                precision_requirement="normal",
                continuation_mode="fresh",
                allow_web_search=False,
                notes=["router_fallback"],
            ))
            continue

        # Build contract same way as extract_agent_query_contract does
        from app.services.intents.router import query_contract_from_router_decision
        from app.services.contracts.normalization import normalize_contract_targets

        regex_targets = extract_targets(tq.query)
        try:
            contract = query_contract_from_router_decision(
                decision=decision,
                clean_query=tq.query,
                session=session,
                extracted_targets=regex_targets,
                normalize_targets=lambda raw_targets, requested_fields: normalize_contract_targets(
                    targets=raw_targets,
                    requested_fields=requested_fields,
                    canonicalize_targets=lambda values: values,  # no retriever yet
                ),
                confidence_floor=0.6,
            )
        except Exception:
            contract = None

        # Sanitize precision_requirement
        if contract is not None and contract.precision_requirement not in ("exact", "high", "normal"):
            contract = contract.model_copy(update={"precision_requirement": "normal"})
        if contract is None:
            contract = QueryContract(
                clean_query=tq.query,
                interaction_mode="research",
                relation="general_question",
                targets=regex_targets,
                answer_slots=["general_answer"],
                requested_fields=["title", "authors", "year"],
                required_modalities=["page_text", "paper_card"],
                answer_shape="narrative",
                precision_requirement="normal",
                continuation_mode="fresh",
                allow_web_search=False,
                notes=["router_miss"],
            )
        # Canonicalize targets against paper titles
        if hasattr(contract, 'targets') and contract.targets:
            # We'll canonicalize after retriever is loaded
            pass
        contracts.append(contract)

        if (i + 1) % 20 == 0:
            print(f"  Router contracts: {i+1}/{len(queries)} done")
    return contracts


# ── Searchers (same as V1) ───────────────────────────────────────────

class DenseSearcher:
    def __init__(self, retriever, settings, use_summary=True):
        self.retriever = retriever
        self.settings = settings
        self.use_summary = use_summary
        self._stripped_embeddings = None
        self._stripped_paper_ids = None
        self._embedding_model = None

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from langchain_openai import OpenAIEmbeddings
            import httpx
            s = self.settings
            eu = str(getattr(s, "embedding_base_url", "") or s.openai_base_url).strip()
            ek = str(getattr(s, "embedding_api_key", "") or "").strip() or s.openai_api_key
            to = max(10.0, float(getattr(s, "embedding_request_timeout_seconds", 30)))
            hc = httpx.Client(trust_env=False, timeout=httpx.Timeout(to, connect=min(20.0, to), read=to, write=to, pool=to),
                             limits=httpx.Limits(max_connections=10, max_keepalive_connections=5))
            self._embedding_model = OpenAIEmbeddings(model=s.embedding_model, api_key=ek, base_url=eu, http_client=hc)
        return self._embedding_model

    def search(self, query, limit=6):
        if self.use_summary:
            docs = self.retriever._paper_dense.search_documents(query, limit=limit)
            allowed = [d for d in docs if self.retriever._is_allowed_library_doc(d)]
            return [{"paper_id": str((d.metadata or {}).get("paper_id", "")),
                     "title": str((d.metadata or {}).get("title", "")),
                     "year": str((d.metadata or {}).get("year", "")),
                     "score": float((d.metadata or {}).get("dense_score", 0))} for d in allowed]
        else:
            return self._brute_force_search(query, limit)

    def _brute_force_search(self, query, limit):
        if self._stripped_embeddings is None:
            self._build_stripped_index()
        model = self._get_embedding_model()
        qv = np.array(model.embed_query(query), dtype=np.float32)
        norms = np.linalg.norm(self._stripped_embeddings, axis=1)
        qn = np.linalg.norm(qv)
        if qn == 0:
            return []
        sims = np.dot(self._stripped_embeddings, qv) / (norms * qn + 1e-10)
        results = []
        for idx in np.argsort(-sims)[:limit]:
            pid = self._stripped_paper_ids[idx]
            doc = self.retriever._paper_docs_by_id.get(pid)
            results.append({"paper_id": pid,
                           "title": str((doc.metadata or {}).get("title", "")) if doc else "",
                           "year": str((doc.metadata or {}).get("year", "")) if doc else "",
                           "score": float(sims[idx])})
        return results

    def _build_stripped_index(self):
        print("  Building stripped-doc dense index (no summary)...")
        model = self._get_embedding_model()
        paper_ids, texts = [], []
        for doc in self.retriever._paper_docs:
            pid = str((doc.metadata or {}).get("paper_id", ""))
            if pid:
                paper_ids.append(pid)
                texts.append(strip_abstract_or_summary(doc.page_content))
        embs = []
        for start in range(0, len(texts), 64):
            batch = texts[start:start + 64]
            for attempt in range(3):
                try:
                    embs.extend([np.array(v, dtype=np.float32) for v in model.embed_documents(batch)])
                    break
                except Exception:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
        self._stripped_embeddings = np.stack(embs)
        self._stripped_paper_ids = paper_ids
        print(f"  Stripped index built: {len(paper_ids)} papers")


class BM25Searcher:
    def __init__(self, retriever, use_summary=True):
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document
        if use_summary:
            docs = list(retriever._paper_docs)
        else:
            docs = [Document(page_content=strip_abstract_or_summary(d.page_content), metadata=d.metadata)
                    for d in retriever._paper_docs]
        self._bm25 = BM25Retriever.from_documents(docs, preprocess_func=_cjk_aware_tokenize)
        self._bm25.k = 12

    def search(self, query, limit=6):
        if self._bm25 is None:
            return []
        docs = self._bm25.invoke(query)[:limit]
        return [{"paper_id": str((d.metadata or {}).get("paper_id", "")),
                 "title": str((d.metadata or {}).get("title", "")),
                 "year": str((d.metadata or {}).get("year", "")),
                 "score": 0.0} for d in docs]


class HybridSearcher:
    def __init__(self, retriever, settings, use_summary=True):
        self.dense = DenseSearcher(retriever, settings, use_summary)
        self.bm25 = BM25Searcher(retriever, use_summary)

    def search(self, query, limit=6):
        bm25_r = self.bm25.search(query, limit=12)
        dense_r = self.dense.search(query, limit=12)
        merged = {}
        for rank, r in enumerate(bm25_r, 1):
            merged[r["paper_id"]] = (r, 1.0 / (60 + rank))
        for rank, r in enumerate(dense_r, 1):
            prev = merged.get(r["paper_id"], (r, 0.0))
            merged[r["paper_id"]] = (prev[0], prev[1] + 1.0 / (60 + rank))
        return [v[0] for v in sorted(merged.values(), key=lambda x: -x[1])[:limit]]


# ── Main ─────────────────────────────────────────────────────────────

def main():
    settings = get_settings()
    retriever = DualIndexRetriever(settings)
    print(f"Loaded {len(retriever._paper_docs)} papers, {len(retriever._block_docs)} blocks")

    queries_path = settings.data_dir / "eval_queries_v3.json"
    queries = load_queries(queries_path)
    closed = [q for q in queries if q.ground_truth_ids]
    print(f"Loaded {len(closed)} closed queries from {queries_path}")

    # Step 1: Build contracts via Router (same as production agent)
    print("\nBuilding QueryContracts via Router (same as production agent)...")
    clients = ModelClients(settings)
    contracts = build_contracts(closed, clients)

    # Canonicalize targets against retriever
    for contract in contracts:
        if contract.targets:
            contract.targets = retriever.canonicalize_targets(contract.targets)

    # Count how many have targets
    with_targets = sum(1 for c in contracts if c.targets)
    print(f"  Contracts with Router targets: {with_targets}/{len(contracts)}")

    # Step 2: Build search queries for +QE and -QE
    qe_queries = []   # paper_query_text(contract) — what the agent does
    noqe_queries = []  # contract.clean_query — true ablation
    for contract in contracts:
        qe_queries.append(paper_query_text(contract))
        noqe_queries.append(contract.clean_query)

    # Show a few examples
    print("\n  Example queries:")
    for i in [0, 1, 2]:
        print(f"    [{closed[i].query}]")
        print(f"      +QE (paper_query_text): '{qe_queries[i]}'")
        print(f"      -QE (clean_query):      '{noqe_queries[i]}'")

    # Step 3: Run 12 configs
    print(f"\nRunning 12 configurations on {len(closed)} queries...\n")

    configs = []
    for use_summary in [True, False]:
        for use_qe in [True, False]:
            queries_list = qe_queries if use_qe else noqe_queries
            sl = "+Sum" if use_summary else "-Sum"
            ql = "+QE" if use_qe else "-QE"
            for stype, sname in [("dense", "Pure Dense"), ("bm25", "Pure BM25"), ("hybrid", "BM25+Dense RRF")]:
                configs.append((f"{sname} ({sl}, {ql})", {
                    "type": stype, "use_summary": use_summary, "queries": queries_list,
                }))

    searcher_cache = {}
    all_results = {}

    for config_name, config in configs:
        key = f"{config['type']}_{config['use_summary']}"
        if key not in searcher_cache:
            if config["type"] == "dense":
                searcher_cache[key] = DenseSearcher(retriever, settings, config["use_summary"])
            elif config["type"] == "bm25":
                searcher_cache[key] = BM25Searcher(retriever, config["use_summary"])
            else:
                searcher_cache[key] = HybridSearcher(retriever, settings, config["use_summary"])
        searcher = searcher_cache[key]

        eval_result = EvalResult(config_name=config_name, total_queries=len(closed))
        all_m = defaultdict(list)
        lats = []

        for i, tq in enumerate(closed):
            sq = config["queries"][i]
            t0 = time.perf_counter()
            retrieved = searcher.search(sq, limit=6)
            lats.append((time.perf_counter() - t0) * 1000)
            m = compute_metrics([r["paper_id"] for r in retrieved], tq.ground_truth_ids)
            for k, v in m.items():
                all_m[k].append(v)

        eval_result.hit_at_1 = avg(all_m["hit@1"])
        eval_result.hit_at_3 = avg(all_m["hit@3"])
        eval_result.hit_at_5 = avg(all_m["hit@5"])
        eval_result.mrr = avg(all_m["rr"])
        eval_result.ndcg_at_5 = avg(all_m["ndcg@5"])
        eval_result.precision_at_5 = avg(all_m["precision@5"])
        eval_result.recall_at_5 = avg(all_m["recall@5"])
        eval_result.avg_latency_ms = avg(lats)
        all_results[config_name] = eval_result
        print(f"  {config_name:<45} Hit@1={eval_result.hit_at_1:.4f}  MRR={eval_result.mrr:.4f}  Lat={eval_result.avg_latency_ms:.0f}ms")

    # Print results matrix
    print("\n" + "=" * 110)
    print("ABLATION BENCHMARK V2 — Using paper_query_text (same as production agent)")
    print(f"Queries: {len(closed)}  |  Embedding: {settings.embedding_model}")
    print("=" * 110)

    strategies = ["Pure Dense", "Pure BM25", "BM25+Dense RRF"]
    conditions = ["+Summary, +QE", "+Summary, -QE", "-Summary, +QE", "-Summary, -QE"]

    hdr = f"{'Strategy':<22} {'Condition':<18} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'MRR':>7} {'NDCG@5':>7} {'P@5':>7} {'R@5':>7} {'Lat(ms)':>8}"
    print(hdr)
    print("-" * 110)
    for strategy in strategies:
        for j, cond in enumerate(conditions):
            r = all_results.get(f"{strategy} ({cond})")
            if r is None:
                continue
            ds = strategy if j == 0 else ""
            print(f"{ds:<22} {cond:<18} {r.hit_at_1:7.4f} {r.hit_at_3:7.4f} {r.hit_at_5:7.4f} {r.mrr:7.4f} {r.ndcg_at_5:7.4f} {r.precision_at_5:7.4f} {r.recall_at_5:7.4f} {r.avg_latency_ms:7.0f}")
        if strategy != strategies[-1]:
            print("-" * 110)

    # Ablation analysis
    print("\n── Summary Ablation (paper_query_text QE) ──")
    for metric, key in [("Hit@1", "hit_at_1"), ("MRR", "mrr")]:
        with_s = avg([getattr(all_results.get(f"{s} (+Summary, +QE)"), key, 0) or 0 for s in strategies] +
                     [getattr(all_results.get(f"{s} (+Summary, -QE)"), key, 0) or 0 for s in strategies])
        without_s = avg([getattr(all_results.get(f"{s} (-Summary, +QE)"), key, 0) or 0 for s in strategies] +
                        [getattr(all_results.get(f"{s} (-Summary, -QE)"), key, 0) or 0 for s in strategies])
        d = with_s - without_s
        pct = f"{d/without_s*100:+.1f}%" if without_s else "N/A"
        print(f"  {metric}: +Summary={with_s:.4f}  -Summary={without_s:.4f}  Δ={d:+.4f} ({pct})")

    print("\n── QE Ablation (paper_query_text) ──")
    for metric, key in [("Hit@1", "hit_at_1"), ("MRR", "mrr")]:
        with_qe = avg([getattr(all_results.get(f"{s} (+Summary, +QE)"), key, 0) or 0 for s in strategies] +
                      [getattr(all_results.get(f"{s} (-Summary, +QE)"), key, 0) or 0 for s in strategies])
        without_qe = avg([getattr(all_results.get(f"{s} (+Summary, -QE)"), key, 0) or 0 for s in strategies] +
                         [getattr(all_results.get(f"{s} (-Summary, -QE)"), key, 0) or 0 for s in strategies])
        d = with_qe - without_qe
        pct = f"{d/without_qe*100:+.1f}%" if without_qe else "N/A"
        print(f"  {metric}: +QE={with_qe:.4f}  -QE={without_qe:.4f}  Δ={d:+.4f} ({pct})")

    print("\n── Per-Strategy QE Impact (Hit@1) ──")
    for strategy in strategies:
        with_qe = avg([getattr(all_results.get(f"{strategy} (+Summary, +QE)"), "hit_at_1", 0) or 0,
                       getattr(all_results.get(f"{strategy} (-Summary, +QE)"), "hit_at_1", 0) or 0])
        without_qe = avg([getattr(all_results.get(f"{strategy} (+Summary, -QE)"), "hit_at_1", 0) or 0,
                          getattr(all_results.get(f"{strategy} (-Summary, -QE)"), "hit_at_1", 0) or 0])
        d = with_qe - without_qe
        print(f"  {strategy:<22}: +QE={with_qe:.4f}  -QE={without_qe:.4f}  Δ={d:+.4f}")

    # Save results
    output = settings.data_dir / "eval_ablation_v2_results.json"
    payload = {}
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
        }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output}")

    retriever.close()
    clients.close()


if __name__ == "__main__":
    main()
