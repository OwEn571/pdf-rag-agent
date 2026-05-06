#!/usr/bin/env python3
"""Retrieval benchmark: Dense vs BM25(jieba)+Dense vs Enhanced(LLM Router).

Usage: python scripts/run_retrieval_benchmark.py [--max 159]
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.domain.models import QueryContract, SessionContext
from app.services.contracts.session_context import (
    agent_session_conversation_context,
    session_llm_history_messages,
)
from app.services.infra.model_clients import ModelClients
from app.services.intents.router import LLMIntentRouter
from app.services.planning.query_shaping import extract_targets
from app.services.retrieval.core import DualIndexRetriever

# ── helpers ──────────────────────────────────────────────────────────


def calc(results, gt_ids):
    gt_set = set(gt_ids)
    rids = [r["paper_id"] for r in results]
    m: dict[str, float] = {}
    for k in [1, 3, 5]:
        m[f"h{k}"] = 1.0 if set(rids[:k]) & gt_set else 0.0
    rr = 0.0
    for rank, pid in enumerate(rids, start=1):
        if pid in gt_set:
            rr = 1.0 / rank
            break
    m["rr"] = rr
    dcg = sum(
        (1.0 if pid in gt_set else 0.0) / math.log2(rank + 2)
        for rank, pid in enumerate(rids[:5])
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt_set), 5))) or 1.0
    m["ndcg"] = dcg / idcg
    return m


def avg(vals):
    return sum(vals) / len(vals) if vals else 0.0


# ── main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=0, help="Max queries to evaluate")
    parser.add_argument("--queries", type=str, default="data/eval_queries_v3.json")
    args = parser.parse_args()

    settings = get_settings()
    print("Loading retriever (BM25 uses jieba)...", flush=True)
    retriever = DualIndexRetriever(settings)
    print(f"  papers={len(retriever._paper_docs)}  blocks={len(retriever._block_docs)}", flush=True)
    print(f"  bm25={retriever._paper_bm25 is not None}", flush=True)

    clients = ModelClients(settings)

    with open(args.queries) as f:
        all_queries = json.load(f)
    if args.max:
        all_queries = all_queries[: args.max]
    valid = [q for q in all_queries if q.get("gt_ids")]
    print(f"Running {len(valid)} queries...\n", flush=True)

    # ── configs ──
    def dense(query, limit=6):
        docs = retriever._paper_dense.search_documents(query, limit=limit)
        return [
            {"paper_id": str((d.metadata or {}).get("paper_id", ""))}
            for d in docs
            if retriever._is_allowed_library_doc(d)
        ]

    def hybrid(query, limit=6):
        bm = retriever._paper_bm25
        bm_docs = bm.invoke(query)[:12] if bm else []
        de_docs = [
            d
            for d in retriever._paper_dense.search_documents(query, limit=12)
            if retriever._is_allowed_library_doc(d)
        ]
        fused = retriever._rrf_fuse([(1.0, bm_docs), (1.0, de_docs)])[:limit]
        return [
            {"paper_id": str((d.metadata or {}).get("paper_id", ""))} for d in fused
        ]

    def enhanced(query, limit=6):
        session = SessionContext(session_id=uuid4().hex[:12])
        router = LLMIntentRouter(
            clients=clients,
            conversation_context=lambda s, mc=24000: agent_session_conversation_context(
                s, settings=settings, max_chars=mc
            ),
            conversation_messages=lambda s: session_llm_history_messages(
                s, max_turns=6, answer_limit=900
            ),
        )
        decision = router.route(query=query, session=session)
        llm = [
            str(t).strip()
            for t in list(decision.args.get("targets", []) or [])
            if str(t).strip()
        ]
        rx = extract_targets(query)
        targets = list(dict.fromkeys(llm + rx))
        contract = QueryContract(
            clean_query=query,
            interaction_mode="research",
            relation="general_question",
            targets=targets,
            answer_slots=["general_answer"],
            requested_fields=["title", "authors", "year"],
            required_modalities=["page_text", "paper_card"],
            answer_shape="narrative",
            precision_requirement="normal",
            continuation_mode="fresh",
            allow_web_search=False,
            notes=["benchmark"],
        )
        candidates = retriever.search_papers(
            query=query, contract=contract, limit=limit
        )
        return [{"paper_id": c.paper_id} for c in candidates]

    configs = {
        "Pure Dense": dense,
        "BM25(jieba)+Dense RRF": hybrid,
        "Enhanced (4-path)": enhanced,
    }
    all_r = {n: defaultdict(list) for n in configs}
    cat_r = {
        d: {n: defaultdict(list) for n in configs}
        for d in ["easy", "medium", "hard"]
    }
    lats = {n: [] for n in configs}

    for i, q in enumerate(valid):
        query = q["query"]
        gt = q["gt_ids"]
        diff = q.get("difficulty", "medium")
        for name, fn in configs.items():
            t0 = time.perf_counter()
            res = fn(query)
            lat = (time.perf_counter() - t0) * 1000
            lats[name].append(lat)
            for k, v in calc(res, gt).items():
                all_r[name][k].append(v)
                cat_r[diff][name][k].append(v)
        if (i + 1) % 20 == 0:
            n = i + 1
            d1 = avg(all_r["Pure Dense"]["h1"])
            b1 = avg(all_r["BM25(jieba)+Dense RRF"]["h1"])
            e1 = avg(all_r["Enhanced (4-path)"]["h1"])
            print(
                f"  [{n}/{len(valid)}] Dense={d1:.3f}  BM25+Dense={b1:.3f}  Enhanced={e1:.3f}",
                flush=True,
            )

    print(flush=True)

    # ── report ──
    labels = {"h1": "Hit@1", "h3": "Hit@3", "h5": "Hit@5", "rr": "MRR", "ndcg": "NDCG@5"}
    sep = "=" * 95

    print(sep, flush=True)
    print(
        f"BENCHMARK: {len(valid)} queries — BM25 uses jieba CJK tokenizer", flush=True
    )
    print(sep, flush=True)
    header = f"{'Metric':<12}"
    for n in configs:
        header += f" {n:<25}"
    print(header, flush=True)
    print("-" * 95, flush=True)
    for m in ["h1", "h3", "h5", "rr", "ndcg"]:
        row = f"{labels[m]:<12}"
        d = avg(all_r["Pure Dense"][m])
        for n in configs:
            row += f" {avg(all_r[n][m]):<25.3f}"
        e = avg(all_r["Enhanced (4-path)"][m])
        delta = (e - d) / d * 100 if d else 0
        row += f" {delta:+.1f}%"
        print(row, flush=True)

    print(f"\n{'Lat(ms)':<12}", end="", flush=True)
    for n in configs:
        print(f" {avg(lats[n]):<25.0f}", end="", flush=True)
    print(flush=True)

    for diff in ["easy", "medium", "hard"]:
        nq = sum(1 for q in valid if q.get("difficulty") == diff)
        print(f"\n── {diff.upper()} ({nq} queries) ──", flush=True)
        for m in ["h1", "rr", "ndcg"]:
            d = avg(cat_r[diff]["Pure Dense"][m])
            b = avg(cat_r[diff]["BM25(jieba)+Dense RRF"][m])
            e = avg(cat_r[diff]["Enhanced (4-path)"][m])
            print(
                f"  {labels[m]}: Dense={d:.3f}  BM25+Dense={b:.3f}  Enhanced={e:.3f}",
                flush=True,
            )

    # ── save ──
    results = {}
    for n in configs:
        results[n] = {
            "config": n,
            "total_queries": len(valid),
            "hit_at_1": round(avg(all_r[n]["h1"]), 4),
            "hit_at_3": round(avg(all_r[n]["h3"]), 4),
            "hit_at_5": round(avg(all_r[n]["h5"]), 4),
            "mrr": round(avg(all_r[n]["rr"]), 4),
            "ndcg_at_5": round(avg(all_r[n]["ndcg"]), 4),
            "avg_latency_ms": round(avg(lats[n]), 0),
        }
    path = settings.data_dir / "eval_retrieval_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {path}", flush=True)

    retriever.close()


if __name__ == "__main__":
    main()
