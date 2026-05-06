#!/usr/bin/env python3
"""Compare retrieval with full vs stripped (no summary) paper_cards."""
from __future__ import annotations

import json, math, os, re, sys, time
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

from app.core.config import get_settings
from app.domain.models import QueryContract, SessionContext
from app.services.contracts.session_context import (
    agent_session_conversation_context,
    session_llm_history_messages,
)
from app.services.infra.model_clients import ModelClients
from app.services.intents.router import LLMIntentRouter
from app.services.planning.query_shaping import extract_targets
from app.services.retrieval.core import DualIndexRetriever, _cjk_aware_tokenize
from app.services.retrieval.vector_index import CollectionVectorIndex
from langchain_community.retrievers import BM25Retriever

# ── helpers ──

def strip_summary(text: str) -> str:
    return re.sub(
        r"abstract_or_summary:\n.*?(?=top_evidence_hints:\n)",
        "abstract_or_summary: [removed for ablation]\n",
        text,
        flags=re.DOTALL,
    )


def calc(results, gt_ids):
    gt_set = set(gt_ids)
    rids = [r["paper_id"] for r in results]
    m = {}
    for k in [1, 3, 5]:
        m[f"h{k}"] = 1.0 if set(rids[:k]) & gt_set else 0.0
    rr = 0.0
    for rank, pid in enumerate(rids, start=1):
        if pid in gt_set:
            rr = 1.0 / rank
            break
    m["rr"] = rr
    dcg = sum((1.0 if pid in gt_set else 0.0) / math.log2(rank + 2) for rank, pid in enumerate(rids[:5]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt_set), 5))) or 1.0
    m["ndcg"] = dcg / idcg
    return m


def avg(vals):
    return sum(vals) / len(vals) if vals else 0.0


# ── main ──

def main():
    settings = get_settings()

    # Load full retriever (with summaries)
    print("Loading FULL retriever...", flush=True)
    full = DualIndexRetriever(settings)
    print(f"  papers={len(full._paper_docs)}", flush=True)

    # Build stripped documents
    print("Building STRIPPED paper docs...", flush=True)
    stripped_docs = []
    for doc in full._paper_docs:
        old_text = str(doc.page_content or "")
        new_text = strip_summary(old_text)
        meta = dict(doc.metadata or {})
        stripped_docs.append(Document(page_content=new_text, metadata=meta))

    # Build stripped BM25
    print("Building stripped BM25...", flush=True)
    stripped_bm25 = BM25Retriever.from_documents(stripped_docs, preprocess_func=_cjk_aware_tokenize)
    stripped_bm25.k = settings.paper_bm25_top_k

    # Build stripped Milvus
    STRIPPED_COLLECTION = "zprag_stripped_ablation"
    print(f"Indexing stripped docs into Milvus collection '{STRIPPED_COLLECTION}'...", flush=True)
    stripped_index = CollectionVectorIndex(settings, collection_name=STRIPPED_COLLECTION)
    # Force rebuild from scratch
    try:
        stripped_index._drop_collection_if_exists()
    except Exception:
        pass
    count = stripped_index.upsert_documents(stripped_docs, force_rebuild=True, batch_size=128)
    print(f"  indexed {count} vectors", flush=True)

    # Clients for enhanced
    clients = ModelClients(settings)

    # ── configs ──

    def dense_full(query, limit=6):
        docs = full._paper_dense.search_documents(query, limit=limit)
        return [{"paper_id": str((d.metadata or {}).get("paper_id", ""))}
                for d in docs if full._is_allowed_library_doc(d)]

    def hybrid_full(query, limit=6):
        bm = full._paper_bm25
        bm_docs = bm.invoke(query)[:12] if bm else []
        de_docs = [d for d in full._paper_dense.search_documents(query, limit=12)
                   if full._is_allowed_library_doc(d)]
        fused = full._rrf_fuse([(1.0, bm_docs), (1.0, de_docs)])[:limit]
        return [{"paper_id": str((d.metadata or {}).get("paper_id", ""))} for d in fused]

    def enhanced_full(query, limit=6):
        session = SessionContext(session_id=uuid4().hex[:12])
        router = LLMIntentRouter(clients=clients,
            conversation_context=lambda s, mc=24000: agent_session_conversation_context(s, settings=settings, max_chars=mc),
            conversation_messages=lambda s: session_llm_history_messages(s, max_turns=6, answer_limit=900))
        decision = router.route(query=query, session=session)
        llm_t = [str(t).strip() for t in list(decision.args.get("targets", []) or []) if str(t).strip()]
        rx_t = extract_targets(query)
        targets = list(dict.fromkeys(llm_t + rx_t))
        contract = QueryContract(clean_query=query, interaction_mode="research", relation="general_question",
            targets=targets, answer_slots=["general_answer"], requested_fields=["title", "authors", "year"],
            required_modalities=["page_text", "paper_card"], answer_shape="narrative",
            precision_requirement="normal", continuation_mode="fresh", allow_web_search=False,
            notes=["benchmark"])
        candidates = full.search_papers(query=query, contract=contract, limit=limit)
        return [{"paper_id": c.paper_id} for c in candidates]

    # Stripped versions
    def dense_stripped(query, limit=6):
        docs = stripped_index.search_documents(query, limit=limit)
        return [{"paper_id": str((d.metadata or {}).get("paper_id", ""))}
                for d in docs if full._is_allowed_library_doc(d)]

    def hybrid_stripped(query, limit=6):
        bm_docs = stripped_bm25.invoke(query)[:12]
        de_docs = [d for d in stripped_index.search_documents(query, limit=12)
                   if full._is_allowed_library_doc(d)]
        fused = full._rrf_fuse([(1.0, bm_docs), (1.0, de_docs)])[:limit]
        return [{"paper_id": str((d.metadata or {}).get("paper_id", ""))} for d in fused]

    def enhanced_stripped(query, limit=6):
        session = SessionContext(session_id=uuid4().hex[:12])
        router = LLMIntentRouter(clients=clients,
            conversation_context=lambda s, mc=24000: agent_session_conversation_context(s, settings=settings, max_chars=mc),
            conversation_messages=lambda s: session_llm_history_messages(s, max_turns=6, answer_limit=900))
        decision = router.route(query=query, session=session)
        llm_t = [str(t).strip() for t in list(decision.args.get("targets", []) or []) if str(t).strip()]
        rx_t = extract_targets(query)
        targets = list(dict.fromkeys(llm_t + rx_t))
        contract = QueryContract(clean_query=query, interaction_mode="research", relation="general_question",
            targets=targets, answer_slots=["general_answer"], requested_fields=["title", "authors", "year"],
            required_modalities=["page_text", "paper_card"], answer_shape="narrative",
            precision_requirement="normal", continuation_mode="fresh", allow_web_search=False,
            notes=["benchmark", "stripped_ablation"])

        # Use stripped dense instead of production dense in the 4-path fusion
        search_text = " ".join(targets) + " " + query
        weighted = []
        anchors = full.title_anchor(targets)
        if anchors:
            weighted.append((1.6, anchors))
        rel_anchors = full.relation_anchor_docs(contract)
        if rel_anchors:
            weighted.append((1.3, rel_anchors))
        bm_docs = stripped_bm25.invoke(search_text)[:12]
        weighted.append((0.9, bm_docs))
        de_docs = [d for d in stripped_index.search_documents(search_text, limit=12)
                   if full._is_allowed_library_doc(d)]
        if de_docs:
            weighted.append((0.8, de_docs))
        fused = full._rrf_fuse(weighted)[:limit]
        return [{"paper_id": str((d.metadata or {}).get("paper_id", ""))} for d in fused]

    # ── Run ──
    with open("data/eval_queries_focused.json") as f:
        queries = json.load(f)
    valid = [q for q in queries if q.get("gt_ids")]
    print(f"\nRunning {len(valid)} queries on FULL + STRIPPED...\n", flush=True)

    configs = {
        "Dense (full)": dense_full, "Hybrid (full)": hybrid_full, "Enhanced (full)": enhanced_full,
        "Dense (stripped)": dense_stripped, "Hybrid (stripped)": hybrid_stripped, "Enhanced (stripped)": enhanced_stripped,
    }
    all_r = {n: defaultdict(list) for n in configs}
    lats = {n: [] for n in configs}

    for i, q in enumerate(valid):
        query = q["query"]; gt = q["gt_ids"]
        for name, fn in configs.items():
            t0 = time.perf_counter()
            res = fn(query)
            lat = (time.perf_counter() - t0) * 1000
            lats[name].append(lat)
            for k, v in calc(res, gt).items():
                all_r[name][k].append(v)
        if (i + 1) % 10 == 0:
            n = i + 1
            df = avg(all_r["Dense (full)"]["h1"])
            ds = avg(all_r["Dense (stripped)"]["h1"])
            ef = avg(all_r["Enhanced (full)"]["h1"])
            es = avg(all_r["Enhanced (stripped)"]["h1"])
            print(f"  [{n}/{len(valid)}] Dense(full)={df:.3f} stripped={ds:.3f}  Enhanced(full)={ef:.3f} stripped={es:.3f}", flush=True)

    # ── Report ──
    labels = {"h1": "Hit@1", "h3": "Hit@3", "h5": "Hit@5", "rr": "MRR", "ndcg": "NDCG@5"}

    print("\n" + "=" * 105, flush=True)
    print(f"ABLATION: Full vs Stripped (no summary) paper_cards — {len(valid)} queries", flush=True)
    print("=" * 105, flush=True)
    header = f"{'Metric':<12} " + " ".join(f"{n:<18}" for n in configs)
    print(header, flush=True)
    print("-" * 105, flush=True)
    for m in ["h1", "h3", "h5", "rr", "ndcg"]:
        row = f"{labels[m]:<12}"
        for n in configs:
            row += f" {avg(all_r[n][m]):<18.3f}"
        print(row, flush=True)

    print(f"\n{'Lat(ms)':<12}", end="", flush=True)
    for n in configs:
        print(f" {avg(lats[n]):<18.0f}", end="", flush=True)
    print(flush=True)

    # Improvement analysis
    print("\n── SUMMARY ABLATION IMPACT ──", flush=True)
    pairs = [("Dense", "Dense (full)", "Dense (stripped)"),
             ("Hybrid", "Hybrid (full)", "Hybrid (stripped)"),
             ("Enhanced", "Enhanced (full)", "Enhanced (stripped)")]
    for label, full_key, stripped_key in pairs:
        for m in ["h1", "rr", "ndcg"]:
            f = avg(all_r[full_key][m])
            s = avg(all_r[stripped_key][m])
            delta = (s - f) / f * 100 if f else 0
            print(f"  {label} {labels[m]}: full={f:.3f} → stripped={s:.3f}  Δ={delta:+.1f}%", flush=True)

    # Save
    results = {}
    for n in configs:
        results[n] = {
            "config": n, "total_queries": len(valid),
            "hit_at_1": round(avg(all_r[n]["h1"]), 4),
            "hit_at_3": round(avg(all_r[n]["h3"]), 4),
            "hit_at_5": round(avg(all_r[n]["h5"]), 4),
            "mrr": round(avg(all_r[n]["rr"]), 4),
            "ndcg_at_5": round(avg(all_r[n]["ndcg"]), 4),
            "avg_latency_ms": round(avg(lats[n]), 0),
        }
    with open("data/eval_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to data/eval_ablation_results.json", flush=True)

    # Cleanup
    try:
        stripped_index._drop_collection_if_exists()
    except Exception:
        pass
    full.close()


if __name__ == "__main__":
    main()
