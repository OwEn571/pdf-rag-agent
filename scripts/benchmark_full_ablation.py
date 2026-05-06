#!/usr/bin/env python3
"""159-query full ablation: Dense/Hybrid/Enhanced × Full/Stripped = 6 configs."""
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


def strip_summary(text: str) -> str:
    return re.sub(
        r"abstract_or_summary:\n.*?(?=top_evidence_hints:\n)",
        "abstract_or_summary: [removed]\n",
        text, flags=re.DOTALL)


def calc(results, gt_ids):
    gt_set = set(gt_ids)
    rids = [r["paper_id"] for r in results]
    m = {}
    for k in [1, 3, 5]:
        m[f"h{k}"] = 1.0 if set(rids[:k]) & gt_set else 0.0
    rr = 0.0
    for rank, pid in enumerate(rids, start=1):
        if pid in gt_set: rr = 1.0 / rank; break
    m["rr"] = rr
    dcg = sum((1.0 if pid in gt_set else 0.0) / math.log2(rank + 2) for rank, pid in enumerate(rids[:5]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt_set), 5))) or 1.0
    m["ndcg"] = dcg / idcg
    return m


def avg(vals): return sum(vals) / len(vals) if vals else 0.0


def main():
    settings = get_settings()

    # ── Full retriever ──
    print("Loading FULL retriever...", flush=True)
    full = DualIndexRetriever(settings)
    print(f"  papers={len(full._paper_docs)}", flush=True)

    # ── Build stripped indexes ──
    print("Building STRIPPED indexes...", flush=True)
    stripped_docs = []
    for doc in full._paper_docs:
        new_text = strip_summary(str(doc.page_content or ""))
        meta = dict(doc.metadata or {})
        stripped_docs.append(Document(page_content=new_text, metadata=meta))

    stripped_bm25 = BM25Retriever.from_documents(stripped_docs, preprocess_func=_cjk_aware_tokenize)
    stripped_bm25.k = settings.paper_bm25_top_k

    STRIPPED_COL = "zprag_stripped_159"
    stripped_dense = CollectionVectorIndex(settings, collection_name=STRIPPED_COL)
    try: stripped_dense._drop_collection_if_exists()
    except: pass
    n = stripped_dense.upsert_documents(stripped_docs, force_rebuild=True, batch_size=128)
    print(f"  indexed {n} stripped vectors", flush=True)

    clients = ModelClients(settings)

    # ── Load queries ──
    with open("data/eval_queries_v3.json") as f:
        queries = json.load(f)
    valid = [q for q in queries if q.get("gt_ids")]
    print(f"Running {len(valid)} queries...\n", flush=True)

    # ── 6 configs ──
    def _enhanced_common(query):
        session = SessionContext(session_id=uuid4().hex[:12])
        router = LLMIntentRouter(clients=clients,
            conversation_context=lambda s, mc=24000: agent_session_conversation_context(s, settings=settings, max_chars=mc),
            conversation_messages=lambda s: session_llm_history_messages(s, max_turns=6, answer_limit=900))
        decision = router.route(query=query, session=session)
        llm_t = [str(t).strip() for t in list(decision.args.get("targets", []) or []) if str(t).strip()]
        rx_t = extract_targets(query)
        return list(dict.fromkeys(llm_t + rx_t)), session

    def dense_f(q):
        docs = full._paper_dense.search_documents(q, limit=6)
        return [{"paper_id": str((d.metadata or {}).get("paper_id",""))} for d in docs if full._is_allowed_library_doc(d)]
    def hyb_f(q):
        bm=full._paper_bm25; bd=bm.invoke(q)[:12] if bm else []
        dd=[d for d in full._paper_dense.search_documents(q,limit=12) if full._is_allowed_library_doc(d)]
        return [{"paper_id":str((d.metadata or {}).get("paper_id",""))} for d in full._rrf_fuse([(1,bd),(1,dd)])[:6]]
    def enh_f(q):
        targets, _ = _enhanced_common(q)
        c = QueryContract(clean_query=q,interaction_mode="research",relation="general_question",targets=targets,
            answer_slots=["general_answer"],requested_fields=["title","authors","year"],
            required_modalities=["page_text","paper_card"],answer_shape="narrative",precision_requirement="normal",
            continuation_mode="fresh",allow_web_search=False,notes=["benchmark"])
        return [{"paper_id":c.paper_id} for c in full.search_papers(query=q,contract=c,limit=6)]

    def dense_s(q):
        docs = stripped_dense.search_documents(q, limit=6)
        return [{"paper_id":str((d.metadata or {}).get("paper_id",""))} for d in docs if full._is_allowed_library_doc(d)]
    def hyb_s(q):
        bd=stripped_bm25.invoke(q)[:12]
        dd=[d for d in stripped_dense.search_documents(q,limit=12) if full._is_allowed_library_doc(d)]
        return [{"paper_id":str((d.metadata or {}).get("paper_id",""))} for d in full._rrf_fuse([(1,bd),(1,dd)])[:6]]
    def enh_s(q):
        targets, _ = _enhanced_common(q)
        st=" ".join(targets)+" "+q
        weighted=[(1.6,full.title_anchor(targets)),(1.3,full.relation_anchor_docs(QueryContract(
            clean_query=q,interaction_mode="research",relation="general_question",targets=targets,
            answer_slots=["general_answer"],requested_fields=[],required_modalities=[],
            answer_shape="narrative",precision_requirement="normal",continuation_mode="fresh",
            allow_web_search=False,notes=["benchmark"])))]
        # Only keep non-empty paths
        weighted=[w for w in weighted if w[1]]
        weighted.append((0.9,stripped_bm25.invoke(st)[:12]))
        dd2=[d for d in stripped_dense.search_documents(st,limit=12) if full._is_allowed_library_doc(d)]
        if dd2: weighted.append((0.8,dd2))
        return [{"paper_id":str((d.metadata or {}).get("paper_id",""))} for d in full._rrf_fuse(weighted)[:6]]

    configs = {"Dense(full)":dense_f,"Hybrid(full)":hyb_f,"Enhanced(full)":enh_f,
               "Dense(stripped)":dense_s,"Hybrid(stripped)":hyb_s,"Enhanced(stripped)":enh_s}
    all_r={n:defaultdict(list) for n in configs}
    lats={n:[] for n in configs}

    for i,q in enumerate(valid):
        query=q["query"]; gt=q["gt_ids"]
        for name,fn in configs.items():
            t0=time.perf_counter(); res=fn(query); lat=(time.perf_counter()-t0)*1000
            lats[name].append(lat)
            for k,v in calc(res,gt).items(): all_r[name][k].append(v)
        if (i+1)%20==0:
            n=i+1; df=avg(all_r["Dense(full)"]["h1"]); ds=avg(all_r["Dense(stripped)"]["h1"])
            ef=avg(all_r["Enhanced(full)"]["h1"]); es=avg(all_r["Enhanced(stripped)"]["h1"])
            print(f"  [{n}/{len(valid)}] D(full)={df:.3f} D(strip)={ds:.3f} E(full)={ef:.3f} E(strip)={es:.3f}",flush=True)

    # ── Report ──
    labels={"h1":"Hit@1","h3":"Hit@3","h5":"Hit@5","rr":"MRR","ndcg":"NDCG@5"}
    cols=["Dense(full)","Hybrid(full)","Enhanced(full)","Dense(stripped)","Hybrid(stripped)","Enhanced(stripped)"]

    print("\n"+"="*120,flush=True)
    print(f"FULL ABLATION: 159 queries — 6 configs (full vs stripped summaries)",flush=True)
    print("="*120,flush=True)
    header=f"{'Metric':<12}"+"".join(f"{c:<20}" for c in cols)
    print(header,flush=True)
    print("-"*120,flush=True)
    for m in ["h1","h3","h5","rr","ndcg"]:
        row=f"{labels[m]:<12}"
        for c in cols: row+=f" {avg(all_r[c][m]):<19.3f}"
        print(row,flush=True)
    print(f"\n{'Lat(ms)':<12}",end="",flush=True)
    for c in cols: print(f" {avg(lats[c]):<19.0f}",end="",flush=True)
    print(flush=True)

    # ── Delta analysis ──
    print("\n--- SUMMARY ABLATION DELTA ---",flush=True)
    for label,fc,sc in [("Dense","Dense(full)","Dense(stripped)"),("Hybrid","Hybrid(full)","Hybrid(stripped)"),("Enhanced","Enhanced(full)","Enhanced(stripped)")]:
        for m in ["h1","rr","ndcg"]:
            fv=avg(all_r[fc][m]); sv=avg(all_r[sc][m])
            d=(sv-fv)/fv*100 if fv else 0
            print(f"  {label} {labels[m]}: {fv:.3f} → {sv:.3f}  Δ={d:+.1f}%",flush=True)

    # Save
    results={}
    for c in cols: results[c]={"config":c,"total_queries":len(valid),"hit_at_1":round(avg(all_r[c]["h1"]),4),"hit_at_3":round(avg(all_r[c]["h3"]),4),"hit_at_5":round(avg(all_r[c]["h5"]),4),"mrr":round(avg(all_r[c]["rr"]),4),"ndcg_at_5":round(avg(all_r[c]["ndcg"]),4),"avg_latency_ms":round(avg(lats[c]),0)}
    with open("data/eval_full_ablation.json","w",encoding="utf-8") as f: json.dump(results,f,ensure_ascii=False,indent=2)
    print(f"\nSaved to data/eval_full_ablation.json",flush=True)

    try: stripped_dense._drop_collection_if_exists()
    except: pass
    full.close()

if __name__=="__main__": main()
