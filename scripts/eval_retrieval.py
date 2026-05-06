"""Retrieval evaluation: baselines vs optimized multi-path fusion.

Generates test queries from the user's Zotero library, runs multiple
retriever configurations, and reports Hit Rate, MRR, NDCG, Precision,
Recall, and latency for each variant.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if str(PROJECT_ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.domain.models import QueryContract  # noqa: E402
from app.services.retrieval.core import DualIndexRetriever  # noqa: E402

# ── Query generation templates ──────────────────────────────────────

QUERY_TEMPLATES_CN = [
    "{acronym}是什么？",
    "{acronym}的定义是什么？",
    "{acronym}的公式是什么？",
    "{acronym}最早由哪篇论文提出？",
    "{acronym}方法是怎么工作的？",
    "{acronym}的核心思想是什么？",
    "解释{acronym}的原理",
    "{acronym}和什么方法有关？",
    "什么是{acronym}？它的目标函数是什么？",
]

QUERY_TEMPLATES_EN = [
    "What is {acronym}?",
    "What is the definition of {acronym}?",
    "What is the formula for {acronym}?",
    "Which paper first proposed {acronym}?",
    "How does {acronym} work?",
    "Explain the mechanism of {acronym}",
    "What is the objective function of {acronym}?",
    "What is the core idea of {acronym}?",
    "What are related methods to {acronym}?",
]


@dataclass(slots=True)
class TestQuery:
    query: str
    ground_truth_paper_ids: list[str]
    acronym: str = ""
    template: str = ""


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


# ── Query generation ────────────────────────────────────────────────


def generate_test_queries(retriever: DualIndexRetriever, max_queries: int = 100) -> list[TestQuery]:
    """Generate test queries from paper metadata (acronyms, aliases, titles)."""
    queries: list[TestQuery] = []
    seen: set[str] = set()

    # Filter for acronyms that look like real ML terms (all uppercase, 2-8 chars)
    import re as _re
    def _is_good_acronym(s: str) -> bool:
        if len(s) < 2 or len(s) > 12:
            return False
        if s.isupper() and _re.fullmatch(r"[A-Z][A-Z0-9\-]{1,11}", s):
            return True
        if _re.fullmatch(r"[A-Z][a-z][A-Za-z0-9\-]{2,}", s) and len(s) <= 20:
            return True
        return False

    for doc in retriever.paper_documents():
        meta = doc.metadata or {}
        paper_id = str(meta.get("paper_id", ""))
        title = str(meta.get("title", ""))

        # Collect candidate acronyms — only use aliases (curated title abbreviations)
        # and body_acronyms that look like real ML terms
        candidates: list[str] = []
        for item in str(meta.get("aliases", "")).split("||"):
            item = item.strip()
            if item and _is_good_acronym(item):
                candidates.append(item)

        # Body acronyms: only keep short, uppercase ones
        for item in str(meta.get("body_acronyms", "")).split("||"):
            item = item.strip()
            if item and item.isupper() and 2 <= len(item) <= 8:
                if item not in candidates:
                    candidates.append(item)

        # Only use first 2 candidates per paper
        for acronym in candidates[:2]:
            templates = QUERY_TEMPLATES_CN if random.random() < 0.6 else QUERY_TEMPLATES_EN
            template = random.choice(templates)
            query_text = template.format(acronym=acronym)

            key = query_text.lower()
            if key in seen:
                continue
            seen.add(key)

            queries.append(
                TestQuery(
                    query=query_text,
                    ground_truth_paper_ids=[paper_id],
                    acronym=acronym,
                    template=template,
                )
            )
            if len(queries) >= max_queries:
                break
        if len(queries) >= max_queries:
            break

    # Add hard queries: Chinese descriptive queries about specific papers
    hard_queries = _generate_hard_queries(retriever, max_hard=max_queries // 4)
    for hq in hard_queries:
        if hq.query.lower() not in seen:
            seen.add(hq.query.lower())
            queries.append(hq)

    return queries


def _generate_hard_queries(retriever: DualIndexRetriever, max_hard: int = 20) -> list[TestQuery]:
    """Generate harder queries where the paper title doesn't directly match the query."""
    hard: list[TestQuery] = []
    papers = retriever.paper_documents()
    random.shuffle(papers)

    templates = [
        "哪篇论文最早提出了{title}？",
        "{title}这篇论文的核心贡献是什么？",
        "{title}中提出的方法是怎么工作的？",
    ]

    for doc in papers[:max_hard]:
        meta = doc.metadata or {}
        title = str(meta.get("title", ""))
        # Extract just the method name (before colon)
        short_title = title.split(":")[0].strip()
        if len(short_title) < 5 or len(short_title) > 60:
            continue
        # Skip if it looks like a generic phrase
        if short_title.lower() in {"improving language understanding", "deep residual learning", "learning transferable visual models"}:
            continue
        template = random.choice(templates)
        query = template.format(title=short_title)
        hard.append(TestQuery(
            query=query,
            ground_truth_paper_ids=[str(meta.get("paper_id", ""))],
            acronym=short_title,
            template="hard_title_query",
        ))

    return hard[:max_hard]


# ── Baseline retrievers ─────────────────────────────────────────────


class BaselineDenseRetriever:
    """Pure Milvus dense search only."""

    def __init__(self, retriever: DualIndexRetriever):
        self.retriever = retriever

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
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


class BaselineBM25Retriever:
    """Pure BM25 sparse search only."""

    def __init__(self, retriever: DualIndexRetriever):
        self.retriever = retriever

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        bm25 = getattr(self.retriever, "_paper_bm25", None)
        if bm25 is None:
            return []
        docs = bm25.invoke(query)[:limit]
        return [
            {
                "paper_id": str((d.metadata or {}).get("paper_id", "")),
                "title": str((d.metadata or {}).get("title", "")),
                "year": str((d.metadata or {}).get("year", "")),
                "score": 0.0,
            }
            for d in docs
        ]


class BaselineHybridRetriever:
    """BM25 + Dense with simple RRF (equal weights, no title/relation anchor)."""

    def __init__(self, retriever: DualIndexRetriever):
        self.retriever = retriever

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        bm25 = getattr(self.retriever, "_paper_bm25", None)
        bm25_docs = bm25.invoke(query)[:12] if bm25 else []
        dense_docs = self.retriever._paper_dense.search_documents(query, limit=12)
        dense_allowed = [d for d in dense_docs if self.retriever._is_allowed_library_doc(d)]

        # Simple RRF (equal weights)
        weighted = [(1.0, bm25_docs), (1.0, dense_allowed)]
        fused = self.retriever._rrf_fuse(weighted)[:limit]

        return [
            {
                "paper_id": str((d.metadata or {}).get("paper_id", "")),
                "title": str((d.metadata or {}).get("title", "")),
                "year": str((d.metadata or {}).get("year", "")),
                "score": 0.0,
            }
            for d in fused
        ]


class OptimizedRetriever:
    """Full multi-path weighted RRF (the current production retriever)."""

    def __init__(self, retriever: DualIndexRetriever, benchmark_mode: bool = True):
        self.retriever = retriever
        self.benchmark_mode = benchmark_mode

    def search(
        self,
        query: str,
        contract: QueryContract | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if contract is None:
            targets = _extract_acronym_targets(query)
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
                notes=["intent_kind=research"],
            )
        if self.benchmark_mode:
            # Bypass screen_papers for fair comparison with baselines
            # (screen_papers relies on LLM-generated contract fields
            # that we can't replicate in auto-generated contracts)
            return self._search_without_screening(query=query, contract=contract, limit=limit)
        candidates = self.retriever.search_papers(query=query, contract=contract, limit=limit)
        return [
            {"paper_id": c.paper_id, "title": c.title, "year": c.year, "score": c.score}
            for c in candidates
        ]

    def _search_without_screening(
        self, *, query: str, contract: QueryContract, limit: int
    ) -> list[dict[str, Any]]:
        """Run the full 4-path RRF fusion but skip screen_papers re-ranking."""
        import re as _re
        target_terms = self.retriever._contract_target_terms(contract)
        target_text = " ".join(target_terms).strip()
        search_text = query.strip()
        if target_text and target_text.lower() not in search_text.lower():
            search_text = f"{target_text} {search_text}".strip()

        weighted_docs: list[tuple[float, list]] = []
        anchors = self.retriever.title_anchor(target_terms)
        if anchors:
            weighted_docs.append((1.6, anchors))
        rel_anchors = self.retriever.relation_anchor_docs(contract)
        if rel_anchors:
            weighted_docs.append((1.3, rel_anchors))
        if self.retriever._paper_bm25 is not None:
            weighted_docs.append((0.9, self.retriever._paper_bm25.invoke(search_text)))
        dense_docs = [
            d for d in self.retriever._paper_dense.search_documents(search_text, limit=12)
            if self.retriever._is_allowed_library_doc(d)
        ]
        if dense_docs:
            weighted_docs.append((0.8, dense_docs))
        fused = self.retriever._rrf_fuse(weighted_docs)[:limit]

        return [
            {
                "paper_id": str((d.metadata or {}).get("paper_id", "")),
                "title": str((d.metadata or {}).get("title", "")),
                "year": str((d.metadata or {}).get("year", "")),
                "score": 0.0,
            }
            for d in fused
            if self.retriever._is_allowed_library_doc(d)
        ]


def _extract_acronym_targets(query: str) -> list[str]:
    """Simple regex extraction matching the system's extract_targets()."""
    targets: list[str] = []
    # Quoted text
    import re
    for pattern in ['[“”](.+?)[“”]', "[''](.+?)['']"]:
        for match in re.finditer(pattern, query):
            c = str(match.group(1) or "").strip()
            if c and c not in targets:
                targets.append(c)
    # Uppercase acronyms
    for token in re.findall(r"[A-Z][A-Z0-9\-]{1,}", query):
        token = token.strip("-")
        if token not in targets:
            targets.append(token)
    # Mixed case tokens
    for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", query):
        if any(ch.isupper() for ch in token[1:]) or "-" in token:
            if token not in targets:
                targets.append(token)
    return targets


# ── Metrics computation ─────────────────────────────────────────────


def compute_metrics(
    results: list[dict[str, Any]],
    ground_truth_ids: list[str],
) -> dict[str, float]:
    """Compute all metrics for a single query result."""
    k_values = [1, 3, 5]
    metrics: dict[str, float] = {}

    gt_set = set(ground_truth_ids)
    retrieved_ids = [r["paper_id"] for r in results]

    # Hit Rate
    for k in k_values:
        top_k = set(retrieved_ids[:k])
        hit = 1.0 if top_k & gt_set else 0.0
        metrics[f"hit@{k}"] = hit

    # MRR
    rr = 0.0
    for rank, pid in enumerate(retrieved_ids, start=1):
        if pid in gt_set:
            rr = 1.0 / rank
            break
    metrics["rr"] = rr

    # NDCG@5
    dcg = 0.0
    for rank, pid in enumerate(retrieved_ids[:5], start=1):
        relevance = 1.0 if pid in gt_set else 0.0
        dcg += relevance / math.log2(rank + 1)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt_set), 5)))
    idcg = ideal_dcg if ideal_dcg > 0 else 1.0
    metrics["ndcg@5"] = dcg / idcg

    # Precision@5
    top_5 = retrieved_ids[:5]
    relevant_in_top5 = sum(1 for pid in top_5 if pid in gt_set)
    metrics["precision@5"] = relevant_in_top5 / max(1, len(top_5))

    # Recall@5
    metrics["recall@5"] = relevant_in_top5 / max(1, len(gt_set))

    return metrics


# ── Evaluation runner ───────────────────────────────────────────────


def run_evaluation(
    retriever: DualIndexRetriever,
    queries: list[TestQuery],
    max_papers: int = 6,
) -> dict[str, EvalResult]:
    """Run full evaluation across all retriever configurations."""

    configs: dict[str, Any] = {
        "Pure Dense": BaselineDenseRetriever(retriever),
        "Pure BM25": BaselineBM25Retriever(retriever),
        "BM25 + Dense (RRF)": BaselineHybridRetriever(retriever),
        "Optimized (4-path Weighted RRF)": OptimizedRetriever(retriever),
    }

    results: dict[str, EvalResult] = {}

    for name, searcher in configs.items():
        eval_result = EvalResult(config_name=name, total_queries=len(queries))
        all_metrics: dict[str, list[float]] = defaultdict(list)
        latencies: list[float] = []

        for tq in queries:
            start = time.perf_counter()
            if name == "Optimized (4-path Weighted RRF)":
                retrieved = searcher.search(query=tq.query, limit=max_papers)
            else:
                retrieved = searcher.search(query=tq.query, limit=max_papers)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

            m = compute_metrics(retrieved, tq.ground_truth_paper_ids)
            for key, value in m.items():
                all_metrics[key].append(value)

            eval_result.per_query.append(
                {
                    "query": tq.query,
                    "acronym": tq.acronym,
                    "ground_truth": tq.ground_truth_paper_ids,
                    "retrieved": [
                        {"paper_id": r["paper_id"], "title": r["title"][:60]}
                        for r in retrieved[:5]
                    ],
                    "metrics": m,
                    "latency_ms": round(elapsed_ms, 1),
                }
            )

        # Aggregate metrics
        def avg(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        eval_result.hit_at_1 = avg(all_metrics["hit@1"])
        eval_result.hit_at_3 = avg(all_metrics["hit@3"])
        eval_result.hit_at_5 = avg(all_metrics["hit@5"])
        eval_result.mrr = avg(all_metrics["rr"])
        eval_result.ndcg_at_5 = avg(all_metrics["ndcg@5"])
        eval_result.precision_at_5 = avg(all_metrics["precision@5"])
        eval_result.recall_at_5 = avg(all_metrics["recall@5"])
        eval_result.avg_latency_ms = avg(latencies)

        results[name] = eval_result

    return results


# ── Reporting ───────────────────────────────────────────────────────


def print_report(
    results: dict[str, EvalResult],
    baseline_name: str = "Pure Dense",
) -> None:
    """Print evaluation report with improvement over baseline."""
    baseline = results.get(baseline_name)
    if baseline is None:
        baseline = list(results.values())[0]

    print()
    print("=" * 90)
    print("RETRIEVAL EVALUATION REPORT")
    print(f"Baseline: {baseline_name}  |  Queries: {baseline.total_queries}")
    print("=" * 90)

    header = (
        f"{'Config':<30} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} "
        f"{'MRR':>7} {'NDCG@5':>7} {'P@5':>7} {'R@5':>7} {'Lat(ms)':>8}"
    )
    print(header)
    print("-" * 90)

    for name in ["Pure Dense", "Pure BM25", "BM25 + Dense (RRF)", "Optimized (4-path Weighted RRF)"]:
        r = results.get(name)
        if r is None:
            continue
        row = (
            f"{name:<30} {r.hit_at_1:7.3f} {r.hit_at_3:7.3f} {r.hit_at_5:7.3f} "
            f"{r.mrr:7.3f} {r.ndcg_at_5:7.3f} {r.precision_at_5:7.3f} "
            f"{r.recall_at_5:7.3f} {r.avg_latency_ms:7.1f}"
        )
        print(row)

    # Improvement section
    optimized = results.get("Optimized (4-path Weighted RRF)")
    if optimized is None:
        return

    print()
    print("-" * 90)
    print("IMPROVEMENT OF OPTIMIZED OVER BASELINES")
    print("-" * 90)
    for baseline_label in ["Pure Dense", "Pure BM25", "BM25 + Dense (RRF)"]:
        bl = results.get(baseline_label)
        if bl is None:
            continue

        def pct(opt: float, base: float) -> str:
            if base == 0:
                return "N/A" if opt == 0 else "+∞"
            change = (opt - base) / base * 100
            sign = "+" if change >= 0 else ""
            return f"{sign}{change:.1f}%"

        print(f"\n  vs {baseline_label}:")
        print(f"    Hit@1:   {bl.hit_at_1:.3f} → {optimized.hit_at_1:.3f}  ({pct(optimized.hit_at_1, bl.hit_at_1)})")
        print(f"    MRR:     {bl.mrr:.3f} → {optimized.mrr:.3f}  ({pct(optimized.mrr, bl.mrr)})")
        print(f"    NDCG@5:  {bl.ndcg_at_5:.3f} → {optimized.ndcg_at_5:.3f}  ({pct(optimized.ndcg_at_5, bl.ndcg_at_5)})")
        print(f"    Recall@5:{bl.recall_at_5:.3f} → {optimized.recall_at_5:.3f}  ({pct(optimized.recall_at_5, bl.recall_at_5)})")
        print(f"    Latency: {bl.avg_latency_ms:.1f}ms → {optimized.avg_latency_ms:.1f}ms  ({pct(optimized.avg_latency_ms, bl.avg_latency_ms)})")

    print()


def save_results(results: dict[str, EvalResult], path: Path) -> None:
    """Save full per-query results to JSON for later analysis."""
    payload: dict[str, Any] = {}
    for name, r in results.items():
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
            "per_query": r.per_query[:10],  # only first 10 in saved output
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Detailed results saved to {path}")


# ── Main ────────────────────────────────────────────────────────────


def load_queries_from_json(path: Path) -> list[TestQuery]:
    """Load hand-crafted test queries from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    queries: list[TestQuery] = []
    for item in raw:
        query_text = str(item.get("query", "")).strip()
        gt_ids = [str(pid).strip() for pid in list(item.get("ground_truth_paper_ids", [])) if str(pid).strip()]
        # Skip open queries without ground truth
        if not query_text:
            continue
        queries.append(
            TestQuery(
                query=query_text,
                ground_truth_paper_ids=gt_ids,
                acronym=item.get("type", ""),
                template=item.get("reason", ""),
            )
        )
    return queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval configurations")
    parser.add_argument("--max-queries", type=int, default=80, help="Max auto-generated test queries (ignored if --queries-json is set)")
    parser.add_argument("--queries-json", type=Path, default=None, help="Load hand-crafted queries from JSON file")
    parser.add_argument("--max-papers", type=int, default=6, help="Papers to retrieve per query (default: 6)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=Path, default=None, help="Save detailed results to JSON")
    parser.add_argument("--only-closed", action="store_true", default=True, help="Only evaluate queries with ground truth (default: True)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    settings = get_settings()
    retriever = DualIndexRetriever(settings)

    print(f"Loaded {len(retriever._paper_docs)} papers, {len(retriever._block_docs)} blocks")

    if args.queries_json and args.queries_json.exists():
        queries = load_queries_from_json(args.queries_json)
        print(f"Loaded {len(queries)} queries from {args.queries_json}")
        # Count open vs closed
        closed = [q for q in queries if q.ground_truth_paper_ids]
        open_q = [q for q in queries if not q.ground_truth_paper_ids]
        print(f"  Closed (with ground truth): {len(closed)}")
        print(f"  Open (no single ground truth): {len(open_q)}")
        if args.only_closed:
            queries = closed
            print(f"  Evaluating on {len(queries)} closed queries only")
        if queries:
            print(f"  Sample: {queries[0].query}")
    else:
        print(f"Generating up to {args.max_queries} test queries...")
        queries = generate_test_queries(retriever, max_queries=args.max_queries)
        print(f"Generated {len(queries)} test queries from paper acronyms/aliases")

    results = run_evaluation(retriever, queries, max_papers=args.max_papers)
    print_report(results, baseline_name="Pure Dense")

    output_path = args.output or (settings.data_dir / "eval_retrieval_results.json")
    save_results(results, output_path)

    retriever.close()


if __name__ == "__main__":
    main()
