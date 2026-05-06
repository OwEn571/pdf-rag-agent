"""Router tool-choice accuracy test — compares chat models on intent routing.

Evaluates whether the LLMIntentRouter correctly routes academic queries to
need_corpus_search (vs answer_directly/need_clarify where it might hallucinate).
Supports multi-turn simulation for queries that get need_clarify first.

Metrics:
  - Final Search Rate: % of academic queries that eventually trigger corpus search
  - Direct Answer Error Rate: % where answer_directly is chosen (hallucination risk)
  - Single-Turn Success Rate: % routed directly to need_corpus_search on first try
  - Average Turns: mean number of turns to reach corpus search
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if str(PROJECT_ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_openai import ChatOpenAI
import httpx

from app.core.config import get_settings
from app.domain.models import SessionContext
from app.services.intents.router import LLMIntentRouter, ROUTER_TOOLS
from app.services.infra.model_clients import ModelClients


# ── Smalltalk / edge-case queries ───────────────────────────────────

SMALLTALK_QUERIES = [
    {"query": "你好", "expected_final": "answer_directly"},
    {"query": "谢谢", "expected_final": "answer_directly"},
    {"query": "你能做什么", "expected_final": "answer_directly"},
    {"query": "你是谁", "expected_final": "answer_directly"},
    {"query": "再见", "expected_final": "answer_directly"},
    {"query": "今天天气怎么样", "expected_final": "answer_directly"},
    {"query": "帮我写一首诗", "expected_final": "answer_directly"},
    {"query": "讲个笑话", "expected_final": "answer_directly"},
    {"query": "你是什么模型", "expected_final": "answer_directly"},
    {"query": "hello", "expected_final": "answer_directly"},
    {"query": "What can you do", "expected_final": "answer_directly"},
    {"query": "你好吗", "expected_final": "answer_directly"},
    {"query": "早上好", "expected_final": "answer_directly"},
    {"query": "你支持什么功能", "expected_final": "answer_directly"},
    {"query": "怎么使用这个系统", "expected_final": "answer_directly"},
]

AMBIGUOUS_QUERIES = [
    {"query": "RL是什么", "expected_final": "need_corpus_search",
     "clarify_reply": "机器学习里的概念"},
    {"query": "什么是PPO", "expected_final": "need_corpus_search",
     "clarify_reply": "论文中的缩写"},
    {"query": "ACE是什么意思", "expected_final": "need_corpus_search",
     "clarify_reply": "机器学习论文里的术语"},
    {"query": "DP是什么", "expected_final": "need_corpus_search",
     "clarify_reply": "就是论文里的概念"},
    {"query": "MLE是啥", "expected_final": "need_corpus_search",
     "clarify_reply": "论文中的方法"},
    {"query": "SFT的定义", "expected_final": "need_corpus_search",
     "clarify_reply": "论文里的术语"},
    {"query": "RLHF里面reward是怎么训练的", "expected_final": "need_corpus_search",
     "clarify_reply": "我说的是论文中的方法"},
    {"query": "CRPO指的是什么", "expected_final": "need_corpus_search",
     "clarify_reply": "论文里的概念"},
    {"query": "APE和DPO有什么区别", "expected_final": "need_corpus_search",
     "clarify_reply": "论文中的概念缩写"},
    {"query": "什么是多模态模型中的对齐", "expected_final": "need_corpus_search",
     "clarify_reply": "请在我的论文库中检索"},
]

# Clarification reply templates (rotated per query)
CLARIFY_REPLIES = [
    "就是论文里的术语/概念缩写",
    "我说的是机器学习论文中的方法",
    "请在我的论文库中检索这个概念",
    "论文中的缩写，帮我查一下",
    "就是AI领域的那个术语",
]


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class RoutingResult:
    query: str
    query_type: str  # "academic", "smalltalk", "ambiguous"
    first_action: str = ""
    first_confidence: float = 0.0
    first_rationale: str = ""
    final_action: str = ""
    turns: int = 0
    path: list[str] = field(default_factory=list)  # action sequence
    error: str = ""


@dataclass
class ModelReport:
    model_name: str
    total_academic: int = 0
    total_smalltalk: int = 0
    total_ambiguous: int = 0
    # Academic queries
    academic_final_search: int = 0       # reached need_corpus_search
    academic_direct_answer_error: int = 0  # answer_directly at any turn
    academic_single_turn: int = 0         # need_corpus_search on first try
    academic_multi_turn_clarify: int = 0  # need_clarify → corpus_search
    academic_clarify_then_hallucinate: int = 0  # need_clarify → answer_directly
    academic_web_search: int = 0
    academic_conversation_tool: int = 0
    academic_total_turns: int = 0
    # Smalltalk
    smalltalk_direct_answer: int = 0
    # Detailed results
    results: list[RoutingResult] = field(default_factory=list)

    @property
    def final_search_rate(self) -> float:
        return self.academic_final_search / max(1, self.total_academic)

    @property
    def direct_answer_error_rate(self) -> float:
        return self.academic_direct_answer_error / max(1, self.total_academic)

    @property
    def single_turn_rate(self) -> float:
        return self.academic_single_turn / max(1, self.total_academic)

    @property
    def avg_turns(self) -> float:
        return self.academic_total_turns / max(1, self.total_academic)


# ── Router test harness ──────────────────────────────────────────────

def _make_clients(settings: Any) -> ModelClients:
    """Create ModelClients from settings."""
    return ModelClients(settings)


def _make_router(clients: ModelClients) -> LLMIntentRouter:
    """Create a Router with minimal conversation context for single-turn testing."""
    return LLMIntentRouter(
        clients=clients,
        conversation_context=lambda session, max_chars=12000: "",
        conversation_messages=lambda session: [],
    )


def _empty_session() -> SessionContext:
    return SessionContext(session_id="router_test")


def _route_with_retry(router: LLMIntentRouter, query: str, session: SessionContext,
                      max_retries: int = 2) -> RoutingResult:
    """Route a query, with up to max_retries retries on API errors."""
    for attempt in range(max_retries + 1):
        try:
            decision = router.route(query=query, session=session)
            return RoutingResult(
                query=query,
                query_type="",
                first_action=decision.action,
                first_confidence=decision.confidence,
                first_rationale=decision.rationale,
                final_action=decision.action,
                turns=1,
                path=[decision.action],
            )
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return RoutingResult(
                    query=query, query_type="", error=str(e)[:200],
                    first_action="error", final_action="error",
                )
    return RoutingResult(query=query, query_type="", error="unknown")


def run_routing_test(queries: list[dict[str, Any]], clients: ModelClients,
                     query_type: str, max_clarify_turns: int = 2) -> list[RoutingResult]:
    """Run routing test for a set of queries with multi-turn simulation."""
    router = _make_router(clients)
    session = _empty_session()
    results: list[RoutingResult] = []

    for i, item in enumerate(queries):
        query = item["query"]
        clarify_reply = item.get("clarify_reply",
                                 CLARIFY_REPLIES[i % len(CLARIFY_REPLIES)])

        path: list[str] = []
        final_action = ""
        turns = 0
        error = ""

        current_query = query
        # Create a fresh session per query for isolation
        current_session = SessionContext(session_id=f"router_test_{i}")

        for turn in range(1, max_clarify_turns + 2):  # +2 for initial + max clarify turns
            turns = turn
            try:
                decision = router.route(query=current_query, session=current_session)
            except Exception as e:
                error = str(e)[:200]
                path.append("error")
                break

            action = decision.action
            path.append(action)

            if action == "need_corpus_search":
                final_action = action
                break
            elif action == "need_clarify":
                # Simulate user clarification response
                current_query = clarify_reply
                # Update session to reflect that we're in a followup
                current_session.continuation_mode = "followup"
                current_session.open_questions.append(query)
                continue
            elif action == "answer_directly":
                final_action = action
                break
            elif action == "need_web":
                final_action = action
                break
            elif action == "need_conversation_tool":
                final_action = action
                break
            else:
                final_action = action
                break

        if turns > 0 and not final_action:
            final_action = path[-1] if path else "unknown"

        results.append(RoutingResult(
            query=query,
            query_type=query_type,
            first_action=path[0] if path else "error",
            first_confidence=0.0,
            final_action=final_action,
            turns=turns,
            path=path,
            error=error,
        ))

        if (i + 1) % 20 == 0:
            print(f"  {query_type}: {i+1}/{len(queries)} done")

    return results


def compute_report(results: list[RoutingResult], model_name: str,
                   query_type_counts: dict[str, int]) -> ModelReport:
    """Compute aggregate metrics from routing results."""
    report = ModelReport(model_name=model_name)
    report.total_academic = query_type_counts.get("academic", 0)
    report.total_smalltalk = query_type_counts.get("smalltalk", 0)
    report.total_ambiguous = query_type_counts.get("ambiguous", 0)
    report.results = results

    for r in results:
        if r.query_type == "academic":
            if r.final_action == "need_corpus_search":
                report.academic_final_search += 1
            if "answer_directly" in r.path:
                report.academic_direct_answer_error += 1
            if r.first_action == "need_corpus_search":
                report.academic_single_turn += 1
            if r.first_action == "need_clarify" and r.final_action == "need_corpus_search":
                report.academic_multi_turn_clarify += 1
            if "need_clarify" in r.path and r.final_action == "answer_directly":
                report.academic_clarify_then_hallucinate += 1
            if r.final_action == "need_web":
                report.academic_web_search += 1
            if r.final_action == "need_conversation_tool":
                report.academic_conversation_tool += 1
            report.academic_total_turns += r.turns
        elif r.query_type == "smalltalk":
            if r.final_action == "answer_directly":
                report.smalltalk_direct_answer += 1

    return report


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    settings = get_settings()
    print(f"Chat model (from .env): {settings.chat_model}")
    print(f"Base URL: {settings.openai_base_url}")

    # Load academic queries
    queries_path = settings.data_dir / "eval_queries_v3.json"
    with open(queries_path, "r", encoding="utf-8") as f:
        raw_queries = json.load(f)

    academic_queries: list[dict[str, Any]] = []
    for item in raw_queries:
        q = str(item.get("query", "")).strip()
        if q:
            academic_queries.append({"query": q, "expected_final": "need_corpus_search"})
    print(f"Loaded {len(academic_queries)} academic queries")

    all_query_groups = [
        ("academic", academic_queries),
        ("smalltalk", SMALLTALK_QUERIES),
        ("ambiguous", AMBIGUOUS_QUERIES),
    ]

    query_type_counts = {
        "academic": len(academic_queries),
        "smalltalk": len(SMALLTALK_QUERIES),
        "ambiguous": len(AMBIGUOUS_QUERIES),
    }

    # Run test
    clients = _make_clients(settings)
    all_results: list[RoutingResult] = []

    print(f"\nRunning Router accuracy test on {sum(query_type_counts.values())} queries...\n")
    start_time = time.perf_counter()

    for qtype, queries in all_query_groups:
        print(f"Testing {qtype} queries ({len(queries)} total)...")
        results = run_routing_test(queries, clients, query_type=qtype)
        all_results.extend(results)

    elapsed = time.perf_counter() - start_time
    print(f"\nTest completed in {elapsed:.1f}s")

    # Compute report
    report = compute_report(all_results, settings.chat_model, query_type_counts)

    # Print report
    print("\n" + "=" * 80)
    print(f"ROUTER TOOL-CHOICE ACCURACY — {settings.chat_model}")
    print(f"Base URL: {settings.openai_base_url}")
    print("=" * 80)

    print(f"\n── Academic Queries ({report.total_academic} total) ──")
    print(f"  Final Search Rate:         {report.academic_final_search}/{report.total_academic} = {report.final_search_rate:.3f}")
    print(f"  Direct Answer Error Rate:  {report.academic_direct_answer_error}/{report.total_academic} = {report.direct_answer_error_rate:.3f}")
    print(f"  Single-Turn Success:       {report.academic_single_turn}/{report.total_academic} = {report.single_turn_rate:.3f}")
    print(f"  Multi-Turn (clarify→search): {report.academic_multi_turn_clarify}/{report.total_academic}")
    print(f"  Clarify→Hallucinate:       {report.academic_clarify_then_hallucinate}/{report.total_academic}")
    print(f"  Web Search:                {report.academic_web_search}/{report.total_academic}")
    print(f"  Conversation Tool:         {report.academic_conversation_tool}/{report.total_academic}")
    print(f"  Average Turns:             {report.avg_turns:.2f}")

    print(f"\n── Smalltalk Queries ({report.total_smalltalk} total) ──")
    print(f"  Correctly answered directly: {report.smalltalk_direct_answer}/{report.total_smalltalk}")

    # Breakdown by action
    print(f"\n── First-Action Distribution (Academic) ──")
    action_counts: dict[str, int] = defaultdict(int)
    for r in all_results:
        if r.query_type == "academic":
            action_counts[r.first_action] += 1
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / max(1, report.total_academic) * 100
        print(f"  {action:<30} {count:>4} ({pct:.1f}%)")

    # Show hallucination-risk examples
    hallucination_cases = [r for r in all_results
                           if r.query_type == "academic" and "answer_directly" in r.path]
    if hallucination_cases:
        print(f"\n── ⚠️  Direct Answer Errors (Hallucination Risk) — {len(hallucination_cases)} cases ──")
        for r in hallucination_cases[:15]:
            path_str = " → ".join(r.path)
            print(f"  [{r.query}] path: {path_str}")

    # Show clarify→hallucinate cases (the GPT-4o GRPO pattern)
    clarify_hallucinate = [r for r in all_results
                           if r.query_type == "academic"
                           and "need_clarify" in r.path
                           and r.final_action == "answer_directly"]
    if clarify_hallucinate:
        print(f"\n── 🔴 Clarify→Hallucinate Pattern (most dangerous) — {len(clarify_hallucinate)} cases ──")
        for r in clarify_hallucinate[:10]:
            path_str = " → ".join(r.path)
            print(f"  [{r.query}] path: {path_str}")

    # Show failed-to-search cases
    failed_search = [r for r in all_results
                     if r.query_type == "academic" and r.final_action != "need_corpus_search"]
    if failed_search:
        print(f"\n── ❌ Academic Queries NOT Routed to Search — {len(failed_search)} cases ──")
        for r in failed_search[:20]:
            path_str = " → ".join(r.path)
            print(f"  [{r.query}] → {r.final_action} ({path_str})")

    # Save detailed results
    output_path = settings.data_dir / "router_accuracy_results.json"
    payload = {
        "model": settings.chat_model,
        "base_url": settings.openai_base_url,
        "total_academic": report.total_academic,
        "total_smalltalk": report.total_smalltalk,
        "total_ambiguous": report.total_ambiguous,
        "final_search_rate": round(report.final_search_rate, 4),
        "direct_answer_error_rate": round(report.direct_answer_error_rate, 4),
        "single_turn_rate": round(report.single_turn_rate, 4),
        "avg_turns": round(report.avg_turns, 2),
        "academic_final_search": report.academic_final_search,
        "academic_direct_answer_error": report.academic_direct_answer_error,
        "academic_single_turn": report.academic_single_turn,
        "academic_multi_turn_clarify": report.academic_multi_turn_clarify,
        "academic_clarify_then_hallucinate": report.academic_clarify_then_hallucinate,
        "smalltalk_direct_answer": report.smalltalk_direct_answer,
        "results": [
            {
                "query": r.query,
                "query_type": r.query_type,
                "first_action": r.first_action,
                "final_action": r.final_action,
                "turns": r.turns,
                "path": r.path,
                "error": r.error,
            }
            for r in all_results
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {output_path}")

    clients.close()


if __name__ == "__main__":
    main()
