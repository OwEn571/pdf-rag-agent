"""Full-pipeline trace comparison: DeepSeek vs GPT-4o for 'GRPO是什么'.

Captures every intermediate output so we can diff exactly where two models diverge.
Uses lightweight composer to avoid wiring every solver/verifier's complex signatures.
"""

from __future__ import annotations

import json, sys, time
from pathlib import Path
from typing import Any

if str(PROJECT_ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings, Settings
from app.domain.models import QueryContract, SessionContext
from app.services.infra.model_clients import ModelClients
from app.services.retrieval.core import DualIndexRetriever
from app.services.planning.query_shaping import paper_query_text, extract_targets
from app.services.intents.router import LLMIntentRouter

QUERY = "GRPO是什么"


def make_clients(chat_model: str, api_key: str, base_url: str) -> ModelClients:
    settings = get_settings()
    override = Settings.model_validate({**settings.model_dump(),
        "chat_model": chat_model, "openai_api_key": api_key, "openai_base_url": base_url})
    return ModelClients(override)


def compose_answer_direct(clients, contract, evidence, candidates) -> str:
    """Lightweight composer — same as what the agent AnswerComposer does."""
    from langchain_core.messages import SystemMessage, HumanMessage
    model = clients.chat
    if model is None:
        return "(no model)"

    ctx: list[str] = []
    for i, p in enumerate(candidates[:6], 1):
        ctx.append(f"[Paper {i}] {p.title} ({p.year}) — {p.paper_id}")
    for i, e in enumerate(evidence[:10], 1):
        ctx.append(f"[Evidence {i}] {e.title} p{e.page} {e.block_type}: {e.snippet[:400]}")

    system_prompt = (
        "你是 Zotero Paper RAG Agent，服务于用户的 ML/AI 论文库。"
        "只基于提供的证据回答。如果证据中没有答案，明确说'在我的论文库中没有找到相关信息'。"
        "绝不编造。引用论文标题。"
    )
    human_prompt = f"问题: {QUERY}\n\n" + "\n\n".join(ctx)

    resp = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    return str(resp.content)


def run_trace(clients: ModelClients, model_name: str, retriever: DualIndexRetriever) -> dict[str, Any]:
    trace: dict[str, Any] = {"model": model_name, "query": QUERY}

    # ── 1. Router ──
    router = LLMIntentRouter(clients=clients,
        conversation_context=lambda s, max_chars=12000: "",
        conversation_messages=lambda s: [])
    session = SessionContext(session_id="trace_diff")

    t0 = time.perf_counter()
    decision = router.route(query=QUERY, session=session)
    trace["1_router_action"] = decision.action
    trace["1_router_confidence"] = round(decision.confidence, 3)
    trace["1_router_args_query"] = str(decision.args.get("query", ""))[:200]
    trace["1_router_args_targets"] = list(decision.args.get("targets", []))
    trace["1_router_args_relation"] = str(decision.args.get("relation", ""))
    trace["1_router_latency_ms"] = round((time.perf_counter() - t0) * 1000)

    # Build contract
    regex_targets = extract_targets(QUERY)
    raw_targets = list(decision.args.get("targets", regex_targets))
    canonical = retriever.canonicalize_targets(raw_targets)

    contract = QueryContract(
        clean_query=QUERY,
        interaction_mode="research" if decision.action == "need_corpus_search" else "conversation",
        relation=str(decision.args.get("relation", "entity_definition")),
        targets=canonical,
        answer_slots=["definition", "mechanism", "entity_type"],
        requested_fields=list(decision.args.get("requested_fields", ["title", "authors", "year"])),
        required_modalities=["page_text", "paper_card"],
        answer_shape="narrative",
        precision_requirement="normal",
        continuation_mode="fresh",
        allow_web_search=False,
    )
    trace["1_contract_targets"] = contract.targets
    trace["1_contract_relation"] = contract.relation

    # ── 2. Search ──
    search_query = paper_query_text(contract)
    trace["2_search_query"] = search_query
    t0 = time.perf_counter()
    candidates = retriever.search_papers(query=search_query, contract=contract, limit=6)
    trace["2_search_latency_ms"] = round((time.perf_counter() - t0) * 1000)
    trace["2_candidates"] = [
        {"rank": i, "title": c.title[:100], "paper_id": c.paper_id,
         "year": c.year, "score": round(c.score, 4)}
        for i, c in enumerate(candidates, 1)
    ]

    # ── 3. Evidence ──
    paper_ids = [c.paper_id for c in candidates[:3]]
    evidence = retriever.expand_evidence(paper_ids=paper_ids, query=QUERY, contract=contract, limit=10)
    trace["3_evidence_count"] = len(evidence)
    trace["3_evidence_titles"] = list(dict.fromkeys(e.title[:80] for e in evidence))

    # ── 4. Answer Composer ──
    t0 = time.perf_counter()
    answer = compose_answer_direct(clients, contract, evidence, candidates)
    trace["4_composer_latency_ms"] = round((time.perf_counter() - t0) * 1000)
    trace["4_answer"] = answer[:2000]
    trace["4_answer_len"] = len(answer)

    # Key content checks
    trace["X_has_DeepSeekMath"] = "DeepSeekMath" in answer
    trace["X_has_Group_Relative"] = "Group Relative Policy Optimization" in answer
    trace["X_has_Generalized"] = "Generalized Reinforcement" in answer  # HALLUCINATION
    trace["X_has_GRPO_definition"] = "GRPO" in answer and ("Group Relative" in answer or "强化学习" in answer)

    return trace


def main():
    settings = get_settings()

    # ── Model A: current .env ──
    a_name = f"{settings.chat_model}"
    print(f"[A] {a_name} ...")
    clients_a = ModelClients(settings)
    retriever = DualIndexRetriever(settings)
    trace_a = run_trace(clients_a, a_name, retriever)
    clients_a.close()

    # ── Model B: the other one ──
    if "deepseek" in settings.openai_base_url.lower():
        b_name = "gpt-4o"
        b_key = "sk-zLS9zYXMVip56ISKLAGcG3JPdCYrTFlMsyiFk48Slwc0a5ax"
        b_url = "https://api.qhaigc.net/v1"
    else:
        b_name = "deepseek-v4-flash"
        b_key = "sk-2b5e0b33851d4ddb841e8e24424567b9"
        b_url = "https://api.deepseek.com/v1"

    print(f"[B] {b_name} ...")
    clients_b = make_clients(b_name, b_key, b_url)
    trace_b = run_trace(clients_b, b_name, retriever)
    clients_b.close()
    retriever.close()

    # ── Diff ──
    print("\n" + "=" * 70)
    print(f"TRACE DIFF: '{QUERY}'")
    print(f"  A = {trace_a['model']}")
    print(f"  B = {trace_b['model']}")
    print("=" * 70)

    keys = [
        ("1_router_action", "Router action"),
        ("1_router_args_targets", "Router targets"),
        ("1_router_args_relation", "Router relation"),
        ("1_contract_targets", "Contract targets (canonicalized)"),
        ("2_search_query", "Search query (paper_query_text)"),
        ("2_candidates", "Top-6 candidates"),
        ("3_evidence_titles", "Evidence paper titles"),
        ("4_answer", "Final answer"),
        ("X_has_DeepSeekMath", "答案含 DeepSeekMath"),
        ("X_has_Group_Relative", "答案含 Group Relative Policy Optimization"),
        ("X_has_Generalized", "答案含 Generalized (幻觉!)"),
    ]

    diffs = 0
    for key, label in keys:
        va = trace_a.get(key)
        vb = trace_b.get(key)
        same = json.dumps(va, ensure_ascii=False, default=str) == json.dumps(vb, ensure_ascii=False, default=str)
        marker = "=" if same else "≠ DIFFER"
        diffs += 0 if same else 1
        print(f"\n── {label} ── [{marker}]")
        if not same:
            print(f"  A: {json.dumps(va, ensure_ascii=False, default=str)[:600]}")
            print(f"  B: {json.dumps(vb, ensure_ascii=False, default=str)[:600]}")

    print(f"\n{'='*70}")
    print(f"Total diffs: {diffs}/{len(keys)}")

    out = {"query": QUERY, "A": trace_a, "B": trace_b}
    path = settings.data_dir / "trace_diff_result.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
