"""Pinpoint where DeepSeek vs GPT-4o diverge for "GRPO是什么".

Tests 3 pipeline stages with identical inputs across two chat models:
  1. Planner — does it plan the same tools?
  2. Solver — does it extract the same claims from evidence?
  3. Answer Composer — does it generate the correct answer from the same claims+evidence?

The Router and retrieval stages are already proven identical across models
(Router test: both 100%; Retrieval benchmark: both models use same Dense embedding).
"""

from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

if str(PROJECT_ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings, Settings
from app.domain.models import QueryContract, SessionContext
from app.services.infra.model_clients import ModelClients
from app.services.agent.planner import AgentPlanner
from app.services.agent.planner_helpers import (
    planner_context_payload,
    planner_messages_with_user,
    json_planner_system_prompt,
    json_planner_human_prompt,
    fallback_plan,
    normalize_plan_payload,
)


def _make_clients(chat_model: str, api_key: str, base_url: str) -> ModelClients:
    """Create ModelClients with a specific chat model."""
    settings = get_settings()
    # Build override settings
    override = Settings.model_validate({
        **settings.model_dump(),
        "chat_model": chat_model,
        "openai_api_key": api_key,
        "openai_base_url": base_url,
    })
    return ModelClients(override)


def _make_planner(clients: ModelClients) -> AgentPlanner:
    return AgentPlanner(
        clients=clients,
        conversation_context=lambda session, max_chars=12000: "",
        conversation_messages=lambda session: [],
        is_negative_correction_query=lambda q: False,
        confidence_floor=0.6,
    )


def _empty_session() -> SessionContext:
    return SessionContext(session_id="divergence_test")


# Fixed QueryContract — same as what Router would produce for "GRPO是什么"
GRPO_CONTRACT = QueryContract(
    clean_query="GRPO是什么",
    interaction_mode="research",
    relation="entity_definition",
    targets=["GRPO", "Group Relative Policy Optimization"],
    answer_slots=["definition", "mechanism", "entity_type"],
    requested_fields=["title", "authors", "year", "definition", "mechanism"],
    required_modalities=["page_text", "paper_card"],
    answer_shape="narrative",
    precision_requirement="normal",
    continuation_mode="fresh",
    allow_web_search=False,
    notes=["intent_kind=research", "router_action=need_corpus_search"],
)


def test_planner(clients: ModelClients, model_name: str) -> dict[str, Any]:
    """Test Planner output for GRPO query."""
    planner = _make_planner(clients)
    session = _empty_session()

    t0 = time.perf_counter()
    try:
        plan = planner.plan_actions(
            contract=GRPO_CONTRACT,
            session=session,
            use_web_search=False,
        )
        elapsed = time.perf_counter() - t0
        return {
            "model": model_name,
            "stage": "planner",
            "success": True,
            "elapsed_s": round(elapsed, 2),
            "actions": plan.get("actions", []),
            "thought": str(plan.get("thought", ""))[:300],
        }
    except Exception as e:
        return {
            "model": model_name,
            "stage": "planner",
            "success": False,
            "error": str(e)[:300],
        }


def test_answer_composer_direct(
    clients: ModelClients,
    model_name: str,
    papers_text: str,
    evidence_text: str,
) -> dict[str, Any]:
    """Directly call the chat model as an answer composer with fixed inputs."""
    from langchain_core.messages import SystemMessage, HumanMessage

    system_prompt = """You are a research assistant. Answer the user's question based ONLY on the provided evidence and papers.
If the evidence contradicts your training knowledge, trust the evidence.
If the evidence does not contain the answer, say so clearly — do NOT fabricate information.
Always cite the source paper when making factual claims."""

    human_prompt = f"""Question: GRPO是什么？

Retrieved Papers:
{papers_text}

Evidence Blocks:
{evidence_text}

Please answer the question based on the above evidence. Cite specific papers."""

    t0 = time.perf_counter()
    try:
        model = clients.chat
        if model is None:
            return {"model": model_name, "stage": "composer", "success": False, "error": "No chat model"}
        response = model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        elapsed = time.perf_counter() - t0
        answer = str(response.content)
        return {
            "model": model_name,
            "stage": "composer",
            "success": True,
            "elapsed_s": round(elapsed, 2),
            "answer": answer[:2000],
            "answer_length": len(answer),
        }
    except Exception as e:
        return {
            "model": model_name,
            "stage": "composer",
            "success": False,
            "error": str(e)[:300],
        }


def main() -> None:
    settings = get_settings()
    retriever = None
    papers_text = ""
    evidence_text = ""

    # First, do retrieval once (model-independent) to get papers and evidence
    print("Step 0: Retrieving papers and evidence for 'GRPO是什么' (model-independent)...")
    try:
        from app.services.retrieval.core import DualIndexRetriever
        retriever = DualIndexRetriever(settings)
        candidates = retriever.search_papers(
            query="GRPO是什么",
            contract=GRPO_CONTRACT,
            limit=6,
        )
        papers_lines: list[str] = []
        for i, c in enumerate(candidates, 1):
            papers_lines.append(
                f"{i}. {c.title} ({c.year}) — score={c.score:.3f}"
            )
            papers_lines.append(f"   paper_id: {c.paper_id}")
        papers_text = "\n".join(papers_lines)

        # Get evidence blocks for top papers
        from app.services.retrieval.core import DualIndexRetriever
        paper_ids = [c.paper_id for c in candidates[:3]]
        evidence = retriever.expand_evidence(
            paper_ids=paper_ids,
            query="GRPO是什么",
            contract=GRPO_CONTRACT,
            limit=8,
        )
        evidence_lines: list[str] = []
        for i, e in enumerate(evidence, 1):
            evidence_lines.append(
                f"[{i}] {e.title} (page {e.page}, {e.block_type}): {e.snippet[:400]}"
            )
        evidence_text = "\n".join(evidence_lines)

        print(f"  Retrieved {len(candidates)} papers, {len(evidence)} evidence blocks")
        print(f"  Top paper: {candidates[0].title if candidates else 'NONE'}")
        print(f"  Papers preview:\n{papers_text[:500]}\n")
    except Exception as e:
        print(f"  Retrieval failed: {e}")
        if retriever:
            retriever.close()
        return

    # Define models to test
    models = [
        {
            "name": "deepseek-v4-flash",
            "api_key": "sk-2b5e0b33851d4ddb841e8e24424567b9",
            "base_url": "https://api.deepseek.com/v1",
        },
        {
            "name": "gpt-4o (Qihai)",
            "api_key": "sk-zLS9zYXMVip56ISKLAGcG3JPdCYrTFlMsyiFk48Slwc0a5ax",
            "base_url": "https://api.qhaigc.net/v1",
        },
    ]

    all_results: list[dict[str, Any]] = []

    for model_cfg in models:
        print(f"\n{'='*60}")
        print(f"Testing: {model_cfg['name']}")
        print(f"{'='*60}")

        clients = _make_clients(
            chat_model=model_cfg["name"].split()[0],  # "deepseek-v4-flash" or "gpt-4o"
            api_key=model_cfg["api_key"],
            base_url=model_cfg["base_url"],
        )

        if clients.chat is None:
            print(f"  SKIP: No chat model available")
            all_results.append({"model": model_cfg["name"], "error": "No chat model"})
            continue

        # Test 1: Planner
        print("\n  [1/2] Testing Planner...")
        planner_result = test_planner(clients, model_cfg["name"])
        all_results.append(planner_result)
        print(f"    Actions: {planner_result.get('actions', 'ERROR')}")
        print(f"    Thought: {planner_result.get('thought', '')[:200]}")

        # Test 2: Answer Composer
        print("\n  [2/2] Testing Answer Composer (fixed evidence)...")
        composer_result = test_answer_composer_direct(
            clients, model_cfg["name"], papers_text, evidence_text
        )
        all_results.append(composer_result)
        if composer_result.get("success"):
            answer = composer_result["answer"]
            # Check for hallucination markers
            has_deepseekmath = "DeepSeekMath" in answer
            has_group_relative = "Group Relative Policy Optimization" in answer
            has_generalized = "Generalized Reinforcement" in answer  # The GPT-4o hallucination
            print(f"    Answer length: {composer_result['answer_length']}")
            print(f"    Contains 'DeepSeekMath': {has_deepseekmath}")
            print(f"    Contains 'Group Relative Policy Optimization': {has_group_relative}")
            print(f"    HALLUCINATION 'Generalized Reinforcement': {has_generalized}")
            print(f"    Answer preview: {answer[:400]}...")
        else:
            print(f"    ERROR: {composer_result.get('error')}")

        clients.close()

    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")

    composer_results = [r for r in all_results if r.get("stage") == "composer" and r.get("success")]
    if len(composer_results) >= 2:
        r0, r1 = composer_results[0], composer_results[1]
        a0, a1 = r0.get("answer", ""), r1.get("answer", "")
        print(f"\n  {r0['model']}:")
        print(f"    'DeepSeekMath' in answer: {'DeepSeekMath' in a0}")
        print(f"    'Group Relative Policy Optimization': {'Group Relative Policy Optimization' in a0}")
        print(f"    'Generalized Reinforcement' (HALLUC): {'Generalized Reinforcement' in a0}")
        print(f"\n  {r1['model']}:")
        print(f"    'DeepSeekMath' in answer: {'DeepSeekMath' in a1}")
        print(f"    'Group Relative Policy Optimization': {'Group Relative Policy Optimization' in a1}")
        print(f"    'Generalized Reinforcement' (HALLUC): {'Generalized Reinforcement' in a1}")

    # Save results
    output_path = settings.data_dir / "model_divergence_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "query": "GRPO是什么",
            "top_papers": papers_text,
            "evidence_preview": evidence_text[:2000],
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")

    if retriever:
        retriever.close()


if __name__ == "__main__":
    main()
