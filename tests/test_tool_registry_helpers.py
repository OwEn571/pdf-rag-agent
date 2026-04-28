from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.domain.models import ActiveResearch, EvidenceBlock, QueryContract, SessionContext, SessionTurn
from app.services.tool_registry_helpers import (
    atomic_search_observation_payload,
    atomic_search_tool_request,
    coerce_int,
    conversation_intent_summary,
    evidence_blocks_from_state,
    evidence_event_payload,
    evidence_result_observation_payload,
    fetch_url_evidence,
    fetch_url_payload,
    fetch_url_tool_payload,
    fetch_url_tool_request,
    focus_values,
    format_fetched_urls_answer,
    format_summaries_answer,
    format_task_results_answer,
    grep_corpus_tool_request,
    library_metadata_observation_payload,
    library_metadata_tool_request,
    normalize_todo_items,
    planned_tool_input_from_state,
    propose_tool_payload,
    query_rewrite_tool_payload,
    query_rewrite_tool_request,
    read_memory_tool_payload,
    read_pdf_page_tool_request,
    remember_tool_payload,
    rerank_observation_payload,
    rerank_tool_request,
    research_intent_summary,
    search_corpus_observation_payload,
    search_corpus_strategy,
    store_claim_check_payload,
    store_fetch_url_evidence_result,
    store_research_evidence_result,
    store_session_todos,
    summary_source_from_state,
    summarize_tool_payload,
    task_result_observation_payload,
    task_tool_request,
    todo_write_tool_payload,
    tool_input_from_state,
    verify_claim_tool_payload,
)
from app.services.url_fetcher import FetchUrlResult


def test_tool_registry_helpers_normalize_and_store_todos() -> None:
    session = SessionContext(session_id="s1", working_memory={"keep": "yes"})
    items = normalize_todo_items(
        [
            {"id": " a ", "text": "  read   papers ", "status": "doing"},
            {"text": "write answer", "status": "unknown"},
            {"id": "empty", "text": ""},
            "bad",
        ]
    )

    assert items == [
        {"id": "a", "text": "read papers", "status": "doing"},
        {"id": "todo-2", "text": "write answer", "status": "pending"},
    ]
    store_session_todos(session, items)
    assert session.working_memory["keep"] == "yes"
    assert session.working_memory["todos"] == items
    tool_items, payload, summary = todo_write_tool_payload(planned_input={"items": items}, session=session)
    assert tool_items == items
    assert payload == {"items": items}
    assert summary == "todos=2"


def test_tool_registry_helpers_read_tool_input_and_intent_summary() -> None:
    state = {"tool_inputs": {"fetch_url": {"url": "https://example.com"}, "bad": "nope"}}
    contract = QueryContract(
        clean_query="上一轮第一篇是什么？",
        interaction_mode="conversation",
        relation="memory_followup",
        requested_fields=["previous_rationale"],
        required_modalities=["page_text"],
        targets=["DPO"],
        notes=["intent_kind=memory_op", "answer_slot=previous_rationale"],
    )

    assert tool_input_from_state(state, "fetch_url") == {"url": "https://example.com"}
    assert tool_input_from_state(state, "bad") == {}
    assert planned_tool_input_from_state(
        {"tool_inputs": {"summarize": {"target_words": 80}}, "current_tool_input": {"target_words": 20}},
        "summarize",
    ) == {"target_words": 80}
    assert planned_tool_input_from_state({"current_tool_input": {"target_words": 20}}, "summarize") == {
        "target_words": 20
    }
    assert conversation_intent_summary(contract) == {
        "kind": "memory_op",
        "answer_slots": ["previous_rationale"],
        "requested_fields": ["previous_rationale"],
        "targets": ["DPO"],
    }
    summary, payload = research_intent_summary(contract)
    assert summary == "previous_rationale"
    assert payload["required_modalities"] == ["page_text"]


def test_tool_registry_helpers_build_library_metadata_payloads() -> None:
    request = library_metadata_tool_request(
        planned_input={"query": " AlignX ", "limit": 5},
        fallback_query="fallback",
    )
    fallback_request = library_metadata_tool_request(planned_input={}, fallback_query=" fallback ")
    summary, payload = library_metadata_observation_payload(
        result={
            "sql": "select * from papers",
            "columns": ["title"],
            "row_count": 2,
            "truncated": False,
            "error": "",
        },
        answer="answer",
    )

    assert request == {"query": "AlignX", "limit": 5}
    assert fallback_request == {"query": "fallback"}
    assert summary == "rows=2"
    assert payload == {
        "sql": "select * from papers",
        "columns": ["title"],
        "row_count": 2,
        "truncated": False,
        "error": "",
        "has_answer": True,
    }


def test_tool_registry_helpers_build_read_memory_payload() -> None:
    session = SessionContext(
        session_id="s1",
        working_memory={"note": "keep"},
        active_research=ActiveResearch(targets=["DPO"], titles=["A", "B", "C"], clean_query="DPO"),
        turns=[SessionTurn(query="q", answer="a")],
    )
    agent = SimpleNamespace(_session_conversation_context=lambda _: {"working_memory": session.working_memory})

    call_arguments, summary, payload = read_memory_tool_payload(agent=agent, session=session, active_title_limit=2)

    assert call_arguments == {"turn_count": 1, "active_targets": ["DPO"]}
    assert summary == "turns=1"
    assert payload["active_research_context"]["titles"] == ["A", "B"]
    assert payload["active_research_context"]["active_titles"] == ["A", "B"]
    assert payload["has_working_memory"] is True


def test_tool_registry_helpers_format_compose_sources() -> None:
    task_answer = format_task_results_answer(
        [
            {"prompt": "A", "answer": "answer A"},
            {"prompt": "B", "answer": ""},
            {"answer": "answer C"},
        ]
    )
    fetched_answer = format_fetched_urls_answer(
        [
            {"ok": True, "url": "https://example.com/a", "title": "A", "text": "body"},
            {"ok": False, "url": "https://example.com/b", "error": "timeout"},
        ]
    )
    summaries_answer = format_summaries_answer([{"summary": " one "}, {"summary": ""}, {"summary": "two"}])

    assert "## 1. A" in task_answer
    assert "## 3. 子任务 3" in task_answer
    assert "### A" in fetched_answer
    assert "读取失败" in fetched_answer
    assert summaries_answer == "one\n\ntwo"


def test_tool_registry_helpers_build_task_request_and_observation() -> None:
    request = task_tool_request(
        planned_input={
            "description": " 子任务 ",
            "prompt": "  explain   DPO ",
            "tools_allowed": [" compose ", "", "fetch_url"],
            "max_steps": 4,
        },
        fallback_prompt="fallback",
    )
    fallback_request = task_tool_request(planned_input={}, fallback_prompt=" fallback query ")
    summary, payload = task_result_observation_payload(
        request=request,
        result={"answer": "done", "verification": {"status": "pass"}},
    )

    assert request == {
        "prompt": "explain DPO",
        "description": " 子任务 ",
        "tools_allowed": ["compose", "fetch_url"],
        "max_steps": 4,
    }
    assert fallback_request["prompt"] == "fallback query"
    assert summary == "task_answer_chars=4"
    assert payload == {"prompt": "explain DPO", "verification": {"status": "pass"}, "answer_chars": 4}


def test_tool_registry_helpers_convert_fetch_result_payload_and_evidence() -> None:
    result = FetchUrlResult(
        ok=True,
        url="https://example.com/a",
        title="Example",
        text="x" * 2000,
        status_code=200,
    )
    failed = FetchUrlResult(ok=False, url="https://example.com/b", error="timeout")

    payload = fetch_url_payload(result)
    state_payload, summary, observation_payload = fetch_url_tool_payload(result)
    evidence = fetch_url_evidence(result)

    assert payload["ok"] is True
    assert payload["status_code"] == 200
    assert fetch_url_tool_request({"url": " https://example.com/a ", "max_chars": 1000}) == {
        "url": "https://example.com/a",
        "max_chars": 1000,
    }
    assert state_payload == payload
    assert summary == "ok"
    assert observation_payload["text"] == "x" * 600
    assert evidence is not None
    assert evidence.doc_id.startswith("web::fetch::")
    assert evidence.paper_id == evidence.doc_id
    assert evidence.snippet == "x" * 1600
    assert evidence.metadata["url"] == "https://example.com/a"
    assert fetch_url_evidence(failed) is None


def test_tool_registry_helpers_collect_evidence_and_summary_source() -> None:
    evidence = EvidenceBlock(
        doc_id="local-a",
        paper_id="P1",
        title="Paper",
        file_path="/tmp/paper.pdf",
        page=1,
        block_type="page_text",
        snippet="local evidence",
    )
    state = {
        "evidence": [evidence],
        "web_evidence": [evidence],
        "fetched_urls": [{"url": "https://example.com", "title": "Example", "text": "fetched text"}],
        "task_results": [{"answer": "task text"}],
    }

    collected = evidence_blocks_from_state(state)
    assert [item.doc_id for item in collected] == ["local-a", "https://example.com"]
    assert summary_source_from_state(state) == "fetched text\ntask text"


def test_tool_registry_helpers_small_coercions() -> None:
    assert focus_values([" A ", "", "B"], ["fallback"]) == ["A", "B"]
    assert focus_values("bad", ["fallback"]) == ["fallback"]
    assert coerce_int("5", default=1, minimum=1, maximum=10) == 5
    assert coerce_int("bad", default=7, minimum=1, maximum=10) == 7
    assert coerce_int(99, default=1, minimum=1, maximum=10) == 10


def test_tool_registry_helpers_build_research_retrieval_requests() -> None:
    contract = QueryContract(clean_query="DPO loss", targets=["DPO"])
    screened = [SimpleNamespace(paper_id="paper-1")]
    state = {"screened_papers": screened, "rewritten_queries": ["rewritten DPO"]}

    read_request = read_pdf_page_tool_request(
        planned_input={"page_from": "2", "page_to": "4", "max_chars": "800"},
        state=state,
    )
    grep_request, grep_payload = grep_corpus_tool_request(
        planned_input={"scope": "blocks", "paper_ids": [" paper-1 ", "", "paper-2"], "max_hits": 5},
        state=state,
    )
    rewrite_request = query_rewrite_tool_request(
        planned_input={"targets": [" DPO ", ""], "max_queries": 99, "mode": "step_back"},
        contract=contract,
    )

    assert read_request == {"paper_id": "paper-1", "page_from": 2, "page_to": 4, "max_chars": 800}
    assert grep_request == {
        "pattern": "rewritten DPO",
        "scope": "blocks",
        "paper_ids": ["paper-1", "paper-2"],
        "max_hits": 5,
    }
    assert grep_payload == {
        "regex": "rewritten DPO",
        "scope": "blocks",
        "paper_ids": ["paper-1", "paper-2"],
        "max_hits": 5,
    }
    assert rewrite_request == {"query": "DPO loss", "targets": ["DPO"], "mode": "step_back", "max_queries": 8}


def test_tool_registry_helpers_store_query_rewrite_payload() -> None:
    state: dict[str, object] = {}
    result = SimpleNamespace(payload=lambda: {"query": "DPO", "mode": "step_back", "queries": ["DPO", "DPO evidence"]})

    payload, summary = query_rewrite_tool_payload(result=result, state=state)

    assert payload["mode"] == "step_back"
    assert state["query_rewrites"] == [payload]
    assert state["rewritten_queries"] == ["DPO", "DPO evidence"]
    assert summary == "queries=2"


def test_tool_registry_helpers_build_search_corpus_observation() -> None:
    state = {"screened_papers": [object(), object()], "evidence": [object()]}

    summary, payload = search_corpus_observation_payload(state)

    assert search_corpus_strategy({"strategy": " hybrid "}) == "hybrid"
    assert search_corpus_strategy({}) == "auto"
    assert summary == "papers=2, evidence=1"
    assert payload == {"paper_count": 2, "evidence_count": 1}


def test_tool_registry_helpers_build_atomic_search_and_rerank_requests() -> None:
    contract = QueryContract(clean_query="DPO objective", targets=["DPO"])
    existing_evidence = EvidenceBlock(
        doc_id="local-1",
        paper_id="paper-1",
        title="DPO",
        file_path="",
        page=1,
        block_type="page_text",
        snippet="DPO objective",
        metadata={"search_source": "bm25_search"},
    )
    state = {"contract": contract, "rewritten_queries": ["rewritten objective"], "evidence": [existing_evidence]}

    atomic_request = atomic_search_tool_request(
        name="hybrid_search",
        planned_input={"paper_ids": [" paper-1 ", ""], "top_k": 3, "alpha": 0.2},
        state=state,
        default_limit=12,
    )
    atomic_summary, atomic_payload = atomic_search_observation_payload(
        request=atomic_request,
        evidence=[existing_evidence],
        paper_count=1,
    )
    rerank_request, rerank_context = rerank_tool_request(
        planned_input={"top_k": "1", "focus": [" DPO "]},
        state=state,
        default_top_k=12,
    )
    rerank_summary, rerank_payload = rerank_observation_payload(
        request=rerank_request,
        payload_context=rerank_context,
        evidence=[existing_evidence],
    )

    assert atomic_request["query"] == "rewritten objective"
    assert atomic_request["paper_ids"] == ["paper-1"]
    assert atomic_request["limit"] == 3
    assert atomic_request["alpha"] == 0.2
    assert atomic_summary == "evidence=1"
    assert atomic_payload["sources"] == ["bm25_search"]
    assert rerank_request["query"] == "DPO objective"
    assert rerank_request["top_k"] == 1
    assert rerank_request["focus"] == ["DPO"]
    assert rerank_context == {"used_explicit_candidates": False, "input_candidate_count": 1}
    assert rerank_summary == "evidence=1"
    assert rerank_payload["top_doc_ids"] == ["local-1"]


def test_tool_registry_helpers_store_research_evidence_result_and_payloads() -> None:
    evidence = EvidenceBlock(
        doc_id="ev-1",
        paper_id="paper-1",
        title="Paper",
        file_path="",
        page=1,
        block_type="page_text",
        snippet="Evidence",
    )
    paper = SimpleNamespace(paper_id="paper-1")

    class _Agent:
        def _merge_evidence(self, existing: list[EvidenceBlock], new: list[EvidenceBlock]) -> list[EvidenceBlock]:
            return [*existing, *new]

        def _candidate_from_paper_id(self, paper_id: str) -> SimpleNamespace | None:
            return paper if paper_id == "paper-1" else None

    state: dict[str, object] = {"screened_papers": [], "candidate_papers": []}

    papers = store_research_evidence_result(agent=_Agent(), state=state, evidence=[evidence])
    event_payload = evidence_event_payload(state["evidence"])  # type: ignore[arg-type]
    summary, observation_payload = evidence_result_observation_payload(
        payload={"paper_id": "paper-1"},
        evidence=[evidence],
        paper_count=len(papers),
    )

    assert papers == [paper]
    assert state["screened_papers"] == [paper]
    assert state["candidate_papers"] == [paper]
    assert event_payload["count"] == 1
    assert event_payload["items"][0]["doc_id"] == "ev-1"
    assert summary == "evidence=1"
    assert observation_payload == {"paper_id": "paper-1", "evidence_count": 1, "paper_count": 1}


def test_tool_registry_helpers_store_fetch_url_evidence_result() -> None:
    evidence = EvidenceBlock(
        doc_id="web-1",
        paper_id="web-1",
        title="Web",
        file_path="https://example.com",
        page=0,
        block_type="web",
        snippet="Fetched text",
    )

    class _Agent:
        def _merge_evidence(self, existing: list[EvidenceBlock], new: list[EvidenceBlock]) -> list[EvidenceBlock]:
            return [*existing, *new]

    state: dict[str, object] = {}

    event_payload = store_fetch_url_evidence_result(agent=_Agent(), state=state, evidence=evidence)
    skipped_payload = store_fetch_url_evidence_result(agent=_Agent(), state=state, evidence=None)

    assert event_payload is not None
    assert event_payload["count"] == 1
    assert event_payload["items"][0]["doc_id"] == "web-1"
    assert state["web_evidence"] == [evidence]
    assert state["evidence"] == [evidence]
    assert skipped_payload is None


def test_tool_registry_helpers_build_remember_payload_and_persist_learning(tmp_path: Path) -> None:
    state: dict[str, object] = {}

    payload, summary = remember_tool_payload(
        data_dir=tmp_path,
        planned_input={"key": " DPO ", "content": " remember this "},
        state=state,
    )

    assert summary == "key=DPO"
    assert payload["key"] == "DPO"
    assert payload["content_chars"] == len("remember this")
    assert state["learnings"] == [{"key": "DPO", "path": payload["path"], "content": "remember this"}]
    assert "remember this" in Path(str(payload["path"])).read_text(encoding="utf-8")


def test_tool_registry_helpers_build_propose_tool_payload(tmp_path: Path) -> None:
    agent = SimpleNamespace(settings=SimpleNamespace(data_dir=tmp_path))

    payload = propose_tool_payload(
        agent,
        {
            "name": "extract_metric",
            "description": "Extract a metric.",
            "input_schema": {"type": "object", "properties": {"metric": {"type": "string"}}},
            "python_code": "async def run(args, ctx, session):\n    return {'ok': True}",
            "rationale": "Reusable metric extraction.",
        },
    )
    rejected = propose_tool_payload(agent, {"name": "../bad"})

    assert payload["status"] == "pending_review"
    assert payload["admin_approval_required"] is True
    assert Path(str(payload["path"])).exists()
    assert rejected["status"] == "rejected"
    assert "tool name" in rejected["error"]


def test_tool_registry_helpers_build_verify_claim_payload_from_state_evidence() -> None:
    state = {
        "evidence": [
            EvidenceBlock(
                doc_id="ev-1",
                paper_id="P1",
                title="Paper",
                file_path="/tmp/paper.pdf",
                page=1,
                block_type="page_text",
                snippet="DPO optimizes preference likelihood without a reward model.",
            )
        ]
    }

    payload, summary = verify_claim_tool_payload(
        planned_input={"claim": "DPO optimizes preference likelihood", "min_overlap": 2},
        state=state,
    )

    assert payload["status"] == "pass"
    assert payload["supporting_evidence_ids"] == ["ev-1"]
    assert payload["min_overlap"] == 2
    assert summary.startswith("pass:")
    store_claim_check_payload(state=state, payload=payload)
    assert state["claim_checks"] == [payload]
    assert state["tool_verifications"] == [payload]


def test_tool_registry_helpers_build_summarize_payload_from_text_and_source_fallback() -> None:
    text_payload = summarize_tool_payload(
        planned_input={"text": "DPO optimizes preferences. Other content.", "target_words": 20},
        state={},
        targets=["DPO"],
        fallback_to_summary_source=False,
    )
    fallback_payload = summarize_tool_payload(
        planned_input={"target_words": 20},
        state={"task_results": [{"answer": "Task answer about RAG."}]},
        targets=["RAG"],
        fallback_to_summary_source=True,
    )

    assert "DPO" in text_payload["summary"]
    assert text_payload["source_chars"] == len("DPO optimizes preferences. Other content.")
    assert "Task answer" in fallback_payload["summary"]
    assert fallback_payload["source_chars"] == len("Task answer about RAG.")
