from __future__ import annotations

from pathlib import Path

from app.domain.models import EvidenceBlock, QueryContract, SessionContext
from app.services.tool_registry_helpers import (
    coerce_int,
    conversation_intent_summary,
    evidence_blocks_from_state,
    fetch_url_evidence,
    fetch_url_payload,
    focus_values,
    format_fetched_urls_answer,
    format_summaries_answer,
    format_task_results_answer,
    normalize_todo_items,
    remember_tool_payload,
    research_intent_summary,
    store_session_todos,
    summary_source_from_state,
    summarize_tool_payload,
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
    assert conversation_intent_summary(contract) == {
        "kind": "memory_op",
        "answer_slots": ["previous_rationale"],
        "requested_fields": ["previous_rationale"],
        "targets": ["DPO"],
    }
    summary, payload = research_intent_summary(contract)
    assert summary == "previous_rationale"
    assert payload["required_modalities"] == ["page_text"]


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
    evidence = fetch_url_evidence(result)

    assert payload["ok"] is True
    assert payload["status_code"] == 200
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
