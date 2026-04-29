from __future__ import annotations

import json

from app.domain.models import QueryContract, SessionContext, VerificationReport
from app.services.compound_task_helpers import (
    compose_compound_comparison_answer,
    comparison_results_with_memory,
    compound_contracts_from_decomposer_payload,
    compound_subtask_contract_from_payload,
    compound_subtask_relation_from_slots,
    compound_research_progress_markdown,
    compound_section_heading,
    compound_task_label,
    compound_task_result_from_task_payload,
    demote_markdown_headings,
    format_compound_section,
    llm_decompose_compound_query,
    merge_redundant_field_subtasks,
)


def test_compound_task_helpers_format_labels_and_sections() -> None:
    formula = QueryContract(clean_query="DPO 公式", relation="formula_lookup", targets=["DPO"])
    comparison = QueryContract(clean_query="比较", relation="comparison_synthesis", targets=["DPO", "PPO"])

    assert compound_task_label(formula) == "查询 DPO 公式"
    assert compound_task_label(comparison) == "比较 DPO 和 PPO"
    assert compound_section_heading(contract=formula, index=2) == "## 2. 查询 DPO 公式"
    assert "好的，我现在去查询 **DPO** 的公式" in compound_research_progress_markdown(contract=formula, index=1)
    assert demote_markdown_headings("# Title\n## Child") == "## Title\n### Child"
    assert format_compound_section(contract=formula, answer="# Answer", index=1).startswith("## 1. 查询 DPO 公式\n\n## Answer")


def test_compound_task_result_from_payload_validates_contract_and_verification() -> None:
    fallback = QueryContract(clean_query="fallback", relation="paper_summary_results")
    payload = {
        "contract": {"clean_query": "DPO", "relation": "formula_lookup", "targets": ["DPO"]},
        "verification": {"status": "clarify", "recommended_action": "ask_human"},
        "answer": "answer",
        "citations": ["c"],
    }

    result = compound_task_result_from_task_payload(payload, fallback_contract=fallback)

    assert isinstance(result["contract"], QueryContract)
    assert result["contract"].relation == "formula_lookup"
    assert isinstance(result["verification"], VerificationReport)
    assert result["verification"].status == "clarify"
    assert result["answer"] == "answer"
    assert result["citations"] == ["c"]


def test_compound_task_result_from_payload_uses_fallbacks_for_bad_values() -> None:
    fallback = QueryContract(clean_query="fallback", relation="paper_summary_results")

    result = compound_task_result_from_task_payload({"contract": "bad", "verification": "bad"}, fallback_contract=fallback)

    assert result["contract"] == fallback
    assert result["verification"].status == "pass"
    assert result["verification"].recommended_action == "task_subagent"


def test_compound_subtask_relation_from_slots_maps_common_slots() -> None:
    assert (
        compound_subtask_relation_from_slots(
            answer_slots=["formula"],
            requested_fields=[],
            targets=["DPO"],
        )
        == "formula_lookup"
    )
    assert (
        compound_subtask_relation_from_slots(
            answer_slots=[],
            requested_fields=["paper_title", "year"],
            targets=["Transformer"],
        )
        == "origin_lookup"
    )


def test_compound_subtask_contract_from_payload_normalizes_formula_defaults() -> None:
    contract = compound_subtask_contract_from_payload(
        {
            "clean_query": " DPO 的公式 ",
            "targets": ["DPO"],
            "answer_slots": "formula",
            "requested_fields": [],
            "required_modalities": [],
            "answer_shape": "unknown",
            "notes": ["from_llm"],
        },
        fallback_query="fallback",
        index=0,
    )

    assert contract is not None
    assert contract.relation == "formula_lookup"
    assert contract.interaction_mode == "research"
    assert contract.requested_fields == ["formula", "variable_explanation"]
    assert contract.required_modalities == ["page_text", "table"]
    assert contract.precision_requirement == "exact"
    assert "compound_subtask" in contract.notes
    assert "answer_slot=formula" in contract.notes


def test_merge_redundant_field_subtasks_merges_same_target_fields() -> None:
    first = QueryContract(
        clean_query="POPI 的核心结论是什么？",
        relation="paper_summary_results",
        targets=["POPI"],
        requested_fields=["core_conclusion"],
        required_modalities=["page_text"],
        precision_requirement="high",
        notes=["first"],
    )
    second = QueryContract(
        clean_query="POPI 的实验结果如何？",
        relation="paper_summary_results",
        targets=["POPI"],
        requested_fields=["experiment_results"],
        required_modalities=["table"],
        precision_requirement="exact",
        notes=["second"],
    )

    merged = merge_redundant_field_subtasks([first, second])

    assert len(merged) == 1
    assert merged[0].requested_fields == ["core_conclusion", "experiment_results"]
    assert merged[0].required_modalities == ["page_text", "table"]
    assert merged[0].precision_requirement == "exact"
    assert "merged_same_target_fields" in merged[0].notes


def test_compound_contracts_from_decomposer_payload_requires_two_valid_subtasks() -> None:
    contracts = compound_contracts_from_decomposer_payload(
        payload={
            "is_compound": True,
            "subtasks": [
                {"clean_query": "DPO 公式", "targets": ["DPO"], "answer_slots": ["formula"]},
                {"clean_query": "PPO 公式", "targets": ["PPO"], "answer_slots": ["formula"]},
            ],
        },
        fallback_query="DPO 和 PPO 公式",
    )

    assert [contract.targets for contract in contracts] == [["DPO"], ["PPO"]]
    assert [contract.relation for contract in contracts] == ["formula_lookup", "formula_lookup"]


def test_llm_decompose_compound_query_uses_message_history_branch() -> None:
    class Clients:
        chat = object()

        def __init__(self) -> None:
            self.messages: list[dict[str, str]] = []

        def invoke_json_messages(self, *, system_prompt: str, messages: list[dict[str, str]], fallback: object) -> object:
            assert "任务分解器" in system_prompt
            self.messages = messages
            payload = json.loads(messages[-1]["content"])
            assert payload["conversation_context"] == {"summary": "previous"}
            return {
                "is_compound": True,
                "subtasks": [
                    {"clean_query": "DPO 公式", "targets": ["DPO"], "answer_slots": ["formula"]},
                    {"clean_query": "PPO 公式", "targets": ["PPO"], "answer_slots": ["formula"]},
                ],
            }

    clients = Clients()
    contracts = llm_decompose_compound_query(
        clean_query="DPO 和 PPO 的公式分别是什么？",
        session=SessionContext(session_id="compound-helper"),
        clients=clients,
        available_tools=[{"name": "search_corpus"}],
        conversation_context=lambda *_args, **_kwargs: {"summary": "previous"},
        history_messages=lambda _session: [{"role": "assistant", "content": "history"}],
    )

    assert len(contracts) == 2
    assert clients.messages[0] == {"role": "assistant", "content": "history"}


def test_comparison_results_with_memory_restores_missing_target_binding() -> None:
    session = SessionContext(
        session_id="comparison-memory",
        working_memory={
            "target_bindings": {
                "ppo": {
                    "target": "PPO",
                    "relation": "formula_lookup",
                    "clean_query": "PPO 的公式是什么？",
                    "requested_fields": ["formula"],
                    "required_modalities": ["page_text", "table"],
                    "answer_preview": "PPO uses a clipped objective.",
                }
            }
        },
    )
    dpo_contract = QueryContract(clean_query="DPO 公式", relation="formula_lookup", targets=["DPO"])
    comparison_contract = QueryContract(
        clean_query="比较 DPO 和 PPO",
        relation="comparison_synthesis",
        targets=["DPO", "PPO"],
    )

    results = comparison_results_with_memory(
        subtask_results=[{"contract": dpo_contract, "answer": "DPO answer"}],
        session=session,
        comparison_contract=comparison_contract,
    )

    assert len(results) == 2
    restored = results[1]
    assert restored["contract"].targets == ["PPO"]
    assert restored["answer"] == "PPO uses a clipped objective."
    assert restored["verification"].recommended_action == "memory_comparison_context"


def test_compose_compound_comparison_answer_fallback_uses_augmented_results() -> None:
    class NoChatClients:
        chat = None

    session = SessionContext(
        session_id="comparison-answer",
        working_memory={
            "target_bindings": {
                "ppo": {
                    "target": "PPO",
                    "relation": "formula_lookup",
                    "answer_preview": "PPO answer from memory.",
                }
            }
        },
    )
    dpo_contract = QueryContract(clean_query="DPO", relation="formula_lookup", targets=["DPO"])
    comparison_contract = QueryContract(clean_query="比较", relation="comparison_synthesis", targets=["DPO", "PPO"])

    answer = compose_compound_comparison_answer(
        query="比较 DPO 和 PPO",
        subtask_results=[{"contract": dpo_contract, "answer": "DPO answer."}],
        session=session,
        comparison_contract=comparison_contract,
        clients=NoChatClients(),
        clean_text=lambda text: text,
    )

    assert "DPO answer" in answer
    assert "PPO answer from memory" in answer
