from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services.entities.type_inference import infer_entity_type


def _evidence(snippet: str) -> EvidenceBlock:
    return EvidenceBlock(
        doc_id="ev-1",
        paper_id="paper-1",
        title="DeepSeekMath",
        file_path="/tmp/deepseekmath.pdf",
        page=13,
        block_type="page_text",
        snippet=snippet,
    )


class _NoChatClients:
    chat = None


class _JsonClients:
    chat = object()

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, str]] = []

    def invoke_json(self, *, system_prompt: str, human_prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"system_prompt": system_prompt, "human_prompt": human_prompt})
        return self.payload


def test_infer_entity_type_prefers_algorithm_signals_without_agent_mixin() -> None:
    label = infer_entity_type(
        clients=_NoChatClients(),
        paper_doc_lookup=lambda paper_id: None,
        contract=QueryContract(clean_query="GRPO 是什么技术？", relation="entity_definition", targets=["GRPO"]),
        papers=[
            CandidatePaper(
                paper_id="paper-1",
                title="DeepSeekMath",
                metadata={"paper_card_text": "GRPO is evaluated on math benchmarks."},
            )
        ],
        evidence=[
            _evidence(
                "GRPO is a variant of PPO that uses relative rewards within a group to compute advantages and removes a value critic."
            )
        ],
    )

    assert label == "优化算法/训练方法"


def test_infer_entity_type_uses_llm_payload_and_paper_summary() -> None:
    clients = _JsonClients({"entity_type": "benchmark", "confidence": 0.9, "rationale": "Dataset signals."})
    paper_doc = SimpleNamespace(metadata={"generated_summary": "AlignX is a personalized preference benchmark."})

    label = infer_entity_type(
        clients=clients,
        paper_doc_lookup=lambda paper_id: paper_doc,
        contract=QueryContract(clean_query="AlignX 是什么？", relation="entity_definition", targets=["AlignX"]),
        papers=[CandidatePaper(paper_id="paper-1", title="AlignX")],
        evidence=[_evidence("AlignX is a large-scale dataset and benchmark for personalized preference alignment.")],
    )

    assert label == "数据集/benchmark"
    assert clients.calls
    assert "AlignX is a personalized preference benchmark." in clients.calls[0]["human_prompt"]
