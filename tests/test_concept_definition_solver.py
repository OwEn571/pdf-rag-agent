from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services.claims.concept_definition_solver import solve_concept_definition_claims


def _evidence(doc_id: str, snippet: str, *, score: float = 0.8) -> EvidenceBlock:
    return EvidenceBlock(
        doc_id=doc_id,
        paper_id="paper-1",
        title="Direct Preference Optimization",
        file_path="/tmp/dpo.pdf",
        page=2,
        block_type="page_text",
        snippet=snippet,
        score=score,
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


def test_concept_definition_solver_builds_fallback_claim_without_agent_mixin() -> None:
    claims = solve_concept_definition_claims(
        clients=_NoChatClients(),
        paper_doc_lookup=lambda paper_id: None,
        contract=QueryContract(clean_query="DPO 是什么？", targets=["DPO"], relation="concept_definition"),
        papers=[CandidatePaper(paper_id="paper-1", title="Direct Preference Optimization")],
        evidence=[
            _evidence(
                "ev-1",
                "Direct Preference Optimization (DPO) is a policy optimization method for aligning language models.",
            ),
            _evidence("ev-2", "DPO optimizes policies directly without fitting a separate reward model.", score=0.7),
        ],
    )

    assert len(claims) == 1
    claim = claims[0]
    assert claim.claim_type == "concept_definition"
    assert claim.entity == "DPO"
    assert claim.structured_data["expansion"] == "Direct Preference Optimization"
    assert claim.structured_data["category"] == "强化学习算法"
    assert claim.evidence_ids == ["ev-1", "ev-2"]


def test_concept_definition_solver_uses_llm_payload_and_selected_evidence() -> None:
    clients = _JsonClients(
        {
            "expansion": "Retrieval-Augmented Generation",
            "category": "framework",
            "definition": "RAG combines retrieval with generation to answer with grounded external context.",
            "supporting_doc_ids": ["ev-2"],
            "confidence": 0.91,
        }
    )
    paper_doc = SimpleNamespace(metadata={"generated_summary": "A paper about retrieval and generation."})

    claims = solve_concept_definition_claims(
        clients=clients,
        paper_doc_lookup=lambda paper_id: paper_doc,
        contract=QueryContract(clean_query="RAG 是什么？", targets=["RAG"], relation="concept_definition"),
        papers=[CandidatePaper(paper_id="paper-1", title="Retrieval-Augmented Generation")],
        evidence=[
            _evidence("ev-1", "RAG is mentioned in the introduction.", score=0.5),
            _evidence("ev-2", "Retrieval-augmented generation uses retrieved passages before generation.", score=0.9),
        ],
    )

    assert len(claims) == 1
    claim = claims[0]
    assert claim.value == "RAG combines retrieval with generation to answer with grounded external context."
    assert claim.structured_data["category"] == "框架/系统"
    assert claim.evidence_ids == ["ev-2"]
    assert claim.paper_ids == ["paper-1"]
    assert claim.confidence == 0.91
    assert clients.calls
    assert "A paper about retrieval and generation." in clients.calls[0]["human_prompt"]
