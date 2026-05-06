from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services.entities.supporting_paper_selector import (
    llm_select_entity_supporting_paper,
    prune_entity_supporting_evidence,
    select_entity_supporting_paper,
)


def _paper(paper_id: str, title: str, *, score: float = 0.5, card: str = "") -> CandidatePaper:
    return CandidatePaper(
        paper_id=paper_id,
        title=title,
        score=score,
        metadata={"paper_card_text": card},
    )


def _evidence(
    doc_id: str,
    paper_id: str,
    title: str,
    snippet: str,
    *,
    definition_score: float = 0.0,
    mechanism_score: float = 0.0,
    application_score: float = 0.0,
    score: float = 0.5,
) -> EvidenceBlock:
    return EvidenceBlock(
        doc_id=doc_id,
        paper_id=paper_id,
        title=title,
        file_path=f"/tmp/{paper_id}.pdf",
        page=1,
        block_type="page_text",
        snippet=snippet,
        score=score,
        metadata={
            "definition_score": definition_score,
            "mechanism_score": mechanism_score,
            "application_score": application_score,
        },
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


def test_select_entity_supporting_paper_prefers_definition_source_without_agent_mixin() -> None:
    papers = [
        _paper("SURVEY", "Survey of RL Algorithms", score=0.4, card="A survey of RL algorithms."),
        _paper("USERLM", "UserLM-R1", score=0.8, card="Uses GRPO during reinforcement learning."),
    ]
    evidence = [
        _evidence(
            "survey-def",
            "SURVEY",
            papers[0].title,
            "Group Relative Policy Optimization (GRPO) introduces a lightweight policy optimization method.",
            definition_score=3.0,
            mechanism_score=2.0,
        ),
        _evidence(
            "userlm-use",
            "USERLM",
            papers[1].title,
            "We employ the GRPO algorithm with rule-based rewards as guiding signals.",
            application_score=1.0,
            score=0.9,
        ),
    ]

    paper, supporting = select_entity_supporting_paper(
        clients=_NoChatClients(),
        paper_doc_lookup=lambda paper_id: None,
        paper_identity_matches_targets=lambda paper, targets: False,
        contract=QueryContract(clean_query="GRPO 是什么技术？", relation="entity_definition", targets=["GRPO"]),
        papers=papers,
        evidence=evidence,
    )

    assert paper is not None
    assert paper.paper_id == "SURVEY"
    assert [item.doc_id for item in supporting] == ["survey-def"]


def test_llm_select_entity_supporting_paper_uses_selected_evidence_and_summary() -> None:
    clients = _JsonClients(
        {
            "paper_id": "ALIGNX",
            "evidence_doc_ids": ["alignx-def"],
            "relation_to_target": "direct_definition",
            "confidence": "high",
        }
    )
    paper_doc = SimpleNamespace(metadata={"generated_summary": "AlignX introduces a personalized preference benchmark."})
    papers = [_paper("ALIGNX", "AlignX"), _paper("TRANSFER", "Transfer Paper")]
    evidence = [
        _evidence("alignx-def", "ALIGNX", "AlignX", "AlignX is a large-scale dataset and benchmark."),
        _evidence("transfer-use", "TRANSFER", "Transfer Paper", "This work evaluates on AlignX."),
    ]

    paper, supporting = llm_select_entity_supporting_paper(
        clients=clients,
        paper_doc_lookup=lambda paper_id: paper_doc,
        contract=QueryContract(clean_query="AlignX 是什么？", relation="entity_definition", targets=["AlignX"]),
        papers=papers,
        matching_evidence=evidence,
    )

    assert paper is not None
    assert paper.paper_id == "ALIGNX"
    assert [item.doc_id for item in supporting] == ["alignx-def"]
    assert clients.calls
    assert "AlignX introduces a personalized preference benchmark." in clients.calls[0]["human_prompt"]


def test_prune_entity_supporting_evidence_drops_noisy_and_deduplicates_pages() -> None:
    evidence = [
        _evidence("formula", "P1", "Paper", "𝜋𝜃 𝑜𝑡 𝑞 𝐴ˆ β ϵ μ | | 12 34"),
        _evidence("definition", "P1", "Paper", "GRPO is a policy optimization method.", definition_score=2.0),
        _evidence("duplicate", "P1", "Paper", "GRPO uses group-relative rewards.", mechanism_score=1.0),
    ]

    pruned = prune_entity_supporting_evidence(evidence)

    assert [item.doc_id for item in pruned] == ["definition"]
