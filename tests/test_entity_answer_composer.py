from __future__ import annotations

from types import SimpleNamespace

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract
from app.services.answers.entity import (
    compose_entity_answer_markdown,
    compose_entity_description,
    entity_clean_lines,
)


def _evidence(
    doc_id: str,
    snippet: str,
    *,
    definition_score: float = 0.0,
    mechanism_score: float = 0.0,
    application_score: float = 0.0,
) -> EvidenceBlock:
    return EvidenceBlock(
        doc_id=doc_id,
        paper_id="paper-1",
        title="DeepSeekMath",
        file_path="/tmp/deepseekmath.pdf",
        page=13,
        block_type="page_text",
        snippet=snippet,
        score=0.8,
        metadata={
            "definition_score": definition_score,
            "mechanism_score": mechanism_score,
            "application_score": application_score,
        },
    )


class _TextClients:
    def __init__(self, text: str = "") -> None:
        self.text = text

    def invoke_text(self, *, system_prompt: str, human_prompt: str, fallback: str) -> str:
        return self.text


def test_entity_clean_lines_skips_formula_noise_without_agent_mixin() -> None:
    cleaned = entity_clean_lines(
        [
            "GRPO foregoes the value model and uses group scores to estimate the baseline.",
            "𝜋𝜃 (𝑜𝑡 |𝑞, 𝑜<𝑡) 𝜋𝜃𝑜𝑙𝑑 (𝑜𝑡 |𝑞, 𝑜<𝑡) 𝐴𝑡 . (15)",
        ],
        limit=3,
    )

    assert cleaned == ["GRPO foregoes the value model and uses group scores to estimate the baseline."]


def test_compose_entity_answer_markdown_uses_claim_support_lines_and_citation() -> None:
    claim = Claim(
        claim_type="entity_definition",
        entity="GRPO",
        value="优化算法/训练方法",
        structured_data={
            "paper_title": "DeepSeekMath",
            "mechanism_lines": [
                "GRPO uses relative rewards within a group to compute advantages and removes the need for a value critic."
            ],
        },
        evidence_ids=["ev-1"],
        paper_ids=["paper-1"],
    )

    answer = compose_entity_answer_markdown(
        contract=QueryContract(
            clean_query="GRPO 的机制是什么？",
            relation="entity_definition",
            targets=["GRPO"],
            requested_fields=["mechanism", "workflow", "reward_signal"],
            continuation_mode="followup",
        ),
        claims=[claim],
        evidence=[
            _evidence(
                "ev-1",
                "GRPO samples a group of outputs, computes rewards, normalizes the group scores, and updates the policy.",
                mechanism_score=2.0,
            )
        ],
        citations=[SimpleNamespace(title="DeepSeekMath", page=13)],
    )

    assert "### GRPO：机制与流程" in answer
    assert "核心机制" in answer
    assert "典型流程" in answer
    assert "主要依据《DeepSeekMath》第 13 页" in answer


def test_compose_entity_description_falls_back_to_paper_summary() -> None:
    paper = CandidatePaper(paper_id="paper-1", title="DeepSeekMath")
    paper_doc = SimpleNamespace(metadata={"generated_summary": "GRPO is a reinforcement learning method for reasoning."})

    description = compose_entity_description(
        clients=_TextClients(""),
        paper_doc_lookup=lambda paper_id: paper_doc,
        contract=QueryContract(clean_query="GRPO 是什么？", relation="entity_definition", targets=["GRPO"]),
        target="GRPO",
        label="优化算法/训练方法",
        paper=paper,
        evidence=[_evidence("ev-1", "GRPO is a policy optimization method.")],
    )

    assert "GRPO 可以定位为 `优化算法/训练方法`" in description
    assert "DeepSeekMath" in description
    assert "reinforcement learning method" in description
