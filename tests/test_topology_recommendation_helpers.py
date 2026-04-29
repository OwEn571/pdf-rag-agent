from __future__ import annotations

from app.services.topology_recommendation_helpers import (
    fallback_topology_recommendation,
    is_unusable_topology_recommendation_text,
    topology_discovery_claim,
    topology_recommendation_from_payload,
    topology_recommendation_human_prompt,
    topology_recommendation_claim,
    topology_recommendation_system_prompt,
)
from app.domain.models import CandidatePaper, EvidenceBlock


def test_unusable_topology_recommendation_text_detects_empty_and_negative_answers() -> None:
    assert is_unusable_topology_recommendation_text("")
    assert is_unusable_topology_recommendation_text("The evidence does not contain a specific comparison.")
    assert is_unusable_topology_recommendation_text("无法确定哪一种 topology 最好")
    assert not is_unusable_topology_recommendation_text("DAG is better when dependencies must be explicit.")


def test_fallback_topology_recommendation_uses_terms_or_default() -> None:
    recommendation = fallback_topology_recommendation(["chain", "DAG"])
    default_recommendation = fallback_topology_recommendation([])

    assert recommendation["engineering_best"] == "DAG"
    assert "chain / DAG" in recommendation["summary"]
    assert "chain / tree / mesh / DAG" in default_recommendation["summary"]


def test_topology_recommendation_prompt_and_payload_helpers() -> None:
    evidence = [
        EvidenceBlock(
            doc_id="ev-1",
            paper_id="p1",
            title="Topology",
            file_path="/tmp/topology.pdf",
            page=1,
            block_type="page_text",
            snippet="DAG is useful when dependencies are explicit.",
        )
    ]
    prompt = topology_recommendation_system_prompt()
    human_prompt = topology_recommendation_human_prompt(topology_terms=["DAG"], evidence=evidence)
    recommendation = topology_recommendation_from_payload(
        {"summary": "Use DAG when dependencies matter.", "engineering_best": "DAG", "rationale": "explicit dependencies"},
        topology_terms=["DAG"],
    )
    fallback = topology_recommendation_from_payload(
        {"summary": "does not contain specific comparison"},
        topology_terms=["chain"],
    )

    assert "topology 证据分析器" in prompt
    assert '"topology_terms": ["DAG"]' in human_prompt
    assert "dependencies are explicit" in human_prompt
    assert recommendation["engineering_best"] == "DAG"
    assert fallback["engineering_best"] == "DAG"
    assert "chain" in fallback["summary"]


def test_topology_discovery_claim_collects_relevant_papers_and_evidence_ids() -> None:
    papers = [
        CandidatePaper(paper_id="p1", title="Topology Paper", year="2025", doc_ids=["paper-doc"]),
        CandidatePaper(paper_id="p2", title="No Evidence", year="2024"),
    ]

    claim = topology_discovery_claim(
        papers=papers,
        topology_terms=["dag", "chain"],
        evidence_ids_for_paper=lambda paper_id: ["ev-1"] if paper_id == "p1" else [],
    )

    assert claim is not None
    assert claim.value == "Topology Paper"
    assert claim.evidence_ids == ["ev-1"]
    assert claim.paper_ids == ["p1"]
    assert claim.structured_data["topology_terms"] == ["dag", "chain"]


def test_topology_recommendation_claim_uses_top_evidence() -> None:
    evidence = [
        EvidenceBlock(
            doc_id=f"ev-{idx}",
            paper_id=f"p{idx % 2}",
            title="Topology",
            file_path="/tmp/topology.pdf",
            page=idx,
            block_type="page_text",
            snippet="DAG topology",
        )
        for idx in range(4)
    ]

    claim = topology_recommendation_claim(
        recommendation={"summary": "Use DAG", "engineering_best": "DAG", "rationale": "explicit dependencies"},
        topology_terms=["dag"],
        evidence=evidence,
    )

    assert claim.value == "Use DAG"
    assert claim.evidence_ids == ["ev-0", "ev-1", "ev-2"]
    assert claim.paper_ids == ["p0", "p1"]
    assert claim.structured_data["engineering_best"] == "DAG"
