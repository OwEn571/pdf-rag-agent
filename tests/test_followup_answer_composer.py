from __future__ import annotations

from app.domain.models import Claim
from app.services.answers.followup import compose_followup_research_answer, followup_public_reason


def test_followup_public_reason_replaces_english_heavy_reason() -> None:
    reason = followup_public_reason(
        {
            "relationship_strength": "direct",
            "reason": "The candidate explicitly evaluates and extends the seed benchmark in a later experiment.",
        }
    )

    assert reason == "结构化证据显示它与种子论文/数据集存在直接使用、评测或扩展关系。"


def test_compose_followup_research_answer_groups_candidates() -> None:
    claim = Claim(
        claim_type="followup_research",
        entity="AlignX",
        structured_data={
            "seed_papers": [{"title": "AlignX", "year": "2025"}],
            "followup_titles": [
                {
                    "title": "Direct Followup",
                    "year": "2026",
                    "relation_type": "评测扩展",
                    "relationship_strength": "direct",
                    "reason": "使用同一 benchmark 继续评测。",
                },
                {
                    "title": "Related Work",
                    "year": "2026",
                    "relation_type": "同主题",
                    "relationship_strength": "strong_related",
                    "reason": "主题和任务设置接近。",
                },
            ],
        },
    )

    answer = compose_followup_research_answer(claims=[claim])

    assert "种子论文是《AlignX》（2025）" in answer
    assert "## 直接后续/使用证据" in answer
    assert "《Direct Followup》（2026）" in answer
    assert "## 强相关延续候选" in answer


def test_compose_followup_research_answer_selected_candidate_verdict() -> None:
    claim = Claim(
        claim_type="followup_research",
        entity="AlignX",
        structured_data={
            "selected_candidate_title": "Candidate Paper",
            "seed_papers": [{"title": "AlignX"}],
            "followup_titles": [
                {
                    "title": "Candidate Paper",
                    "relationship_strength": "strong_related",
                    "relation_type": "相关延续候选",
                    "classification": "strong_related",
                    "evidence_ids": ["ev-1", "ev-2"],
                    "reason": "主题和任务设置接近。",
                    "strict_followup": False,
                }
            ],
        },
    )

    answer = compose_followup_research_answer(claims=[claim])

    assert "## 判断" in answer
    assert "更适合写成强相关延续候选" in answer
    assert "证据范围" in answer
    assert "严格后续：否/证据不足" in answer
