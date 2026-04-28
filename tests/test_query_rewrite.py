from __future__ import annotations

from app.services.query_rewrite import rewrite_query


def test_rewrite_query_adds_target_and_multi_query_variants() -> None:
    result = rewrite_query(query="核心公式", targets=["DPO"], mode="multi_query", max_queries=4)

    assert result.query == "核心公式"
    assert result.targets == ["DPO"]
    assert result.queries[0] == "核心公式"
    assert "DPO 核心公式" in result.queries
    assert any("evidence formula" in item for item in result.queries)


def test_rewrite_query_supports_step_back_mode_and_limit() -> None:
    result = rewrite_query(query="PPO clipping objective", mode="step_back", max_queries=2)

    assert result.mode == "step_back"
    assert len(result.queries) == 2
    assert result.queries[1].startswith("background concepts")


def test_rewrite_query_handles_invalid_limit_and_mode() -> None:
    result = rewrite_query(query="  AlignX   results  ", mode="unknown", max_queries="bad")  # type: ignore[arg-type]

    assert result.mode == "multi_query"
    assert result.queries[0] == "AlignX results"
    assert len(result.queries) <= 3
