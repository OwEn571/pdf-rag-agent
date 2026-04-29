from __future__ import annotations

from app.services.agent_mixins.solver_pipeline import SOLVER_PIPELINE_MARKERS
from app.services.intent_marker_matching import query_matches_any


def test_solver_pipeline_markers_cover_origin_intro_profile() -> None:
    assert query_matches_any("we introduce AlignX", "", SOLVER_PIPELINE_MARKERS["origin_intro"])
    assert query_matches_any("this paper proposed DPO", "", SOLVER_PIPELINE_MARKERS["origin_intro"])
