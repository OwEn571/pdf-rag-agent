from __future__ import annotations

from app.services.claims.origin_selection import ORIGIN_INTRO_MARKER_RE


def test_solver_pipeline_markers_cover_origin_intro_profile() -> None:
    assert ORIGIN_INTRO_MARKER_RE.search("we introduce AlignX")
    assert ORIGIN_INTRO_MARKER_RE.search("this paper proposed DPO")
