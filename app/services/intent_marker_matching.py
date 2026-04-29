from __future__ import annotations

from collections.abc import Iterable


MarkerProfile = tuple[str, ...]


def normalized_query_text(query: str) -> tuple[str, str]:
    normalized = " ".join(str(query or "").lower().split())
    return normalized, normalized.replace(" ", "")


def query_matches_any(lowered: str, compact: str, markers: Iterable[str]) -> bool:
    return any(marker in lowered or marker in compact for marker in markers)
