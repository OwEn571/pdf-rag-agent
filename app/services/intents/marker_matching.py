from __future__ import annotations

import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any


MarkerProfile = tuple[str, ...]


def normalized_query_text(query: str) -> tuple[str, str]:
    normalized = " ".join(str(query or "").lower().split())
    return normalized, normalized.replace(" ", "")


def query_matches_any(lowered: str, compact: str, markers: Iterable[str]) -> bool:
    return any(marker in lowered or marker in compact for marker in markers)


@lru_cache(maxsize=1)
def marker_profiles() -> dict[str, Any]:
    path = Path(__file__).with_name("intent_marker_profiles.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("intent marker profiles must be a JSON object")
    return payload


def marker_profile(section: str, key: str) -> MarkerProfile:
    value = marker_profiles().get(section, {}).get(key, ())
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def marker_profile_map(section: str) -> dict[str, MarkerProfile]:
    value = marker_profiles().get(section, {})
    if not isinstance(value, dict):
        return {}
    return {
        str(key): tuple(str(item) for item in markers)
        for key, markers in value.items()
        if isinstance(markers, list)
    }
