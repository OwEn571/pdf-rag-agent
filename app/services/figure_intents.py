from __future__ import annotations

from app.services.intent_marker_matching import MarkerProfile, query_matches_any


FIGURE_SIGNAL_TOKENS: MarkerProfile = (
    "figure 1",
    "benchmark performance",
    "aime",
    "codeforces",
    "gpqa",
    "math-500",
    "mmlu",
    "swe-bench",
)

EXPLICIT_FIGURE_REFERENCE_TOKENS: MarkerProfile = ("figure 1", "fig. 1", "fig 1", "figure1", "图1")

FIGURE_BENCHMARK_ALIASES: tuple[tuple[str, MarkerProfile], ...] = (
    ("AIME 2024", ("aime 2024", "aime")),
    ("Codeforces", ("codeforces",)),
    ("GPQA Diamond", ("gpqa diamond", "gpqa")),
    ("MATH-500", ("math-500",)),
    ("MMLU", ("mmlu",)),
    ("SWE-bench Verified", ("swe-bench verified", "swe-bench")),
    ("LiveCodeBench", ("livecodebench",)),
)

FIGURE_INTENT_MARKERS: dict[str, MarkerProfile] = {
    "signal": FIGURE_SIGNAL_TOKENS,
    "explicit_reference": EXPLICIT_FIGURE_REFERENCE_TOKENS,
}


def figure_signal_score(text: str) -> int:
    haystack = str(text or "").lower()
    return sum(1 for token in FIGURE_INTENT_MARKERS["signal"] if token in haystack)


def has_explicit_figure_reference(text: str) -> bool:
    haystack = str(text or "").lower()
    return query_matches_any(haystack, "", FIGURE_INTENT_MARKERS["explicit_reference"])


def extract_figure_benchmarks(text: str) -> list[str]:
    haystack = str(text or "").lower()
    found: list[str] = []
    for label, tokens in FIGURE_BENCHMARK_ALIASES:
        if query_matches_any(haystack, "", tokens):
            found.append(label)
    return found
