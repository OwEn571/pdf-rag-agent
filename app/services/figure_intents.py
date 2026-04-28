from __future__ import annotations


FIGURE_SIGNAL_TOKENS = [
    "figure 1",
    "benchmark performance",
    "aime",
    "codeforces",
    "gpqa",
    "math-500",
    "mmlu",
    "swe-bench",
]

EXPLICIT_FIGURE_REFERENCE_TOKENS = ["figure 1", "fig. 1", "fig 1", "figure1", "图1"]

FIGURE_BENCHMARK_ALIASES = [
    ("AIME 2024", ["aime 2024", "aime"]),
    ("Codeforces", ["codeforces"]),
    ("GPQA Diamond", ["gpqa diamond", "gpqa"]),
    ("MATH-500", ["math-500"]),
    ("MMLU", ["mmlu"]),
    ("SWE-bench Verified", ["swe-bench verified", "swe-bench"]),
    ("LiveCodeBench", ["livecodebench"]),
]


def figure_signal_score(text: str) -> int:
    haystack = str(text or "").lower()
    return sum(1 for token in FIGURE_SIGNAL_TOKENS if token in haystack)


def has_explicit_figure_reference(text: str) -> bool:
    haystack = str(text or "").lower()
    return any(token in haystack for token in EXPLICIT_FIGURE_REFERENCE_TOKENS)


def extract_figure_benchmarks(text: str) -> list[str]:
    haystack = str(text or "").lower()
    found: list[str] = []
    for label, tokens in FIGURE_BENCHMARK_ALIASES:
        if any(token in haystack for token in tokens):
            found.append(label)
    return found
