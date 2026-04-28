from __future__ import annotations

from app.services.figure_intents import (
    extract_figure_benchmarks,
    figure_signal_score,
    has_explicit_figure_reference,
)


def test_figure_intents_detect_reference_and_benchmark_signals() -> None:
    text = "Figure 1 | Benchmark performance on AIME, Codeforces, GPQA, MATH-500, MMLU and SWE-bench."

    assert has_explicit_figure_reference(text)
    assert figure_signal_score(text) >= 6
    assert extract_figure_benchmarks(text) == [
        "AIME 2024",
        "Codeforces",
        "GPQA Diamond",
        "MATH-500",
        "MMLU",
        "SWE-bench Verified",
    ]
