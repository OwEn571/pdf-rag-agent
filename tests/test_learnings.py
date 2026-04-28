from __future__ import annotations

from app.services.learnings import load_learnings, remember_learning


def test_remember_learning_appends_and_loads_recent_content(tmp_path) -> None:
    first = remember_learning(data_dir=tmp_path, key="Citation Strategy", content="Prefer verified citation counts.")
    second = remember_learning(data_dir=tmp_path, key="Citation Strategy", content="Mention snapshot timing.")

    text = load_learnings(data_dir=tmp_path, max_chars=2000)

    assert first == second
    assert first.name == "citation-strategy.md"
    assert "Prefer verified citation counts." in text
    assert "Mention snapshot timing." in text
