from __future__ import annotations

from types import SimpleNamespace

from app.core.config import Settings
from app.services.tools.dynamic_context import load_agent_dynamic_tool_manifests


def _settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="",
        data_dir=tmp_path / "data",
        paper_store_path=tmp_path / "data" / "papers.jsonl",
        block_store_path=tmp_path / "data" / "blocks.jsonl",
        ingestion_state_path=tmp_path / "data" / "state.json",
        session_store_path=tmp_path / "data" / "sessions.sqlite3",
        eval_cases_path=tmp_path / "evals" / "cases.yaml",
    )


def test_load_agent_dynamic_tool_manifests_is_disabled_by_default(tmp_path, monkeypatch) -> None:
    import app.services.tools.dynamic_context as dynamic_tool_context

    def fail_if_called(**_: object) -> list[dict[str, object]]:
        raise AssertionError("runtime manifests should not load when dynamic tools are disabled")

    monkeypatch.setattr(dynamic_tool_context, "load_runtime_tool_manifests", fail_if_called)

    assert load_agent_dynamic_tool_manifests(
        settings=_settings(tmp_path),
        agent_settings=SimpleNamespace(dynamic_tools_enabled=False),
    ) == []


def test_load_agent_dynamic_tool_manifests_returns_empty_on_invalid_runtime_manifest(tmp_path, monkeypatch) -> None:
    import app.services.tools.dynamic_context as dynamic_tool_context

    def raise_bad_manifest(**_: object) -> list[dict[str, object]]:
        raise ValueError("bad manifest")

    monkeypatch.setattr(dynamic_tool_context, "load_runtime_tool_manifests", raise_bad_manifest)

    assert load_agent_dynamic_tool_manifests(
        settings=_settings(tmp_path),
        agent_settings=SimpleNamespace(dynamic_tools_enabled=True),
    ) == []


def test_load_agent_dynamic_tool_manifests_passes_deployment_scope(tmp_path, monkeypatch) -> None:
    import app.services.tools.dynamic_context as dynamic_tool_context

    captured: dict[str, object] = {}

    def fake_load_runtime_manifests(**kwargs: object) -> list[dict[str, object]]:
        captured.update(kwargs)
        return [{"name": "scoped_tool"}]

    monkeypatch.setattr(dynamic_tool_context, "load_runtime_tool_manifests", fake_load_runtime_manifests)

    manifests = load_agent_dynamic_tool_manifests(
        settings=_settings(tmp_path),
        agent_settings=SimpleNamespace(dynamic_tools_enabled=True, dynamic_tool_deployment_id="prod-east"),
    )

    assert manifests == [{"name": "scoped_tool"}]
    assert captured["deployment_id"] == "prod-east"
