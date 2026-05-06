from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app
from app.services.tools.proposals import propose_tool


def _settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="",
        admin_api_key="secret",
        data_dir=tmp_path / "data",
        paper_store_path=tmp_path / "data" / "papers.jsonl",
        block_store_path=tmp_path / "data" / "blocks.jsonl",
        ingestion_state_path=tmp_path / "data" / "state.json",
        session_store_path=tmp_path / "data" / "sessions.sqlite3",
        eval_cases_path=tmp_path / "evals" / "cases.yaml",
    )


def test_dynamic_tool_admin_api_lists_sandboxes_and_approves_runtime(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.ensure_runtime_dirs()
    proposal = propose_tool(
        data_dir=settings.data_dir,
        name="double_metric",
        description="Double a numeric metric.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
        },
        python_code="async def run(args, ctx, session):\n    return {'value': args['value'] * 2}",
        rationale="Admin API smoke test.",
    )
    proposal_payload = proposal.payload()
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        client = TestClient(app)
        headers = {"X-API-Key": "secret"}

        unauthorized = client.get("/api/v1/v4/admin/tools/proposals")
        assert unauthorized.status_code == 401

        listed = client.get("/api/v1/v4/admin/tools/proposals", headers=headers)
        assert listed.status_code == 200
        listed_item = listed.json()["items"][0]
        assert listed_item["proposal_id"] == proposal_payload["proposal_id"]
        assert "python_code" not in listed_item

        detail = client.get(f"/api/v1/v4/admin/tools/proposals/{proposal_payload['proposal_id']}", headers=headers)
        assert detail.status_code == 200
        assert "async def run" in detail.json()["python_code"]

        sandbox_status = client.post(
            f"/api/v1/v4/admin/tools/proposals/{proposal_payload['proposal_id']}/status",
            headers=headers,
            json={
                "next_status": "approved_for_sandbox_test",
                "code_sha256": proposal_payload["code_sha256"],
                "reviewer": "admin",
                "note": "static review passed",
            },
        )
        assert sandbox_status.status_code == 200
        assert sandbox_status.json()["status"] == "approved_for_sandbox_test"

        sandbox_report = client.post(
            f"/api/v1/v4/admin/tools/proposals/{proposal_payload['proposal_id']}/sandbox",
            headers=headers,
            json={"args": {"value": 4}},
        )
        assert sandbox_report.status_code == 200
        report = sandbox_report.json()
        assert report["status"] == "pass"
        assert report["result"] == {"value": 8}
        assert report["sandbox"]["resource_limits"]["open_files"] == 32

        runtime_status = client.post(
            f"/api/v1/v4/admin/tools/proposals/{proposal_payload['proposal_id']}/status",
            headers=headers,
            json={
                "next_status": "approved_for_runtime",
                "code_sha256": proposal_payload["code_sha256"],
                "reviewer": "admin",
                "sandbox_report": report,
            },
        )
        assert runtime_status.status_code == 200
        assert runtime_status.json()["status"] == "approved_for_runtime"
        assert runtime_status.json()["review_log"][-1]["to"] == "approved_for_runtime"
    finally:
        app.dependency_overrides.pop(get_settings, None)


def test_dynamic_tool_admin_api_rejects_unknown_proposal(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.ensure_runtime_dirs()
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        client = TestClient(app)
        response = client.get("/api/v1/v4/admin/tools/proposals/not-found", headers={"X-API-Key": "secret"})
        assert response.status_code == 404
    finally:
        app.dependency_overrides.pop(get_settings, None)
