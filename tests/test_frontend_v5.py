from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_root_redirects_to_v5_frontend() -> None:
    client = TestClient(app, follow_redirects=False)

    response = client.get("/")

    assert response.status_code == 307
    assert response.headers["location"] == "/v5"


def test_v5_frontend_serves_rebranded_workspace() -> None:
    client = TestClient(app)

    response = client.get("/v5")

    assert response.status_code == 200
    assert "Paper Research Agent V5" in response.text
    assert "V5 Research Console" in response.text
    assert "zotero-agent-v5-conversations" in response.text
    # SSE event handling
    assert "todo_update" in response.text
    assert "tool_use" in response.text
    assert "confidence" in response.text
    assert "runGraph" in response.text
    assert "flowchart" in response.text
    assert "agent_step" in response.text
    assert "renderRunGraph" in response.text
    assert "buildRunFlowGraph" in response.text
    # Tool admin
    assert "Tool Review" in response.text
    assert "ADMIN_API_KEY" in response.text
    assert "admin/tools/proposals" in response.text
    assert "zotero-agent-v5-admin-key" in response.text
    assert "approved_for_sandbox_test" in response.text
    assert "approved_for_runtime" in response.text
