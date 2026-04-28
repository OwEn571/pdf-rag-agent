from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.core.config import Settings
from app.core.security import require_pdf_access
from app.services.agent import _subprocess_command_allowed


def _request(*, host: str = "127.0.0.1", query_string: bytes = b"", headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/v4/library/papers/PAPER/pdf",
            "headers": headers or [],
            "query_string": query_string,
            "client": (host, 4321),
        }
    )


def test_pdf_access_allows_local_preview_when_no_key_is_configured() -> None:
    settings = Settings(_env_file=None, admin_api_key="", library_api_key="", allow_local_pdf_without_api_key=True)

    require_pdf_access(_request(host="127.0.0.1"), settings=settings, authorization=None, x_api_key=None)


def test_pdf_access_still_blocks_remote_preview_when_no_key_is_configured() -> None:
    settings = Settings(_env_file=None, admin_api_key="", library_api_key="", allow_local_pdf_without_api_key=True)

    with pytest.raises(HTTPException) as exc_info:
        require_pdf_access(_request(host="203.0.113.10"), settings=settings, authorization=None, x_api_key=None)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "LIBRARY_API_KEY or ADMIN_API_KEY is not configured"


def test_pdf_access_allows_same_origin_browser_preview_when_no_key_is_configured() -> None:
    settings = Settings(
        _env_file=None,
        admin_api_key="",
        library_api_key="",
        allow_local_pdf_without_api_key=True,
        allow_same_origin_pdf_without_api_key=True,
    )
    headers = [
        (b"host", b"agent.example.test"),
        (b"referer", b"https://agent.example.test/v4"),
        (b"sec-fetch-site", b"same-origin"),
    ]

    require_pdf_access(
        _request(host="203.0.113.10", headers=headers),
        settings=settings,
        authorization=None,
        x_api_key=None,
    )


def test_pdf_access_requires_configured_key_even_for_local_requests() -> None:
    settings = Settings(_env_file=None, admin_api_key="", library_api_key="secret", allow_local_pdf_without_api_key=True)

    with pytest.raises(HTTPException) as exc_info:
        require_pdf_access(_request(host="127.0.0.1"), settings=settings, authorization=None, x_api_key=None)

    assert exc_info.value.status_code == 401
    require_pdf_access(
        _request(host="127.0.0.1", query_string=b"api_key=secret"),
        settings=settings,
        authorization=None,
        x_api_key=None,
    )


def test_agent_subprocess_allowlist_only_allows_bare_pdftoppm() -> None:
    assert _subprocess_command_allowed(["pdftoppm", "-png"]) is True
    assert _subprocess_command_allowed(["/usr/bin/pdftoppm", "-png"]) is False
    assert _subprocess_command_allowed(["python", "-c", "print(1)"]) is False
    assert _subprocess_command_allowed([]) is False
