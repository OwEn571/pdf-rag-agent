from __future__ import annotations

from collections import defaultdict, deque
from hmac import compare_digest
from ipaddress import ip_address
from threading import RLock
import time
from urllib.parse import urlparse

from fastapi import Depends, Header, HTTPException, Request, status

from app.core.config import Settings, get_settings


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = RLock()

    def check(self, *, scope: str, key: str, limit: int, window_seconds: int) -> None:
        if limit <= 0:
            return
        now = time.monotonic()
        window = max(1, int(window_seconds))
        with self._lock:
            bucket = self._hits[(scope, key)]
            while bucket and now - bucket[0] > window:
                bucket.popleft()
            if len(bucket) >= limit:
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate limit exceeded")
            bucket.append(now)


_rate_limiter = InMemoryRateLimiter()


def _request_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
    if forwarded_for:
        return forwarded_for
    if request.client is not None and request.client.host:
        return request.client.host
    return "unknown"


def _extract_api_key(authorization: str | None, x_api_key: str | None) -> str:
    if x_api_key:
        return x_api_key.strip()
    value = str(authorization or "").strip()
    if value.lower().startswith("bearer "):
        return value[7:].strip()
    return ""


def _require_api_key(*, expected: str, provided: str, missing_detail: str) -> None:
    if not expected:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=missing_detail)
    if not provided or not compare_digest(provided, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")


def _is_local_request(request: Request) -> bool:
    forwarded_for = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
    host = forwarded_for or (request.client.host if request.client is not None else "")
    host = str(host or "").strip().lower()
    if host in {"localhost", "testclient"}:
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _request_host(request: Request) -> str:
    host = str(request.headers.get("host") or request.url.netloc or "").strip().lower()
    return host.rstrip("/")


def _origin_host(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    host = parsed.netloc or parsed.path
    return host.strip().lower().rstrip("/")


def _is_same_origin_browser_request(request: Request) -> bool:
    sec_fetch_site = str(request.headers.get("sec-fetch-site", "") or "").strip().lower()
    if sec_fetch_site == "same-origin":
        return True
    host = _request_host(request)
    if not host:
        return False
    for header_name in ("origin", "referer"):
        origin_host = _origin_host(request.headers.get(header_name, ""))
        if origin_host and origin_host == host:
            return True
    return False


def require_admin_access(
    request: Request,
    settings: Settings = Depends(get_settings),
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    provided = _extract_api_key(authorization, x_api_key)
    _require_api_key(
        expected=settings.admin_api_key,
        provided=provided,
        missing_detail="ADMIN_API_KEY is not configured",
    )
    _rate_limiter.check(
        scope="admin",
        key=_request_key(request),
        limit=settings.admin_rate_limit_per_window,
        window_seconds=settings.api_rate_limit_window_seconds,
    )


def require_pdf_access(
    request: Request,
    settings: Settings = Depends(get_settings),
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    provided = str(request.query_params.get("api_key", "") or "").strip() or _extract_api_key(authorization, x_api_key)
    expected = settings.library_api_key or settings.admin_api_key
    if not expected and settings.allow_local_pdf_without_api_key and _is_local_request(request):
        _rate_limiter.check(
            scope="pdf",
            key=_request_key(request),
            limit=settings.pdf_rate_limit_per_window,
            window_seconds=settings.api_rate_limit_window_seconds,
        )
        return
    if not expected and settings.allow_same_origin_pdf_without_api_key and _is_same_origin_browser_request(request):
        _rate_limiter.check(
            scope="pdf",
            key=_request_key(request),
            limit=settings.pdf_rate_limit_per_window,
            window_seconds=settings.api_rate_limit_window_seconds,
        )
        return
    _require_api_key(
        expected=expected,
        provided=provided,
        missing_detail="LIBRARY_API_KEY or ADMIN_API_KEY is not configured",
    )
    _rate_limiter.check(
        scope="pdf",
        key=_request_key(request),
        limit=settings.pdf_rate_limit_per_window,
        window_seconds=settings.api_rate_limit_window_seconds,
    )
