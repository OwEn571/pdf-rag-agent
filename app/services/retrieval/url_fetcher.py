from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class FetchUrlResult:
    ok: bool
    url: str
    title: str = ""
    text: str = ""
    error: str = ""
    status_code: int = 0


def validate_fetch_url(raw_url: str) -> tuple[bool, str, str]:
    url = str(raw_url or "").strip()
    if not url:
        return False, "", "empty_url"
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        return False, url, "only_https_urls_are_allowed"
    host = str(parsed.hostname or "").strip().lower()
    if not host:
        return False, url, "missing_host"
    if _is_blocked_host(host):
        return False, url, "blocked_private_or_local_host"
    return True, url, ""


def fetch_url(*, client: Any, url: str, max_chars: int = 4000, timeout_seconds: float = 10.0) -> FetchUrlResult:
    ok, normalized_url, error = validate_fetch_url(url)
    if not ok:
        return FetchUrlResult(ok=False, url=normalized_url or url, error=error)
    max_chars = max(200, min(int(max_chars or 4000), 20000))
    try:
        response = client.get(normalized_url, timeout=timeout_seconds, follow_redirects=True)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return FetchUrlResult(ok=False, url=normalized_url, error=f"fetch_failed:{exc}")
    text = str(getattr(response, "text", "") or "")
    content_type = str(getattr(response, "headers", {}).get("content-type", "") or "").lower()
    if "html" in content_type or "<html" in text[:500].lower():
        title, body = _html_text(text)
    else:
        title, body = "", text
    body = " ".join(body.split())[:max_chars]
    return FetchUrlResult(
        ok=True,
        url=str(getattr(response, "url", "") or normalized_url),
        title=title[:240],
        text=body,
        status_code=int(getattr(response, "status_code", 0) or 0),
    )


def _is_blocked_host(host: str) -> bool:
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return True
    try:
        parsed_ip = ip_address(host.strip("[]"))
    except ValueError:
        return False
    return any(
        (
            parsed_ip.is_loopback,
            parsed_ip.is_private,
            parsed_ip.is_link_local,
            parsed_ip.is_multicast,
            parsed_ip.is_reserved,
            parsed_ip.is_unspecified,
        )
    )


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.skip_depth = 0
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        _ = attrs
        lowered = tag.lower()
        if lowered == "title":
            self.in_title = True
        if lowered in {"script", "style", "noscript"}:
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered == "title":
            self.in_title = False
        if lowered in {"script", "style", "noscript"} and self.skip_depth > 0:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        text = " ".join(str(data or "").split())
        if not text:
            return
        if self.in_title:
            self.title_parts.append(text)
        elif self.skip_depth == 0:
            self.text_parts.append(text)


def _html_text(raw_html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(raw_html)
    return " ".join(parser.title_parts).strip(), " ".join(parser.text_parts).strip()
