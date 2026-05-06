from __future__ import annotations

from app.services.retrieval.url_fetcher import fetch_url, validate_fetch_url


class _FakeResponse:
    status_code = 200
    url = "https://example.com/page"
    headers = {"content-type": "text/html; charset=utf-8"}
    text = "<html><head><title>Example</title><script>bad()</script></head><body><h1>Hello</h1><p>Readable text.</p></body></html>"

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return _FakeResponse()


def test_validate_fetch_url_rejects_non_https_and_private_hosts() -> None:
    assert validate_fetch_url("http://example.com")[2] == "only_https_urls_are_allowed"
    assert validate_fetch_url("https://localhost/test")[2] == "blocked_private_or_local_host"
    assert validate_fetch_url("https://127.0.0.1/test")[2] == "blocked_private_or_local_host"
    assert validate_fetch_url("https://10.0.0.2/test")[2] == "blocked_private_or_local_host"


def test_fetch_url_extracts_readable_html_text() -> None:
    client = _FakeClient()

    result = fetch_url(client=client, url="https://example.com/page", max_chars=1000)

    assert result.ok is True
    assert result.title == "Example"
    assert "Hello" in result.text
    assert "Readable text." in result.text
    assert "bad()" not in result.text
    assert client.calls[0]["url"] == "https://example.com/page"
