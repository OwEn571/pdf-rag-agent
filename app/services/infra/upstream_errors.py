from __future__ import annotations


UPSTREAM_RETRY_MESSAGE = "上游模型服务临时失败，请等待 10 秒后重新发起请求。"


def user_facing_error_message(exc: Exception) -> str:
    """Return a short user-facing message for transient provider failures."""
    raw = str(exc).strip()
    lowered = raw.lower()
    if (
        "qhai_api_error" in lowered
        or "upstream service request failed" in lowered
        or "请求上游服务失败" in raw
    ):
        return UPSTREAM_RETRY_MESSAGE
    return raw or "请求失败。"
