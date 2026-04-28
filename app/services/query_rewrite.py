from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class QueryRewriteResult:
    query: str
    mode: str
    queries: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, object]:
        return {
            "query": self.query,
            "mode": self.mode,
            "queries": list(self.queries),
            "targets": list(self.targets),
        }


def rewrite_query(
    *,
    query: str,
    targets: list[str] | None = None,
    mode: str = "multi_query",
    max_queries: int = 3,
) -> QueryRewriteResult:
    clean_query = " ".join(str(query or "").split())
    clean_targets = [str(item).strip() for item in list(targets or []) if str(item).strip()]
    clean_mode = mode if mode in {"multi_query", "hyde", "step_back"} else "multi_query"
    try:
        parsed_limit = int(max_queries or 3)
    except (TypeError, ValueError):
        parsed_limit = 3
    limit = max(1, min(8, parsed_limit))
    variants: list[str] = []
    _append_unique(variants, clean_query)
    target_text = " ".join(clean_targets)
    if target_text and target_text.lower() not in clean_query.lower():
        _append_unique(variants, f"{target_text} {clean_query}")
    if clean_mode == "hyde":
        _append_unique(variants, f"hypothetical answer evidence for: {clean_query}")
    elif clean_mode == "step_back":
        _append_unique(variants, f"background concepts and evidence needed to answer: {clean_query}")
    else:
        _append_unique(variants, f"{clean_query} evidence formula table figure result")
    if clean_targets:
        _append_unique(variants, " ".join([*clean_targets, "method result evidence"]))
    return QueryRewriteResult(
        query=clean_query,
        mode=clean_mode,
        queries=variants[:limit],
        targets=clean_targets,
    )


def _append_unique(items: list[str], value: str) -> None:
    text = " ".join(str(value or "").split())
    if text and text not in items:
        items.append(text)
