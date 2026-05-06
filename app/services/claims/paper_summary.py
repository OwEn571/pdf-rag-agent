from __future__ import annotations

from typing import Any, Callable


PaperDocLookup = Callable[[str], Any | None]


def paper_summary_text(
    paper_id: str,
    *,
    paper_doc_lookup: PaperDocLookup,
    content_limit: int = 400,
) -> str:
    doc = paper_doc_lookup(paper_id)
    if doc is None:
        return ""
    meta = dict(getattr(doc, "metadata", None) or {})
    page_content = str(getattr(doc, "page_content", "") or "")
    return str(meta.get("generated_summary") or meta.get("abstract_note") or page_content[:content_limit]).strip()
