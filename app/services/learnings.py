from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


def remember_learning(*, data_dir: Path, key: str, content: str) -> Path:
    learning_dir = data_dir / "learnings"
    learning_dir.mkdir(parents=True, exist_ok=True)
    safe_key = _safe_key(key)
    path = learning_dir / f"{safe_key}.md"
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = f"\n\n## {timestamp}\n\n{str(content or '').strip()}\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    return path


def load_learnings(*, data_dir: Path, max_chars: int = 4000) -> str:
    learning_dir = data_dir / "learnings"
    if not learning_dir.exists():
        return ""
    parts: list[str] = []
    total = 0
    for path in sorted(learning_dir.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        header = f"# {path.stem}\n"
        chunk = f"{header}{text}\n"
        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining > len(header) + 80:
                parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts).strip()


def _safe_key(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in str(value or "").strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:80] or "general"
