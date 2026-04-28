from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

from app.domain.models import ActiveResearch, SessionContext, SessionTurn

logger = logging.getLogger(__name__)


def _trim_context_history(context: SessionContext, *, max_turns: int) -> SessionContext:
    context.normalize_active_research()
    if len(context.turns) <= max_turns:
        return context
    older = context.turns[: -max_turns]
    older_lines = [
        (
            f"用户问了：{turn.query}；系统围绕 {turn.relation or 'general'} 回答；"
            f"目标={','.join(turn.targets) or '无'}；字段={','.join(turn.requested_fields) or '无'}；"
            f"引用/论文={','.join(turn.titles[:2]) or '无'}。"
        )
        for turn in older
    ]
    context.summary = " ".join(part for part in [context.summary.strip(), " ".join(older_lines)] if part).strip()
    context.turns = context.turns[-max_turns:]
    return context


def _apply_active_research(context: SessionContext, active: ActiveResearch | None) -> None:
    if active is None:
        return
    context.set_active_research(
        relation=active.relation,
        targets=list(active.targets),
        titles=list(active.titles),
        requested_fields=list(active.requested_fields),
        required_modalities=list(active.required_modalities),
        answer_shape=active.answer_shape,
        precision_requirement=active.precision_requirement,
        clean_query=active.clean_query,
        last_topic_signature=active.last_topic_signature,
    )


class InMemorySessionStore:
    def __init__(self, max_turns: int = 8) -> None:
        self.max_turns = max(1, max_turns)
        self._sessions: dict[str, SessionContext] = {}

    def get(self, session_id: str) -> SessionContext:
        return self._sessions.setdefault(session_id, SessionContext(session_id=session_id))

    def upsert(self, context: SessionContext) -> None:
        self._sessions[context.session_id] = _trim_context_history(context, max_turns=self.max_turns)

    def append_turn(self, session_id: str, turn: SessionTurn) -> SessionContext:
        context = self.get(session_id)
        return self.commit_turn(context, turn)

    def commit_turn(self, context: SessionContext, turn: SessionTurn, active: ActiveResearch | None = None) -> SessionContext:
        _apply_active_research(context, active)
        context.turns.append(turn)
        self.upsert(context)
        return context


class SQLiteSessionStore:
    def __init__(self, db_path: str | Path, max_turns: int = 8) -> None:
        self.db_path = Path(db_path)
        self.max_turns = max(1, max_turns)
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get(self, session_id: str) -> SessionContext:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT context_json FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            if row is None:
                context = SessionContext(session_id=session_id)
                self.upsert(context)
                return context
            raw = str(row["context_json"] or "")
            try:
                return SessionContext.model_validate_json(raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning("session context is corrupt; recreating session_id=%s err=%s", session_id, exc)
                context = SessionContext(session_id=session_id)
                self.upsert(context)
                return context

    def upsert(self, context: SessionContext) -> None:
        context = _trim_context_history(context, max_turns=self.max_turns)
        payload = context.model_dump_json()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions(session_id, context_json, updated_at)
                    VALUES(?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                    ON CONFLICT(session_id) DO UPDATE SET
                        context_json = excluded.context_json,
                        updated_at = excluded.updated_at
                    """,
                    (context.session_id, payload),
                )
                conn.commit()

    def append_turn(self, session_id: str, turn: SessionTurn) -> SessionContext:
        with self._lock:
            context = self.get(session_id)
            return self.commit_turn(context, turn)

    def commit_turn(self, context: SessionContext, turn: SessionTurn, active: ActiveResearch | None = None) -> SessionContext:
        with self._lock:
            _apply_active_research(context, active)
            context.turns.append(turn)
            self.upsert(context)
            return context

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    context_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)")
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn


SessionStore = InMemorySessionStore | SQLiteSessionStore
