"""
services/operator/memory.py

Persistent Trading Partner memory — in the SAME SQLite file as the rest of
the bot (settings.MEMORY_DB_PATH, default data/memory.db). No second
storage engine.

Tables (created idempotently):
  operator_conversations  one row per conversation (id, title, created_at)
  operator_messages       user/assistant turns + tool-call summaries +
                          the system-state snapshot at answer time
  system_events           lightweight diagnostic event history the operator
                          writes after each significant investigation

Never stores secrets: messages pass through schemas.sanitize() before
persistence, and tool-call summaries store only tool names/labels/status.
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import settings
from services.operator.schemas import sanitize

logger = logging.getLogger("operator")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS operator_conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_activity_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS operator_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL,               -- user | assistant
    content TEXT NOT NULL,
    tool_calls TEXT,                  -- JSON [{tool, label, ok}]
    system_state TEXT,                -- JSON snapshot (sanitized)
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_op_msgs_conv
    ON operator_messages(conversation_id, id);

CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    event_type TEXT NOT NULL,         -- e.g. diagnosis, degradation, recovery
    severity TEXT NOT NULL,           -- info | warning | error
    component TEXT,
    message TEXT NOT NULL,
    metadata TEXT,                    -- JSON (sanitized)
    correlation_id TEXT               -- cycle id / conversation id when known
);
CREATE INDEX IF NOT EXISTS idx_sys_events_ts ON system_events(ts);
"""

_init_lock = threading.Lock()
_initialized = False


@contextmanager
def _conn():
    db_path = Path(str(settings.MEMORY_DB_PATH))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    global _initialized
    with _init_lock:
        if _initialized:
            return
        with _conn() as conn:
            conn.executescript(_SCHEMA)
        _initialized = True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def create_conversation(title: str) -> int:
    init_db()
    now = _now()
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO operator_conversations (title, created_at, last_activity_at) "
            "VALUES (?, ?, ?)", (str(title)[:200], now, now))
        return int(cur.lastrowid)


def append_message(conversation_id: int, role: str, content: str,
                   tool_calls: list = None, system_state: dict = None) -> None:
    init_db()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO operator_messages (conversation_id, role, content, "
            "tool_calls, system_state, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (int(conversation_id), role,
             str(sanitize(content) or "")[:20000],
             json.dumps(sanitize(tool_calls or []))[:20000],
             json.dumps(sanitize(system_state or {}))[:20000],
             _now()))
        conn.execute(
            "UPDATE operator_conversations SET last_activity_at = ? WHERE id = ?",
            (_now(), int(conversation_id)))


def get_messages(conversation_id: int, limit: int = 100) -> list:
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, role, content, tool_calls, system_state, created_at "
            "FROM operator_messages WHERE conversation_id = ? "
            "ORDER BY id DESC LIMIT ?", (int(conversation_id), int(limit))
        ).fetchall()
    out = []
    for r in reversed(rows):
        msg = dict(r)
        for key in ("tool_calls", "system_state"):
            try:
                msg[key] = json.loads(msg.get(key) or "null")
            except (ValueError, TypeError):
                msg[key] = None
        out.append(msg)
    return out


def list_conversations(limit: int = 30) -> list:
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, last_activity_at "
            "FROM operator_conversations ORDER BY last_activity_at DESC LIMIT ?",
            (int(limit),)).fetchall()
    return [dict(r) for r in rows]


def delete_conversation(conversation_id: int) -> bool:
    init_db()
    with _conn() as conn:
        cur = conn.execute(
            "DELETE FROM operator_messages WHERE conversation_id = ?",
            (int(conversation_id),))
        conn.execute(
            "DELETE FROM operator_conversations WHERE id = ?", (int(conversation_id),))
        return cur.rowcount > 0 or True


# ---------------------------------------------------------------------------
# System events (diagnostic history)
# ---------------------------------------------------------------------------

def record_system_event(event_type: str, severity: str, component: str,
                        message: str, metadata: dict = None,
                        correlation_id: str = None) -> None:
    init_db()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO system_events (ts, event_type, severity, component, "
            "message, metadata, correlation_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_now(), str(event_type)[:60], str(severity)[:20],
             str(component or "system")[:80], str(sanitize(message) or "")[:2000],
             json.dumps(sanitize(metadata or {}))[:20000],
             str(correlation_id)[:80] if correlation_id else None))


def get_system_events(limit: int = 50, severity: str = None,
                      component: str = None) -> list:
    init_db()
    query = "SELECT * FROM system_events"
    clauses, params = [], []
    if severity:
        clauses.append("severity = ?"); params.append(severity)
    if component:
        clauses.append("component = ?"); params.append(component)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with _conn() as conn:
        rows = conn.execute(query, params).fetchall()
    out = []
    for r in rows:
        evt = dict(r)
        try:
            evt["metadata"] = json.loads(evt.get("metadata") or "null")
        except (ValueError, TypeError):
            evt["metadata"] = None
        out.append(evt)
    return out


def prune_old_data() -> None:
    """Drops conversations and events past the retention window."""
    init_db()
    keep_days = max(1, int(settings.OPERATOR_HISTORY_KEEP_DAYS or 30))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).isoformat()
    with _conn() as conn:
        stale = [r[0] for r in conn.execute(
            "SELECT id FROM operator_conversations WHERE last_activity_at < ?",
            (cutoff,)).fetchall()]
        for conv_id in stale:
            conn.execute("DELETE FROM operator_messages WHERE conversation_id = ?",
                         (conv_id,))
            conn.execute("DELETE FROM operator_conversations WHERE id = ?", (conv_id,))
        conn.execute("DELETE FROM system_events WHERE ts < ?", (cutoff,))
