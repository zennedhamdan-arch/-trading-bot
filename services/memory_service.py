"""
services/memory_service.py

Gives the bot persistent memory across restarts, using a local SQLite
file (zero cost, zero external dependency). Two things live here:

1. Decision log — every CIO decision, with which agents said what,
   gets written to disk. When a position is later closed, we compute
   the realized P&L and attach it to that decision record.

2. Adaptive agent weighting — tracks each agent's (technical/news/
   fundamentals/risk) historical hit rate and produces a weight the
   CIO agent can use to lean more/less on a given agent's read. This
   is NOT model fine-tuning; it's a simple accuracy-tracking feedback
   loop implemented in plain Python/SQL.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from config import settings

logger = logging.getLogger("memory_service")

DB_PATH = Path(settings.MEMORY_DB_PATH)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decision TEXT NOT NULL,
    confidence REAL,
    notional_usd REAL,
    reasoning TEXT,
    tech_signal TEXT,
    news_sentiment TEXT,
    fundamentals_signal TEXT,
    bull_case TEXT,
    bear_case TEXT,
    entry_price REAL,
    exit_price REAL,
    realized_pl_pct REAL,
    closed_at TEXT,
    status TEXT NOT NULL DEFAULT 'OPEN'  -- OPEN | CLOSED
);

CREATE TABLE IF NOT EXISTS agent_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER NOT NULL,
    agent_name TEXT NOT NULL,
    call_direction TEXT NOT NULL,  -- BULLISH | BEARISH | NEUTRAL
    confidence REAL,
    FOREIGN KEY(decision_id) REFERENCES decisions(id)
);
"""


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as conn:
        conn.executescript(_SCHEMA)
    logger.info(f"Memory DB ready at {DB_PATH}")


def record_decision(symbol: str, decision_report: dict, tech_report: dict,
                     news_report: dict, fundamentals_report: dict,
                     debate_report: dict, entry_price: float | None) -> int:
    """Writes a new decision + each agent's directional call. Returns the row id."""
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO decisions
               (symbol, created_at, decision, confidence, notional_usd, reasoning,
                tech_signal, news_sentiment, fundamentals_signal, bull_case, bear_case,
                entry_price, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')""",
            (
                symbol,
                datetime.now(timezone.utc).isoformat(),
                decision_report.get("decision"),
                decision_report.get("confidence"),
                decision_report.get("notional_usd"),
                decision_report.get("reasoning"),
                tech_report.get("signal"),
                news_report.get("sentiment"),
                fundamentals_report.get("signal") if fundamentals_report else None,
                debate_report.get("bull_summary") if debate_report else None,
                debate_report.get("bear_summary") if debate_report else None,
                entry_price,
            ),
        )
        decision_id = cur.lastrowid

        agent_calls = [
            ("technical", tech_report.get("signal"), tech_report.get("confidence")),
            ("news", news_report.get("sentiment"), news_report.get("confidence")),
        ]
        if fundamentals_report:
            agent_calls.append(("fundamentals", fundamentals_report.get("signal"),
                                 fundamentals_report.get("confidence")))
        for agent_name, direction, confidence in agent_calls:
            if direction:
                conn.execute(
                    """INSERT INTO agent_calls (decision_id, agent_name, call_direction, confidence)
                       VALUES (?, ?, ?, ?)""",
                    (decision_id, agent_name, direction, confidence or 0.0),
                )
        return decision_id


def close_open_decision(symbol: str, exit_price: float):
    """
    Called when a position in `symbol` is closed. Finds the most recent
    OPEN decision for that symbol, computes realized P&L%, and marks it
    CLOSED. This is what lets the bot "learn" -- every closed trade
    becomes a labeled data point for agent accuracy scoring.
    """
    with _conn() as conn:
        row = conn.execute(
            """SELECT id, entry_price, decision FROM decisions
               WHERE symbol = ? AND status = 'OPEN' AND decision = 'BUY'
               ORDER BY created_at DESC LIMIT 1""",
            (symbol,),
        ).fetchone()
        if not row or not row["entry_price"]:
            return None

        pl_pct = ((exit_price - row["entry_price"]) / row["entry_price"]) * 100
        conn.execute(
            """UPDATE decisions SET exit_price = ?, realized_pl_pct = ?,
               closed_at = ?, status = 'CLOSED' WHERE id = ?""",
            (exit_price, round(pl_pct, 3), datetime.now(timezone.utc).isoformat(), row["id"]),
        )
        return {"decision_id": row["id"], "realized_pl_pct": round(pl_pct, 3)}


def get_agent_accuracy(agent_name: str, lookback: int = 20) -> dict:
    """
    Computes agent_name's hit rate over its last `lookback` closed calls.
    A "hit" = the agent's directional call (BULLISH/BEARISH) matched the
    sign of the trade's realized P&L. Returns a weight in [0.5, 1.5]
    the CIO can multiply that agent's confidence by: 1.0 = neutral,
    >1.0 = agent has been reliable, <1.0 = agent has been unreliable.
    """
    with _conn() as conn:
        rows = conn.execute(
            """SELECT ac.call_direction, d.realized_pl_pct
               FROM agent_calls ac
               JOIN decisions d ON d.id = ac.decision_id
               WHERE ac.agent_name = ? AND d.status = 'CLOSED' AND d.realized_pl_pct IS NOT NULL
               ORDER BY d.closed_at DESC LIMIT ?""",
            (agent_name, lookback),
        ).fetchall()

    if not rows:
        return {"agent": agent_name, "sample_size": 0, "hit_rate": None, "weight": 1.0}

    hits = 0
    for r in rows:
        was_bullish_call = r["call_direction"] == "BULLISH"
        was_profitable = r["realized_pl_pct"] > 0
        if was_bullish_call == was_profitable:
            hits += 1

    hit_rate = hits / len(rows)
    # Map hit rate (0..1) to a weight range (0.5..1.5), centered on 0.5 hit rate = 1.0 weight.
    weight = round(0.5 + hit_rate, 3)
    return {"agent": agent_name, "sample_size": len(rows), "hit_rate": round(hit_rate, 3), "weight": weight}


def get_recent_outcomes_summary(symbol: str | None = None, lookback: int = 10) -> str:
    """Human-readable summary of recent closed trades, for injecting into
    the CIO's prompt as memory context."""
    with _conn() as conn:
        if symbol:
            rows = conn.execute(
                """SELECT symbol, decision, realized_pl_pct, closed_at FROM decisions
                   WHERE status = 'CLOSED' AND symbol = ? ORDER BY closed_at DESC LIMIT ?""",
                (symbol, lookback),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT symbol, decision, realized_pl_pct, closed_at FROM decisions
                   WHERE status = 'CLOSED' ORDER BY closed_at DESC LIMIT ?""",
                (lookback,),
            ).fetchall()

    if not rows:
        return "No closed trades yet -- no historical performance data available."

    wins = sum(1 for r in rows if r["realized_pl_pct"] and r["realized_pl_pct"] > 0)
    lines = [f"Last {len(rows)} closed trades{f' for ' + symbol if symbol else ''}: {wins}/{len(rows)} profitable."]
    for r in rows[:5]:
        lines.append(f"  {r['symbol']} {r['decision']}: {r['realized_pl_pct']:+.2f}%")
    return "\n".join(lines)
