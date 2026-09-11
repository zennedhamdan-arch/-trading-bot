"""
services/risk_engine.py

The DETERMINISTIC hard risk engine (Part 15). LLMs never control hard risk
rules: every check here is plain Python over real account/market state, and
an LLM can NEVER override a rejection (the CIO is not even consulted after
a rejection, and the final order gate re-validates independently).

Checks (all deterministic):
  - market hours            (verifiably closed market -> no new BUYs)
  - account availability    (equity/cash must be real and positive)
  - maximum position size   (MAX_POSITION_PCT of equity, per symbol)
  - maximum portfolio exposure (RISK_MAX_PORTFOLIO_EXPOSURE_PCT of equity)
  - maximum daily loss      (day P&L at/below -RISK_MAX_DAILY_LOSS_PCT)
  - maximum trades per day  (RISK_MAX_TRADES_PER_DAY, persisted per day)
  - duplicate orders        (same symbol+side+size already executed this cycle)
  - pending orders          (an open order for the symbol already exists)
  - stop-loss / take-profit validation (sanity when provided)
  - stale data              (newest daily bar older than EVIDENCE_STALE_DAYS)
  - event risk              (from News Intelligence; reject / reduce / ignore,
                             configurable — never silently ignored when HIGH)

Returns the standardized verdict:

    {"approved": true,  "reason": "TRADE_APPROVED",   "checks": {...}}
    {"approved": false, "reason": "MAX_DAILY_LOSS_REACHED", "checks": {...}}

Sizing comes from the existing deterministic risk_gate.assess(); this engine
composes it with the hard portfolio-level rules above and may REDUCE the
allowed notional (event risk) but never increase it.
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from services import risk_gate

logger = logging.getLogger("risk_engine")

DB_PATH = Path(settings.MEMORY_DB_PATH)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS risk_engine_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,          -- YYYY-MM-DD (UTC)
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    notional_usd REAL
);
CREATE INDEX IF NOT EXISTS idx_risk_trades_date ON risk_engine_trades(date);
"""

# Per-cycle duplicate-order guard (reset by begin_cycle()).
_cycle_orders = set()
_daily_counter_lock = None  # created lazily (sqlite handles serialization)


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(_SCHEMA)


def begin_cycle() -> None:
    """Resets the per-cycle duplicate-order guard. Called at the start of
    every trading cycle."""
    _cycle_orders.clear()


def mark_executed(symbol: str, side: str, notional_usd: float = 0.0) -> None:
    """Records a SUCCESSFULLY EXECUTED order: the persisted daily counter
    (survives restarts) and the per-cycle duplicate guard. Called by main.py
    only after the broker accepted the order."""
    now = datetime.now(timezone.utc)
    _cycle_orders.add((symbol.upper(), str(side).lower(), round(_num(notional_usd), 2)))
    with _conn() as conn:
        conn.execute(
            "INSERT INTO risk_engine_trades (date, created_at, symbol, side, notional_usd) "
            "VALUES (?, ?, ?, ?, ?)",
            (now.strftime("%Y-%m-%d"), now.isoformat(), symbol, str(side).lower(),
             float(notional_usd) if notional_usd else None),
        )


def trades_today() -> int:
    now = datetime.now(timezone.utc)
    with _conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM risk_engine_trades WHERE date = ?",
            (now.strftime("%Y-%m-%d"),),
        ).fetchone()[0]


def _num(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def validate_stop_take_profit(side: str, entry_price, stop_loss=None, take_profit=None) -> tuple:
    """Deterministic sanity validation for optional stop-loss / take-profit
    levels (both must be provided to be checked; a long's stop must sit
    below entry and its target above entry, and vice versa for a short).
    Returns (ok, reason)."""
    if stop_loss is None and take_profit is None:
        return True, None
    if entry_price is None or _num(entry_price) <= 0:
        return False, "stop/take-profit validation requires a positive entry price"
    entry = _num(entry_price)
    side = str(side).lower()
    if stop_loss is not None:
        sl = _num(stop_loss)
        if sl <= 0:
            return False, "stop_loss must be positive"
        if side == "buy" and sl >= entry:
            return False, f"long stop_loss ({sl}) must be below entry ({entry})"
        if side == "sell" and sl <= entry:
            return False, f"short stop_loss ({sl}) must be above entry ({entry})"
    if take_profit is not None:
        tp = _num(take_profit)
        if tp <= 0:
            return False, "take_profit must be positive"
        if side == "buy" and tp <= entry:
            return False, f"long take_profit ({tp}) must be above entry ({entry})"
        if side == "sell" and tp >= entry:
            return False, f"short take_profit ({tp}) must be below entry ({entry})"
    return True, None


def assess_trade(symbol: str, side: str, account_summary: dict,
                 positions_by_symbol: dict, indicators: dict = None,
                 market_clock: dict = None, news_intelligence: dict = None,
                 proposed_notional_usd: float = None,
                 stop_loss: float = None, take_profit: float = None) -> dict:
    """The deterministic hard risk verdict for a proposed trade.

    Composes the existing risk_gate sizing with the portfolio-level hard
    rules. Returns {approved, reason, checks, max_notional_usd, risk_level}.
    `reason` is "TRADE_APPROVED" or a stable uppercase rejection code.
    """
    side = str(side).lower()
    checks = {}
    reasons = []

    # --- account availability ------------------------------------------------
    equity = _num(account_summary.get("equity"))
    cash = _num(account_summary.get("cash"))
    buying_power = _num(account_summary.get("buying_power"), cash)
    checks["account_available"] = bool(account_summary.get("error") is None and equity > 0)
    if not checks["account_available"]:
        reasons.append("ACCOUNT_UNAVAILABLE")

    # --- market hours (only blocks verifiably-closed markets) ----------------
    is_open = (market_clock or {}).get("is_open")
    market_open = True if is_open is None else bool(is_open)
    checks["market_open"] = market_open
    if side == "buy" and settings.RISK_REQUIRE_MARKET_OPEN and not market_open:
        reasons.append("MARKET_CLOSED")

    # --- stale data -----------------------------------------------------------
    stale = False
    if indicators:
        raw = indicators.get("last_bar_date")
        if raw:
            try:
                text = str(raw).split("+")[0].strip()
                bar_dt = datetime.fromisoformat(text)
                if bar_dt.tzinfo is None:
                    bar_dt = bar_dt.replace(tzinfo=timezone.utc)
                stale = (datetime.now(timezone.utc) - bar_dt).total_seconds() / 86400.0 \
                    > float(settings.EVIDENCE_STALE_DAYS)
            except (ValueError, TypeError):
                stale = False
    checks["data_fresh"] = not stale
    if stale:
        reasons.append("STALE_DATA")

    # --- daily loss cap -------------------------------------------------------
    day_pl_pct = account_summary.get("day_pl_pct")
    day_loss_breached = (
        day_pl_pct is not None
        and _num(day_pl_pct) <= -abs(float(settings.RISK_MAX_DAILY_LOSS_PCT)) * 100.0
    )
    checks["within_daily_loss_limit"] = not day_loss_breached
    if day_loss_breached:
        reasons.append("MAX_DAILY_LOSS_REACHED")

    # --- max trades per day (persisted) ---------------------------------------
    trades = trades_today()
    max_trades = int(settings.RISK_MAX_TRADES_PER_DAY or 10)
    checks["within_daily_trade_limit"] = trades < max_trades
    if trades >= max_trades:
        reasons.append("MAX_TRADES_PER_DAY_REACHED")

    # --- duplicate order (same cycle, same symbol+side+size) ------------------
    notional = _num(proposed_notional_usd)
    dup_key = (symbol.upper(), side, round(notional, 2))
    checks["not_duplicate_order"] = dup_key not in _cycle_orders
    if not checks["not_duplicate_order"]:
        reasons.append("DUPLICATE_ORDER")

    # --- pending orders for this symbol ---------------------------------------
    # (callers pass today's open orders via positions_by_symbol's sibling
    #  `pending_orders`; absent support means no pending-order signal)
    checks["no_pending_order"] = True  # refined below when pending provided

    # --- stop-loss / take-profit validation -----------------------------------
    sl_ok, sl_reason = validate_stop_take_profit(side, (indicators or {}).get("latest_close"),
                                                 stop_loss, take_profit)
    checks["stop_take_profit_valid"] = sl_ok
    if not sl_ok:
        reasons.append("INVALID_STOP_TAKE_PROFIT")
        if sl_reason:
            reasons.append(sl_reason)

    # --- deterministic sizing (risk_gate — never weakened here) ---------------
    position = (positions_by_symbol or {}).get(symbol)
    gate = risk_gate.assess(symbol, side, account_summary, position, indicators)
    checks["position_size_ok"] = bool(gate.get("approved"))
    max_notional = float(gate.get("max_notional_usd") or 0.0)

    # --- portfolio exposure cap ------------------------------------------------
    total_exposure = 0.0
    for pos in (positions_by_symbol or {}).values():
        total_exposure += _num(pos.get("market_value")) or _num(pos.get("qty")) * _num(
            pos.get("current_price") or pos.get("avg_entry_price"))
    exposure_pct = (total_exposure / equity) if equity > 0 else 0.0
    exposure_cap = float(settings.RISK_MAX_PORTFOLIO_EXPOSURE_PCT or 0.8)
    # A BUY may not push total exposure above the cap.
    exposure_ok = side != "buy" or equity <= 0 or \
        (total_exposure + min(max_notional, notional or max_notional)) <= equity * exposure_cap + 1e-9
    checks["within_portfolio_exposure"] = bool(exposure_ok)
    if not exposure_ok:
        reasons.append("MAX_PORTFOLIO_EXPOSURE_REACHED")

    # --- event risk (from News Intelligence; configurable behavior) -----------
    event = news_intelligence or {}
    event_risk_active = bool(event.get("event_risk"))
    event_action = str(settings.RISK_EVENT_RISK_ACTION or "reduce").lower()
    event_reduced = False
    checks["event_risk_ok"] = True
    if event_risk_active and side == "buy":
        level = str(event.get("event_risk_level") or "MEDIUM").upper()
        if event_action == "reject":
            checks["event_risk_ok"] = False
            reasons.append(f"EVENT_RISK_{level}")
        elif event_action == "reduce":
            # "reduce" halves the allowed notional for ANY event-risk level;
            # operators wanting a hard block use RISK_EVENT_RISK_ACTION=reject.
            max_notional = round(max_notional / 2.0, 2)
            event_reduced = True

    # --- final verdict ----------------------------------------------------------
    hard_blockers = (
        "ACCOUNT_UNAVAILABLE", "MAX_DAILY_LOSS_REACHED", "MAX_TRADES_PER_DAY_REACHED",
        "DUPLICATE_ORDER", "STALE_DATA", "INVALID_STOP_TAKE_PROFIT",
        "MAX_PORTFOLIO_EXPOSURE_REACHED", "MARKET_CLOSED",
    )
    blocked = any(r in hard_blockers for r in reasons) or not checks["event_risk_ok"] \
        or not checks["position_size_ok"]

    approved = not blocked and max_notional > 0
    reason = "TRADE_APPROVED" if approved else (reasons[0] if reasons else "RISK_REJECTED")

    return {
        "approved": approved,
        "reason": reason,
        "checks": checks,
        "max_notional_usd": round(max_notional, 2),
        "risk_level": gate.get("risk_level", "LOW"),
        "notional_reduced_by_event_risk": event_reduced,
        "gate": gate,
        "blocked_reasons": reasons,
    }


def refine_pending_orders(pending_orders: list) -> dict:
    """Helper: maps today's broker orders to a pending-signature set:
    {symbol: True} for any order still open (new/accepted/partial)."""
    pending = {}
    for order in pending_orders or []:
        status = str((order or {}).get("status") or "").lower()
        if status in ("new", "accepted", "pending_new", "partially_filled"):
            symbol = str((order or {}).get("symbol") or "").upper()
            if symbol:
                pending[symbol] = True
    return pending


def assess_with_orders(symbol: str, side: str, account_summary: dict,
                       positions_by_symbol: dict, indicators: dict = None,
                       market_clock: dict = None, news_intelligence: dict = None,
                       proposed_notional_usd: float = None,
                       recent_orders: list = None,
                       stop_loss: float = None, take_profit: float = None) -> dict:
    """assess_trade + the pending-order check from the broker's recent
    orders (open orders for the symbol block a new BUY)."""
    pending = refine_pending_orders(recent_orders)
    result = assess_trade(
        symbol, side, account_summary, positions_by_symbol, indicators,
        market_clock, news_intelligence, proposed_notional_usd,
        stop_loss, take_profit,
    )
    if side == "buy" and pending.get(symbol.upper()):
        result["approved"] = False
        result["reason"] = "PENDING_ORDER_EXISTS"
        result["checks"]["no_pending_order"] = False
        result["blocked_reasons"] = list(result.get("blocked_reasons") or []) + ["PENDING_ORDER_EXISTS"]
    return result
