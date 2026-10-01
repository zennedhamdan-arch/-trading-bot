"""
operator_tools/market.py + portfolio.py equivalents (market state, paper
portfolio) — all read-only, all through the EXISTING service abstractions.
"""

from services import alpaca_service, market_data_service
from services.operator.operator_tools._registry import tool
from services.operator.schemas import sanitize

UNIVERSE = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]


def _universe():
    from config import settings
    return list(settings.TRADE_UNIVERSE or UNIVERSE)


@tool("get_current_market_state", "market",
      "Current market state for a symbol: latest price snapshot (bid/ask/"
      "last trade), market clock (open/closed, next open/close).",
      args={"symbol": {"type": "string", "description": "ticker symbol",
                       "required": True}},
      label="Checking market state")
def _get_current_market_state(args):
    symbol = str(args.get("symbol") or "").strip().upper()
    clock = alpaca_service.get_clock()
    snapshot = alpaca_service.get_snapshot(symbol)
    return sanitize({"symbol": symbol, "clock": clock, "snapshot": snapshot})


@tool("get_recent_bars", "market",
      "Recent daily closes for a symbol from the existing market-data "
      "abstraction (the bot computes its indicators from the same series).",
      args={"symbol": {"type": "string", "description": "ticker", "required": True},
            "limit": {"type": "integer", "description": "max closes (default 10)"}},
      label="Fetching recent bars")
def _get_recent_bars(args):
    symbol = str(args.get("symbol") or "").strip().upper()
    limit = min(int(args.get("limit") or 10), 60)
    indicators = alpaca_service.get_indicators(symbol)
    closes = list(indicators.get("recent_closes") or [])[-limit:]
    return sanitize({"symbol": symbol, "timeframe": "1d",
                     "latest_close": indicators.get("latest_close"),
                     "last_bar_date": indicators.get("last_bar_date"),
                     "closes": closes,
                     "error": indicators.get("error")})


@tool("get_indicators", "market",
      "The deterministic technical indicator suite for a symbol (RSI, "
      "SMAs, EMA, MACD, ATR, volatility, drawdown, returns, rule-based "
      "signal) — computed by the bot's own engine, never an LLM.",
      args={"symbol": {"type": "string", "description": "ticker", "required": True}},
      label="Computing indicators")
def _get_indicators(args):
    symbol = str(args.get("symbol") or "").strip().upper()
    return sanitize({"symbol": symbol,
                    "indicators": alpaca_service.get_indicators(symbol)})


@tool("get_market_regime", "market",
      "Market regime across the trade universe: the deterministic technical "
      "signal per symbol from the most recent cycle (with staleness noted).",
      label="Checking market regime")
def _get_market_regime(args):
    import main
    record = main.cycle_history[0] if main.cycle_history else None
    per_symbol = {}
    if record:
        for entry in (record.get("decisions") or []):
            per_symbol[entry.get("symbol")] = {
                "decision": entry.get("decision"),
                "blocked_reason": entry.get("blocked_reason"),
                "evidence_quality": entry.get("evidence_quality"),
            }
    tech_logs = {}
    for log in list(main.agent_logs)[:400]:
        if log.get("agent") == "technical" and log.get("symbol") \
                and log.get("symbol") not in tech_logs and log.get("data"):
            tech_logs[log["symbol"]] = {
                "signal": log["data"].get("signal"),
                "summary": log["data"].get("summary"),
            }
    return {"cycle_id": (record or {}).get("id"),
            "cycle_at": (record or {}).get("started_at"),
            "per_symbol_decisions": per_symbol,
            "latest_technical_reads": tech_logs,
            "note": ("reads the most recent cycle; signals are the "
                     "deterministic engine's, not LLM output") if record
            else "no cycle has run yet in this process"}


@tool("get_market_data_health", "market",
      "Health of the market-data layer: configured feed, probe status, "
      "subscription limitations.", label="Checking market data health")
def _get_market_data_health(args):
    from services import health_service
    startup = health_service.get_startup_report()
    return sanitize({
        "market_data": startup.get("market_data"),
        "indicators": startup.get("indicators"),
        "feed_config": market_data_service.feed_config(),
    })
