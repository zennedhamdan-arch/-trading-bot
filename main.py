"""
main.py

FastAPI server for the 24/7 Automated Multi-Agent AI Trading Bot.

Serves the static dashboard, exposes REST endpoints for portfolio data,
agent deliberation logs, and bot control, and runs an APScheduler
background job that executes the full agent decision cycle on a
fixed interval for every symbol in the trade universe.

Agent pipeline per symbol, per cycle:
  1. Technical analysis          (tech_agent, existing)
  2. News/sentiment analysis     (news_agent, existing)
  3. Fundamentals analysis       (fundamentals_agent, NEW -- free via yfinance)
  4. Bull vs Bear debate         (debate_agent, NEW -- surfaces one-sided reasoning)
  5. Risk assessment             (risk_agent, existing)
  6. Executive decision          (cio_agent, now also given debate + fundamentals +
                                   each agent's historical accuracy weight + recent
                                   outcome memory, so its context grows every cycle)
  7. Execute if actionable, and record the decision to memory_service.
  8. If a previously open position was closed since the last cycle,
     compute realized P&L and feed it back into memory_service so the
     next cycle's agent-accuracy weights reflect it.

Run with:  python main.py
Dashboard: http://localhost:8000
"""

import logging
from collections import deque
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import settings
from services import alpaca_service, memory_service
from agents import news_agent, tech_agent, risk_agent, cio_agent, fundamentals_agent, debate_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# In-memory state (fine for a single-process paper trading dashboard).
# Restarting the server clears the live log feed and uptime, but decision
# history and agent accuracy now persist in SQLite via memory_service --
# that's what survives restarts and is what the "learning" loop reads from.
# ---------------------------------------------------------------------------
MAX_LOG_ENTRIES = 200
agent_logs = deque(maxlen=MAX_LOG_ENTRIES)

bot_state = {
    "running": False,
    "started_at": None,
    "last_cycle_at": None,
    "last_cycle_status": "NEVER_RUN",
}

scheduler = AsyncIOScheduler()

# Tracks symbol -> qty as of the end of the previous cycle, so we can detect
# when a position disappears (closed) between cycles and credit/debit the
# outcome to the decision that opened it.
_previous_positions: dict = {}


def _log_event(entry: dict):
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    agent_logs.appendleft(entry)


def _placeholder_headlines(symbol: str) -> list:
    """
    Fetches recent news headlines for a symbol via Alpaca's News API
    (included with alpaca-py). Returns an empty list on any failure so
    the news agent can gracefully report NEUTRAL rather than crash.
    """
    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest

        client = NewsClient(api_key=settings.ALPACA_API_KEY, secret_key=settings.ALPACA_SECRET_KEY)
        req = NewsRequest(symbols=symbol, limit=10)
        news_set = client.get_news(req)
        headlines = [item.headline for item in news_set.data.get("news", [])] if hasattr(news_set, "data") else []
        if not headlines and hasattr(news_set, "news"):
            headlines = [item.headline for item in news_set.news]
        return headlines
    except Exception as e:
        logger.warning(f"Could not fetch news for {symbol}: {e}")
        return []


def _detect_closed_positions(current_positions_by_symbol: dict):
    """
    Compares this cycle's open positions against last cycle's. Any symbol
    that had a position before and doesn't now was closed since the last
    cycle -- compute realized P&L and feed it back into memory_service.
    This is the mechanism that makes the "learning" loop actually fire:
    every closed trade becomes a labeled data point for agent accuracy.
    """
    if not settings.ENABLE_MEMORY:
        return
    for symbol, prev in _previous_positions.items():
        if symbol not in current_positions_by_symbol:
            exit_price = prev.get("current_price") or prev.get("avg_entry_price")
            if exit_price:
                result = memory_service.close_open_decision(symbol, exit_price)
                if result:
                    _log_event({
                        "agent": "memory", "symbol": symbol, "level": "INFO",
                        "message": f"Trade closed. Realized P&L: {result['realized_pl_pct']:+.2f}%. "
                                   f"Recorded for agent accuracy scoring.",
                        "data": result,
                    })


def _get_agent_weights() -> dict:
    if not settings.ENABLE_MEMORY:
        return {}
    weights = {}
    for agent_name in ("technical", "news", "fundamentals"):
        weights[agent_name] = memory_service.get_agent_accuracy(
            agent_name, lookback=settings.AGENT_ACCURACY_LOOKBACK
        )
    return weights


async def run_trading_cycle(triggered_by: str = "scheduler") -> dict:
    """
    Executes one full agent decision cycle across the entire trade
    universe. This is the core loop used by both the scheduled job and
    the manual '/api/bot/run-now' endpoint.
    """
    cycle_summary = {"triggered_by": triggered_by, "symbols_processed": [], "errors": []}
    logger.info(f"Starting trading cycle (triggered by: {triggered_by})")

    account_summary = alpaca_service.get_account_summary()
    if account_summary.get("error"):
        msg = f"Skipping cycle: could not fetch account summary ({account_summary['error']})"
        logger.error(msg)
        _log_event({"agent": "system", "symbol": None, "level": "ERROR", "message": msg})
        cycle_summary["errors"].append(msg)
        bot_state["last_cycle_status"] = "ERROR"
        bot_state["last_cycle_at"] = datetime.now(timezone.utc).isoformat()
        return cycle_summary

    positions_data = alpaca_service.get_open_positions()
    positions_by_symbol = {p["symbol"]: p for p in positions_data.get("positions", [])}

    # Learning loop: check what closed since last cycle before doing anything else.
    _detect_closed_positions(positions_by_symbol)
    agent_weights = _get_agent_weights()

    for symbol in settings.TRADE_UNIVERSE:
        try:
            # 1. Technical analysis
            indicators = alpaca_service.get_indicators(symbol)
            tech_report = tech_agent.analyze_technicals(symbol, indicators)
            _log_event({"agent": "technical", "symbol": symbol, "level": "INFO",
                        "message": tech_report["summary"], "data": tech_report})

            # 2. News/sentiment analysis
            headlines = _placeholder_headlines(symbol)
            news_report = news_agent.analyze_news(symbol, headlines)
            _log_event({"agent": "news", "symbol": symbol, "level": "INFO",
                        "message": news_report["summary"], "data": news_report})

            # 3. Fundamentals analysis (new -- free via yfinance)
            fundamentals_report = None
            if settings.ENABLE_FUNDAMENTALS_AGENT:
                fundamentals_data = fundamentals_agent.get_fundamentals(symbol)
                fundamentals_report = fundamentals_agent.analyze_fundamentals(symbol, fundamentals_data)
                _log_event({"agent": "fundamentals", "symbol": symbol, "level": "INFO",
                            "message": fundamentals_report["summary"], "data": fundamentals_report})

            # 4. Bull vs Bear debate (new)
            debate_report = None
            if settings.ENABLE_DEBATE:
                debate_report = debate_agent.run_debate(symbol, tech_report, news_report, fundamentals_report)
                _log_event({"agent": "debate", "symbol": symbol, "level": "INFO",
                            "message": f"Bull({debate_report['bull_strength']}) vs "
                                       f"Bear({debate_report['bear_strength']}), edge={debate_report['edge']}",
                            "data": debate_report})

            # 5. Risk assessment (only meaningful ahead of a potential BUY,
            # but we compute it every cycle so the CIO always has it)
            existing_position = positions_by_symbol.get(symbol)
            risk_report = risk_agent.assess_risk(symbol, "buy", account_summary, existing_position)
            _log_event({"agent": "risk", "symbol": symbol, "level": "INFO",
                        "message": risk_report["reasoning"], "data": risk_report})

            # 6. Executive decision, now with fundamentals + debate + memory context
            memory_summary = (
                memory_service.get_recent_outcomes_summary(symbol=symbol, lookback=10)
                if settings.ENABLE_MEMORY else ""
            )
            decision_report = cio_agent.make_decision(
                symbol, news_report, tech_report, risk_report,
                fundamentals_report=fundamentals_report,
                debate_report=debate_report,
                agent_weights=agent_weights,
                memory_summary=memory_summary,
            )
            _log_event({"agent": "cio", "symbol": symbol, "level": "INFO",
                        "message": f"{decision_report['decision']}: {decision_report['reasoning']}",
                        "data": decision_report})

            # 7. Execute if actionable
            trade_result = None
            decision = decision_report["decision"]
            if decision == "BUY" and decision_report["notional_usd"] > 0:
                trade_result = alpaca_service.execute_order(
                    symbol, "buy", notional_usd=decision_report["notional_usd"]
                )
            elif decision == "SELL" and existing_position:
                trade_result = alpaca_service.execute_order(
                    symbol, "sell", qty=existing_position["qty"]
                )

            if trade_result:
                level = "INFO" if trade_result.get("success") else "ERROR"
                _log_event({
                    "agent": "execution", "symbol": symbol, "level": level,
                    "message": (
                        f"Order {'submitted' if trade_result.get('success') else 'FAILED'}: "
                        f"{decision} {symbol}"
                        + (f" ({trade_result.get('error')})" if not trade_result.get("success") else "")
                    ),
                    "data": trade_result,
                })

            # 8. Record this decision to persistent memory for future learning
            if settings.ENABLE_MEMORY and decision == "BUY" and trade_result and trade_result.get("success"):
                entry_price = indicators.get("latest_close")
                memory_service.record_decision(
                    symbol, decision_report, tech_report, news_report,
                    fundamentals_report or {}, debate_report or {}, entry_price,
                )

            cycle_summary["symbols_processed"].append(symbol)

        except Exception as e:
            logger.error(f"Cycle error for {symbol}: {e}")
            _log_event({"agent": "system", "symbol": symbol, "level": "ERROR", "message": str(e)})
            cycle_summary["errors"].append(f"{symbol}: {e}")

    _previous_positions.clear()
    _previous_positions.update(positions_by_symbol)

    bot_state["last_cycle_at"] = datetime.now(timezone.utc).isoformat()
    bot_state["last_cycle_status"] = "OK" if not cycle_summary["errors"] else "PARTIAL_ERROR"
    logger.info(f"Cycle complete. Processed: {cycle_summary['symbols_processed']}")
    return cycle_summary


async def _scheduled_job():
    if bot_state["running"]:
        await run_trading_cycle(triggered_by="scheduler")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if settings.ENABLE_MEMORY:
        memory_service.init_db()

    warnings = settings.validate()
    for w in warnings:
        logger.warning(w)
        _log_event({"agent": "system", "symbol": None, "level": "WARNING", "message": w})

    scheduler.add_job(
        _scheduled_job,
        trigger=IntervalTrigger(minutes=settings.CYCLE_INTERVAL_MINUTES),
        id="trading_cycle",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(f"Scheduler started. Cycle interval: {settings.CYCLE_INTERVAL_MINUTES} min.")
    yield
    # Shutdown
    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped.")


app = FastAPI(title="AI Trading Bot Dashboard", lifespan=lifespan)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/portfolio")
async def api_portfolio():
    account = alpaca_service.get_account_summary()
    positions = alpaca_service.get_open_positions()
    orders = alpaca_service.get_recent_orders(limit=15)
    history = alpaca_service.get_portfolio_history()
    return JSONResponse({
        "account": account,
        "positions": positions["positions"],
        "orders": orders["orders"],
        "history": history["points"],
        "bot_state": {
            **bot_state,
            "uptime_status": "ONLINE" if bot_state["running"] else "OFFLINE",
        },
        "trade_universe": settings.TRADE_UNIVERSE,
    })


@app.get("/api/logs")
async def api_logs():
    return JSONResponse({"logs": list(agent_logs)})


@app.get("/api/agent-accuracy")
async def api_agent_accuracy():
    """New endpoint: shows the learning loop's current state -- each
    agent's hit rate and weight based on closed-trade outcomes."""
    if not settings.ENABLE_MEMORY:
        return JSONResponse({"enabled": False, "agents": {}})
    return JSONResponse({"enabled": True, "agents": _get_agent_weights()})


@app.post("/api/bot/start")
async def api_bot_start():
    if not bot_state["running"]:
        bot_state["running"] = True
        bot_state["started_at"] = datetime.now(timezone.utc).isoformat()
        _log_event({"agent": "system", "symbol": None, "level": "INFO",
                    "message": "Bot started. Automated cycle is now active."})
    return JSONResponse({"running": bot_state["running"], "started_at": bot_state["started_at"]})


@app.post("/api/bot/stop")
async def api_bot_stop():
    bot_state["running"] = False
    _log_event({"agent": "system", "symbol": None, "level": "INFO",
                "message": "Bot stopped. Automated cycle is paused."})
    return JSONResponse({"running": bot_state["running"]})


@app.post("/api/bot/run-now")
async def api_bot_run_now():
    result = await run_trading_cycle(triggered_by="manual")
    return JSONResponse(result)


@app.get("/api/config")
async def api_config():
    """Non-sensitive config info for the dashboard to display (no secrets)."""
    warnings = settings.validate()
    return JSONResponse({
        "trade_universe": settings.TRADE_UNIVERSE,
        "cycle_interval_minutes": settings.CYCLE_INTERVAL_MINUTES,
        "max_position_pct": settings.MAX_POSITION_PCT,
        "enable_fundamentals_agent": settings.ENABLE_FUNDAMENTALS_AGENT,
        "enable_debate": settings.ENABLE_DEBATE,
        "enable_memory": settings.ENABLE_MEMORY,
        "warnings": warnings,
    })


# ---------------------------------------------------------------------------
# Static dashboard
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
