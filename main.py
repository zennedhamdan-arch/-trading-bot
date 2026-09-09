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
import time
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

# Real cycle history (in-memory, most recent first) so the dashboard can show
# what every autonomous cycle actually did. Cleared on restart, same as logs.
MAX_CYCLE_ENTRIES = 100
cycle_history = deque(maxlen=MAX_CYCLE_ENTRIES)
_cycle_counter = {"n": 0}
_active_cycle = None  # set while a cycle is running; used to tally warnings

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
    if _active_cycle is not None and entry.get("level") == "WARNING":
        _active_cycle["warnings"] += 1
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


PIPELINE_STAGES = ("market_data", "technical", "news", "fundamentals", "debate", "risk", "cio", "execution", "memory")


def _new_symbol_status() -> dict:
    """Per-symbol, per-agent execution status for one cycle.
    Values: OK | ERROR | UNAVAILABLE | SKIPPED."""
    return {stage: "SKIPPED" for stage in PIPELINE_STAGES}


def _report_level(report: dict) -> str:
    """Agent reports are logged to the dashboard feed at ERROR level when the
    agent itself reported an error — failures must be visible, not hidden
    behind INFO entries."""
    return "ERROR" if report and report.get("error") else "INFO"


def _compute_cycle_status(cycle_record: dict, cycle_summary: dict) -> str:
    """Honest cycle status.

    ERROR        - the cycle aborted (e.g. broker unreachable) or every
                   symbol failed outright.
    PARTIAL_ERROR- any pipeline stage across any symbol errored or its data
                   source was unavailable. The scheduler's "job executed
                   successfully" NEVER implies cycle OK.
    OK           - every enabled stage ran cleanly for every symbol.
    """
    if not cycle_summary["symbols_processed"] and cycle_summary["errors"]:
        return "ERROR"
    agent_status = cycle_record.get("agent_status") or {}
    for symbol_status in agent_status.values():
        for stage, status in symbol_status.items():
            if status in ("ERROR", "UNAVAILABLE"):
                return "PARTIAL_ERROR"
    return "OK"


async def run_trading_cycle(triggered_by: str = "scheduler") -> dict:
    """
    Executes one full agent decision cycle across the entire trade
    universe. This is the core loop used by both the scheduled job and
    the manual '/api/bot/run-now' endpoint.

    Every stage for every symbol is tracked in cycle_record["agent_status"]
    (OK / ERROR / UNAVAILABLE / SKIPPED) and the final cycle status is
    computed from those — a cycle is only OK when every enabled stage
    actually succeeded.
    """
    global _active_cycle
    cycle_summary = {"triggered_by": triggered_by, "symbols_processed": [], "errors": []}
    logger.info(f"Starting trading cycle (triggered by: {triggered_by})")

    # Real cycle record for the dashboard's cycle monitor.
    _cycle_counter["n"] += 1
    _active_cycle = {"warnings": 0}
    cycle_record = {
        "id": _cycle_counter["n"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "duration_s": None,
        "triggered_by": triggered_by,
        "status": None,
        "symbols_processed": [],
        "decisions": [],
        "orders": [],
        "agent_status": {},
        "warnings": 0,
        "errors": [],
    }
    _t0 = time.monotonic()

    def _finish_cycle_record(status: str):
        cycle_record["finished_at"] = datetime.now(timezone.utc).isoformat()
        cycle_record["duration_s"] = round(time.monotonic() - _t0, 1)
        cycle_record["status"] = status
        cycle_record["symbols_processed"] = cycle_summary["symbols_processed"]
        cycle_record["errors"] = cycle_summary["errors"]
        cycle_record["warnings"] = _active_cycle["warnings"] if _active_cycle else 0
        cycle_history.appendleft(cycle_record)

    account_summary = alpaca_service.get_account_summary()
    if account_summary.get("error"):
        msg = f"Skipping cycle: could not fetch account summary ({account_summary['error']})"
        logger.error(msg)
        _log_event({"agent": "system", "symbol": None, "level": "ERROR", "message": msg})
        cycle_summary["errors"].append(msg)
        bot_state["last_cycle_status"] = "ERROR"
        bot_state["last_cycle_at"] = datetime.now(timezone.utc).isoformat()
        _finish_cycle_record("ERROR")
        return cycle_summary

    positions_data = alpaca_service.get_open_positions()
    positions_by_symbol = {p["symbol"]: p for p in positions_data.get("positions", [])}

    # Learning loop: check what closed since last cycle before doing anything else.
    _detect_closed_positions(positions_by_symbol)
    agent_weights = _get_agent_weights()

    for symbol in settings.TRADE_UNIVERSE:
        status = _new_symbol_status()
        cycle_record["agent_status"][symbol] = status
        current_stage = "market_data"
        try:
            # 1. Market data (bars + indicators)
            indicators = alpaca_service.get_indicators(symbol)
            status["market_data"] = "ERROR" if indicators.get("error") else "OK"

            # 2. Technical analysis
            current_stage = "technical"
            tech_report = tech_agent.analyze_technicals(symbol, indicators)
            status["technical"] = "ERROR" if tech_report.get("error") else "OK"
            _log_event({"agent": "technical", "symbol": symbol, "level": _report_level(tech_report),
                        "message": tech_report["summary"], "data": tech_report})

            # 3. News/sentiment analysis
            current_stage = "news"
            headlines = _placeholder_headlines(symbol)
            news_report = news_agent.analyze_news(symbol, headlines)
            status["news"] = "ERROR" if news_report.get("error") else "OK"
            _log_event({"agent": "news", "symbol": symbol, "level": _report_level(news_report),
                        "message": news_report["summary"], "data": news_report})

            # 4. Fundamentals analysis (free via yfinance)
            current_stage = "fundamentals"
            fundamentals_report = None
            if settings.ENABLE_FUNDAMENTALS_AGENT:
                fundamentals_data = fundamentals_agent.get_fundamentals(symbol)
                fundamentals_report = fundamentals_agent.analyze_fundamentals(symbol, fundamentals_data)
                # Distinguish provider unavailability (Yahoo 429 etc.) from an
                # agent/LLM failure — both are recorded honestly.
                if fundamentals_data.get("error"):
                    status["fundamentals"] = "UNAVAILABLE"
                else:
                    status["fundamentals"] = "ERROR" if fundamentals_report.get("error") else "OK"
                _log_event({"agent": "fundamentals", "symbol": symbol, "level": _report_level(fundamentals_report),
                            "message": fundamentals_report["summary"], "data": fundamentals_report})

            # 5. Bull vs Bear debate
            current_stage = "debate"
            debate_report = None
            if settings.ENABLE_DEBATE:
                debate_report = debate_agent.run_debate(symbol, tech_report, news_report, fundamentals_report)
                status["debate"] = "ERROR" if debate_report.get("error") else "OK"
                _log_event({"agent": "debate", "symbol": symbol, "level": _report_level(debate_report),
                            "message": f"Bull({debate_report['bull_strength']}) vs "
                                       f"Bear({debate_report['bear_strength']}), edge={debate_report['edge']}",
                            "data": debate_report})

            # 6. Risk assessment (only meaningful ahead of a potential BUY,
            # but we compute it every cycle so the CIO always has it)
            current_stage = "risk"
            existing_position = positions_by_symbol.get(symbol)
            risk_report = risk_agent.assess_risk(symbol, "buy", account_summary, existing_position)
            status["risk"] = "ERROR" if risk_report.get("error") else "OK"
            _log_event({"agent": "risk", "symbol": symbol, "level": _report_level(risk_report),
                        "message": risk_report["reasoning"], "data": risk_report})

            # 7. Executive decision, with fundamentals + debate + memory context
            current_stage = "cio"
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
            status["cio"] = "ERROR" if decision_report.get("error") else "OK"
            _log_event({"agent": "cio", "symbol": symbol, "level": _report_level(decision_report),
                        "message": f"{decision_report['decision']}: {decision_report['reasoning']}",
                        "data": decision_report})

            cycle_record["decisions"].append({
                "symbol": symbol,
                "decision": decision_report["decision"],
                "confidence": decision_report.get("confidence"),
                "notional_usd": decision_report.get("notional_usd"),
            })

            # 8. Execute if actionable
            current_stage = "execution"
            trade_result = None
            decision = decision_report["decision"]
            order_attempted = False
            if decision == "BUY" and decision_report["notional_usd"] > 0:
                order_attempted = True
                trade_result = alpaca_service.execute_order(
                    symbol, "buy", notional_usd=decision_report["notional_usd"]
                )
            elif decision == "SELL" and existing_position:
                order_attempted = True
                trade_result = alpaca_service.execute_order(
                    symbol, "sell", qty=existing_position["qty"]
                )

            if order_attempted:
                status["execution"] = "OK" if trade_result.get("success") else "ERROR"
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
                cycle_record["orders"].append({
                    "symbol": symbol,
                    "side": decision.lower(),
                    "order_id": trade_result.get("order_id"),
                    "status": trade_result.get("status"),
                    "success": bool(trade_result.get("success")),
                    "notional_usd": trade_result.get("notional_usd"),
                    "qty": trade_result.get("qty"),
                    "error": trade_result.get("error"),
                })
            # else: no order required (HOLD / no position to sell) -> stays SKIPPED

            # 9. Record this decision to persistent memory for future learning
            current_stage = "memory"
            if settings.ENABLE_MEMORY and decision == "BUY" and trade_result and trade_result.get("success"):
                entry_price = indicators.get("latest_close")
                memory_service.record_decision(
                    symbol, decision_report, tech_report, news_report,
                    fundamentals_report or {}, debate_report or {}, entry_price,
                )
                status["memory"] = "OK"

            cycle_summary["symbols_processed"].append(symbol)

        except Exception as e:
            logger.error(f"Cycle error for {symbol} at stage '{current_stage}': {e}")
            status[current_stage] = "ERROR"
            _log_event({"agent": "system", "symbol": symbol, "level": "ERROR",
                        "message": f"Pipeline aborted at stage '{current_stage}': {e}"})
            cycle_summary["errors"].append(f"{symbol} ({current_stage}): {e}")

    _previous_positions.clear()
    _previous_positions.update(positions_by_symbol)

    cycle_status = _compute_cycle_status(cycle_record, cycle_summary)
    bot_state["last_cycle_at"] = datetime.now(timezone.utc).isoformat()
    bot_state["last_cycle_status"] = cycle_status
    _finish_cycle_record(cycle_status)
    logger.info(
        f"Cycle complete. Status: {cycle_status}. "
        f"Processed: {cycle_summary['symbols_processed']}. Errors: {cycle_summary['errors']}"
    )
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


@app.get("/api/cycles")
async def api_cycles():
    """Recent autonomous cycle executions (real records, most recent first).
    In-memory only: cleared on server restart, same as the live log feed."""
    return JSONResponse({"cycles": list(cycle_history)})


@app.get("/api/orders")
async def api_orders(limit: int = 50):
    """Recent broker orders with a caller-controlled limit (paper account)."""
    orders = alpaca_service.get_recent_orders(limit=max(1, min(limit, 500)))
    return JSONResponse({"orders": orders["orders"], "error": orders["error"]})


@app.get("/api/history")
async def api_history(period: str = "1M"):
    """Portfolio equity history for the chart, with a period passthrough."""
    history = alpaca_service.get_portfolio_history(period=period)
    return JSONResponse({"points": history["points"], "period": period, "error": history["error"]})


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
        "trading_mode": "PAPER",  # this application is hardcoded to paper trading
        "alpaca_data_feed": settings.ALPACA_DATA_FEED,
        "memory_db_path": settings.MEMORY_DB_PATH,
        "agent_accuracy_lookback": settings.AGENT_ACCURACY_LOOKBACK,
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
