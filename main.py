"""
main.py

FastAPI server for the 24/7 Automated Multi-Agent AI Trading Bot.

Serves the static dashboard, exposes REST endpoints for portfolio data,
agent deliberation logs, and bot control, and runs an APScheduler
background job that executes the full agent decision cycle on a
fixed interval for every symbol in the trade universe.

Architecture (reliability-focused; AI never does basic financial math):

    MARKET DATA (Alpaca primary: bars/quotes/snapshots/news/account)
        |
    NORMALIZED DATA LAYER (market_data_service, per-symbol data_quality)
        |
    DETERMINISTIC ANALYTICS (RSI/SMA/EMA/MACD/ATR/volatility/drawdown/
        returns, portfolio exposure, position sizing -- all in Python)
        |
    AI REASONING (technical/news/fundamentals interpretation, bull-vs-bear
        debate, CIO synthesis -- via the centralized LLM router)
        |
    VALIDATION + RISK GATE (risk_gate: deterministic order validation,
        position limits, buying power -- AI cannot bypass it)
        |
    ALPACA PAPER EXECUTION (paper trading only, always)

Per symbol, per cycle: market data bundle -> technical -> news ->
fundamentals (DATA_UNAVAILABLE when no provider is configured) ->
debate -> risk (deterministic gate + optional LLM reasoning) -> CIO
decision -> validated paper execution -> memory feedback.

Run with:  python main.py
Dashboard: http://localhost:8000
"""

import asyncio
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
from services import (alpaca_service, memory_service, llm_service,
                      market_data_service, risk_gate, health_service,
                      realtime_service, fundamentals_service, evidence as evidence_service)
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

# Data-provider outcome extras for the most recent cycle's provider_results
# (market data / fundamentals summaries), set by run_trading_cycle.
_provider_extras: dict = {}


def _log_event(entry: dict):
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    if _active_cycle is not None and entry.get("level") == "WARNING":
        _active_cycle["warnings"] += 1
    agent_logs.appendleft(entry)


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


def _stage_status(report: dict) -> str:
    """Maps an agent report to its pipeline-stage status.

    OK          the agent produced a valid result
    UNAVAILABLE the LLM provider was unavailable: no API key configured,
                quota exhausted (PROVIDER_QUOTA_EXCEEDED), model not found,
                auth failure, or network unreachable
    ERROR       any other failure (provider error, unparseable response)
    """
    if not report:
        return "SKIPPED"
    if not report.get("error"):
        return "OK"
    llm_status = report.get("llm_status")
    if llm_status in ("NOT_CONFIGURED", "PROVIDER_QUOTA_EXCEEDED", "MODEL_NOT_FOUND",
                      "AUTH_ERROR", "NETWORK_ERROR"):
        return "UNAVAILABLE"
    return "ERROR"


def _llm_error_type(llm_status) -> str:
    """Compact error type for structured cycle errors (e.g. the provider
    keeps PROVIDER_QUOTA_EXCEEDED, the cycle error carries QUOTA_EXCEEDED)."""
    if llm_status == "PROVIDER_QUOTA_EXCEEDED":
        return "QUOTA_EXCEEDED"
    return str(llm_status or "ERROR")


def _compute_cycle_status(cycle_record: dict, cycle_summary: dict) -> str:
    """Honest cycle status.

    ERROR        - the cycle aborted (e.g. broker unreachable) or every
                   symbol failed outright.
    PARTIAL_ERROR- any pipeline stage across any symbol errored or its data
                   source / LLM provider was unavailable (including quota
                   exhaustion and missing models). The scheduler's "job
                   executed successfully" NEVER implies cycle OK.
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
    universe: MARKET DATA -> NORMALIZED DATA -> DETERMINISTIC ANALYTICS ->
    AI REASONING -> VALIDATION -> RISK GATE -> PAPER EXECUTION.

    Every stage for every symbol is tracked in cycle_record["agent_status"]
    (OK / ERROR / UNAVAILABLE / SKIPPED); agent_results summarizes per-agent
    outcomes (e.g. technical: 5/5, news: 0/5); provider_results summarizes
    per-provider LLM usage/circuits plus data-provider outcomes; and EVERY
    failed operation is represented in cycle_record["errors"] as
    {"provider", "type", "agent", "symbol", "message"} — a PARTIAL_ERROR
    cycle never has an empty error list again.
    """
    global _active_cycle, _provider_extras
    _provider_extras = {}
    cycle_summary = {
        "triggered_by": triggered_by,
        "symbols_processed": [],
        "errors": [],  # structured: {provider, type, agent, symbol, message}
    }
    logger.info(f"Starting trading cycle (triggered by: {triggered_by})")

    # Per-cycle LLM usage accounting (provider/model/call counts).
    llm_service.reset_cycle_usage()

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
        "agent_results": None,
        "provider_results": None,
        "llm_usage": None,
        "warnings": 0,
        "errors": [],
    }
    _t0 = time.monotonic()

    def _add_error(provider: str, err_type: str, agent: str, symbol, message: str):
        """Structured, non-lossy error aggregation: every failed operation
        lands in the cycle summary, keyed by provider/type/agent/symbol."""
        entry = {
            "provider": provider,
            "type": err_type,
            "agent": agent,
            "symbol": symbol,
            "message": str(message)[:500],
        }
        cycle_summary["errors"].append(entry)

    def _finish_cycle_record(status: str):
        cycle_record["finished_at"] = datetime.now(timezone.utc).isoformat()
        cycle_record["duration_s"] = round(time.monotonic() - _t0, 1)
        cycle_record["status"] = status
        cycle_record["symbols_processed"] = cycle_summary["symbols_processed"]
        cycle_record["errors"] = cycle_summary["errors"]
        cycle_record["warnings"] = _active_cycle["warnings"] if _active_cycle else 0
        cycle_record["llm_usage"] = llm_service.cycle_usage()
        cycle_record["agent_results"] = _compute_agent_results(cycle_record)
        cycle_record["provider_results"] = _compute_provider_results(cycle_record)
        cycle_history.appendleft(cycle_record)

    account_summary = alpaca_service.get_account_summary()
    if account_summary.get("error"):
        msg = f"Skipping cycle: could not fetch account summary ({account_summary['error']})"
        logger.error(msg)
        _log_event({"agent": "system", "symbol": None, "level": "ERROR", "message": msg})
        cycle_summary["errors"].append({
            "provider": "alpaca", "type": "PROVIDER_ERROR", "agent": "system",
            "symbol": None, "message": msg,
        })
        bot_state["last_cycle_status"] = "ERROR"
        bot_state["last_cycle_at"] = datetime.now(timezone.utc).isoformat()
        _finish_cycle_record("ERROR")
        return _cycle_summary_payload(cycle_summary, "ERROR")

    positions_data = alpaca_service.get_open_positions()
    if positions_data.get("error"):
        _add_error("alpaca", "PROVIDER_ERROR", "system", None,
                   f"Could not fetch open positions: {positions_data['error']}")
    positions_by_symbol = {p["symbol"]: p for p in positions_data.get("positions", [])}

    # Learning loop: check what closed since last cycle before doing anything else.
    _detect_closed_positions(positions_by_symbol)
    agent_weights = _get_agent_weights()

    fundamentals_provider_results = {"symbols_ok": 0, "symbols_unavailable": 0}
    market_data_results = {"symbols_ok": 0, "symbols_unavailable": 0, "feed": settings.ALPACA_DATA_FEED}

    for symbol in settings.TRADE_UNIVERSE:
        status = _new_symbol_status()
        cycle_record["agent_status"][symbol] = status
        current_stage = "market_data"
        try:
            # ------------------------------------------------------------
            # 1. MARKET DATA + NORMALIZATION + DETERMINISTIC ANALYTICS
            #    (one normalized bundle per symbol; agents never call
            #    external data APIs directly)
            # ------------------------------------------------------------
            data = market_data_service.get_symbol_data(symbol)
            indicators = data["indicators"]
            dq = data["data_quality"]

            if dq["bars"] == "OK":
                if dq["price"] == "OK":
                    status["market_data"] = "OK"
                    market_data_results["symbols_ok"] += 1
                else:
                    status["market_data"] = "OK"  # bars OK; price fell back or absent
                    market_data_results["symbols_ok"] += 1
                    _add_error("alpaca", "DATA_UNAVAILABLE", "market_data", symbol,
                               data.get("price_error") or "price source unavailable")
            else:
                bars_error = data.get("bars_error") or "no bar data"
                if "SUBSCRIPTION_FEED_UNAVAILABLE" in str(bars_error):
                    # Configured feed not permitted by the subscription —
                    # honest UNAVAILABLE, never a silent feed switch.
                    status["market_data"] = "UNAVAILABLE"
                    _add_error("alpaca", "SUBSCRIPTION_FEED_UNAVAILABLE", "market_data",
                               symbol, bars_error)
                else:
                    status["market_data"] = "ERROR"
                    _add_error("alpaca", "DATA_ERROR", "market_data", symbol, bars_error)
                market_data_results["symbols_unavailable"] += 1

            # ------------------------------------------------------------
            # 2. AI REASONING: technical interpretation (evidence is
            #    pre-computed deterministically)
            # ------------------------------------------------------------
            current_stage = "technical"
            tech_report = tech_agent.analyze_technicals(symbol, indicators)
            status["technical"] = _stage_status(tech_report)
            if tech_report.get("error"):
                _add_error(tech_report.get("provider", "llm"), _llm_error_type(tech_report.get("llm_status")),
                           "technical", symbol, tech_report["error"])
            _log_event({"agent": "technical", "symbol": symbol, "level": _report_level(tech_report),
                        "message": tech_report["summary"], "data": tech_report})

            # 3. News/sentiment interpretation
            current_stage = "news"
            headlines = data.get("news", [])
            if data.get("news_error"):
                # News DATA source unavailable — the agent still runs with
                # what it has; the data failure is recorded honestly.
                _add_error("alpaca", "DATA_UNAVAILABLE", "news", symbol, data["news_error"])
                status["news"] = "UNAVAILABLE"
            news_report = news_agent.analyze_news(symbol, headlines)
            if status["news"] != "UNAVAILABLE":
                status["news"] = _stage_status(news_report)
            if news_report.get("error"):
                _add_error(news_report.get("provider", "llm"), _llm_error_type(news_report.get("llm_status")),
                           "news", symbol, news_report["error"])
            _log_event({"agent": "news", "symbol": symbol, "level": _report_level(news_report),
                        "message": news_report["summary"], "data": news_report})

            # 4. Fundamentals: normalized provider data + LLM interpretation.
            #    No provider configured -> SKIPPED (configured-off, visible
            #    in provider_results); provider failure -> UNAVAILABLE.
            current_stage = "fundamentals"
            fundamentals_report = None
            if settings.ENABLE_FUNDAMENTALS_AGENT:
                fundamentals_data = data["fundamentals"]
                # Always run the agent: it fail-safes internally (no LLM call
                # when data is unavailable) so the debate/CIO still receive an
                # explicit "fundamentals unavailable" report.
                fundamentals_report = fundamentals_agent.analyze_fundamentals(symbol, fundamentals_data)
                f_status = fundamentals_data.get("status")
                if f_status == "OK":
                    fundamentals_provider_results["symbols_ok"] += 1
                    status["fundamentals"] = _stage_status(fundamentals_report)
                    if fundamentals_report.get("error"):
                        _add_error(fundamentals_report.get("provider", "llm"),
                                   _llm_error_type(fundamentals_report.get("llm_status")),
                                   "fundamentals", symbol, fundamentals_report["error"])
                elif fundamentals_data.get("reason") == "NO_PROVIDER_CONFIGURED":
                    # Deliberately unconfigured (FUNDAMENTALS_PROVIDER=none):
                    # visible in provider_results, does not degrade the cycle.
                    fundamentals_provider_results["symbols_unavailable"] += 1
                    status["fundamentals"] = "SKIPPED"
                else:
                    fundamentals_provider_results["symbols_unavailable"] += 1
                    status["fundamentals"] = "UNAVAILABLE"
                    _add_error(f"fundamentals:{fundamentals_data.get('provider', 'unknown')}",
                               "DATA_UNAVAILABLE", "fundamentals", symbol,
                               fundamentals_data.get("reason") or fundamentals_data.get("error") or f_status)
                _log_event({"agent": "fundamentals", "symbol": symbol, "level": _report_level(fundamentals_report),
                            "message": fundamentals_report["summary"], "data": fundamentals_report})

            # 5. Evidence quality snapshot (which evidence is actually
            #    available) — computed BEFORE the debate so the debate
            #    knows exactly what is real and what is missing, and
            #    re-computed after it for the risk/CIO/execution stages.
            current_stage = "debate"
            fundamentals_evidence_report = (
                fundamentals_report if settings.ENABLE_FUNDAMENTALS_AGENT else None
            )
            analyst_evidence = evidence_service.assess(
                tech_report, news_report,
                fundamentals_evidence_report,
                debate_report=None,
                indicators=indicators,
            )

            debate_report = None
            if settings.ENABLE_DEBATE:
                debate_report = debate_agent.run_debate(
                    symbol, tech_report, news_report, fundamentals_report,
                    evidence=analyst_evidence,
                )
                status["debate"] = _stage_status(debate_report)
                if debate_report.get("error"):
                    _add_error(debate_report.get("provider", "llm"), _llm_error_type(debate_report.get("llm_status")),
                               "debate", symbol, debate_report["error"])
                _log_event({"agent": "debate", "symbol": symbol, "level": _report_level(debate_report),
                            "message": f"Bull({debate_report['bull_strength']}) vs "
                                       f"Bear({debate_report['bear_strength']}), edge={debate_report['edge']}",
                            "data": debate_report})

            # Final evidence snapshot including the debate outcome.
            evidence = evidence_service.assess(
                tech_report, news_report,
                fundamentals_evidence_report,
                debate_report=debate_report if settings.ENABLE_DEBATE else None,
                indicators=indicators,
            )

            # 6. Risk: deterministic PORTFOLIO-RISK gate + optional LLM
            #    reasoning (informed about decision quality — a separate
            #    prerequisite enforced before execution below).
            current_stage = "risk"
            existing_position = positions_by_symbol.get(symbol)
            risk_report = risk_agent.assess_risk(
                symbol, "buy", account_summary, existing_position, indicators,
                evidence=evidence,
            )
            status["risk"] = _stage_status(risk_report)
            if risk_report.get("error"):
                _add_error(risk_report.get("provider", "llm"), _llm_error_type(risk_report.get("llm_status")),
                           "risk", symbol, risk_report["error"])
            _log_event({"agent": "risk", "symbol": symbol, "level": _report_level(risk_report),
                        "message": risk_report["reasoning"], "data": risk_report})

            # 7. Executive decision (LLM; HOLD fail-safe on any failure)
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
                evidence=evidence,
            )
            status["cio"] = _stage_status(decision_report)
            if decision_report.get("error"):
                _add_error(decision_report.get("provider", "llm"), _llm_error_type(decision_report.get("llm_status")),
                           "cio", symbol, decision_report["error"])
            _log_event({"agent": "cio", "symbol": symbol, "level": _report_level(decision_report),
                        "message": f"{decision_report['decision']}: {decision_report['reasoning']}",
                        "data": decision_report})

            cycle_record["decisions"].append({
                "symbol": symbol,
                "decision": decision_report["decision"],
                "confidence": decision_report.get("confidence"),
                "notional_usd": decision_report.get("notional_usd"),
                "reasoning": decision_report.get("reasoning", ""),
                "evidence_quality": evidence.get("quality"),
                "evidence": {
                    "agents": {name: entry["state"]
                               for name, entry in (evidence.get("agents") or {}).items()},
                    "available": evidence.get("available"),
                    "missing": evidence.get("missing"),
                    "staleness": evidence.get("staleness"),
                },
            })
            cycle_record.setdefault("evidence", {})[symbol] = {
                "quality": evidence.get("quality"),
                "agents": {name: entry["state"]
                           for name, entry in (evidence.get("agents") or {}).items()},
                "staleness": evidence.get("staleness"),
            }

            # ------------------------------------------------------------
            # 8. DECISION-QUALITY PREREQUISITE + VALIDATION + RISK GATE +
            #    PAPER EXECUTION.
            #    Deterministic final gates, in order:
            #      a. evidence quality — a trade may never be justified
            #         solely because the portfolio-risk cap permits the
            #         notional; missing/stale critical evidence => HOLD.
            #      b. order validation: action, symbol, quantity, buying
            #         power, position limits, risk constraints. AI cannot
            #         bypass either.
            # ------------------------------------------------------------
            current_stage = "execution"
            trade_result = None
            decision = decision_report["decision"]
            order_attempted = False
            validation = None

            # a. DECISION-QUALITY PREREQUISITE: portfolio risk (buying
            #    power, position caps) and decision quality are separate.
            #    With INSUFFICIENT evidence quality a BUY is held even if
            #    the deterministic risk gate would permit the notional.
            if decision == "BUY" and not evidence.get("trade_allowed", True):
                reason_txt = "; ".join(evidence.get("reasons") or ["evidence quality insufficient"])
                decision = "HOLD"
                decision_report["decision"] = "HOLD"
                decision_report["notional_usd"] = 0.0
                if cycle_record["decisions"]:
                    cycle_record["decisions"][-1]["decision"] = "HOLD"
                    cycle_record["decisions"][-1]["notional_usd"] = 0.0
                    cycle_record["decisions"][-1]["blocked_reason"] = "EVIDENCE_INSUFFICIENT"
                status["execution"] = "UNAVAILABLE"
                _add_error("system", "EVIDENCE_INSUFFICIENT", "execution", symbol,
                           f"BUY held — decision quality INSUFFICIENT: {reason_txt}")
                _log_event({
                    "agent": "execution", "symbol": symbol, "level": "WARNING",
                    "message": (
                        f"BUY held: evidence quality INSUFFICIENT ({reason_txt}). "
                        f"The portfolio-risk cap alone never justifies a trade."
                    ),
                    "data": {"blocked_reason": "EVIDENCE_INSUFFICIENT", "evidence": evidence},
                })

            if decision == "BUY" and decision_report["notional_usd"] > 0:
                validation = risk_gate.validate_order(
                    symbol, "buy", account_summary, positions_by_symbol,
                    notional_usd=decision_report["notional_usd"],
                    trade_universe=settings.TRADE_UNIVERSE,
                )
                if not validation["valid"]:
                    # Unsafe/impossible order blocked by the deterministic
                    # gate: recorded, never executed, decision downgraded.
                    decision = "HOLD"
                    cycle_record["decisions"][-1]["decision"] = "HOLD"
                    status["execution"] = "ERROR"
                    _add_error("system", "VALIDATION_FAILED", "execution", symbol,
                               validation["reason"])
                    _log_event({"agent": "execution", "symbol": symbol, "level": "ERROR",
                                "message": f"Order blocked by validation gate: {validation['reason']}",
                                "data": validation})
                else:
                    order_attempted = True
                    trade_result = alpaca_service.execute_order(
                        symbol, "buy", notional_usd=decision_report["notional_usd"]
                    )
            elif decision == "SELL" and existing_position:
                validation = risk_gate.validate_order(
                    symbol, "sell", account_summary, positions_by_symbol,
                    qty=existing_position["qty"], trade_universe=settings.TRADE_UNIVERSE,
                )
                if not validation["valid"]:
                    decision = "HOLD"
                    cycle_record["decisions"][-1]["decision"] = "HOLD"
                    status["execution"] = "ERROR"
                    _add_error("system", "VALIDATION_FAILED", "execution", symbol,
                               validation["reason"])
                    _log_event({"agent": "execution", "symbol": symbol, "level": "ERROR",
                                "message": f"Order blocked by validation gate: {validation['reason']}",
                                "data": validation})
                else:
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
                if not trade_result.get("success"):
                    _add_error("alpaca", "ORDER_REJECTED", "execution", symbol,
                               trade_result.get("error") or "order failed")
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
            # else: no order required (HOLD / no position / blocked) -> SKIPPED

            # 9. Record the decision to persistent memory for future learning
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
            _add_error("system", "PIPELINE_ERROR", current_stage, symbol, str(e))

    _previous_positions.clear()
    _previous_positions.update(positions_by_symbol)

    # Data-provider outcomes for this cycle's provider_results — must be set
    # before the record is finalized (provider_results reads them).
    # (_provider_extras is declared global at the top of this function.)
    _provider_extras = {
        "market_data": market_data_results,
        "fundamentals": {
            "provider": settings.FUNDAMENTALS_PROVIDER,
            **fundamentals_provider_results,
        },
    }

    cycle_status = _compute_cycle_status(cycle_record, cycle_summary)
    bot_state["last_cycle_at"] = datetime.now(timezone.utc).isoformat()
    bot_state["last_cycle_status"] = cycle_status
    _finish_cycle_record(cycle_status)

    logger.info(
        f"Cycle complete. Status: {cycle_status}. "
        f"Processed: {cycle_summary['symbols_processed']}. Errors: {len(cycle_summary['errors'])}."
    )
    return _cycle_summary_payload(cycle_summary, cycle_status)


def _cycle_summary_payload(cycle_summary: dict, cycle_status: str) -> dict:
    """The run-now/API view of a finished cycle: status, processed symbols,
    per-agent results, per-provider results and every structured error."""
    record = cycle_history[0] if cycle_history else {}
    return {
        "triggered_by": cycle_summary["triggered_by"],
        "status": cycle_status,
        "processed_symbols": cycle_summary["symbols_processed"],
        "symbols_processed": cycle_summary["symbols_processed"],  # legacy key
        "agent_results": record.get("agent_results"),
        "provider_results": record.get("provider_results"),
        "llm_usage": record.get("llm_usage"),
        "errors": cycle_summary["errors"],
    }


def _compute_agent_results(cycle_record: dict) -> dict:
    """Per-agent outcome counts across symbols, e.g. technical: 5/5,
    news: 0/5 — the honest summary of who succeeded this cycle."""
    agent_status = cycle_record.get("agent_status") or {}
    results = {}
    for stage in PIPELINE_STAGES:
        counts = {"ok": 0, "unavailable": 0, "error": 0, "skipped": 0, "total": 0}
        for symbol_status in agent_status.values():
            value = symbol_status.get(stage, "SKIPPED")
            counts["total"] += 1
            if value == "OK":
                counts["ok"] += 1
            elif value == "UNAVAILABLE":
                counts["unavailable"] += 1
            elif value == "ERROR":
                counts["error"] += 1
            else:
                counts["skipped"] += 1
        results[stage] = counts
    return results


def _compute_provider_results(cycle_record: dict) -> dict:
    """Per-provider outcomes: LLM usage + circuit states + data providers
    (Alpaca market data, fundamentals provider)."""
    extras = _provider_extras or {}
    return {
        "llm": cycle_record.get("llm_usage") or llm_service.cycle_usage(),
        "llm_states": llm_service.provider_states(),
        "market_data": extras.get("market_data", {"feed": settings.ALPACA_DATA_FEED}),
        "fundamentals": extras.get("fundamentals", {"provider": settings.FUNDAMENTALS_PROVIDER}),
    }


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

    # Startup health check: Alpaca (credentials/account), market data
    # (configured feed actually permitted), fundamentals provider, and every
    # LLM provider (credentials / model availability / connectivity). Logs
    # an aligned table and exposes it via /api/health. Never crashes boot.
    try:
        health_service.run_startup_checks()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Startup health check failed (non-fatal): {exc}")

    # Real-time layer (Alpaca WebSockets). Each SDK stream runs in its own
    # dedicated thread — the SDK's run() owns its event loop, so it must
    # never be awaited inside the FastAPI loop. Ticks are bridged to this
    # loop through a thread-safe queue. Purely observational: ticks never
    # trigger LLM calls.
    try:
        await realtime_service.start_async()
    except Exception as exc:  # noqa: BLE001 — optional layer, never fatal
        logger.error(f"Real-time layer failed to start (non-fatal): {exc}")

    # Verify every configured LLM model id against each provider's LIVE
    # catalog (Groq/OpenRouter/Gemini). Providers without keys are skipped
    # (already reported by settings.validate). Missing models are surfaced
    # as warnings — they will fail per-request with MODEL_NOT_FOUND, never
    # silently.
    validation = llm_service.validate_models()
    for provider, entry in validation["providers"].items():
        for missing in entry.get("missing", []):
            msg = f"LLM model validation ({provider}): {missing} is not in the provider's current model list."
            logger.warning(msg)
            _log_event({"agent": "system", "symbol": None, "level": "WARNING", "message": msg})
        error = entry.get("error")
        if error and "not configured" not in error:
            msg = f"LLM model validation ({provider}): {error}"
            logger.warning(msg)
            _log_event({"agent": "system", "symbol": None, "level": "WARNING", "message": msg})

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
    try:
        await realtime_service.shutdown()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Real-time layer shutdown issue (non-fatal): {exc}")
    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped. Real-time layer stopped.")


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
    """Portfolio equity history for the chart, with a period passthrough.
    Invalid points (null/zero equity, epoch-0 timestamps) are dropped and
    counted — missing history is never represented as $0."""
    history = alpaca_service.get_portfolio_history(period=period)
    return JSONResponse({
        "points": history["points"],
        "period": period,
        "error": history["error"],
        "dropped_invalid_points": history.get("dropped_invalid_points", 0),
    })


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


@app.get("/api/health")
async def api_health():
    """Live system health: startup checks, provider circuit states,
    market clock, realtime status, fundamentals provider. An HTTP 200 here
    does NOT mean the trading system is healthy — read the statuses."""
    return JSONResponse(health_service.live_health())


@app.get("/api/realtime")
async def api_realtime():
    """Normalized real-time market state (Alpaca WebSockets): latest
    trades/quotes/minute bars per symbol plus the news stream. Ticks never
    trigger LLM calls."""
    payload = realtime_service.get_state()
    payload["market_clock"] = alpaca_service.get_clock()
    return JSONResponse(payload)


@app.get("/api/config")
async def api_config():
    """Non-sensitive config info for the dashboard to display (no secrets).
    Includes the LLM routing table (which provider/model each agent uses),
    live model-validation results and per-provider quota posture."""
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
        "llm_routes": llm_service.llm_routes(),
        "llm_model_validation": llm_service.last_validation(),
        "llm_quota": llm_service.quota_state(),
        "llm_provider_states": llm_service.provider_states(),
        "fundamentals": fundamentals_service.provider_config(),
        "market_data": market_data_service.feed_config(),
        "realtime_enabled": settings.REALTIME_ENABLED,
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
