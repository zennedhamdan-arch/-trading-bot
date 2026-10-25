"""
operator_tools/system.py — whole-system state tools (read-only).
"""

from services import health_service
from services.operator.memory import get_system_events
from services.operator.operator_tools._registry import tool
from services.operator.schemas import sanitize


def _latest_cycle():
    """The most recent trading-cycle record (main holds it in memory)."""
    import main
    return (main.cycle_history[0] if main.cycle_history else None)


@tool("get_system_health", "system",
      "Live health of the whole system: overall status (HEALTHY/DEGRADED/"
      "OFFLINE) with reasons, all startup check rows (Alpaca, market data, "
      "indicators, realtime, fundamentals, news intelligence, every LLM "
      "provider), market clock and realtime state.",
      label="Checking system health")
def _get_system_health(args):
    return sanitize(health_service.live_health())


@tool("get_system_status", "system",
      "Bot operating status: running flag, last cycle time/status/trigger, "
      "cycle interval, trade universe, trading mode (paper-only) and the "
      "current overall health verdict.",
      label="Checking system status")
def _get_system_status(args):
    import main
    from config import settings
    overall = health_service.overall_status(live=True)
    cycle = _latest_cycle()
    return sanitize({
        "running": bool(main.bot_state.get("running")),
        "last_cycle_at": main.bot_state.get("last_cycle_at"),
        "last_cycle_status": main.bot_state.get("last_cycle_status"),
        "last_cycle_triggered_by": main.bot_state.get("last_cycle_triggered_by"),
        "cycle_interval_minutes": settings.CYCLE_INTERVAL_MINUTES,
        "trade_universe": settings.TRADE_UNIVERSE,
        "trading_mode": "PAPER" if settings.PAPER_TRADING_ONLY else "PAPER",
        "paper_trading_only": True,
        "overall": overall,
        "last_cycle_summary": ({
            "id": cycle.get("id"), "status": cycle.get("status"),
            "triggered_by": cycle.get("triggered_by"),
            "symbols_processed": cycle.get("symbols_processed"),
            "orders": len(cycle.get("orders") or []),
            "errors": len(cycle.get("errors") or []),
        } if cycle else None),
    })


@tool("get_recent_errors", "system",
      "Most recent ERROR entries from the running process (agent, symbol, "
      "message, timestamp) plus errors recorded in the latest cycle.",
      args={"limit": {"type": "integer", "description": "max entries (default 20)"}},
      label="Checking recent errors")
def _get_recent_errors(args):
    import main
    limit = min(int(args.get("limit") or 20), 60)
    errors = [{"agent": e.get("agent"), "symbol": e.get("symbol"),
               "level": e.get("level"), "message": e.get("message"),
               "timestamp": e.get("timestamp")}
              for e in list(main.agent_logs) if e.get("level") == "ERROR"][:limit]
    cycle = _latest_cycle()
    cycle_errors = (cycle.get("errors") or [])[:limit] if cycle else []
    return {"errors": errors, "count": len(errors),
            "latest_cycle_errors": cycle_errors,
            "latest_cycle_error_count": len(cycle_errors)}


@tool("get_recent_warnings", "system",
      "Most recent WARNING entries from the running process.",
      args={"limit": {"type": "integer", "description": "max entries (default 20)"}},
      label="Checking recent warnings")
def _get_recent_warnings(args):
    import main
    limit = min(int(args.get("limit") or 20), 60)
    warnings = [{"agent": e.get("agent"), "symbol": e.get("symbol"),
                 "message": e.get("message"), "timestamp": e.get("timestamp")}
                for e in list(main.agent_logs) if e.get("level") == "WARNING"][:limit]
    return {"warnings": warnings, "count": len(warnings)}


@tool("get_worker_status", "system",
      "Background worker status: the news intelligence worker (last refresh, "
      "articles fetched/deduped/analyzed, cache counts) and scheduler jobs.",
      label="Checking background workers")
def _get_worker_status(args):
    from services import news_worker
    return sanitize(news_worker.stats())


@tool("get_service_status", "system",
      "Status of one named service/component with its dependencies. "
      "Names: alpaca_account, market_data, indicators, realtime, fundamentals, "
      "news_intelligence, memory, unorouter, groq, nvidia, gemini, openrouter.",
      args={"service_name": {"type": "string", "description": "service name",
                             "required": True}},
      label="Checking service status")
def _get_service_status(args):
    from services.operator.operator_tools.agents import component_status
    name = str(args.get("service_name") or "").strip().lower()
    result = component_status(name)
    if result is None:
        return {"error": f"unknown service '{name}'",
                "known_services": ["alpaca_account", "market_data", "indicators",
                                   "realtime", "fundamentals", "news_intelligence",
                                   "memory", "unorouter", "groq", "nvidia",
                                   "gemini", "openrouter"]}
    return result


@tool("get_application_version", "system",
      "Application identity and version: the git commit the server is running, "
      "process start time and uptime.", label="Checking version")
def _get_application_version(args):
    import time as _time
    from pathlib import Path
    commit = "unknown"
    try:
        git = Path(__file__).resolve().parents[3] / ".git"
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref_file = git / head.split(" ", 1)[1]
            commit = ref_file.read_text().strip()
        else:
            commit = head
    except Exception:  # noqa: BLE001 — version is best-effort
        pass
    import main
    return {"application": "ai-trader (paper trading only)",
            "git_commit": commit[:12], "branch": "arena/01a08520-trading-bot",
            "trading_mode": "PAPER"}


@tool("get_recent_system_events", "system",
      "Recent persisted system/diagnostic events (operator-recorded findings, "
      "degradations, recoveries) with severity, component and correlation id.",
      args={"limit": {"type": "integer", "description": "max events (default 20)"}},
      label="Checking system events")
def _get_recent_system_events(args):
    limit = min(int(args.get("limit") or 20), 100)
    events = get_system_events(limit=limit)
    return {"events": events, "count": len(events)}
