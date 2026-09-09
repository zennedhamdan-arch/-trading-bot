"""
services/health_service.py

Startup + live health checks for every dependency. At startup the bot
verifies (and logs, and exposes via /api/health):

  ALPACA       credentials / account / paper mode
  MARKET DATA  configured feed actually permitted by the subscription
  FUNDAMENTALS configured provider (or honest DATA_UNAVAILABLE)
  GROQ         credentials / model availability / circuit state
  NVIDIA       credentials / model availability (optional — NOT_CONFIGURED ok)
  GEMINI       credentials / model availability / quota state
  OPENROUTER   credentials / model availability (optional)

Rules:
  - A failing check NEVER prevents the app from booting.
  - Secrets are never exposed — only key PRESENCE and error reasons.
  - HTTP 200 from an API endpoint does not mean the system is healthy;
    these checks say what actually works.
"""

import logging
from datetime import datetime, timezone

from config import settings
from services import alpaca_service, fundamentals_service, llm_service, realtime_service

logger = logging.getLogger("health_service")

_startup_report = None  # cached result of run_startup_checks()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _probe_market_data() -> dict:
    """Probes the CONFIGURED data feed with one light bars request.
    Detects subscription limitations honestly (SUBSCRIPTION_FEED_UNAVAILABLE)
    — never switches feeds silently."""
    probe_symbol = (settings.TRADE_UNIVERSE or ["SPY"])[0]
    indicators = alpaca_service.get_indicators(probe_symbol, lookback_days=5)
    if indicators.get("error"):
        err = str(indicators["error"])
        if "SUBSCRIPTION_FEED_UNAVAILABLE" in err:
            return {"status": "SUBSCRIPTION_FEED_UNAVAILABLE", "detail": err}
        return {"status": "ERROR", "detail": err, "symbol": probe_symbol}
    return {
        "status": "READY",
        "detail": f"bars OK for {probe_symbol} on feed={indicators.get('feed', settings.ALPACA_DATA_FEED)}",
        "symbol": probe_symbol,
        "feed": indicators.get("feed"),
    }


def run_startup_checks() -> dict:
    """Runs all checks, logs the aligned table, caches and returns the report."""
    rows = []

    # --- Alpaca account -----------------------------------------------------
    if not settings.ALPACA_API_KEY or not settings.ALPACA_SECRET_KEY:
        alpaca = {"status": "NOT_CONFIGURED", "detail": "Alpaca API keys not set"}
    else:
        account = alpaca_service.get_account_summary()
        if account.get("error"):
            alpaca = {"status": "ERROR", "detail": str(account["error"])}
        else:
            alpaca = {
                "status": "READY",
                "detail": f"paper account OK (equity ${account.get('equity', 0):.2f}, "
                          f"status {account.get('status')})",
                "equity": account.get("equity"),
                "account_status": account.get("status"),
            }
    rows.append({"component": "ALPACA", **alpaca})

    # --- Market data feed ----------------------------------------------------
    if alpaca["status"] != "READY":
        market_data = {"status": "NOT_CONFIGURED", "detail": "requires Alpaca credentials"}
    else:
        market_data = _probe_market_data()
    rows.append({"component": "MARKET DATA", **market_data})

    # --- Fundamentals provider ------------------------------------------------
    fundamentals = fundamentals_service.health_check()
    rows.append({"component": "FUNDAMENTALS", **{
        "status": fundamentals.get("status", "DATA_UNAVAILABLE"),
        "detail": fundamentals.get("detail", ""),
    }})

    # --- LLM providers ---------------------------------------------------------
    validation = llm_service.validate_models()
    states = llm_service.provider_states()
    llm_providers = {}
    for provider in ("groq", "nvidia", "gemini", "openrouter"):
        state = states.get(provider, {})
        entry = validation.get("providers", {}).get(provider, {})
        missing = entry.get("missing") or []
        if state.get("state") == "NOT_CONFIGURED":
            status = "NOT_CONFIGURED"
            detail = state.get("detail", "")
        elif missing:
            status = "MODEL_UNAVAILABLE"
            detail = "; ".join(missing)
        elif state.get("state") != "READY":
            status = state.get("state")
            detail = state.get("detail", "")
        else:
            checked = entry.get("checked") or {}
            verified = sum(1 for v in checked.values() if v)
            status = "READY"
            detail = f"{verified} model id(s) verified against the live catalog"
        llm_providers[provider] = {"status": status, "detail": detail,
                                   "circuit": state.get("state"),
                                   "models_checked": entry.get("checked", {})}
        rows.append({"component": provider.upper(), "status": status, "detail": detail})

    report = {
        "checked_at": _now_iso(),
        "rows": rows,
        "alpaca": alpaca,
        "market_data": {**market_data, "configured_feed": settings.ALPACA_DATA_FEED},
        "fundamentals": fundamentals,
        "llm": {
            "providers": llm_providers,
            "validation": validation,
        },
    }
    _startup_report = report

    # Aligned table — exactly what the logs should show at boot.
    width = max(len(r["component"]) for r in rows) + 2
    lines = ["startup health check:"]
    for r in rows:
        lines.append(f"  {r['component']:<{width}}{r['status']:<22}{r.get('detail', '')}")
    logger.info("\n".join(lines))
    return report


def get_startup_report() -> dict:
    """The cached startup report (or a fresh one if startup hasn't run)."""
    if _startup_report is None:
        return run_startup_checks()
    return _startup_report


def live_health() -> dict:
    """/api/health payload: startup checks + LIVE provider circuit states,
    market clock and realtime status. No secrets."""
    startup = get_startup_report()
    return {
        "checked_at": _now_iso(),
        "startup": startup,
        "providers": llm_service.provider_states(),
        "market_clock": alpaca_service.get_clock(),
        "realtime": realtime_service.get_state(),
        "fundamentals": fundamentals_service.provider_config(),
        "configured_feed": settings.ALPACA_DATA_FEED,
    }
