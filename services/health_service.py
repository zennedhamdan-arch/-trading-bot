"""
services/health_service.py

Startup + live health checks for every dependency. At startup the bot
verifies (and logs, and exposes via /api/health):

  ALPACA ACCOUNT     credentials / account / paper mode
  ALPACA MARKET DATA configured feed actually permitted by the subscription
  INDICATORS         full indicator pipeline computes on the configured feed
  REAL-TIME STREAM   Alpaca WebSocket layer state
  FUNDAMENTALS       configured provider (or honest DATA_UNAVAILABLE)
  GROQ / GEMINI / NVIDIA / OPENROUTER  credentials / models / circuits

Top-level status (HEALTHY / DEGRADED / OFFLINE) — HTTP 200 from an API
endpoint does NOT mean the system is healthy:

  HEALTHY   account ready + market data ready + indicators ready +
            real-time connected + every ROUTED LLM provider ready
  DEGRADED  backend running but one or more non-critical subsystems are
            unavailable (real-time disconnected, a routed LLM provider
            down, fundamentals provider failing, ...)
  OFFLINE   critical trading infrastructure unavailable (Alpaca account
            or market data/indicators down) — the bot cannot trade.

Rules:
  - A failing check NEVER prevents the app from booting.
  - Secrets are never exposed — only key PRESENCE and error reasons.
  - Fundamentals DATA_UNAVAILABLE with FUNDAMENTALS_PROVIDER=none is
    "off by configuration" (acceptable per spec) and does NOT degrade
    the system; a CONFIGURED provider that fails DOES.
"""

import logging
import threading
import time
from datetime import datetime, timezone

from config import settings
from services import alpaca_service, fundamentals_service, llm_service, realtime_service

logger = logging.getLogger("health_service")

_startup_report = None  # cached result of run_startup_checks()

# Short TTL caches so /api/health (polled ~45s by the dashboard) stays
# cheap while the top-level status still reflects LIVE reality rather
# than boot-time results.
_PROBE_TTL_S = 60.0
_probe_lock = threading.Lock()
_probe_cache = {
    "ts": 0.0,
    "account": None,       # alpaca_service.get_account_summary() result
    "market_data": None,   # _probe_market_data() result
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _probe_market_data() -> dict:
    """Probes the CONFIGURED data feed AND the full indicator pipeline.

    get_indicators() always requests enough bars for the complete suite
    (SMA200 max window) and validates the returned bar count, so a
    successful probe proves both the feed and indicator computation. It
    detects subscription limitations honestly
    (SUBSCRIPTION_FEED_UNAVAILABLE) and insufficient history
    (DATA_UNAVAILABLE / insufficient_bars) — never switches feeds
    silently, never fabricates indicator values."""
    probe_symbol = (settings.TRADE_UNIVERSE or ["SPY"])[0]
    indicators = alpaca_service.get_indicators(probe_symbol)
    if indicators.get("error"):
        err = str(indicators["error"])
        if "SUBSCRIPTION_FEED_UNAVAILABLE" in err:
            return {"status": "SUBSCRIPTION_FEED_UNAVAILABLE", "detail": err}
        if indicators.get("status") == "DATA_UNAVAILABLE":
            return {
                "status": "DATA_UNAVAILABLE",
                "detail": err,
                "symbol": probe_symbol,
                "available_bars": indicators.get("available_bars"),
                "required_bars": indicators.get("required_bars"),
            }
        return {"status": "ERROR", "detail": err, "symbol": probe_symbol}
    return {
        "status": "READY",
        "detail": (
            f"bars + indicators OK for {probe_symbol} on feed="
            f"{indicators.get('feed', settings.ALPACA_DATA_FEED)} "
            f"({indicators.get('bars_available')} bars)"
        ),
        "symbol": probe_symbol,
        "feed": indicators.get("feed"),
        "bars": indicators.get("bars_available"),
    }


def _refresh_live_probes(force: bool = False) -> dict:
    """Account + market-data/indicators probes with a TTL cache (thread-
    safe): cheap enough for /api/health, live enough to be honest."""
    with _probe_lock:
        now = time.time()
        if (not force and _probe_cache["ts"]
                and now - _probe_cache["ts"] < _PROBE_TTL_S
                and _probe_cache["account"] is not None):
            return _probe_cache
        account = alpaca_service.get_account_summary()
        market = _probe_market_data()
        _probe_cache.update({"ts": now, "account": account, "market_data": market})
        return _probe_cache


def _realtime_row() -> dict:
    """Real-time stream row (live state at check time). A stream whose last
    tick is stale is reported as STALE WITH ITS AGE IN SECONDS — never as
    CONNECTED/healthy. (Staleness is judged while the market is open: a
    silent closed market is normal and stays CONNECTED with its tick age.)"""
    state = realtime_service.get_state()
    status = state.get("connection_status") or state.get("status") or "UNKNOWN"
    age = state.get("seconds_since_last_tick")
    age_txt = f"{age:.0f}s ago" if age is not None else "no tick yet"
    threshold = float(settings.REALTIME_STALE_TICK_SECONDS or 0)
    if status == "STALE":
        detail = (
            f"STALE — last tick {age_txt} (threshold {threshold:.0f}s, "
            f"market open); the supervisor is recycling the stream"
        )
    elif status == "CONNECTED":
        detail = (
            f"feed={state.get('feed')}, {len(state.get('symbols') or [])} symbols, "
            f"last tick {age_txt}"
        )
    elif status == "DISABLED":
        detail = "disabled via REALTIME_ENABLED=false"
    elif status == "NO_KEYS":
        detail = "Alpaca API keys not configured"
    else:
        detail = state.get("last_error") or status
    return {
        "component": "REAL-TIME STREAM",
        "status": status,
        "detail": detail,
        "last_tick_age_s": age,
        "stale_threshold_s": threshold or None,
    }


def _routed_providers() -> list:
    """Providers the ACTIVE request chain depends on: every provider an LLM
    task is routed to, plus Groq when UnoRouter is routed (Groq is the
    chain-final provider fallback). These are REQUIRED for the AI pipeline;
    unrouted optional providers are not."""
    try:
        return llm_service.active_chain_providers()
    except Exception:  # noqa: BLE001 — health must never crash
        routed = set()
        try:
            for route in (llm_service.llm_routes() or {}).values():
                provider = (route or {}).get("provider")
                if provider:
                    routed.add(str(provider).lower())
        except Exception:  # noqa: BLE001
            pass
        return sorted(routed)


def overall_status(live: bool = True) -> dict:
    """Top-level system state: HEALTHY / DEGRADED / OFFLINE plus the
    reasons. Derived from existing health semantics — no new probes
    beyond the cheap cached ones.

    Critical (=> OFFLINE when down): Alpaca account, market data/indicators.
    Non-critical (=> DEGRADED when down): real-time stream, routed LLM
    providers, configured fundamentals provider.
    """
    if live:
        probes = _refresh_live_probes()
        account, market = probes["account"], probes["market_data"]
    else:
        startup = _startup_report or {}
        account, market = startup.get("alpaca") or {}, startup.get("market_data") or {}
    return _overall_from(account, market)


def _overall_from(account: dict, market: dict) -> dict:
    """Core top-level classification from an account summary dict and a
    market-data/indicators probe dict."""
    reasons = []

    # --- critical trading infrastructure -------------------------------
    acct_error = account.get("error")
    acct_status = "ERROR" if acct_error else (
        "NOT_CONFIGURED" if not (settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY) else "READY"
    )
    if acct_status != "READY":
        return {
            "status": "OFFLINE",
            "reasons": [f"Alpaca account {acct_status}"
                        + (f": {str(acct_error)[:200]}" if acct_error else "")],
            "critical": ["alpaca_account"],
        }
    if market.get("status") not in ("READY", None):
        return {
            "status": "OFFLINE",
            "reasons": [f"market data/indicators {market.get('status')}: "
                        f"{str(market.get('detail', ''))[:200]}"],
            "critical": ["market_data"],
        }

    # --- non-critical subsystems ----------------------------------------
    status = "HEALTHY"

    rt = realtime_service.get_state()
    rt_state = rt.get("connection_status") or rt.get("status")
    if rt_state not in ("CONNECTED", "DISABLED", "NO_KEYS"):
        # Real-time is an observability layer: disconnected/STALE/error
        # degrades the system but never takes it offline. A stale stream
        # (last tick older than REALTIME_STALE_TICK_SECONDS while the market
        # is open) is NEVER reported as connected/healthy — its age in
        # seconds is part of the reason.
        status = "DEGRADED"
        attempt = rt.get("reconnect_attempt")
        age = rt.get("seconds_since_last_tick")
        age_txt = f", last tick {age:.0f}s ago" if age is not None else ""
        reasons.append(
            f"real-time stream {rt_state}{age_txt}"
            + (f" (reconnect attempt #{attempt})" if attempt else "")
        )

    # --- required LLM providers: circuit state AND catalog usability -------
    validation = (llm_service.last_validation() or {}).get("providers", {})
    usable_providers = []
    for provider in _routed_providers():
        state = (llm_service.provider_states().get(provider) or {})
        circuit = state.get("state")
        if circuit not in ("READY",):
            status = "DEGRADED"
            reasons.append(
                f"LLM provider {provider} {circuit}"
                + (f": {state.get('detail', '')}" if state.get("detail") else "")
            )
            continue
        ventry = validation.get(provider) or {}
        vstatus = ventry.get("status")
        if vstatus == "NO_USABLE_MODEL":
            # Endpoint reachable but ZERO configured models are in the live
            # catalog — the provider cannot serve anything as configured.
            status = "DEGRADED"
            missing = ", ".join(ventry.get("missing_models") or []) or "none configured"
            reasons.append(
                f"LLM provider {provider} NO_USABLE_MODEL "
                f"(configured models missing from the live catalog: {missing})"
            )
            continue
        if vstatus in ("DEGRADED", "ERROR"):
            status = "DEGRADED"
            reasons.append(
                f"LLM provider {provider} model validation {vstatus}"
                + (f": {ventry.get('error')}" if ventry.get("error") else "")
            )
            continue
        usable_providers.append(provider)

    # No usable configured LLM model ANYWHERE on the active chain.
    if _routed_providers() and not usable_providers:
        status = "DEGRADED"
        reasons.append(
            "no usable configured LLM model on any active-chain provider "
            f"({', '.join(_routed_providers())})"
        )

    fundamentals = fundamentals_service.health_check()
    provider_name = str(fundamentals.get("provider") or settings.FUNDAMENTALS_PROVIDER or "none").lower()
    if provider_name not in ("", "none"):
        # A CONFIGURED provider failing is a real degradation; provider=none
        # is off-by-configuration and acceptable by design.
        if fundamentals.get("status") not in ("READY",):
            status = "DEGRADED"
            reasons.append(
                f"fundamentals provider {provider_name} {fundamentals.get('status')}"
            )

    return {"status": status, "reasons": reasons, "critical": []}


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
    rows.append({"component": "ALPACA ACCOUNT", **alpaca})
    alpaca_row = dict(alpaca)

    # --- Market data feed + indicators --------------------------------------
    if alpaca_row["status"] != "READY":
        market_data = {"status": "NOT_CONFIGURED", "detail": "requires Alpaca credentials"}
    else:
        market_data = _probe_market_data()
    rows.append({"component": "ALPACA MARKET DATA", **market_data})

    # Indicators get their own row: the probe above runs the FULL
    # indicator suite (bars + RSI/SMA/EMA/MACD/ATR) on the probe symbol.
    if alpaca_row["status"] != "READY":
        indicators_row = {"status": "NOT_CONFIGURED", "detail": "requires Alpaca credentials"}
    elif market_data.get("status") == "READY":
        indicators_row = {
            "status": "READY",
            "detail": f"full suite computed ({market_data.get('bars')} bars, feed={market_data.get('feed')})",
        }
    else:
        indicators_row = {
            "status": market_data.get("status", "ERROR"),
            "detail": str(market_data.get("detail", ""))[:200],
        }
    rows.append({"component": "INDICATORS", **indicators_row})

    # --- Real-time stream (state at boot; live state via /api/health) -------
    rows.append(_realtime_row())

    # --- Fundamentals provider ------------------------------------------------
    fundamentals = fundamentals_service.health_check()
    rows.append({"component": "FUNDAMENTALS", **{
        "status": fundamentals.get("status", "DATA_UNAVAILABLE"),
        "detail": fundamentals.get("detail", ""),
    }})

    # --- News Intelligence (worker state; the cache survives restarts) ------
    try:
        from services import news_worker
        news_stats = news_worker.stats()
        if not settings.NEWS_ENABLED:
            news_row = {"status": "NOT_CONFIGURED", "detail": "NEWS_ENABLED=false"}
        elif news_stats.get("last_refresh"):
            news_row = {
                "status": "READY",
                "detail": (
                    f"last refresh {news_stats['last_refresh'][:19]} — "
                    f"{news_stats.get('articles_analyzed', 0)} analyzed, "
                    f"{news_stats.get('duplicates_ignored', 0)} duplicates ignored, "
                    f"{news_stats.get('symbols_with_intelligence', 0)} symbol(s) cached"
                ),
            }
        else:
            news_row = {"status": "PENDING", "detail": "worker has not completed a refresh yet"}
    except Exception as exc:  # noqa: BLE001 — health rows never crash boot
        news_row = {"status": "ERROR", "detail": str(exc)[:200]}
    rows.append({"component": "NEWS INTELLIGENCE", **news_row})

    # --- LLM providers ---------------------------------------------------------
    # One /v1/models fetch per provider per catalog-TTL window; each fetch is
    # timeout-bounded so a hanging provider (e.g. Gemini) is marked DEGRADED
    # and the check continues. Statuses reflect catalog usability:
    # a reachable provider with ZERO matching configured models is
    # NO_USABLE_MODEL — never READY.
    validation = llm_service.validate_models()
    states = llm_service.provider_states()
    llm_providers = {}
    for provider in ("unorouter", "groq", "nvidia", "gemini", "openrouter"):
        state = states.get(provider, {})
        entry = validation.get("providers", {}).get(provider, {})
        if state.get("state") == "NOT_CONFIGURED":
            status, detail = "NOT_CONFIGURED", state.get("detail", "")
        elif entry.get("status") == "NO_USABLE_MODEL":
            status = "NO_USABLE_MODEL"
            detail = (
                f"endpoint OK ({entry.get('models_found', 0)} live models) but "
                f"ZERO configured models match: "
                f"{', '.join(entry.get('missing_models') or []) or 'none configured'}"
            )
        elif entry.get("status") in ("DEGRADED", "ERROR"):
            status = entry["status"]
            detail = str(entry.get("error") or entry["status"])
        elif state.get("state") != "READY":
            status, detail = state.get("state"), state.get("detail", "")
        else:
            matched = entry.get("matched") or []
            missing = entry.get("missing_models") or []
            status = "READY"
            detail = (
                f"{len(matched)}/{len(entry.get('configured') or [])} configured "
                f"model(s) matched in the live catalog "
                f"({entry.get('models_found', 0)} live)"
                + (f"; missing: {', '.join(missing)}" if missing else "")
                + (f"; selected fallback: {entry['selected_fallback']}"
                   if provider == "unorouter" and entry.get("selected_fallback") else "")
            )
        llm_providers[provider] = {
            "status": status, "detail": detail,
            "circuit": state.get("state"),
            "models_checked": entry.get("checked", {}),
            "catalog": {
                "live_models": entry.get("models_found", 0),
                "configured": entry.get("configured", []),
                "matched": entry.get("matched", []),
                "missing": entry.get("missing_models", []),
                "selected_fallback": entry.get("selected_fallback"),
                "age_s": entry.get("catalog_age_s"),
                "cached": entry.get("catalog_cached", False),
                "timed_out": entry.get("timed_out", False),
            },
        }
        rows.append({"component": provider.upper(), "status": status, "detail": detail})

    # Top-level status from the rows computed above (no recursion).
    overall = _overall_from(alpaca_row, market_data)

    report = {
        "checked_at": _now_iso(),
        "rows": rows,
        "overall": overall,
        "alpaca": alpaca_row,
        "market_data": {**market_data, "configured_feed": settings.ALPACA_DATA_FEED},
        "indicators": indicators_row,
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
        lines.append(f"  {r['component']:<{width}}{r['status']:<26}{r.get('detail', '')}")
    lines.append(f"  {'OVERALL':<{width}}{overall['status']}")
    if overall.get("reasons"):
        for reason in overall["reasons"]:
            lines.append(f"  {'':<{width}}  - {reason}")
    logger.info("\n".join(lines))
    return report


def get_startup_report() -> dict:
    """The cached startup report (or a fresh one if startup hasn't run)."""
    if _startup_report is None:
        return run_startup_checks()
    return _startup_report


def live_health() -> dict:
    """/api/health payload: LIVE overall status + top-level reasons,
    startup checks, live provider circuit states, market clock and
    realtime status. No secrets. An HTTP 200 here does NOT mean all
    subsystems are healthy — read `overall` and the rows."""
    startup = get_startup_report()
    overall = overall_status(live=True)
    return {
        "checked_at": _now_iso(),
        "overall": overall,
        "startup": startup,
        "providers": llm_service.provider_states(),
        "market_clock": alpaca_service.get_clock(),
        "realtime": realtime_service.get_state(),
        "fundamentals": fundamentals_service.provider_config(),
        "configured_feed": settings.ALPACA_DATA_FEED,
    }
