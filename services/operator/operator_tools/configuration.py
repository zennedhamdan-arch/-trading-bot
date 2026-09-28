"""
operator_tools/configuration.py + history.py — safe configuration inspection
(redacted) and diagnostic history.

Secrets are redacted at TWO layers: the curated key list below never
includes key VALUES (only presence), and schemas.sanitize() redacts any
secret-looking key that slips through. There is no tool that WRITES
configuration — inspection only.
"""

from services import fundamentals_service, llm_service, market_data_service
from services.operator.memory import (get_messages, get_system_events,
                                      list_conversations)
from services.operator.operator_tools._registry import tool
from services.operator.schemas import sanitize
from services.operator.operator_tools.agents import DEPENDENCY_GRAPH

# Safe settings: (settings attr, label). Only these are ever exposed; every
# *_API_KEY / *_SECRET attribute is exposed as presence, never value.
_SAFE_SETTINGS = (
    ("PAPER_TRADING_ONLY", "paper trading enforced (constant true)"),
    ("ALPACA_BASE_URL", "alpaca endpoint (paper)"),
    ("ALPACA_DATA_FEED", "market data feed"),
    ("TRADE_UNIVERSE", "trade universe"),
    ("CYCLE_INTERVAL_MINUTES", "cycle interval (min)"),
    ("MAX_POSITION_PCT", "max position size (% of equity)"),
    ("RISK_MAX_PORTFOLIO_EXPOSURE_PCT", "max portfolio exposure"),
    ("RISK_MAX_DAILY_LOSS_PCT", "max daily loss"),
    ("RISK_MAX_TRADES_PER_DAY", "max trades per day"),
    ("RISK_EVENT_RISK_ACTION", "event-risk action"),
    ("LLM_REQUEST_TIMEOUT_SECONDS", "LLM request timeout (s)"),
    ("LLM_MAX_RETRIES", "LLM SDK retries"),
    ("LLM_MAX_MODEL_ATTEMPTS", "LLM model attempts"),
    ("LLM_MODEL_MIN_INTERVAL_SECONDS_UNOROUTER", "per-model min interval (s)"),
    ("LLM_CYCLE_MAX_REQUESTS", "per-cycle LLM send budget"),
    ("NEWS_ENABLED", "news intelligence enabled"),
    ("NEWS_REFRESH_MINUTES", "news worker interval (min)"),
    ("NEWS_RELEVANCE_THRESHOLD", "news relevance threshold"),
    ("ENABLE_FUNDAMENTALS_AGENT", "fundamentals agent enabled"),
    ("ENABLE_DEBATE", "debate enabled"),
    ("ENABLE_MEMORY", "memory/learning enabled"),
    ("CIO_AGENT_ENABLED", "CIO agent enabled"),
    ("TECH_LLM_INTERPRETATION_ENABLED", "technical LLM interpretation"),
    ("OPERATOR_ENABLED", "trading partner enabled"),
    ("OPERATOR_LLM_PROVIDER", "trading partner LLM provider"),
    ("OPERATOR_LLM_MODEL", "trading partner model override"),
    ("FUNDAMENTALS_PROVIDER", "fundamentals DATA provider"),
    ("FUNDAMENTALS_FALLBACK_PROVIDER", "fundamentals fallback provider"),
    ("REALTIME_ENABLED", "realtime stream enabled"),
    ("REALTIME_STALE_TICK_SECONDS", "realtime staleness threshold (s)"),
)

_SECRET_PRESENCE_KEYS = (
    "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "UNOROUTER_API_KEY", "GROQ_API_KEY",
    "GEMINI_API_KEY", "NVIDIA_API_KEY", "OPENROUTER_API_KEY",
)


@tool("get_safe_config", "configuration",
      "The system configuration, secrets REDACTED (keys show only as "
      "'configured'/'not set'). Includes config validation warnings.",
      label="Checking configuration")
def _get_safe_config(args):
    from config import settings
    values = {}
    for attr, label in _SAFE_SETTINGS:
        try:
            values[attr] = getattr(settings, attr)
        except AttributeError:
            continue
    secrets = {k: ("***configured***" if getattr(settings, k, "") else "***not set***")
               for k in _SECRET_PRESENCE_KEYS}
    return sanitize({"values": values, "secret_presence": secrets,
                     "config_warnings": settings.validate(),
                     "note": "read-only inspection; the operator cannot modify "
                             "configuration"})


@tool("get_provider_configuration", "configuration",
      "Provider configuration: LLM routing per task, provider key presence "
      "(redacted), fundamentals DATA provider config, market data feed.",
      label="Checking provider configuration")
def _get_provider_configuration(args):
    from config import settings
    return sanitize({
        "llm_routes": llm_service.llm_routes(),
        "llm_key_presence": {k: ("***configured***" if getattr(settings, k, "") else "***not set***")
                             for k in _SECRET_PRESENCE_KEYS if k.endswith("API_KEY")},
        "fundamentals": fundamentals_service.provider_config(),
        "fundamentals_registered_providers": fundamentals_service.available_providers(),
        "market_data": market_data_service.feed_config(),
        "active_chain_providers": llm_service.active_chain_providers(),
    })


@tool("get_model_configuration", "configuration",
      "Model configuration per provider: configured ids, live-catalog "
      "matches/misses, the UnoRouter fallback chain and selected fallback.",
      label="Checking model configuration")
def _get_model_configuration(args):
    validation = llm_service.last_validation() or llm_service.validate_models()
    providers = {}
    for provider, entry in (validation.get("providers") or {}).items():
        providers[provider] = {
            "status": entry.get("status"),
            "configured_models": entry.get("configured"),
            "matched_models": entry.get("matched"),
            "missing_models": entry.get("missing_models"),
            "selected_fallback": entry.get("selected_fallback"),
            "live_catalog_models": entry.get("models_found"),
        }
    return {"providers": providers,
            "unorouter_configured_chain": llm_service.unorouter_model_chain(),
            "unorouter_effective_chain": llm_service.unorouter_effective_chain()}


@tool("get_feature_flags", "configuration",
      "Every feature toggle in one view (agents, news, debate, memory, "
      "realtime, operator) + the paper-only guarantee.",
      label="Checking feature flags")
def _get_feature_flags(args):
    from config import settings
    return sanitize({
        "feature_flags": {
            "fundamentals_agent": bool(settings.ENABLE_FUNDAMENTALS_AGENT),
            "debate": bool(settings.ENABLE_DEBATE),
            "memory": bool(settings.ENABLE_MEMORY),
            "cio_agent": bool(settings.CIO_AGENT_ENABLED),
            "tech_llm_interpretation": bool(settings.TECH_LLM_INTERPRETATION_ENABLED),
            "news_intelligence": bool(settings.NEWS_ENABLED),
            "realtime": bool(settings.REALTIME_ENABLED),
            "operator": bool(settings.OPERATOR_ENABLED),
        },
        "paper_trading_only": True,
        "live_trading_possible": False,
    })


# ---------------------------------------------------------------------------
# history tools
# ---------------------------------------------------------------------------

@tool("get_conversation_history", "history",
      "Previous Trading Partner conversations (titles + messages), so past "
      "investigations can be recalled.",
      args={"limit": {"type": "integer", "description": "max messages (default 20)"}},
      label="Checking conversation history")
def _get_conversation_history(args):
    limit = min(int(args.get("limit") or 20), 100)
    conversations = list_conversations(limit=10)
    out = []
    for conv in conversations[:5]:
        out.append({"conversation_id": conv["id"], "title": conv["title"],
                    "messages": get_messages(conv["id"], limit=limit)})
    return {"conversations": out, "count": len(out)}


@tool("get_system_event_history", "history",
      "Persisted diagnostic/system events (operator findings, degradations, "
      "recoveries) with severity and component filters.",
      args={"limit": {"type": "integer", "description": "max events (default 20)"},
            "severity": {"type": "string", "description": "info|warning|error"}},
      label="Checking event history")
def _get_system_event_history(args):
    limit = min(int(args.get("limit") or 20), 100)
    severity = args.get("severity")
    events = get_system_events(limit=limit, severity=severity)
    return {"events": events, "count": len(events)}


@tool("get_failure_comparison", "history",
      "Compares failures between two periods (defaults: today vs "
      "yesterday): LLM request failures by status, cycle error counts and "
      "degradation labels per day.", label="Comparing failures")
def _get_failure_comparison(args):
    from datetime import datetime, timedelta, timezone
    from services.operator.memory import _conn
    import main

    def _day_stats(day: str):
        stats = {"date": day, "llm_failures_by_status": {}, "llm_sends": 0,
                 "cycles": 0, "cycle_errors": 0, "degraded_cycles": 0}
        try:
            with _conn() as conn:
                for row in conn.execute(
                        "SELECT kind, status, COUNT(*) AS n FROM llm_request_log "
                        "WHERE date=? GROUP BY kind, status", (day,)):
                    if row["kind"] == "send":
                        stats["llm_sends"] += row["n"]
                        status = row["status"] or "UNKNOWN"
                        if status != "OK":
                            stats["llm_failures_by_status"][status] = row["n"]
        except Exception:  # noqa: BLE001 — log may not exist yet
            pass
        return stats

    today = datetime.now(timezone.utc)
    days = {}
    for label, offset in (("today", 0), ("yesterday", 1), ("day_before", 2)):
        day = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
        days[label] = _day_stats(day)

    # in-process cycle stats per day
    for record in list(main.cycle_history):
        day = str(record.get("started_at") or "")[:10]
        for label, stats in days.items():
            if stats["date"] == day:
                stats["cycles"] += 1
                stats["cycle_errors"] += len(record.get("errors") or [])
                if record.get("status") == "PARTIAL_ERROR":
                    stats["degraded_cycles"] += 1
    return {"periods": days,
            "note": "LLM failures come from the persisted request log; cycle "
                    "stats from this process's in-memory history"}
