"""
config.py
Centralized environment/configuration loader for the AI Trading Bot.
All other modules should import settings from here rather than
calling os.getenv() directly, so there is a single source of truth.
"""

import os
from dotenv import load_dotenv

# Load variables from a local .env file if present.
load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_str(name: str, default: str = "") -> str:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip()


class Settings:
    # --- Alpaca (primary market data + paper execution) ---
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    # This app is hardcoded to PAPER trading. There is deliberately NO
    # configuration that can enable live-money trading: an ALPACA_BASE_URL
    # pointing at a live endpoint is ignored with a warning. PAPER_TRADING_ONLY
    # is a display constant and cannot be turned off.
    _PAPER_BASE_URL = "https://paper-api.alpaca.markets"
    _env_base_url = os.getenv("ALPACA_BASE_URL", _PAPER_BASE_URL).strip()
    if _env_base_url and "paper-api.alpaca.markets" not in _env_base_url:
        import logging as _logging
        _logging.getLogger("config").warning(
            "ALPACA_BASE_URL points at a non-paper endpoint and was IGNORED — "
            "this application is paper-trading only."
        )
    ALPACA_BASE_URL: str = _PAPER_BASE_URL
    ALPACA_PAPER: bool = True
    PAPER_TRADING_ONLY: bool = True  # constant; env value never disables it
    # Market-data feed. The free/paper subscription only permits IEX;
    # requesting SIP fails with "subscription does not permit querying
    # recent SIP data". The feed is NEVER switched silently: if a
    # configured feed is not permitted, data calls fail honestly with
    # DATA_UNAVAILABLE (SUBSCRIPTION_FEED_UNAVAILABLE).
    ALPACA_DATA_FEED: str = os.getenv("ALPACA_DATA_FEED", "IEX").strip().upper() or "IEX"

    # --- LLM provider credentials ---
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")

    # --- UnoRouter (OpenAI-compatible gateway; the PRIMARY LLM provider) ---
    # The application implements its OWN model fallback chain — UnoRouter
    # never chooses models for us. Model ids are verified against the live
    # catalog at startup; an id that no longer exists is skipped at runtime
    # with LLM_MODEL_UNAVAILABLE and the next configured model is tried.
    UNOROUTER_ENABLED: bool = _get_bool("UNOROUTER_ENABLED", True)
    UNOROUTER_API_KEY: str = os.getenv("UNOROUTER_API_KEY", "")
    UNOROUTER_BASE_URL: str = _get_str("UNOROUTER_BASE_URL", "https://api.unorouter.com/v1")
    UNOROUTER_PRIMARY_MODEL: str = _get_str("UNOROUTER_PRIMARY_MODEL", "glm-5.3-flash-thinking:free")
    # Generic Groq model used when the chain falls back to Groq for a task
    # that has no task-specific Groq model configured.
    GROQ_FALLBACK_MODEL: str = _get_str("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
    UNOROUTER_DAILY_REQUEST_LIMIT: int = _get_int("UNOROUTER_DAILY_REQUEST_LIMIT", 0)
    UNOROUTER_QUOTA_BACKOFF_MINUTES: float = _get_float("UNOROUTER_QUOTA_BACKOFF_MINUTES", 10)
    # Comma-separated application-level fallback models (tried in order).
    # NOTE (verified 2026-09-11 against the live UnoRouter catalog):
    # qwen3.8-flash-next:free is NOT in the catalog; it is skipped at runtime
    # and the chain continues to glm-5.3-flash:free. The closest valid free
    # Qwen3.8 id is qwen3.8-27b:free.
    UNOROUTER_FALLBACK_MODELS: list = [
        m.strip() for m in os.getenv(
            "UNOROUTER_FALLBACK_MODELS",
            "qwen3.8-flash-next:free,glm-5.3-flash:free",
        ).split(",") if m.strip()
    ]
    # Provider enable switches (a disabled provider is skipped in the chain).
    GROQ_ENABLED: bool = _get_bool("GROQ_ENABLED", True)

    # --- LLM routing (provider per task; resolved at call time) ---
    # Any task can be pointed at any configured provider. Defaults keep
    # every reasoning task on Groq (verified models) except news/fundamentals
    # interpretation, which stays on Gemini. OpenRouter became optional after
    # its free risk model was withdrawn (404 MODEL_NOT_FOUND in production).
    # V2: every reasoning task defaults to UnoRouter (primary model chain),
    # with Groq as the final provider fallback in the unified chain.
    LLM_TECH_PROVIDER: str = _get_str("LLM_TECH_PROVIDER", "unorouter").lower()
    LLM_DEBATE_PROVIDER: str = _get_str("LLM_DEBATE_PROVIDER", "unorouter").lower()
    LLM_RISK_PROVIDER: str = _get_str("LLM_RISK_PROVIDER", "unorouter").lower()
    LLM_CIO_PROVIDER: str = _get_str("LLM_CIO_PROVIDER", "unorouter").lower()
    LLM_NEWS_PROVIDER: str = _get_str("LLM_NEWS_PROVIDER", "unorouter").lower()
    LLM_FUNDAMENTALS_PROVIDER: str = _get_str("LLM_FUNDAMENTALS_PROVIDER", "unorouter").lower()

    # --- Model identifiers (all env-configurable; defaults verified against
    # the providers' live catalogs in September 2026) ---
    # Groq shut down llama-3.1-8b-instant / llama-3.3-70b-versatile on
    # 2026-08-16 (Enterprise-only). The defaults below are Groq's recommended
    # replacements (https://console.groq.com/docs/deprecations) and are
    # re-verified against the live /models catalog at startup.
    GROQ_TECH_MODEL: str = _get_str("GROQ_TECH_MODEL", "openai/gpt-oss-20b")
    GROQ_DEBATE_MODEL: str = _get_str("GROQ_DEBATE_MODEL", "openai/gpt-oss-20b")
    GROQ_RISK_MODEL: str = _get_str("GROQ_RISK_MODEL", "openai/gpt-oss-20b")
    GROQ_CIO_MODEL: str = _get_str("GROQ_CIO_MODEL", "openai/gpt-oss-120b")
    # OpenRouter: OPTIONAL. Only used when a model is EXPLICITLY configured
    # and verified available for the account. Empty default = not routed.
    # (OPENROUTER_RISK_MODEL is read as a legacy fallback.)
    OPENROUTER_MODEL: str = _get_str("OPENROUTER_MODEL", "") or _get_str("OPENROUTER_RISK_MODEL", "")
    OPENROUTER_BASE_URL: str = _get_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    # NVIDIA: OPTIONAL secondary provider (OpenAI-compatible NIM endpoint;
    # base URL overridable). Usable as fallback/alternative/selected tasks.
    NVIDIA_MODEL: str = _get_str("NVIDIA_MODEL", "")
    NVIDIA_BASE_URL: str = _get_str("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    # Gemini (news/fundamentals interpretation; NOT the high-frequency
    # backbone — its free tier allows 20 requests/day, protected below).
    GEMINI_MODEL: str = _get_str("GEMINI_MODEL", "gemini-3.6-flash")

    # --- Global LLM fallback route (optional, explicit, verified) ---
    # Used when a task's primary route fails with MODEL_NOT_FOUND /
    # PROVIDER_ERROR / AUTH_ERROR / NETWORK_ERROR — never for quota errors
    # (quota exhaustion is reported honestly, not routed around) and never
    # as a silent switch. Both values must be set; the fallback model is
    # verified against the fallback provider's live catalog at startup.
    LLM_FALLBACK_PROVIDER: str = _get_str("LLM_FALLBACK_PROVIDER", "").lower()
    LLM_FALLBACK_MODEL: str = _get_str("LLM_FALLBACK_MODEL", "")

    # --- LLM quota protection (free-tier friendly) ---
    # Local rolling-24h request budgets per provider. 0 disables the local
    # cap. The observed Gemini free tier limit for gemini-3.6-flash is
    # 20 requests/day (quota metric GenerateContentFreeTierRequests).
    GEMINI_DAILY_REQUEST_LIMIT: int = _get_int("GEMINI_DAILY_REQUEST_LIMIT", 20)
    GROQ_DAILY_REQUEST_LIMIT: int = _get_int("GROQ_DAILY_REQUEST_LIMIT", 0)
    OPENROUTER_DAILY_REQUEST_LIMIT: int = _get_int("OPENROUTER_DAILY_REQUEST_LIMIT", 0)
    NVIDIA_DAILY_REQUEST_LIMIT: int = _get_int("NVIDIA_DAILY_REQUEST_LIMIT", 0)
    # Minutes to stop calling a provider after a server-side quota-exhausted
    # (429) error that carries no usable retry delay.
    GEMINI_QUOTA_BACKOFF_MINUTES: float = _get_float("GEMINI_QUOTA_BACKOFF_MINUTES", 60)
    GROQ_QUOTA_BACKOFF_MINUTES: float = _get_float("GROQ_QUOTA_BACKOFF_MINUTES", 10)
    OPENROUTER_QUOTA_BACKOFF_MINUTES: float = _get_float("OPENROUTER_QUOTA_BACKOFF_MINUTES", 60)
    NVIDIA_QUOTA_BACKOFF_MINUTES: float = _get_float("NVIDIA_QUOTA_BACKOFF_MINUTES", 60)
    # Retries ONLY for transient rate limits (429 with a short explicit
    # retry delay); quota exhaustion is never retried. Default 0 per the V2
    # no-long-retries policy — a failed attempt immediately moves to the
    # next model/provider in the chain.
    LLM_RATE_LIMIT_MAX_RETRIES: int = _get_int("LLM_RATE_LIMIT_MAX_RETRIES", 0)
    LLM_RATE_LIMIT_MAX_WAIT_SECONDS: float = _get_float("LLM_RATE_LIMIT_MAX_WAIT_SECONDS", 30)
    # Hard per-request timeout (seconds) applied to every LLM HTTP call, and
    # the SDK-level retry count (0 = the SDK never retries; the application's
    # own model/provider fallback handles failures instead of waiting).
    LLM_REQUEST_TIMEOUT_SECONDS: float = _get_float("LLM_REQUEST_TIMEOUT_SECONDS", 10.0)
    LLM_MAX_RETRIES: int = _get_int("LLM_MAX_RETRIES", 0)
    # Maximum model attempts per request across the fallback chain
    # (primary + fallback models; never unlimited model guessing).
    LLM_MAX_MODEL_ATTEMPTS: int = _get_int("LLM_MAX_MODEL_ATTEMPTS", 3)
    # Classic per-provider circuit breaker (CLOSED -> OPEN -> HALF_OPEN).
    LLM_CIRCUIT_BREAKER_ENABLED: bool = _get_bool("LLM_CIRCUIT_BREAKER_ENABLED", True)
    LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = _get_int("LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD", 3)
    LLM_CIRCUIT_BREAKER_BACKOFF_SECONDS: float = _get_float("LLM_CIRCUIT_BREAKER_BACKOFF_SECONDS", 600.0)
    # Router-level response cache: identical (task, prompt) within the TTL is
    # served from cache; on total chain failure the last cached response is
    # the "cached result" stop before the deterministic fallback.
    LLM_RESPONSE_CACHE_TTL_MINUTES: float = _get_float("LLM_RESPONSE_CACHE_TTL_MINUTES", 240.0)
    LLM_RESPONSE_CACHE_MAX_ENTRIES: int = _get_int("LLM_RESPONSE_CACHE_MAX_ENTRIES", 500)

    # --- LLM circuit breakers ---
    # After a 404 MODEL_NOT_FOUND, the (provider, model) pair is short-
    # circuited for this long — no repeated requests to a dead model.
    MODEL_UNAVAILABLE_COOLDOWN_MINUTES: float = _get_float("MODEL_UNAVAILABLE_COOLDOWN_MINUTES", 30)
    # After an authentication failure (401/403), the provider is short-
    # circuited for this long — no repeated auth failures.
    AUTH_COOLDOWN_MINUTES: float = _get_float("AUTH_COOLDOWN_MINUTES", 60)

    # --- News Intelligence V2 (persistent, worker-driven) ---
    NEWS_ENABLED: bool = _get_bool("NEWS_ENABLED", True)
    # The independent news worker refresh interval. News processing NEVER
    # runs inside a trading cycle; cycles read the persistent cache.
    NEWS_REFRESH_MINUTES: float = _get_float("NEWS_REFRESH_MINUTES", 30.0)
    # Intelligence older than this is still used but marked is_stale=true.
    NEWS_CACHE_MAX_AGE_MINUTES: float = _get_float("NEWS_CACHE_MAX_AGE_MINUTES", 60.0)
    # Only articles at/above this relevance level are sent to an LLM
    # (HIGH | MEDIUM | LOW; IRRELEVANT is never analyzed).
    NEWS_RELEVANCE_THRESHOLD: str = _get_str("NEWS_RELEVANCE_THRESHOLD", "MEDIUM").upper()
    NEWS_MAX_ARTICLES_PER_SYMBOL: int = _get_int("NEWS_MAX_ARTICLES_PER_SYMBOL", 20)

    # --- LLM analysis reuse (avoid re-sending identical context) ---
    # The news and fundamentals agents cache their LLM analysis keyed by the
    # exact input. While the input is unchanged, the cached analysis is
    # reused and NO new request is sent. This is what keeps a 15-minute
    # cycle cadence inside the Gemini free tier's 20 requests/day.
    NEWS_ANALYSIS_CACHE_TTL_MINUTES: float = _get_float("NEWS_ANALYSIS_CACHE_TTL_MINUTES", 240)
    FUNDAMENTALS_ANALYSIS_CACHE_TTL_MINUTES: float = _get_float("FUNDAMENTALS_ANALYSIS_CACHE_TTL_MINUTES", 480)

    # --- Fundamentals data providers ---
    # "none" (default): no fundamentals provider configured — the fundamentals
    # agent reports DATA_UNAVAILABLE and the cycle continues on price data,
    # technicals, news and risk. Providers are pluggable (implement
    # services/fundamentals_service.FundamentalsProvider); nothing scrapes
    # Yahoo Finance or relies on Yahoo cookies/crumbs. yfinance is GONE.
    FUNDAMENTALS_PROVIDER: str = _get_str("FUNDAMENTALS_PROVIDER", "none").lower()
    FUNDAMENTALS_FALLBACK_PROVIDER: str = _get_str("FUNDAMENTALS_FALLBACK_PROVIDER", "").lower()
    # Seconds a fundamentals result (success or unavailability) is cached per
    # symbol so one cycle never asks twice.
    FUNDAMENTALS_RESULT_CACHE_TTL_SECONDS: float = _get_float("FUNDAMENTALS_RESULT_CACHE_TTL_SECONDS", 60)

    # --- Real-time market data (Alpaca WebSockets; optional) ---
    # Trades + quotes + minute bars + news stream for the trade universe.
    # Purely a dashboard/observability layer: ticks NEVER trigger LLM calls;
    # the AI cycle stays on its scheduled interval.
    REALTIME_ENABLED: bool = _get_bool("REALTIME_ENABLED", True)
    # Base delay for the supervisor's reconnect backoff. The delay grows
    # exponentially per consecutive failed attempt (2s, 4s, 8s, 16s, ...)
    # with jitter, capped at REALTIME_RECONNECT_MAX_SECONDS — never a fixed
    # every-N-seconds-forever loop.
    REALTIME_RECONNECT_SECONDS: float = _get_float("REALTIME_RECONNECT_SECONDS", 2.0)
    REALTIME_RECONNECT_MAX_SECONDS: float = _get_float("REALTIME_RECONNECT_MAX_SECONDS", 60.0)
    # A live stream that receives no market tick for this many seconds
    # WHILE THE MARKET IS OPEN is considered connected-but-silent and is
    # recycled. Outside market hours IEX legitimately sends nothing, so no
    # reconnect storm happens while the market is closed. 0 disables.
    REALTIME_STALE_TICK_SECONDS: float = _get_float("REALTIME_STALE_TICK_SECONDS", 180.0)
    # Optional opt-in passthrough to the SDK's own data_timeout (detects a
    # silent socket at the transport level). Disabled by default: with no
    # market data outside trading hours the SDK would otherwise force an
    # endless reconnect cycle on quiet markets. The market-aware staleness
    # check above is the default mechanism.
    REALTIME_DATA_TIMEOUT_SECONDS: float = _get_float("REALTIME_DATA_TIMEOUT_SECONDS", 0.0)

    # --- Evidence quality (decision-quality gate) ---
    # Daily-bar analytics older than this many calendar days are STALE
    # (long weekends/holidays fit comfortably below the threshold).
    EVIDENCE_STALE_DAYS: int = _get_int("EVIDENCE_STALE_DAYS", 5)

    # --- Deterministic risk engine (hard rules; LLM can never override) ---
    # Total open-position exposure as a fraction of equity.
    RISK_MAX_PORTFOLIO_EXPOSURE_PCT: float = _get_float("RISK_MAX_PORTFOLIO_EXPOSURE_PCT", 0.80)
    # Reject all NEW buys when the day's P&L is at or below this fraction.
    RISK_MAX_DAILY_LOSS_PCT: float = _get_float("RISK_MAX_DAILY_LOSS_PCT", 0.03)
    # Maximum executed orders per calendar day (persisted across restarts).
    RISK_MAX_TRADES_PER_DAY: int = _get_int("RISK_MAX_TRADES_PER_DAY", 10)
    # Event-risk behavior: "reject" (no new trades on HIGH event risk),
    # "reduce" (halve the allowed notional), or "ignore".
    RISK_EVENT_RISK_ACTION: str = _get_str("RISK_EVENT_RISK_ACTION", "reduce").lower()
    # Reject new BUYs while the market is verifiably closed.
    RISK_REQUIRE_MARKET_OPEN: bool = _get_bool("RISK_REQUIRE_MARKET_OPEN", True)

    # --- Bot behavior ---
    TRADE_UNIVERSE: list = [
        t.strip().upper()
        for t in os.getenv("TRADE_UNIVERSE", "AAPL,MSFT,NVDA,TSLA,SPY").split(",")
        if t.strip()
    ]
    CYCLE_INTERVAL_MINUTES: int = _get_int("CYCLE_INTERVAL_MINUTES", 15)
    MAX_POSITION_PCT: float = _get_float("MAX_POSITION_PCT", 0.10)

    # --- Feature toggles ---
    ENABLE_FUNDAMENTALS_AGENT: bool = _get_bool("ENABLE_FUNDAMENTALS_AGENT", True)
    # Debate is OFF by default (V2): it only runs on a valid setup with
    # conflicting evidence and a healthy provider. Legacy env name
    # ENABLE_DEBATE is still honored.
    ENABLE_DEBATE: bool = _get_bool("DEBATE_AGENT_ENABLED",
                                    _get_bool("ENABLE_DEBATE", False))
    # Minimum technical confidence for the debate to run (when the technical
    # report is deterministic and carries no LLM confidence, setup existence
    # itself is the qualifier).
    DEBATE_MIN_TECH_CONFIDENCE: float = _get_float("DEBATE_MIN_TECH_CONFIDENCE", 0.65)
    DEBATE_REQUIRE_CONFLICT: bool = _get_bool("DEBATE_REQUIRE_CONFLICT", True)
    # CIO AI review runs ONLY on a valid setup that passed the hard risk
    # engine (the deterministic fail-safe below it never needs an LLM).
    CIO_AGENT_ENABLED: bool = _get_bool("CIO_AGENT_ENABLED", True)
    # Optional richer LLM interpretation of technicals (default off: the
    # deterministic rule engine produces the setup verdict).
    TECH_LLM_INTERPRETATION_ENABLED: bool = _get_bool("TECH_LLM_INTERPRETATION_ENABLED", False)
    ENABLE_MEMORY: bool = _get_bool("ENABLE_MEMORY", True)
    MEMORY_DB_PATH: str = os.getenv("MEMORY_DB_PATH", "data/memory.db")
    AGENT_ACCURACY_LOOKBACK: int = _get_int("AGENT_ACCURACY_LOOKBACK", 20)

    def validate(self) -> list:
        """Returns a list of human-readable warnings for missing config.
        Intentionally does not raise, so the dashboard can still boot and
        show the user what's missing instead of crashing on import."""
        warnings = []
        if not self.ALPACA_API_KEY or not self.ALPACA_SECRET_KEY:
            warnings.append("Alpaca API keys are not set. Trading functions will fail.")
        if not self.GEMINI_API_KEY:
            warnings.append("GEMINI_API_KEY is not set. News sentiment interpretation will be unavailable.")
        if self.UNOROUTER_ENABLED and not self.UNOROUTER_API_KEY:
            warnings.append("UNOROUTER_API_KEY is not set. The primary LLM chain will skip UnoRouter and fall back to Groq.")
        if not self.GROQ_API_KEY:
            warnings.append("GROQ_API_KEY is not set. The Groq fallback (and any Groq-routed tasks) will be unavailable.")
        if not self.GROQ_ENABLED and not (self.UNOROUTER_ENABLED and self.UNOROUTER_API_KEY):
            warnings.append("No LLM provider available: Groq disabled and UnoRouter not configured. Agents will use cached/deterministic fallbacks.")
        if not self.OPENROUTER_API_KEY:
            warnings.append("OPENROUTER_API_KEY is not set. OpenRouter will be unavailable (optional provider).")
        if self.OPENROUTER_API_KEY and not self.OPENROUTER_MODEL:
            warnings.append("OPENROUTER_API_KEY is set but OPENROUTER_MODEL is not; OpenRouter is not routed to any task.")
        if not self.NVIDIA_API_KEY:
            warnings.append("NVIDIA_API_KEY is not set. NVIDIA LLM provider is not configured (optional).")
        if self.FUNDAMENTALS_PROVIDER == "none":
            warnings.append("No fundamentals provider configured (FUNDAMENTALS_PROVIDER=none); fundamentals analysis will report DATA_UNAVAILABLE.")
        return warnings


settings = Settings()
