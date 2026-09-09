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
    ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    # This app is hardcoded to paper trading. Do not point it at a live endpoint.
    ALPACA_PAPER: bool = True
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

    # --- LLM routing (provider per task; resolved at call time) ---
    # Any task can be pointed at any configured provider. Defaults keep
    # every reasoning task on Groq (verified models) except news/fundamentals
    # interpretation, which stays on Gemini. OpenRouter became optional after
    # its free risk model was withdrawn (404 MODEL_NOT_FOUND in production).
    LLM_TECH_PROVIDER: str = _get_str("LLM_TECH_PROVIDER", "groq").lower()
    LLM_DEBATE_PROVIDER: str = _get_str("LLM_DEBATE_PROVIDER", "groq").lower()
    LLM_RISK_PROVIDER: str = _get_str("LLM_RISK_PROVIDER", "groq").lower()
    LLM_CIO_PROVIDER: str = _get_str("LLM_CIO_PROVIDER", "groq").lower()
    LLM_NEWS_PROVIDER: str = _get_str("LLM_NEWS_PROVIDER", "gemini").lower()
    LLM_FUNDAMENTALS_PROVIDER: str = _get_str("LLM_FUNDAMENTALS_PROVIDER", "gemini").lower()

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
    # retry delay); quota exhaustion is never retried.
    LLM_RATE_LIMIT_MAX_RETRIES: int = _get_int("LLM_RATE_LIMIT_MAX_RETRIES", 1)
    LLM_RATE_LIMIT_MAX_WAIT_SECONDS: float = _get_float("LLM_RATE_LIMIT_MAX_WAIT_SECONDS", 30)

    # --- LLM circuit breakers ---
    # After a 404 MODEL_NOT_FOUND, the (provider, model) pair is short-
    # circuited for this long — no repeated requests to a dead model.
    MODEL_UNAVAILABLE_COOLDOWN_MINUTES: float = _get_float("MODEL_UNAVAILABLE_COOLDOWN_MINUTES", 30)
    # After an authentication failure (401/403), the provider is short-
    # circuited for this long — no repeated auth failures.
    AUTH_COOLDOWN_MINUTES: float = _get_float("AUTH_COOLDOWN_MINUTES", 60)

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
    REALTIME_RECONNECT_SECONDS: float = _get_float("REALTIME_RECONNECT_SECONDS", 30)

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
    ENABLE_DEBATE: bool = _get_bool("ENABLE_DEBATE", True)
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
        if not self.GROQ_API_KEY:
            warnings.append("GROQ_API_KEY is not set. Technical/Debate/Risk/CIO agents will be unavailable.")
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
