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


def _get_bool(name: str, default: bool = False) -> bool:
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
    # --- Alpaca ---
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    # This app is hardcoded to paper trading. Do not point it at a live endpoint.
    ALPACA_PAPER: bool = True

    # --- LLM Providers ---
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

    # --- Model identifiers (all env-configurable) ---
    # Defaults are verified against the providers' CURRENT catalogs
    # (September 2026). Groq shut down llama-3.1-8b-instant and
    # llama-3.3-70b-versatile on 2026-08-16 (they are Enterprise-only now);
    # Groq's recommended replacements are the GPT-OSS models below.
    # https://console.groq.com/docs/deprecations
    # OpenRouter's deepseek/deepseek-r1:free free variant is no longer
    # usable; openai/gpt-oss-20b:free is verified free today.
    # Model ids are validated against each provider's live /models endpoint
    # at startup (services/llm_service.validate_models).
    GEMINI_MODEL: str = _get_str("GEMINI_MODEL", "gemini-3.6-flash")
    GROQ_TECH_MODEL: str = _get_str("GROQ_TECH_MODEL", "openai/gpt-oss-20b")
    GROQ_DEBATE_MODEL: str = _get_str("GROQ_DEBATE_MODEL", "openai/gpt-oss-20b")
    GROQ_CIO_MODEL: str = _get_str("GROQ_CIO_MODEL", "openai/gpt-oss-120b")
    OPENROUTER_RISK_MODEL: str = _get_str("OPENROUTER_RISK_MODEL", "openai/gpt-oss-20b:free")
    OPENROUTER_BASE_URL: str = _get_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # Optional explicit fallback models. Empty (default) = NO fallback: an
    # agent whose model fails is marked failed/unavailable, never silently
    # rerouted. A configured fallback is used ONLY after startup validation
    # confirms it exists on the same provider, and never for quota errors
    # (a fallback on the same provider shares the account's quota).
    GROQ_TECH_FALLBACK_MODEL: str = _get_str("GROQ_TECH_FALLBACK_MODEL", "")
    GROQ_DEBATE_FALLBACK_MODEL: str = _get_str("GROQ_DEBATE_FALLBACK_MODEL", "")
    GROQ_CIO_FALLBACK_MODEL: str = _get_str("GROQ_CIO_FALLBACK_MODEL", "")
    OPENROUTER_RISK_FALLBACK_MODEL: str = _get_str("OPENROUTER_RISK_FALLBACK_MODEL", "")
    GEMINI_FALLBACK_MODEL: str = _get_str("GEMINI_FALLBACK_MODEL", "")

    # --- LLM quota protection (free-tier friendly) ---
    # Local rolling-24h request budgets per provider. When the budget is
    # reached, further requests short-circuit with PROVIDER_QUOTA_EXCEEDED
    # instead of hammering the provider into 429s. 0 disables the local cap.
    # The observed Gemini free tier allows 20 requests/day for
    # gemini-3.6-flash (quota metric GenerateContentFreeTierRequests),
    # hence the default of 20.
    GEMINI_DAILY_REQUEST_LIMIT: int = _get_int("GEMINI_DAILY_REQUEST_LIMIT", 20)
    GROQ_DAILY_REQUEST_LIMIT: int = _get_int("GROQ_DAILY_REQUEST_LIMIT", 0)
    OPENROUTER_DAILY_REQUEST_LIMIT: int = _get_int("OPENROUTER_DAILY_REQUEST_LIMIT", 0)
    # How long to stop calling a provider after a server-side quota-exhausted
    # (429) error that carries no usable retry delay.
    GEMINI_QUOTA_BACKOFF_MINUTES: float = _get_float("GEMINI_QUOTA_BACKOFF_MINUTES", 60)
    GROQ_QUOTA_BACKOFF_MINUTES: float = _get_float("GROQ_QUOTA_BACKOFF_MINUTES", 10)
    OPENROUTER_QUOTA_BACKOFF_MINUTES: float = _get_float("OPENROUTER_QUOTA_BACKOFF_MINUTES", 60)
    # Retries are ONLY for transient rate limits (429 with a short retry
    # delay). Quota exhaustion is never retried. Default: at most 1 retry,
    # and never wait longer than 30s for it.
    LLM_RATE_LIMIT_MAX_RETRIES: int = _get_int("LLM_RATE_LIMIT_MAX_RETRIES", 1)
    LLM_RATE_LIMIT_MAX_WAIT_SECONDS: float = _get_float("LLM_RATE_LIMIT_MAX_WAIT_SECONDS", 30)

    # --- LLM analysis reuse (avoid re-sending identical context) ---
    # The news and fundamentals agents cache their LLM analysis keyed by the
    # exact input (headlines / metrics). While the input is unchanged, the
    # cached analysis is reused and NO new request is sent. These TTLs bound
    # how long an unchanged-input analysis stays valid. This is what keeps a
    # 15-minute cycle cadence inside the Gemini free tier's 20 requests/day.
    NEWS_ANALYSIS_CACHE_TTL_MINUTES: float = _get_float("NEWS_ANALYSIS_CACHE_TTL_MINUTES", 240)
    FUNDAMENTALS_ANALYSIS_CACHE_TTL_MINUTES: float = _get_float("FUNDAMENTALS_ANALYSIS_CACHE_TTL_MINUTES", 480)

    # --- Bot behavior ---
    TRADE_UNIVERSE: list = [
        t.strip().upper()
        for t in os.getenv("TRADE_UNIVERSE", "AAPL,MSFT,NVDA,TSLA,SPY").split(",")
        if t.strip()
    ]
    CYCLE_INTERVAL_MINUTES: int = _get_int("CYCLE_INTERVAL_MINUTES", 15)
    MAX_POSITION_PCT: float = _get_float("MAX_POSITION_PCT", 0.10)

    # --- Market data ---
    # The free Alpaca paper-trading subscription only permits the IEX feed;
    # requesting the SIP feed fails with "subscription does not permit
    # querying recent SIP data". Default to IEX; set ALPACA_DATA_FEED=SIP on
    # a paid subscription.
    ALPACA_DATA_FEED: str = os.getenv("ALPACA_DATA_FEED", "IEX").strip().upper() or "IEX"

    # --- Fundamentals data layer (yfinance rate-limit protection) ---
    # How long a successful fundamentals fetch is cached per symbol.
    FUNDAMENTALS_CACHE_TTL_MINUTES: int = _get_int("FUNDAMENTALS_CACHE_TTL_MINUTES", 60)
    # Minimum spacing between two live yfinance calls (they share one
    # rate-limit budget across symbols).
    FUNDAMENTALS_MIN_INTERVAL_SECONDS: float = _get_float("FUNDAMENTALS_MIN_INTERVAL_SECONDS", 2.0)
    # After an HTTP 429, back off from yfinance entirely for this long.
    FUNDAMENTALS_RATE_LIMIT_BACKOFF_MINUTES: float = _get_float("FUNDAMENTALS_RATE_LIMIT_BACKOFF_MINUTES", 10)

    # --- New feature toggles (all free to run) ---
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
            warnings.append("GEMINI_API_KEY is not set. News/Sentiment agent will be disabled.")
        if not self.GROQ_API_KEY:
            warnings.append("GROQ_API_KEY is not set. Technical + CIO agents will be disabled.")
        if not self.OPENROUTER_API_KEY:
            warnings.append("OPENROUTER_API_KEY is not set. Risk agent will be disabled.")
        return warnings


settings = Settings()
