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

    # --- Model identifiers ---
    # Current default: Gemini 3.6 Flash via the Interactions API.
    # Override with the GEMINI_MODEL environment variable ("models/..." prefix
    # is tolerated and stripped).
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    GROQ_TECH_MODEL: str = "llama-3.1-8b-instant"
    GROQ_CIO_MODEL: str = "llama-3.3-70b-versatile"
    OPENROUTER_RISK_MODEL: str = "deepseek/deepseek-r1:free"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # --- Bot behavior ---
    TRADE_UNIVERSE: list = [
        t.strip().upper()
        for t in os.getenv("TRADE_UNIVERSE", "AAPL,MSFT,NVDA,TSLA,SPY").split(",")
        if t.strip()
    ]
    CYCLE_INTERVAL_MINUTES: int = _get_int("CYCLE_INTERVAL_MINUTES", 15)
    MAX_POSITION_PCT: float = _get_float("MAX_POSITION_PCT", 0.10)

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
