"""
agents/fundamentals_agent.py

Fundamentals Analyst -- adapted from TradingAgents' fundamentals analyst
role. Pulls basic company financials via yfinance (free, no API key)
and has an LLM (Gemini, GEMINI_MODEL, via the Interactions API -- same
free tier already used by news_agent) interpret them into a directional
signal for the CIO.

Data-layer behavior (Yahoo Finance is aggressively rate-limited, especially
from cloud IPs):
  - Successful fetches are cached per symbol (FUNDAMENTALS_CACHE_TTL_MINUTES,
    default 60) so repeated cycles do not re-request identical data.
  - Live calls are spaced by FUNDAMENTALS_MIN_INTERVAL_SECONDS.
  - HTTP 429 / rate-limit responses (including yfinance's JSONDecodeError
    symptom of an HTML error page) trigger a global backoff
    (FUNDAMENTALS_RATE_LIMIT_BACKOFF_MINUTES) during which no further
    yfinance calls are attempted.
  - Any failure returns a clean {"error": "DATA_UNAVAILABLE: ..."} result —
    never a fabricated value, and never an exception that could take down
    the trading cycle.
"""

import logging
import time

from config import settings
from services import gemini_service

logger = logging.getLogger("fundamentals_agent")

SYSTEM_INSTRUCTIONS = """You are a fundamentals analyst for equities.
You will be given basic financial metrics for a company: P/E ratio,
revenue growth, profit margins, and debt-to-equity ratio.

Judge whether the company's fundamentals support a bullish, bearish, or
neutral medium-term outlook. High debt with weak revenue growth is a
bearish signal. Strong revenue growth with reasonable valuation is bullish.
An extremely high P/E with slowing growth is a caution flag.

Respond ONLY with a single valid JSON object, no markdown fences, no preamble:
{
  "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": <float 0.0 to 1.0>,
  "summary": "<one to two sentence explanation referencing the actual numbers>"
}
"""


def _extract_json(text: str) -> dict:
    """Tolerant JSON extraction — kept for local use/testing; the live path
    parses through services.gemini_service (same logic)."""
    return gemini_service._extract_json(text)


# ---------------------------------------------------------------------------
# Data layer: cache + rate limiting for yfinance
# ---------------------------------------------------------------------------

# symbol -> (expires_at_monotonic, fundamentals_dict)
_fundamentals_cache: dict = {}

# monotonic timestamps for pacing and 429 backoff
_last_yf_call: float = 0.0
_yf_backoff_until: float = 0.0


def _is_rate_limit_error(exc: Exception) -> bool:
    """True when the exception represents Yahoo rate limiting (HTTP 429).

    Covers yfinance's dedicated YFRateLimitError, HTTP 429 errors, and the
    JSONDecodeError ('Expecting value: line 1 column 1') that surfaces when
    yfinance tries to parse an HTML/empty error page returned alongside a
    429.
    """
    try:
        from yfinance.exceptions import YFRateLimitError
        if isinstance(exc, YFRateLimitError):
            return True
    except ImportError:
        pass
    msg = str(exc)
    if "429" in msg or "Too Many Requests" in msg or "rate" in msg.lower():
        return True
    # json.JSONDecodeError and its message shape
    if exc.__class__.__name__ == "JSONDecodeError" or "Expecting value" in msg:
        return True
    # exceptions may be wrapped; check the chain
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause is not None and cause is not exc:
        return _is_rate_limit_error(cause)
    return False


def reset_fundamentals_cache() -> None:
    """Test/introspection hook: clears the cache and backoff state."""
    global _yf_backoff_until, _last_yf_call
    _fundamentals_cache.clear()
    _yf_backoff_until = 0.0
    _last_yf_call = 0.0


def get_fundamentals(symbol: str) -> dict:
    """Fetches basic fundamentals via yfinance. Free, no API key required.

    Cached, rate-limited, and fully defensive: any provider failure
    (including HTTP 429) returns {"symbol": ..., "error": "DATA_UNAVAILABLE: ..."}
    rather than raising.
    """
    now = time.monotonic()
    global _last_yf_call, _yf_backoff_until

    # 1. Serve from cache while fresh.
    cached = _fundamentals_cache.get(symbol)
    if cached and cached[0] > now:
        return cached[1]

    # 2. Honor a global 429 backoff — do not hammer Yahoo while limited.
    if now < _yf_backoff_until:
        logger.warning(
            f"Fundamentals for {symbol}: DATA_UNAVAILABLE (yfinance rate-limit backoff active)."
        )
        return {"symbol": symbol, "error": "DATA_UNAVAILABLE: Yahoo Finance rate-limited (HTTP 429); backing off."}

    # 3. Pace live calls.
    min_interval = max(0.0, float(settings.FUNDAMENTALS_MIN_INTERVAL_SECONDS))
    wait = _last_yf_call + min_interval - now
    if wait > 0:
        time.sleep(wait)
    _last_yf_call = time.monotonic()

    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        if not info:
            raise ValueError("yfinance returned an empty info dict.")

        result = {
            "symbol": symbol,
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "revenue_growth": info.get("revenueGrowth"),
            "profit_margin": info.get("profitMargins"),
            "debt_to_equity": info.get("debtToEquity"),
            "return_on_equity": info.get("returnOnEquity"),
            "error": None,
        }
        # Cache success for the full TTL.
        ttl = max(60.0, float(settings.FUNDAMENTALS_CACHE_TTL_MINUTES) * 60.0)
        _fundamentals_cache[symbol] = (time.monotonic() + ttl, result)
        return result

    except Exception as e:
        if _is_rate_limit_error(e):
            backoff = max(30.0, float(settings.FUNDAMENTALS_RATE_LIMIT_BACKOFF_MINUTES) * 60.0)
            _yf_backoff_until = time.monotonic() + backoff
            logger.warning(
                f"Fundamentals for {symbol}: Yahoo rate limited (429). "
                f"Backing off {backoff/60:.0f} min. Error: {e}"
            )
            result = {"symbol": symbol, "error": "DATA_UNAVAILABLE: Yahoo Finance rate-limited (HTTP 429)."}
        else:
            logger.error(f"get_fundamentals failed for {symbol}: {e}")
            result = {"symbol": symbol, "error": f"DATA_UNAVAILABLE: {e}"}
        # Cache the unavailability briefly so a single cycle does not retry
        # the same failing symbol repeatedly.
        _fundamentals_cache[symbol] = (time.monotonic() + 60.0, result)
        return result


def analyze_fundamentals(symbol: str, fundamentals: dict) -> dict:
    """
    Args:
        symbol: ticker symbol
        fundamentals: output of get_fundamentals()

    Returns:
        {
          "agent": "fundamentals",
          "symbol": symbol,
          "signal": "BULLISH"/"BEARISH"/"NEUTRAL",
          "confidence": float,
          "summary": str,
          "error": str | None
        }
    """
    base_result = {
        "agent": "fundamentals",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "confidence": 0.0,
        "summary": "",
        "error": None,
    }

    if not settings.ENABLE_FUNDAMENTALS_AGENT:
        base_result["summary"] = "Fundamentals agent disabled via config."
        return base_result

    if fundamentals.get("error"):
        base_result["error"] = fundamentals["error"]
        base_result["summary"] = "No usable fundamentals data available."
        return base_result

    if not settings.GEMINI_API_KEY:
        base_result["error"] = "GEMINI_API_KEY not configured."
        base_result["summary"] = "Fundamentals agent disabled: missing API key."
        return base_result

    try:
        metrics_text = (
            f"Symbol: {symbol}\n"
            f"P/E ratio: {fundamentals.get('pe_ratio')}\n"
            f"Forward P/E: {fundamentals.get('forward_pe')}\n"
            f"Revenue growth (YoY): {fundamentals.get('revenue_growth')}\n"
            f"Profit margin: {fundamentals.get('profit_margin')}\n"
            f"Debt-to-equity: {fundamentals.get('debt_to_equity')}\n"
            f"Return on equity: {fundamentals.get('return_on_equity')}\n"
        )

        parsed = gemini_service.generate_json(
            system_instructions=SYSTEM_INSTRUCTIONS,
            input_text=metrics_text,
        )

        base_result["signal"] = str(parsed.get("signal", "NEUTRAL")).upper()
        base_result["confidence"] = float(parsed.get("confidence", 0.0))
        base_result["summary"] = str(parsed.get("summary", ""))
        return base_result

    except Exception as e:
        logger.error(f"Fundamentals agent failed for {symbol}: {e}")
        base_result["error"] = str(e)
        base_result["summary"] = "Fundamentals agent encountered an error; defaulting to NEUTRAL."
        return base_result
