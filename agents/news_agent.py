"""
agents/news_agent.py

Sentiment/News Agent. Analyzes market news headlines for a given ticker
and returns a structured sentiment assessment used by the CIO agent.

Provider/model: configured centrally via services/llm_service.py
(default Gemini — see GEMINI_MODEL in config.py).

Gemini free-tier protection: the analysis is cached per (symbol, exact
headline set) for NEWS_ANALYSIS_CACHE_TTL_MINUTES. While the headlines
are unchanged, the cached analysis is reused and NO new Gemini request is
sent — the identical context is never re-sent. Cache hits are labeled
("cached": true) so reuse is always visible, never hidden.
"""

import hashlib
import json
import logging
import time

from config import settings
from services import llm_service

logger = logging.getLogger("news_agent")

SYSTEM_INSTRUCTIONS = """You are a financial news sentiment analyst.
You will be given a stock ticker and a list of recent news headlines.
Analyze the overall sentiment and its likely short-term impact on the stock price.

Respond ONLY with a single valid JSON object, no markdown fences, no preamble, in this exact shape:
{
  "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": <float 0.0 to 1.0>,
  "summary": "<one to two sentence explanation>",
  "key_headline": "<the single most impactful headline, or empty string if none provided>"
}
"""


def _extract_json(text: str) -> dict:
    """Tolerant JSON extraction — kept for local use/testing; the live path
    parses through services.llm_service (same logic)."""
    return llm_service.extract_json(text)


# ---------------------------------------------------------------------------
# Analysis cache: identical headlines -> reuse the previous LLM analysis
# ---------------------------------------------------------------------------

# (symbol, headlines_hash) -> (expires_at_monotonic, report_fields)
_analysis_cache: dict = {}


def reset_news_analysis_cache() -> None:
    """Test/introspection hook: clears the analysis cache."""
    _analysis_cache.clear()


def _cache_key(symbol: str, headlines: list) -> tuple:
    capped = [str(h) for h in headlines[:15]]
    payload = json.dumps(capped, ensure_ascii=False, sort_keys=True)
    return (symbol, hashlib.sha1(payload.encode("utf-8")).hexdigest())


def analyze_news(symbol: str, headlines: list) -> dict:
    """
    Args:
        symbol: ticker symbol, e.g. "AAPL"
        headlines: list of recent headline strings for this ticker

    Returns a dict:
        {
          "agent": "news",
          "symbol": symbol,
          "sentiment": "BULLISH"/"BEARISH"/"NEUTRAL",
          "confidence": float,
          "summary": str,
          "key_headline": str,
          "error": str | null,
          "provider": str, "model": str, "llm_status": str,
          "latency_ms": float | null, "cached": bool
        }
    """
    route = llm_service.route_info("news")
    base_result = {
        "agent": "news",
        "symbol": symbol,
        "sentiment": "NEUTRAL",
        "confidence": 0.0,
        "summary": "",
        "key_headline": "",
        "error": None,
        "provider": route["provider"],
        "model": route["model"],
        "llm_status": "NOT_CONFIGURED",
        "latency_ms": None,
        "cached": False,
    }

    if not settings.GEMINI_API_KEY:
        base_result["error"] = "GEMINI_API_KEY not configured."
        base_result["summary"] = "News agent disabled: missing API key."
        return base_result

    if not headlines:
        base_result["llm_status"] = "SKIPPED_NO_DATA"
        base_result["summary"] = "No recent headlines available for this symbol."
        return base_result

    # 1. Reuse the previous analysis while the exact headlines are unchanged
    # and the TTL has not expired — this is what keeps a 15-minute cycle
    # cadence inside the Gemini free tier's daily request quota.
    key = _cache_key(symbol, headlines)
    ttl = max(0.0, float(settings.NEWS_ANALYSIS_CACHE_TTL_MINUTES) * 60.0)
    cached = _analysis_cache.get(key)
    if cached and cached[0] > time.monotonic():
        report = dict(cached[1])
        report["cached"] = True
        report["cache_age_s"] = round(time.monotonic() - cached[2], 1) if len(cached) > 2 else None
        return report

    headlines_block = "\n".join(f"- {h}" for h in headlines[:15])
    prompt = f"Ticker: {symbol}\nHeadlines:\n{headlines_block}"

    result = llm_service.call_json(
        "news",
        system=SYSTEM_INSTRUCTIONS,
        user=prompt,
        temperature=0.2,
        max_tokens=400,
        symbol=symbol,
    )

    base_result["llm_status"] = result.status
    base_result["model"] = result.model
    base_result["latency_ms"] = result.latency_ms

    if not result.ok:
        logger.error(f"News agent failed for {symbol}: {result.error}")
        base_result["error"] = result.error
        base_result["summary"] = "News agent encountered an error; defaulting to NEUTRAL."
        return base_result

    parsed = result.parsed
    base_result["sentiment"] = str(parsed.get("sentiment", "NEUTRAL")).upper()
    base_result["confidence"] = float(parsed.get("confidence", 0.0))
    base_result["summary"] = str(parsed.get("summary", ""))
    base_result["key_headline"] = str(parsed.get("key_headline", ""))

    # Cache the successful analysis (failures are never cached as successes).
    if ttl > 0:
        stored = dict(base_result)
        _analysis_cache[key] = (time.monotonic() + ttl, stored, time.monotonic())
    return base_result
