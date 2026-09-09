"""
agents/fundamentals_agent.py

Fundamentals Analyst. Consumes the NORMALIZED fundamentals from the
FundamentalsProvider abstraction (services/fundamentals_service.py) and has
an LLM interpret them into a directional signal for the CIO.

yfinance is gone: no Yahoo scraping, no cookies/crumbs, no unofficial
endpoints, and no trading decision depends on Yahoo being available. When
no fundamentals provider is configured (FUNDAMENTALS_PROVIDER=none, the
default) the agent returns an explicit DATA_UNAVAILABLE result and the
cycle continues on price data, technicals, news and risk.

Provider/model for the interpretation: routed via services/llm_service.py
(LLM_FUNDAMENTALS_PROVIDER, default Gemini / GEMINI_MODEL).

LLM-call reduction: the interpretation is cached per (symbol, exact
metrics); unchanged metrics are never re-sent (labeled "cached": true).
"""

import hashlib
import json
import logging
import time

from config import settings
from services import llm_service

logger = logging.getLogger("fundamentals_agent")

SYSTEM_INSTRUCTIONS = """You are a fundamentals analyst for equities.
You will be given the fundamental metrics available for a company (any
metric may be missing/unavailable — treat missing data as unknown, never
as zero): P/E ratio, EPS, revenue, profit margin, return on equity, and
debt-to-equity ratio, plus market capitalization for scale.

Judge whether the company's fundamentals support a bullish, bearish, or
neutral medium-term outlook. High debt with weak revenue growth is a
bearish signal. Strong revenue growth with reasonable valuation is bullish.
An extremely high P/E with slowing growth is a caution flag. If most
metrics are unavailable, say so and lean NEUTRAL rather than guessing.

Respond ONLY with a single valid JSON object, no markdown fences, no preamble:
{
  "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": <float 0.0 to 1.0>,
  "summary": "<one to two sentence explanation referencing the actual numbers>"
}
"""


def _extract_json(text: str) -> dict:
    """Tolerant JSON extraction — kept for local use/testing; the live path
    parses through services.llm_service (same logic)."""
    return llm_service.extract_json(text)


# (symbol, metrics_hash) -> (expires_at_monotonic, report_fields, stored_at)
_analysis_cache: dict = {}


def reset_fundamentals_cache() -> None:
    """Test/introspection hook: clears the LLM analysis cache."""
    _analysis_cache.clear()


def _is_unavailable(fundamentals: dict) -> bool:
    """True when the normalized fundamentals payload says the data is not
    available (handles both the new normalized shape and the legacy
    {"error": ...} shape used by older tests)."""
    if not fundamentals:
        return True
    if fundamentals.get("error"):
        return True
    status = fundamentals.get("status")
    if status and status != "OK":
        return True
    return False


def analyze_fundamentals(symbol: str, fundamentals: dict) -> dict:
    """
    Args:
        symbol: ticker symbol
        fundamentals: normalized output of fundamentals_service.get_fundamentals()

    Returns:
        {
          "agent": "fundamentals",
          "symbol": symbol,
          "signal": "BULLISH"/"BEARISH"/"NEUTRAL",
          "confidence": float,
          "summary": str,
          "error": str | None,
          "provider": str, "model": str, "llm_status": str,
          "latency_ms": float | null, "cached": bool
        }
    """
    route = llm_service.route_info("fundamentals")
    base_result = {
        "agent": "fundamentals",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "confidence": 0.0,
        "summary": "",
        "error": None,
        "provider": route["provider"],
        "model": route["model"],
        "llm_status": "NOT_CONFIGURED",
        "latency_ms": None,
        "cached": False,
    }

    if not settings.ENABLE_FUNDAMENTALS_AGENT:
        base_result["llm_status"] = "SKIPPED_DISABLED"
        base_result["summary"] = "Fundamentals agent disabled via config."
        return base_result

    if _is_unavailable(fundamentals):
        base_result["llm_status"] = "SKIPPED_NO_DATA"
        reason = (
            fundamentals.get("reason")
            or fundamentals.get("error")
            or f"status={fundamentals.get('status', 'MISSING')}"
        )
        base_result["error"] = f"DATA_UNAVAILABLE: {reason}"
        base_result["summary"] = "No usable fundamentals data available."
        return base_result

    # 1. Reuse the previous LLM interpretation while the exact metrics are
    # unchanged — fundamentals change at most daily.
    metrics_payload = json.dumps(fundamentals, sort_keys=True, default=str)
    key = (symbol, hashlib.sha1(metrics_payload.encode("utf-8")).hexdigest())
    ttl = max(0.0, float(settings.FUNDAMENTALS_ANALYSIS_CACHE_TTL_MINUTES) * 60.0)
    cached = _analysis_cache.get(key)
    if cached and cached[0] > time.monotonic():
        report = dict(cached[1])
        report["cached"] = True
        return report

    # 2. Build the evidence text from PRESENT fields only; unavailable
    # metrics are labeled as such — never fabricated, never defaulted.
    def _fmt(label, value, fmt="{}"):
        return f"{label}: " + (fmt.format(value) if value is not None else "not available")

    metrics_text = (
        f"Symbol: {symbol}\n"
        f"Data provider: {fundamentals.get('provider', 'unknown')}\n"
        + "\n".join([
            _fmt("Market cap", fundamentals.get("market_cap"), "{:,.0f}"),
            _fmt("P/E ratio", fundamentals.get("pe_ratio")),
            _fmt("EPS", fundamentals.get("eps")),
            _fmt("Revenue", fundamentals.get("revenue"), "{:,.0f}"),
            _fmt("Profit margin", fundamentals.get("profit_margin")),
            _fmt("Return on equity", fundamentals.get("roe")),
            _fmt("Debt-to-equity", fundamentals.get("debt_to_equity")),
        ])
        + "\n"
    )

    result = llm_service.call_json(
        "fundamentals",
        system=SYSTEM_INSTRUCTIONS,
        user=metrics_text,
        temperature=0.2,
        max_tokens=400,
        symbol=symbol,
    )

    base_result["llm_status"] = result.status
    base_result["model"] = result.model
    base_result["latency_ms"] = result.latency_ms

    if not result.ok:
        logger.error(f"Fundamentals agent failed for {symbol}: {result.error}")
        base_result["error"] = result.error
        base_result["summary"] = (
            "Fundamentals agent disabled: missing API key."
            if result.status == "NOT_CONFIGURED"
            else "Fundamentals agent encountered an error; defaulting to NEUTRAL."
        )
        return base_result

    parsed = result.parsed
    base_result["signal"] = str(parsed.get("signal", "NEUTRAL")).upper()
    base_result["confidence"] = float(parsed.get("confidence", 0.0))
    base_result["summary"] = str(parsed.get("summary", ""))

    if ttl > 0:
        stored = dict(base_result)
        _analysis_cache[key] = (time.monotonic() + ttl, stored, time.monotonic())
    return base_result
