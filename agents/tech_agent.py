"""
agents/tech_agent.py

Technical Agent. Interprets RSI, moving averages, and MACD for a given
symbol and returns a structured technical read used by the CIO agent.

Provider/model: configured centrally via services/llm_service.py
(default Groq — see GROQ_TECH_MODEL in config.py; no model id lives in
this file). The report carries provider/model/llm_status/latency_ms so
every cycle record says exactly which model produced it and how the
request fared.
"""

import logging

from config import settings
from services import llm_service

logger = logging.getLogger("tech_agent")

SYSTEM_INSTRUCTIONS = """You are a technical analysis expert for equities.
You will be given the latest technical indicator readings for a stock.
Interpret RSI (overbought/oversold), the relationship between price and
its 50-day/200-day moving averages (trend direction, golden/death cross),
and MACD vs its signal line (momentum).

Respond ONLY with a single valid JSON object, no markdown fences, no preamble, in this exact shape:
{
  "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": <float 0.0 to 1.0>,
  "summary": "<one to two sentence explanation referencing the actual numbers>"
}
"""


def analyze_technicals(symbol: str, indicators: dict) -> dict:
    """
    Args:
        symbol: ticker symbol
        indicators: dict as returned by alpaca_service.get_indicators()

    Returns:
        {
          "agent": "technical",
          "symbol": symbol,
          "signal": "BULLISH"/"BEARISH"/"NEUTRAL",
          "confidence": float,
          "summary": str,
          "error": str | None,
          "provider": str, "model": str, "llm_status": str, "latency_ms": float
        }
    """
    route = llm_service.route_info("technical")
    base_result = {
        "agent": "technical",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "confidence": 0.0,
        "summary": "",
        "error": None,
        "provider": route["provider"],
        "model": route["model"],
        "llm_status": "NOT_CONFIGURED",
        "latency_ms": None,
    }

    if not settings.GROQ_API_KEY:
        base_result["error"] = "GROQ_API_KEY not configured."
        base_result["summary"] = "Technical agent disabled: missing API key."
        return base_result

    if indicators.get("error"):
        base_result["llm_status"] = "SKIPPED_NO_DATA"
        base_result["error"] = indicators["error"]
        base_result["summary"] = "No usable indicator data available."
        return base_result

    indicator_text = (
        f"Symbol: {symbol}\n"
        f"Latest close: {indicators.get('latest_close')}\n"
        f"RSI(14): {indicators.get('rsi_14')}\n"
        f"SMA(50): {indicators.get('sma_50')}\n"
        f"SMA(200): {indicators.get('sma_200')}\n"
        f"MACD: {indicators.get('macd')}\n"
        f"MACD Signal: {indicators.get('macd_signal')}\n"
        f"Recent closes (oldest to newest): {indicators.get('recent_closes')}\n"
    )

    result = llm_service.call_json(
        "technical",
        system=SYSTEM_INSTRUCTIONS,
        user=indicator_text,
        temperature=0.2,
        max_tokens=400,
        symbol=symbol,
    )

    base_result["llm_status"] = result.status
    base_result["model"] = result.model
    base_result["latency_ms"] = result.latency_ms

    if not result.ok:
        logger.error(f"Technical agent failed for {symbol}: {result.error}")
        base_result["error"] = result.error
        base_result["summary"] = "Technical agent encountered an error; defaulting to NEUTRAL."
        return base_result

    parsed = result.parsed
    base_result["signal"] = str(parsed.get("signal", "NEUTRAL")).upper()
    base_result["confidence"] = float(parsed.get("confidence", 0.0))
    base_result["summary"] = str(parsed.get("summary", ""))
    return base_result
