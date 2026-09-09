"""
agents/tech_agent.py

Technical Agent. Interprets the DETERMINISTIC analytics computed by
services/alpaca_service.get_indicators() (RSI, SMAs, EMA, MACD, ATR,
volatility, drawdown, returns, rule-based signal) into a structured
technical read used by the CIO agent. All financial math happens in
Python; the LLM only reasons over pre-computed evidence.

Provider/model: routed via services/llm_service.py (LLM_TECH_PROVIDER,
default Groq / GROQ_TECH_MODEL; no model id lives in this file). The
report carries provider/model/llm_status/latency_ms so every cycle
record says exactly which model produced it and how the request fared.
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
          "evidence_status": "AVAILABLE"|"UNAVAILABLE"|"ERROR",
          "signal": "BULLISH"/"BEARISH"/"NEUTRAL" | None (None unless AVAILABLE),
          "confidence": float | None,
          "summary": str,
          "error": str | None,
          "provider": str, "model": str, "llm_status": str, "latency_ms": float
        }

    NEUTRAL is only ever a VERDICT on successfully analyzed evidence; a
    failed agent yields evidence_status UNAVAILABLE/ERROR with a null
    signal and null confidence — never a fake NEUTRAL.
    """
    route = llm_service.route_info("technical")
    base_result = {
        "agent": "technical",
        "symbol": symbol,
        "evidence_status": None,
        "signal": None,
        "confidence": None,
        "summary": "",
        "error": None,
        "provider": route["provider"],
        "model": route["model"],
        "llm_status": "NOT_CONFIGURED",
        "latency_ms": None,
    }

    if indicators.get("error"):
        base_result["llm_status"] = "SKIPPED_NO_DATA"
        base_result["evidence_status"] = "UNAVAILABLE"
        base_result["error"] = indicators["error"]
        base_result["summary"] = "No usable indicator data available — no technical verdict."
        return base_result

    # Deterministic evidence (computed in Python, NOT by the LLM): the model
    # interprets these numbers; it never calculates them.
    indicator_text = (
        f"Symbol: {symbol}\n"
        f"Data feed: {indicators.get('feed', 'iex')}\n"
        f"Latest close: {indicators.get('latest_close')}\n"
        f"RSI(14): {indicators.get('rsi_14')}\n"
        f"SMA(20): {indicators.get('sma_20')}\n"
        f"SMA(50): {indicators.get('sma_50')}\n"
        f"SMA(200): {indicators.get('sma_200')}\n"
        f"EMA(20): {indicators.get('ema_20')}\n"
        f"MACD: {indicators.get('macd')}\n"
        f"MACD Signal: {indicators.get('macd_signal')}\n"
        f"ATR(14): {indicators.get('atr_14')}\n"
        f"Annualized volatility: {indicators.get('volatility_annualized')}\n"
        f"Max drawdown: {indicators.get('max_drawdown')}\n"
        f"Return 1d/5d/20d: {indicators.get('return_1d')} / "
        f"{indicators.get('return_5d')} / {indicators.get('return_20d')}\n"
        f"Rule-based signal (deterministic): {indicators.get('technical_signal')} "
        f"components={indicators.get('technical_components')}\n"
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
        from services.evidence import state_for_llm_status
        base_result["evidence_status"] = state_for_llm_status(result.status)
        base_result["error"] = result.error
        base_result["summary"] = "Technical agent encountered an error — no technical verdict."
        return base_result

    parsed = result.parsed
    base_result["evidence_status"] = "AVAILABLE"
    base_result["signal"] = str(parsed.get("signal", "NEUTRAL")).upper()
    base_result["confidence"] = float(parsed.get("confidence", 0.0))
    base_result["summary"] = str(parsed.get("summary", ""))
    return base_result
