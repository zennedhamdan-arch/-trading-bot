"""
agents/tech_agent.py

Technical Agent powered by Groq (llama-3.1-8b-instant).
Interprets RSI, moving averages, and MACD for a given symbol and
returns a structured technical read used by the CIO agent.
"""

import json
import logging
import re

from config import settings

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


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")
    return json.loads(match.group(0))


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
          "error": str | None
        }
    """
    base_result = {
        "agent": "technical",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "confidence": 0.0,
        "summary": "",
        "error": None,
    }

    if not settings.GROQ_API_KEY:
        base_result["error"] = "GROQ_API_KEY not configured."
        base_result["summary"] = "Technical agent disabled: missing API key."
        return base_result

    if indicators.get("error"):
        base_result["error"] = indicators["error"]
        base_result["summary"] = "No usable indicator data available."
        return base_result

    try:
        from groq import Groq

        client = Groq(api_key=settings.GROQ_API_KEY)

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

        completion = client.chat.completions.create(
            model=settings.GROQ_TECH_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": indicator_text},
            ],
            temperature=0.2,
            max_tokens=400,
        )

        raw_text = completion.choices[0].message.content or ""
        parsed = _extract_json(raw_text)

        base_result["signal"] = str(parsed.get("signal", "NEUTRAL")).upper()
        base_result["confidence"] = float(parsed.get("confidence", 0.0))
        base_result["summary"] = str(parsed.get("summary", ""))
        return base_result

    except Exception as e:
        logger.error(f"Technical agent failed for {symbol}: {e}")
        base_result["error"] = str(e)
        base_result["summary"] = "Technical agent encountered an error; defaulting to NEUTRAL."
        return base_result
