"""
agents/fundamentals_agent.py

Fundamentals Analyst -- adapted from TradingAgents' fundamentals analyst
role. Pulls basic company financials via yfinance (free, no API key)
and has an LLM (Gemini, same free tier already used by news_agent)
interpret them into a directional signal for the CIO.
"""

import json
import logging
import re

from config import settings

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
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")
    return json.loads(match.group(0))


def get_fundamentals(symbol: str) -> dict:
    """Fetches basic fundamentals via yfinance. Free, no API key required."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        return {
            "symbol": symbol,
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "revenue_growth": info.get("revenueGrowth"),
            "profit_margin": info.get("profitMargins"),
            "debt_to_equity": info.get("debtToEquity"),
            "return_on_equity": info.get("returnOnEquity"),
            "error": None,
        }
    except Exception as e:
        logger.error(f"get_fundamentals failed for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


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
        from google import genai

        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        metrics_text = (
            f"Symbol: {symbol}\n"
            f"P/E ratio: {fundamentals.get('pe_ratio')}\n"
            f"Forward P/E: {fundamentals.get('forward_pe')}\n"
            f"Revenue growth (YoY): {fundamentals.get('revenue_growth')}\n"
            f"Profit margin: {fundamentals.get('profit_margin')}\n"
            f"Debt-to-equity: {fundamentals.get('debt_to_equity')}\n"
            f"Return on equity: {fundamentals.get('return_on_equity')}\n"
        )
        prompt = f"{SYSTEM_INSTRUCTIONS}\n\n{metrics_text}"

        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
        )

        raw_text = response.text or ""
        parsed = _extract_json(raw_text)

        base_result["signal"] = str(parsed.get("signal", "NEUTRAL")).upper()
        base_result["confidence"] = float(parsed.get("confidence", 0.0))
        base_result["summary"] = str(parsed.get("summary", ""))
        return base_result

    except Exception as e:
        logger.error(f"Fundamentals agent failed for {symbol}: {e}")
        base_result["error"] = str(e)
        base_result["summary"] = "Fundamentals agent encountered an error; defaulting to NEUTRAL."
        return base_result
