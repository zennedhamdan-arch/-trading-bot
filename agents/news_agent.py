"""
agents/news_agent.py

Sentiment/News Agent powered by Google Gemini (GEMINI_MODEL, default
gemini-3.6-flash, via the Interactions API).
Analyzes market news headlines for a given ticker and returns a
structured sentiment assessment used by the CIO agent.
"""

import logging

from config import settings
from services import gemini_service

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
    parses through services.gemini_service (same logic)."""
    return gemini_service._extract_json(text)


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
          "error": str | None
        }
    """
    base_result = {
        "agent": "news",
        "symbol": symbol,
        "sentiment": "NEUTRAL",
        "confidence": 0.0,
        "summary": "",
        "key_headline": "",
        "error": None,
    }

    if not settings.GEMINI_API_KEY:
        base_result["error"] = "GEMINI_API_KEY not configured."
        base_result["summary"] = "News agent disabled: missing API key."
        return base_result

    if not headlines:
        base_result["summary"] = "No recent headlines available for this symbol."
        return base_result

    try:
        headlines_block = "\n".join(f"- {h}" for h in headlines[:15])
        prompt = f"Ticker: {symbol}\nHeadlines:\n{headlines_block}"

        parsed = gemini_service.generate_json(
            system_instructions=SYSTEM_INSTRUCTIONS,
            input_text=prompt,
        )

        base_result["sentiment"] = str(parsed.get("sentiment", "NEUTRAL")).upper()
        base_result["confidence"] = float(parsed.get("confidence", 0.0))
        base_result["summary"] = str(parsed.get("summary", ""))
        base_result["key_headline"] = str(parsed.get("key_headline", ""))
        return base_result

    except Exception as e:
        logger.error(f"News agent failed for {symbol}: {e}")
        base_result["error"] = str(e)
        base_result["summary"] = "News agent encountered an error; defaulting to NEUTRAL."
        return base_result
