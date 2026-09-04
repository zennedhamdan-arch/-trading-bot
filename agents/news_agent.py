"""
agents/news_agent.py

Sentiment/News Agent powered by Google Gemini (gemini-2.5-flash).
Analyzes market news headlines for a given ticker and returns a
structured sentiment assessment used by the CIO agent.
"""

import json
import logging
import re

from config import settings

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
    """Best-effort extraction of a JSON object from a model response,
    tolerating stray markdown fences or extra text around the JSON."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")
    return json.loads(match.group(0))


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
        from google import genai

        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        headlines_block = "\n".join(f"- {h}" for h in headlines[:15])
        prompt = f"{SYSTEM_INSTRUCTIONS}\n\nTicker: {symbol}\nHeadlines:\n{headlines_block}"

        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
        )

        raw_text = response.text or ""
        parsed = _extract_json(raw_text)

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
