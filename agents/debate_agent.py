"""
agents/debate_agent.py

Bull vs Bear debate -- adapted from TradingAgents' bull/bear researcher
debate step. Instead of feeding the CIO a single pass per agent, two
opposing personas independently argue the strongest possible case for
buying vs. avoiding/selling the symbol, using the SAME underlying data
(technical + news + fundamentals) that the other agents already saw.

This surfaces one-sided reasoning: if the bear case is much stronger
than the bull case despite a nominally "BULLISH" technical read, the
CIO sees that tension directly instead of it being averaged away.

Uses Groq's free tier (same provider as tech_agent/cio_agent), just two
extra short completions per symbol per cycle.
"""

import json
import logging
import re

from config import settings

logger = logging.getLogger("debate_agent")

BULL_INSTRUCTIONS = """You are the Bull Researcher on a trading desk. Your job is to
build the STRONGEST possible case for why this stock should be bought or held,
using the data provided. Be persuasive but honest -- do not invent facts not
supported by the data. If the data genuinely does not support a bull case,
say so plainly rather than forcing one.

Respond ONLY with a single valid JSON object, no markdown fences, no preamble:
{
  "strength": <float 0.0 to 1.0, how strong the bull case actually is>,
  "summary": "<two to three sentence bull argument>"
}
"""

BEAR_INSTRUCTIONS = """You are the Bear Researcher on a trading desk. Your job is to
build the STRONGEST possible case for why this stock should be avoided or sold,
using the data provided. Be persuasive but honest -- do not invent facts not
supported by the data. If the data genuinely does not support a bear case,
say so plainly rather than forcing one.

Respond ONLY with a single valid JSON object, no markdown fences, no preamble:
{
  "strength": <float 0.0 to 1.0, how strong the bear case actually is>,
  "summary": "<two to three sentence bear argument>"
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


def _build_context(symbol: str, tech_report: dict, news_report: dict,
                    fundamentals_report: dict) -> str:
    parts = [
        f"Symbol: {symbol}",
        f"Technical read: {tech_report.get('signal')} ({tech_report.get('summary')})",
        f"News sentiment: {news_report.get('sentiment')} ({news_report.get('summary')})",
    ]
    if fundamentals_report:
        parts.append(
            f"Fundamentals: {fundamentals_report.get('signal')} ({fundamentals_report.get('summary')})"
        )
    return "\n".join(parts)


def _run_side(system_instructions: str, context_text: str) -> dict:
    if not settings.GROQ_API_KEY:
        return {"strength": 0.0, "summary": "Debate disabled: missing GROQ_API_KEY.", "error": "GROQ_API_KEY not configured."}

    try:
        from groq import Groq

        client = Groq(api_key=settings.GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=settings.GROQ_TECH_MODEL,  # reuse the cheap/fast free-tier model
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": context_text},
            ],
            temperature=0.4,
            max_tokens=300,
        )
        raw_text = completion.choices[0].message.content or ""
        parsed = _extract_json(raw_text)
        return {
            "strength": float(parsed.get("strength", 0.0)),
            "summary": str(parsed.get("summary", "")),
            "error": None,
        }
    except Exception as e:
        logger.error(f"Debate side failed: {e}")
        return {"strength": 0.0, "summary": "Debate agent encountered an error.", "error": str(e)}


def run_debate(symbol: str, tech_report: dict, news_report: dict,
               fundamentals_report: dict = None) -> dict:
    """
    Runs the bull and bear cases independently and returns both plus a
    simple "edge" score (bull_strength - bear_strength) the CIO can use
    as an extra signal alongside the individual agent reports.

    Returns:
        {
          "agent": "debate",
          "symbol": symbol,
          "bull_strength": float,
          "bull_summary": str,
          "bear_strength": float,
          "bear_summary": str,
          "edge": float,  # positive = bull case wins, negative = bear case wins
        }
    """
    if not settings.ENABLE_DEBATE:
        return {
            "agent": "debate", "symbol": symbol,
            "bull_strength": 0.0, "bull_summary": "Debate disabled via config.",
            "bear_strength": 0.0, "bear_summary": "Debate disabled via config.",
            "edge": 0.0,
        }

    context_text = _build_context(symbol, tech_report, news_report, fundamentals_report)
    bull = _run_side(BULL_INSTRUCTIONS, context_text)
    bear = _run_side(BEAR_INSTRUCTIONS, context_text)

    return {
        "agent": "debate",
        "symbol": symbol,
        "bull_strength": bull["strength"],
        "bull_summary": bull["summary"],
        "bear_strength": bear["strength"],
        "bear_summary": bear["summary"],
        "edge": round(bull["strength"] - bear["strength"], 3),
    }
