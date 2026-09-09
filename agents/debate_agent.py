"""
agents/debate_agent.py

Bull vs Bear debate -- adapted from TradingAgents' bull/bear researcher
debate step. Instead of feeding the CIO a single pass per agent, two
opposing personas argue the strongest possible case for buying vs.
avoiding/selling the symbol, using the SAME underlying data (technical +
news + fundamentals) that the other agents already saw.

This surfaces one-sided reasoning: if the bear case is much stronger
than the bull case despite a nominally "BULLISH" technical read, the
CIO sees that tension directly instead of it being averaged away.

Provider/model: configured centrally via services/llm_service.py
(GROQ_DEBATE_MODEL; no model id lives in this file).

LLM-call efficiency: bull and bear cases are produced by ONE structured
request per symbol (the model returns both sides in a single JSON
object) rather than two separate completions with identical context —
same reasoning quality, half the calls.
"""

import logging

from config import settings
from services import llm_service

logger = logging.getLogger("debate_agent")

DEBATE_INSTRUCTIONS = """You are running a bull vs. bear debate on a trading desk for one stock.
Using ONLY the data provided, write the STRONGEST possible case FOR buying or holding
the stock (the bull case), and the STRONGEST possible case AGAINST it (the bear case).
Be persuasive but honest -- do not invent facts not supported by the data. If the data
genuinely does not support a case, say so plainly rather than forcing one.

Respond ONLY with a single valid JSON object, no markdown fences, no preamble, in this exact shape:
{
  "bull_strength": <float 0.0 to 1.0, how strong the bull case actually is>,
  "bull_summary": "<two to three sentence bull argument>",
  "bear_strength": <float 0.0 to 1.0, how strong the bear case actually is>,
  "bear_summary": "<two to three sentence bear argument>"
}
"""


def _extract_json(text: str) -> dict:
    """Tolerant JSON extraction — kept for local use/testing; the live path
    parses through services.llm_service (same logic)."""
    return llm_service.extract_json(text)


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


def run_debate(symbol: str, tech_report: dict, news_report: dict,
               fundamentals_report: dict = None) -> dict:
    """
    Runs the bull and bear cases in one structured request and returns both
    plus a simple "edge" score (bull_strength - bear_strength) the CIO can
    use as an extra signal alongside the individual agent reports.

    Returns:
        {
          "agent": "debate",
          "symbol": symbol,
          "bull_strength": float,
          "bull_summary": str,
          "bear_strength": float,
          "bear_summary": str,
          "edge": float,  # positive = bull case wins, negative = bear case wins
          "error": str | null,
          "provider": str, "model": str, "llm_status": str, "latency_ms": float | null
        }
    """
    route = llm_service.route_info("debate")
    base_result = {
        "agent": "debate",
        "symbol": symbol,
        "bull_strength": 0.0,
        "bull_summary": "",
        "bear_strength": 0.0,
        "bear_summary": "",
        "edge": 0.0,
        "error": None,
        "provider": route["provider"],
        "model": route["model"],
        "llm_status": "NOT_CONFIGURED",
        "latency_ms": None,
    }

    if not settings.ENABLE_DEBATE:
        base_result["llm_status"] = "SKIPPED_DISABLED"
        base_result["bull_summary"] = "Debate disabled via config."
        base_result["bear_summary"] = "Debate disabled via config."
        return base_result

    if not settings.GROQ_API_KEY:
        base_result["error"] = "GROQ_API_KEY not configured."
        base_result["bull_summary"] = "Debate disabled: missing GROQ_API_KEY."
        base_result["bear_summary"] = "Debate disabled: missing GROQ_API_KEY."
        return base_result

    context_text = _build_context(symbol, tech_report, news_report, fundamentals_report)

    result = llm_service.call_json(
        "debate",
        system=DEBATE_INSTRUCTIONS,
        user=context_text,
        temperature=0.4,
        max_tokens=600,
        symbol=symbol,
    )

    base_result["llm_status"] = result.status
    base_result["model"] = result.model
    base_result["latency_ms"] = result.latency_ms

    if not result.ok:
        logger.error(f"Debate failed for {symbol}: {result.error}")
        base_result["error"] = result.error
        base_result["bull_summary"] = "Debate agent encountered an error."
        base_result["bear_summary"] = "Debate agent encountered an error."
        return base_result

    parsed = result.parsed
    base_result["bull_strength"] = float(parsed.get("bull_strength", 0.0))
    base_result["bull_summary"] = str(parsed.get("bull_summary", ""))
    base_result["bear_strength"] = float(parsed.get("bear_strength", 0.0))
    base_result["bear_summary"] = str(parsed.get("bear_summary", ""))
    base_result["edge"] = round(base_result["bull_strength"] - base_result["bear_strength"], 3)
    return base_result
