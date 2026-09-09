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

Each evidence source is labeled with its state:
- AVAILABLE: the source produced a real analysis this cycle -- use it.
- UNAVAILABLE / ERROR / STALE: the source produced NOTHING this cycle. This is
  MISSING INFORMATION -- it is not neutral, not mildly bearish and not mildly
  bullish, and you must not count it as support for either side. Argue only
  from the sources marked AVAILABLE and say plainly what is not known.

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
                   fundamentals_report: dict, evidence: dict = None) -> str:
    """Builds the debate context with EXPLICIT evidence states so the
    debate never mistakes missing evidence for neutral evidence."""
    from services import evidence as evidence_service

    def _entry(name, report, stance_key):
        state = (evidence or {}).get("agents", {}).get(name, {}).get("state") \
            if evidence else None
        if not state:
            state = evidence_service._state_from_report(report)
        if state != "AVAILABLE" or not report:
            detail = (report or {}).get("summary") or "no analysis produced"
            return f"{state}: {detail}"
        stance = report.get(stance_key)
        conf = report.get("confidence")
        conf_txt = f", confidence {conf:.2f}" if isinstance(conf, (int, float)) else ""
        return f"AVAILABLE: {stance}{conf_txt} ({report.get('summary')})"

    parts = [f"Symbol: {symbol}"]
    parts.append(f"Technical evidence: {_entry('technical', tech_report, 'signal')}")
    parts.append(f"News evidence: {_entry('news', news_report, 'sentiment')}")
    if fundamentals_report is not None:
        parts.append(
            f"Fundamentals evidence: {_entry('fundamentals', fundamentals_report, 'signal')}"
        )
    else:
        parts.append("Fundamentals evidence: OFF (not part of this pipeline)")
    if evidence:
        parts.append("")
        parts.append("Evidence quality: " + str(evidence.get("quality")))
        if evidence.get("missing"):
            parts.append(
                "Missing evidence sources: " + ", ".join(evidence["missing"])
                + " — treat as missing information, NOT neutral."
            )
    return "\n".join(parts)


def run_debate(symbol: str, tech_report: dict, news_report: dict,
               fundamentals_report: dict = None, evidence: dict = None) -> dict:
    """
    Runs the bull and bear cases in one structured request and returns both
    plus a simple "edge" score (bull_strength - bear_strength) the CIO can
    use as an extra signal alongside the individual agent reports.

    `evidence` is the evidence-quality snapshot (services/evidence.py): the
    debate is told exactly which evidence is actually available and must
    never treat unavailable evidence as neutral.

    Returns:
        {
          "agent": "debate",
          "symbol": symbol,
          "evidence_status": "AVAILABLE"|"ERROR",
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
        "evidence_status": None,
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
        base_result["evidence_status"] = "OFF"
        base_result["bull_summary"] = "Debate disabled via config."
        base_result["bear_summary"] = "Debate disabled via config."
        return base_result

    context_text = _build_context(symbol, tech_report, news_report,
                                  fundamentals_report, evidence=evidence)

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
        base_result["evidence_status"] = "ERROR"
        base_result["error"] = result.error
        if result.status == "NOT_CONFIGURED":
            base_result["bull_summary"] = "Debate disabled: missing API key."
            base_result["bear_summary"] = "Debate disabled: missing API key."
        else:
            base_result["bull_summary"] = "Debate agent encountered an error."
            base_result["bear_summary"] = "Debate agent encountered an error."
        return base_result

    parsed = result.parsed
    base_result["evidence_status"] = "AVAILABLE"
    base_result["bull_strength"] = float(parsed.get("bull_strength", 0.0))
    base_result["bull_summary"] = str(parsed.get("bull_summary", ""))
    base_result["bear_strength"] = float(parsed.get("bear_strength", 0.0))
    base_result["bear_summary"] = str(parsed.get("bear_summary", ""))
    base_result["edge"] = round(base_result["bull_strength"] - base_result["bear_strength"], 3)
    return base_result
