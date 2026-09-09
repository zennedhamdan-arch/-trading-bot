"""
agents/cio_agent.py

Executive (CIO) Agent. Weighs the outputs of the News, Technical,
Fundamentals, Risk, and Bull/Bear Debate agents and issues the final
BUY / SELL / HOLD decision for a symbol.

Provider/model: configured centrally via services/llm_service.py
(GROQ_CIO_MODEL; no model id lives in this file).

Also incorporates:
  - Adaptive agent weighting: each agent's historical hit rate (from
    memory_service) is shown to the CIO so it can lean more/less on
    an agent that's been reliable/unreliable recently.
  - Recent outcome memory: a summary of the last several closed trades
    for this symbol, so the CIO has continuity across cycles instead
    of deciding from scratch every time.

Safety (unchanged): the risk agent's approval is a hard constraint, and
the notional is clamped deterministically regardless of what the model
says. If the CIO cannot obtain a valid LLM response, it HOLDs (fail-safe)
with an explicit error — never a fabricated decision.
"""

import json
import logging

from config import settings
from services import llm_service

logger = logging.getLogger("cio_agent")

SYSTEM_INSTRUCTIONS = """You are the Chief Investment Officer (CIO) of an automated trading desk.
You will receive reports from your team:
1. A news/sentiment agent's read on recent headlines.
2. A technical agent's read on RSI, moving averages, and MACD.
3. A fundamentals agent's read on valuation, growth, and balance sheet health (if available).
4. A bull researcher's strongest case FOR the trade, and a bear researcher's
   strongest case AGAINST it -- pay attention to which case is actually stronger,
   not just which agents nominally lean bullish/bearish.
5. A risk agent's verdict on whether a trade is approved and the max dollar amount allowed.
6. Each agent's historical accuracy weight (1.0 = neutral track record, above 1.0 =
   has been reliable recently, below 1.0 = has been unreliable recently) -- use this
   to lean more heavily on agents that have been right and discount ones that haven't.
7. A summary of recent closed-trade outcomes for this symbol, for continuity.

The risk agent's approval is a hard constraint: if the risk agent did NOT approve
the trade, you may NOT issue a BUY (you may still issue SELL if there is an existing
position and technicals/news/debate are strongly bearish, or HOLD).
Favor HOLD when signals conflict or confidence is low across agents, especially
when the bull and bear cases are close in strength (a close debate = genuine uncertainty).

Respond ONLY with a single valid JSON object, no markdown fences, no preamble, in this exact shape:
{
  "decision": "BUY" | "SELL" | "HOLD",
  "confidence": <float 0.0 to 1.0>,
  "notional_usd": <float, dollar amount to trade if BUY, 0 otherwise>,
  "reasoning": "<two to three sentence explanation synthesizing all inputs, including the debate>"
}
"""


def _extract_json(text: str) -> dict:
    """Tolerant JSON extraction — kept for local use/testing; the live path
    parses through services.llm_service (same logic)."""
    return llm_service.extract_json(text)


def make_decision(symbol: str, news_report: dict, tech_report: dict, risk_report: dict,
                   fundamentals_report: dict = None, debate_report: dict = None,
                   agent_weights: dict = None, memory_summary: str = "") -> dict:
    """
    Args:
        symbol: ticker symbol
        news_report: output of news_agent.analyze_news()
        tech_report: output of tech_agent.analyze_technicals()
        risk_report: output of risk_agent.assess_risk()
        fundamentals_report: output of fundamentals_agent.analyze_fundamentals(), optional
        debate_report: output of debate_agent.run_debate(), optional
        agent_weights: dict of {agent_name: weight} from memory_service.get_agent_accuracy(), optional
        memory_summary: human-readable recent outcome summary string, optional

    Returns:
        {
          "agent": "cio",
          "symbol": symbol,
          "decision": "BUY"/"SELL"/"HOLD",
          "confidence": float,
          "notional_usd": float,
          "reasoning": str,
          "error": str | None,
          "provider": str, "model": str, "llm_status": str, "latency_ms": float | null
        }
    """
    route = llm_service.route_info("cio")
    base_result = {
        "agent": "cio",
        "symbol": symbol,
        "decision": "HOLD",
        "confidence": 0.0,
        "notional_usd": 0.0,
        "reasoning": "",
        "error": None,
        "provider": route["provider"],
        "model": route["model"],
        "llm_status": "NOT_CONFIGURED",
        "latency_ms": None,
    }

    context_parts = [
        f"Symbol: {symbol}\n",
        f"--- News/Sentiment Agent Report ---\n{json.dumps(news_report, indent=2)}\n",
        f"--- Technical Agent Report ---\n{json.dumps(tech_report, indent=2)}\n",
    ]
    if fundamentals_report:
        context_parts.append(
            f"--- Fundamentals Agent Report ---\n{json.dumps(fundamentals_report, indent=2)}\n"
        )
    if debate_report:
        context_parts.append(
            f"--- Bull vs Bear Debate ---\n"
            f"Bull case (strength {debate_report.get('bull_strength')}): {debate_report.get('bull_summary')}\n"
            f"Bear case (strength {debate_report.get('bear_strength')}): {debate_report.get('bear_summary')}\n"
            f"Net edge (bull - bear): {debate_report.get('edge')}\n"
        )
    context_parts.append(
        f"--- Risk Agent Report ---\n{json.dumps(risk_report, indent=2)}\n"
    )
    if agent_weights:
        weight_lines = "\n".join(
            f"  {name}: weight={w.get('weight')} (hit rate {w.get('hit_rate')}, n={w.get('sample_size')})"
            for name, w in agent_weights.items()
        )
        context_parts.append(f"--- Agent Historical Accuracy Weights ---\n{weight_lines}\n")
    if memory_summary:
        context_parts.append(f"--- Recent Outcome History ---\n{memory_summary}\n")

    context_text = "\n".join(context_parts)

    result = llm_service.call_json(
        "cio",
        system=SYSTEM_INSTRUCTIONS,
        user=context_text,
        temperature=0.2,
        max_tokens=700,
        symbol=symbol,
    )

    base_result["llm_status"] = result.status
    base_result["model"] = result.model
    base_result["latency_ms"] = result.latency_ms

    if not result.ok:
        logger.error(f"CIO agent failed for {symbol}: {result.error}")
        base_result["error"] = result.error
        base_result["reasoning"] = (
            "CIO agent disabled: missing API key. Defaulting to HOLD."
            if result.status == "NOT_CONFIGURED"
            else "CIO agent encountered an error; defaulting to HOLD (fail-safe)."
        )
        return base_result

    parsed = result.parsed
    decision = str(parsed.get("decision", "HOLD")).upper()
    notional = float(parsed.get("notional_usd", 0.0))

    # Hard safety enforcement: never allow a BUY if risk agent didn't approve,
    # and never exceed the risk agent's approved notional, regardless of
    # what the CIO model says.
    if decision == "BUY":
        if not risk_report.get("approved", False):
            decision = "HOLD"
            notional = 0.0
        else:
            notional = min(notional, risk_report.get("max_notional_usd", 0.0))
    else:
        notional = 0.0

    base_result["decision"] = decision
    base_result["confidence"] = float(parsed.get("confidence", 0.0))
    base_result["notional_usd"] = round(max(0.0, notional), 2)
    base_result["reasoning"] = str(parsed.get("reasoning", ""))
    return base_result
