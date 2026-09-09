"""
agents/risk_agent.py

Risk Agent. Enforces portfolio position sizing and risk limits before any
trade is allowed to proceed.

Provider/model: configured centrally via services/llm_service.py via
OPENROUTER_RISK_MODEL (default: a model verified free on OpenRouter's
live catalog at the time of writing — see config.py; no model id lives
in this file). If the configured model is unavailable, the provider
layer returns a clear MODEL_NOT_FOUND / PROVIDER_QUOTA_EXCEEDED status
and this agent BLOCKS the trade — it never silently produces a fake
risk verdict.

The hard numeric safety clamp (never exceed max position pct of equity,
never exceed available cash) is deterministic code, independent of what
any LLM says.
"""

import json
import logging

from config import settings
from services import llm_service

logger = logging.getLogger("risk_agent")

SYSTEM_INSTRUCTIONS = """You are a strict portfolio risk manager for a paper trading account.
You will be given: the proposed trade direction, current portfolio equity,
current cash, existing position (if any) in the symbol, and the maximum
allowed position size as a percentage of equity.

Your job is to decide whether the proposed trade is within acceptable risk
limits and, if approved, what dollar amount (notional) should be allocated.
Never approve a notional amount that would push the position above the
max position percentage of total equity. Be conservative when data is
incomplete or portfolio equity is very low.

Respond ONLY with a single valid JSON object, no markdown fences, no preamble, in this exact shape:
{
  "approved": true | false,
  "max_notional_usd": <float, dollar amount approved for this trade, 0 if not approved>,
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "reasoning": "<one to two sentence explanation>"
}
"""


def _extract_json(text: str) -> dict:
    """Tolerant JSON extraction — kept for local use/testing; the live path
    parses through services.llm_service (same logic)."""
    return llm_service.extract_json(text)


def assess_risk(symbol: str, proposed_side: str, account_summary: dict,
                 existing_position: dict = None) -> dict:
    """
    Args:
        symbol: ticker symbol
        proposed_side: "buy" or "sell"
        account_summary: dict from alpaca_service.get_account_summary()
        existing_position: dict for this symbol from get_open_positions(), or None

    Returns:
        {
          "agent": "risk",
          "symbol": symbol,
          "approved": bool,
          "max_notional_usd": float,
          "risk_level": "LOW"/"MEDIUM"/"HIGH",
          "reasoning": str,
          "error": str | None,
          "provider": str, "model": str, "llm_status": str, "latency_ms": float | null
        }
    """
    route = llm_service.route_info("risk")
    base_result = {
        "agent": "risk",
        "symbol": symbol,
        "approved": False,
        "max_notional_usd": 0.0,
        "risk_level": "HIGH",
        "reasoning": "",
        "error": None,
        "provider": route["provider"],
        "model": route["model"],
        "llm_status": "NOT_CONFIGURED",
        "latency_ms": None,
    }

    if not settings.OPENROUTER_API_KEY:
        base_result["error"] = "OPENROUTER_API_KEY not configured."
        base_result["reasoning"] = "Risk agent disabled: missing API key. Trade blocked by default."
        return base_result

    equity = account_summary.get("equity", 0.0)
    cash = account_summary.get("cash", 0.0)

    if equity <= 0:
        base_result["llm_status"] = "SKIPPED_NO_DATA"
        base_result["reasoning"] = "Portfolio equity is zero or unavailable; blocking trade."
        return base_result

    context_text = (
        f"Symbol: {symbol}\n"
        f"Proposed side: {proposed_side}\n"
        f"Portfolio equity: ${equity:.2f}\n"
        f"Available cash: ${cash:.2f}\n"
        f"Max position percent allowed: {settings.MAX_POSITION_PCT * 100:.1f}%\n"
        f"Existing position in {symbol}: {json.dumps(existing_position) if existing_position else 'None'}\n"
    )

    result = llm_service.call_json(
        "risk",
        system=SYSTEM_INSTRUCTIONS,
        user=context_text,
        temperature=0.1,
        max_tokens=500,
        symbol=symbol,
    )

    base_result["llm_status"] = result.status
    base_result["model"] = result.model
    base_result["latency_ms"] = result.latency_ms

    if not result.ok:
        # Provider/model failure -> explicit, visible error; trade blocked.
        logger.error(f"Risk agent failed for {symbol}: {result.error}")
        base_result["error"] = result.error
        base_result["reasoning"] = f"Risk agent provider error ({result.status}); blocking trade by default (fail-safe)."
        return base_result

    parsed = result.parsed

    # Hard safety clamp: never trust the LLM's number beyond our own ceiling.
    hard_cap = equity * settings.MAX_POSITION_PCT
    requested = float(parsed.get("max_notional_usd", 0.0))
    clamped_notional = max(0.0, min(requested, hard_cap, cash))

    base_result["approved"] = bool(parsed.get("approved", False)) and clamped_notional > 0
    base_result["max_notional_usd"] = round(clamped_notional, 2)
    base_result["risk_level"] = str(parsed.get("risk_level", "HIGH")).upper()
    base_result["reasoning"] = str(parsed.get("reasoning", ""))
    return base_result
