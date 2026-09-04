"""
agents/risk_agent.py

Risk Agent powered by OpenRouter (deepseek/deepseek-r1:free).
Enforces portfolio position sizing and risk limits before any trade
is allowed to proceed. Uses the OpenAI-compatible client pointed at
OpenRouter's base URL, as recommended by OpenRouter's docs.
"""

import json
import logging
import re

from config import settings

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
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")
    return json.loads(match.group(0))


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
          "error": str | None
        }
    """
    base_result = {
        "agent": "risk",
        "symbol": symbol,
        "approved": False,
        "max_notional_usd": 0.0,
        "risk_level": "HIGH",
        "reasoning": "",
        "error": None,
    }

    if not settings.OPENROUTER_API_KEY:
        base_result["error"] = "OPENROUTER_API_KEY not configured."
        base_result["reasoning"] = "Risk agent disabled: missing API key. Trade blocked by default."
        return base_result

    equity = account_summary.get("equity", 0.0)
    cash = account_summary.get("cash", 0.0)

    if equity <= 0:
        base_result["reasoning"] = "Portfolio equity is zero or unavailable; blocking trade."
        return base_result

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
        )

        context_text = (
            f"Symbol: {symbol}\n"
            f"Proposed side: {proposed_side}\n"
            f"Portfolio equity: ${equity:.2f}\n"
            f"Available cash: ${cash:.2f}\n"
            f"Max position percent allowed: {settings.MAX_POSITION_PCT * 100:.1f}%\n"
            f"Existing position in {symbol}: {json.dumps(existing_position) if existing_position else 'None'}\n"
        )

        completion = client.chat.completions.create(
            model=settings.OPENROUTER_RISK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": context_text},
            ],
            temperature=0.1,
            max_tokens=500,
        )

        raw_text = completion.choices[0].message.content or ""
        parsed = _extract_json(raw_text)

        # Hard safety clamp: never trust the LLM's number beyond our own ceiling.
        hard_cap = equity * settings.MAX_POSITION_PCT
        requested = float(parsed.get("max_notional_usd", 0.0))
        clamped_notional = max(0.0, min(requested, hard_cap, cash))

        base_result["approved"] = bool(parsed.get("approved", False)) and clamped_notional > 0
        base_result["max_notional_usd"] = round(clamped_notional, 2)
        base_result["risk_level"] = str(parsed.get("risk_level", "HIGH")).upper()
        base_result["reasoning"] = str(parsed.get("reasoning", ""))
        return base_result

    except Exception as e:
        logger.error(f"Risk agent failed for {symbol}: {e}")
        base_result["error"] = str(e)
        base_result["reasoning"] = "Risk agent encountered an error; blocking trade by default (fail-safe)."
        return base_result
