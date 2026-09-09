"""
services/risk_gate.py

Deterministic risk layer. ALL financial arithmetic for risk decisions and
order validation happens here, in plain Python — never in an LLM:

  - position percentage / portfolio exposure
  - max position validation
  - position sizing (allowed notional under equity %, cash and
    concentration caps)
  - buying power checks
  - final pre-execution order validation (action, symbol, quantity,
    limits, risk constraints)

The AI reasoning layer (risk agent) receives this computed evidence and may
add qualitative reasoning, and may VETO (block) a trade — but it can never
approve a trade or a size that the deterministic gate did not already allow.
If AI reasoning is unavailable, the deterministic gate result stands and is
reported honestly.
"""

import logging

from config import settings

logger = logging.getLogger("risk_gate")


def _num(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def assess(symbol: str, side: str, account_summary: dict,
           position: dict = None, indicators: dict = None) -> dict:
    """Deterministic risk assessment for a proposed trade.

    Args:
        symbol: ticker
        side: "buy" or "sell"
        account_summary: from alpaca_service.get_account_summary()
        position: existing position dict (or None)
        indicators: deterministic analytics (volatility/drawdown evidence)

    Returns a dict with approved / max_notional_usd / risk_level / reasons /
    checks — everything computed, nothing fabricated.
    """
    equity = _num(account_summary.get("equity"))
    cash = _num(account_summary.get("cash"))
    buying_power = _num(account_summary.get("buying_power"), cash)
    max_pct = float(settings.MAX_POSITION_PCT)

    existing_value = 0.0
    existing_qty = 0.0
    if position:
        existing_qty = _num(position.get("qty"))
        existing_value = _num(position.get("market_value")) or existing_qty * _num(
            position.get("current_price") or position.get("avg_entry_price")
        )
    position_pct = (existing_value / equity) if equity > 0 else 0.0

    volatility = (indicators or {}).get("volatility_annualized")
    max_drawdown = (indicators or {}).get("max_drawdown")

    checks = {
        "equity_positive": equity > 0,
        "side_valid": str(side).lower() in ("buy", "sell"),
    }
    reasons = []

    if not checks["side_valid"]:
        reasons.append(f"invalid side '{side}'")
    if not checks["equity_positive"]:
        reasons.append("portfolio equity is zero or unavailable")

    if str(side).lower() == "sell":
        # Selling is bounded by the existing position; no new exposure.
        approved = existing_qty > 0
        if not approved:
            reasons.append("no existing position to sell")
        return {
            "symbol": symbol, "side": "sell",
            "approved": approved,
            "max_notional_usd": 0.0,  # sell sizing is by qty, not notional
            "sell_max_qty": existing_qty if approved else 0.0,
            "risk_level": "LOW" if approved else "HIGH",
            "position_pct": round(position_pct, 4),
            "reasons": reasons,
            "checks": checks,
            "equity": equity, "cash": cash, "buying_power": buying_power,
            "max_position_pct": max_pct,
        }

    # --- BUY: deterministic sizing under equity %, cash, concentration ----
    hard_cap = equity * max_pct
    room_under_cap = max(0.0, hard_cap - existing_value)
    allowed_notional = round(max(0.0, min(room_under_cap, cash)), 2)

    checks["cash_available"] = cash > 0
    checks["room_under_position_cap"] = room_under_cap > 0
    checks["within_buying_power"] = allowed_notional <= buying_power

    if not checks["cash_available"]:
        reasons.append("no cash available")
    if not checks["room_under_position_cap"]:
        reasons.append(
            f"position already at max ({position_pct * 100:.1f}% of equity; "
            f"cap {max_pct * 100:.1f}%)"
        )

    approved = allowed_notional > 0 and all(checks.values())

    # Deterministic risk level from computed evidence (volatility/drawdown/
    # concentration) — the AI layer may add reasoning but not loosen this.
    risk_level = "LOW"
    if (volatility is not None and volatility > 0.60) or (
        max_drawdown is not None and max_drawdown < -0.40
    ):
        risk_level = "HIGH"
    elif (volatility is not None and volatility > 0.35) or (
        position_pct >= 0.8 * max_pct
    ):
        risk_level = "MEDIUM"

    return {
        "symbol": symbol, "side": "buy",
        "approved": approved,
        "max_notional_usd": allowed_notional,
        "risk_level": risk_level,
        "position_pct": round(position_pct, 4),
        "reasons": reasons,
        "checks": checks,
        "equity": equity, "cash": cash, "buying_power": buying_power,
        "max_position_pct": max_pct,
        "existing_position_value": round(existing_value, 2),
        "volatility": volatility,
        "max_drawdown": max_drawdown,
    }


def validate_order(symbol: str, side: str, account_summary: dict,
                   positions_by_symbol: dict, notional_usd: float = None,
                   qty: float = None, trade_universe: list = None) -> dict:
    """FINAL deterministic gate before any paper order is submitted.
    Validates action, symbol, quantity, buying power, position limits and
    risk constraints. AI output can never bypass this.

    Returns {valid, reason, checks}.
    """
    side = str(side).lower()
    checks = {
        "action_valid": side in ("buy", "sell"),
        "symbol_valid": bool(symbol) and (
            trade_universe is None or symbol in trade_universe
        ),
        "size_positive": (_num(notional_usd) > 0) or (_num(qty) > 0),
        "not_exclusive": not (_num(notional_usd) > 0 and _num(qty) > 0),
    }

    equity = _num(account_summary.get("equity"))
    cash = _num(account_summary.get("cash"))
    buying_power = _num(account_summary.get("buying_power"), cash)
    max_pct = float(settings.MAX_POSITION_PCT)
    position = (positions_by_symbol or {}).get(symbol)
    existing_value = 0.0
    existing_qty = 0.0
    if position:
        existing_qty = _num(position.get("qty"))
        existing_value = _num(position.get("market_value")) or existing_qty * _num(
            position.get("current_price") or position.get("avg_entry_price")
        )

    if side == "buy":
        notional = _num(notional_usd)
        post_trade_value = existing_value + notional
        checks["buying_power_sufficient"] = notional <= buying_power
        checks["cash_sufficient"] = notional <= cash
        checks["within_max_position_pct"] = equity <= 0 or post_trade_value <= equity * max_pct + 1e-9
    elif side == "sell":
        checks["has_position"] = existing_qty > 0
        checks["qty_within_position"] = _num(qty) <= existing_qty + 1e-9
    else:
        checks["buying_power_sufficient"] = False
        checks["cash_sufficient"] = False
        checks["within_max_position_pct"] = False

    reason = None
    if not checks["action_valid"]:
        reason = f"invalid action '{side}'"
    elif not checks["symbol_valid"]:
        reason = f"symbol '{symbol}' is not in the trade universe"
    elif not checks["size_positive"]:
        reason = "order size must be positive (notional_usd or qty)"
    elif not checks["not_exclusive"]:
        reason = "specify either notional_usd or qty, not both"
    elif side == "buy":
        if not checks["buying_power_sufficient"]:
            reason = f"notional ${_num(notional_usd):.2f} exceeds buying power ${buying_power:.2f}"
        elif not checks["cash_sufficient"]:
            reason = f"notional ${_num(notional_usd):.2f} exceeds cash ${cash:.2f}"
        elif not checks["within_max_position_pct"]:
            reason = (
                f"order would push {symbol} above the max position "
                f"{max_pct * 100:.1f}% of equity"
            )
    else:  # sell
        if not checks["has_position"]:
            reason = f"no open position in {symbol} to sell"
        elif not checks["qty_within_position"]:
            reason = f"sell qty {_num(qty)} exceeds position qty {existing_qty}"

    valid = all(checks.values())
    return {"valid": valid, "reason": reason, "checks": checks,
            "symbol": symbol, "side": side}
