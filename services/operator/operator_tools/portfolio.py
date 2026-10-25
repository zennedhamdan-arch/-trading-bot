"""
operator_tools/portfolio.py — READ-ONLY paper-account tools.

This system is PAPER TRADING ONLY. There is deliberately NO tool that
submits, modifies or cancels an order — the operator can look, never touch.
"""

from services import alpaca_service
from services.operator.operator_tools._registry import tool
from services.operator.schemas import sanitize

OPEN_STATES = ("new", "accepted", "pending_new", "partially_filled")


@tool("get_paper_account", "portfolio",
      "The paper trading account summary: equity, cash, buying power, day "
      "P&L, account status. (This system is paper-only; there is no live "
      "account and no way to enable one.)", label="Checking paper account")
def _get_paper_account(args):
    return sanitize({"trading_mode": "PAPER",
                     "account": alpaca_service.get_account_summary()})


@tool("get_positions", "portfolio",
      "Currently open positions in the paper account.",
      label="Checking positions")
def _get_positions(args):
    return sanitize(alpaca_service.get_open_positions())


@tool("get_open_orders", "portfolio",
      "Orders in the paper account that are still open/pending.",
      label="Checking open orders")
def _get_open_orders(args):
    result = alpaca_service.get_recent_orders(limit=100)
    orders = [o for o in (result.get("orders") or [])
              if str(o.get("status") or "").lower() in OPEN_STATES]
    return {"orders": orders, "count": len(orders),
            "error": result.get("error")}


@tool("get_recent_orders", "portfolio",
      "The most recent orders in the paper account (any status).",
      args={"limit": {"type": "integer", "description": "max orders (default 20)"}},
      label="Checking recent orders")
def _get_recent_orders(args):
    limit = min(int(args.get("limit") or 20), 100)
    return sanitize(alpaca_service.get_recent_orders(limit=limit))


@tool("get_recent_fills", "portfolio",
      "Recently FILLED orders in the paper account.",
      label="Checking recent fills")
def _get_recent_fills(args):
    result = alpaca_service.get_recent_orders(limit=100)
    fills = [o for o in (result.get("orders") or [])
             if str(o.get("status") or "").lower() in ("filled", "partially_filled")]
    return {"fills": fills[:30], "count": len(fills),
            "error": result.get("error")}


@tool("get_portfolio_history", "portfolio",
      "Paper portfolio equity history.",
      args={"period": {"type": "string", "description": "1D|1W|1M|3M (default 1M)"}},
      label="Checking portfolio history")
def _get_portfolio_history(args):
    period = str(args.get("period") or "1M").upper()[:3]
    return sanitize(alpaca_service.get_portfolio_history(period=period))


@tool("get_pnl_summary", "portfolio",
      "P&L summary of the paper account: day P&L, unrealized P&L per open "
      "position (market value vs cost basis) and total equity.",
      label="Checking P&L")
def _get_pnl_summary(args):
    account = alpaca_service.get_account_summary()
    positions = alpaca_service.get_open_positions()
    unrealized = []
    total_unrealized = 0.0
    for pos in (positions.get("positions") or []):
        try:
            mv = float(pos.get("market_value") or 0.0)
            cost = float(pos.get("cost_basis") or 0.0)
            pnl = round(mv - cost, 2)
            total_unrealized += pnl
            unrealized.append({"symbol": pos.get("symbol"), "qty": pos.get("qty"),
                               "market_value": mv, "cost_basis": cost,
                               "unrealized_pnl": pnl,
                               "unrealized_pnl_pct": round(pnl / cost * 100, 2) if cost else None})
        except (TypeError, ValueError):
            continue
    return sanitize({
        "equity": account.get("equity"), "cash": account.get("cash"),
        "day_pnl": account.get("day_pl"), "day_pnl_pct": account.get("day_pl_pct"),
        "total_unrealized_pnl": round(total_unrealized, 2),
        "per_position": unrealized,
        "account_error": account.get("error"),
        "positions_error": positions.get("error"),
    })


@tool("get_exposure", "portfolio",
      "Portfolio exposure: each position's market value as a share of "
      "equity, total exposure, and cash share.",
      label="Checking exposure")
def _get_exposure(args):
    account = alpaca_service.get_account_summary()
    positions = alpaca_service.get_open_positions()
    equity = float(account.get("equity") or 0.0)
    per_symbol, total_mv = {}, 0.0
    for pos in (positions.get("positions") or []):
        try:
            mv = float(pos.get("market_value") or 0.0)
        except (TypeError, ValueError):
            continue
        total_mv += mv
        per_symbol[pos.get("symbol")] = {
            "market_value": mv,
            "pct_of_equity": round(mv / equity * 100, 2) if equity else None,
        }
    return sanitize({
        "equity": equity,
        "total_market_value": round(total_mv, 2),
        "total_exposure_pct": round(total_mv / equity * 100, 2) if equity else None,
        "cash": account.get("cash"),
        "cash_pct": round(float(account.get("cash") or 0) / equity * 100, 2) if equity else None,
        "per_symbol": per_symbol,
        "position_count": len(per_symbol),
        "account_error": account.get("error"),
    })
