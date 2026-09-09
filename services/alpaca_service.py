"""
services/alpaca_service.py

Wraps the alpaca-py SDK to provide:
  - Account / portfolio metrics
  - Open positions and recent orders
  - Portfolio equity history (dashboard performance chart)
  - Historical candle data + technical indicators (RSI, SMA50, SMA200)
  - Market order execution (buy/sell)

Every public function is defensive: if the Alpaca API is unreachable,
misconfigured, or returns an error, functions return a structured
dict with an "error" key rather than raising, so the FastAPI server
and the scheduler loop never crash because of a broker hiccup.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, GetPortfolioHistoryRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    ALPACA_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - only hit if dependency missing
    ALPACA_SDK_AVAILABLE = False


try:
    import ta
    TA_AVAILABLE = True
except ImportError:  # pragma: no cover
    TA_AVAILABLE = False

from config import settings

logger = logging.getLogger("alpaca_service")

_trading_client: Optional["TradingClient"] = None
_data_client: Optional["StockHistoricalDataClient"] = None


def _resolve_feed() -> "DataFeed":
    """Resolves the market-data feed from configuration.

    The default is IEX: the free/paper Alpaca subscription does not permit
    the SIP feed, and requesting it fails with
    "subscription does not permit querying recent SIP data". Paid accounts
    can set ALPACA_DATA_FEED=SIP.
    """
    if not ALPACA_SDK_AVAILABLE:
        return None
    wanted = (settings.ALPACA_DATA_FEED or "IEX").upper()
    if wanted == "SIP":
        return DataFeed.SIP
    if wanted == "OTC":
        return DataFeed.OTC
    if wanted != "IEX":
        logger.warning(f"Unknown ALPACA_DATA_FEED '{wanted}'; falling back to IEX.")
    return DataFeed.IEX


def _get_trading_client():
    """Lazily instantiate and cache the Alpaca trading client."""
    global _trading_client
    if not ALPACA_SDK_AVAILABLE:
        raise RuntimeError("alpaca-py SDK is not installed.")
    if _trading_client is not None:
        return _trading_client
    if not settings.ALPACA_API_KEY or not settings.ALPACA_SECRET_KEY:
        raise RuntimeError("Alpaca API keys are not configured.")
    if _trading_client is None:
        _trading_client = TradingClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            paper=True,  # This application is hardcoded to paper trading only.
        )
    return _trading_client


def _get_data_client():
    """Lazily instantiate and cache the Alpaca historical data client."""
    global _data_client
    if not ALPACA_SDK_AVAILABLE:
        raise RuntimeError("alpaca-py SDK is not installed.")
    if _data_client is not None:
        return _data_client
    if not settings.ALPACA_API_KEY or not settings.ALPACA_SECRET_KEY:
        raise RuntimeError("Alpaca API keys are not configured.")
    if _data_client is None:
        _data_client = StockHistoricalDataClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
        )
    return _data_client


def get_account_summary() -> dict:
    """Returns cash, equity, buying power, and P&L for the paper account."""
    try:
        client = _get_trading_client()
        account = client.get_account()
        equity = float(account.equity)
        last_equity = float(account.last_equity) if account.last_equity else equity
        day_pl = equity - last_equity
        day_pl_pct = (day_pl / last_equity * 100) if last_equity else 0.0
        return {
            "cash": float(account.cash),
            "equity": equity,
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "day_pl": round(day_pl, 2),
            "day_pl_pct": round(day_pl_pct, 3),
            "status": account.status.value if hasattr(account.status, "value") else str(account.status),
            "error": None,
        }
    except Exception as e:
        logger.error(f"get_account_summary failed: {e}")
        return {
            "cash": 0.0,
            "equity": 0.0,
            "buying_power": 0.0,
            "portfolio_value": 0.0,
            "day_pl": 0.0,
            "day_pl_pct": 0.0,
            "status": "UNKNOWN",
            "error": str(e),
        }


def get_open_positions() -> dict:
    """Returns all currently held positions with unrealized P&L."""
    try:
        client = _get_trading_client()
        positions = client.get_all_positions()
        result = []
        for p in positions:
            result.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price) if p.current_price else None,
                "market_value": float(p.market_value) if p.market_value else None,
                "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl else 0.0,
                "unrealized_plpc": float(p.unrealized_plpc) * 100 if p.unrealized_plpc else 0.0,
                "side": p.side.value if hasattr(p.side, "value") else str(p.side),
            })
        return {"positions": result, "error": None}
    except Exception as e:
        logger.error(f"get_open_positions failed: {e}")
        return {"positions": [], "error": str(e)}


def get_recent_orders(limit: int = 20) -> dict:
    """Returns the most recent orders (filled, pending, canceled, etc.)."""
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        client = _get_trading_client()
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
        orders = client.get_orders(req)
        result = []
        for o in orders:
            result.append({
                "id": str(o.id),
                "symbol": o.symbol,
                "side": o.side.value if hasattr(o.side, "value") else str(o.side),
                "qty": float(o.qty) if o.qty else None,
                "status": o.status.value if hasattr(o.status, "value") else str(o.status),
                "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                "submitted_at": o.submitted_at.isoformat() if o.submitted_at else None,
            })
        return {"orders": result, "error": None}
    except Exception as e:
        logger.error(f"get_recent_orders failed: {e}")
        return {"orders": [], "error": str(e)}


def get_portfolio_history(period: str = "1M", timeframe: Optional[str] = None) -> dict:
    """Returns historical equity curve data points for charting.

    Uses TradingClient.get_portfolio_history() with a GetPortfolioHistoryRequest
    (the correct alpaca-py API — this service previously called a method that
    does not exist on the installed SDK and always failed). Equity values the
    API reports as null are filtered out; nothing is interpolated or fabricated.
    """
    try:
        client = _get_trading_client()
        if timeframe is None:
            # Intraday bars make the 1D chart meaningful; daily for longer ranges.
            timeframe = "5Min" if period == "1D" else "1D"
        history = client.get_portfolio_history(
            history_filter=GetPortfolioHistoryRequest(period=period, timeframe=timeframe)
        )

        if isinstance(history, dict):
            timestamps = history.get("timestamp") or []
            equity = history.get("equity") or []
        else:
            timestamps = getattr(history, "timestamp", None) or []
            equity = getattr(history, "equity", None) or []
        points = [
            {"timestamp": ts, "equity": eq}
            for ts, eq in zip(timestamps, equity)
            if eq is not None
        ]
        return {"points": points, "error": None}
    except Exception as e:
        logger.error(f"get_portfolio_history failed: {e}")
        return {"points": [], "error": str(e)}


def get_indicators(symbol: str, lookback_days: int = 250) -> dict:
    """
    Fetches daily bars for `symbol` and computes RSI(14), SMA50, SMA200,
    and MACD. Returns the latest values plus recent close price history.
    """
    if not TA_AVAILABLE:
        return {"error": "The 'ta' library is not installed.", "symbol": symbol}

    try:
        data_client = _get_data_client()
        end = datetime.utcnow()
        start = end - timedelta(days=lookback_days * 1.6)  # buffer for weekends/holidays

        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=_resolve_feed(),
        )
        bars = data_client.get_stock_bars(req)
        df = bars.df

        if df is None or df.empty:
            return {"error": f"No bar data returned for {symbol}", "symbol": symbol}

        # alpaca-py returns a MultiIndex (symbol, timestamp) when multiple
        # symbols could be present; normalize to a flat frame for this symbol.
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)

        df = df.sort_index()
        closes = df["close"]

        rsi = ta.momentum.RSIIndicator(close=closes, window=14).rsi()
        sma50 = ta.trend.SMAIndicator(close=closes, window=50).sma_indicator()
        sma200 = ta.trend.SMAIndicator(close=closes, window=200).sma_indicator()
        macd_ind = ta.trend.MACD(close=closes)
        macd_line = macd_ind.macd()
        macd_signal = macd_ind.macd_signal()

        latest_close = float(closes.iloc[-1])

        def _safe_last(series):
            val = series.dropna()
            return float(val.iloc[-1]) if not val.empty else None

        return {
            "symbol": symbol,
            "latest_close": latest_close,
            "rsi_14": _safe_last(rsi),
            "sma_50": _safe_last(sma50),
            "sma_200": _safe_last(sma200),
            "macd": _safe_last(macd_line),
            "macd_signal": _safe_last(macd_signal),
            "recent_closes": [round(float(c), 2) for c in closes.tail(10).tolist()],
            "error": None,
        }
    except Exception as e:
        logger.error(f"get_indicators failed for {symbol}: {e}")
        return {"error": str(e), "symbol": symbol}


def execute_order(symbol: str, side: str, notional_usd: Optional[float] = None,
                   qty: Optional[float] = None) -> dict:
    """
    Executes a market order. Provide either notional_usd (dollar amount)
    or qty (share count) - not both. side must be 'buy' or 'sell'.
    """
    try:
        if side.lower() not in ("buy", "sell"):
            return {"success": False, "error": f"Invalid side: {side}"}
        if not notional_usd and not qty:
            return {"success": False, "error": "Must specify notional_usd or qty"}

        client = _get_trading_client()
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        order_kwargs = {
            "symbol": symbol,
            "side": order_side,
            "time_in_force": TimeInForce.DAY,
        }
        if notional_usd:
            order_kwargs["notional"] = round(notional_usd, 2)
        else:
            order_kwargs["qty"] = qty

        order_request = MarketOrderRequest(**order_kwargs)
        order = client.submit_order(order_request)

        return {
            "success": True,
            "order_id": str(order.id),
            "symbol": order.symbol,
            "side": side,
            "qty": float(order.qty) if order.qty else None,
            "notional_usd": round(notional_usd, 2) if notional_usd else None,
            "status": order.status.value if hasattr(order.status, "value") else str(order.status),
            "error": None,
        }
    except Exception as e:
        logger.error(f"execute_order failed for {symbol} {side}: {e}")
        return {"success": False, "error": str(e)}
