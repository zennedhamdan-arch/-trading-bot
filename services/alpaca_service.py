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
    from alpaca.trading.requests import (MarketOrderRequest, GetPortfolioHistoryRequest,
                                         GetCorporateAnnouncementsRequest)
    from alpaca.trading.enums import OrderSide, TimeInForce, CorporateActionType
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
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

# ---------------------------------------------------------------------------
# Indicator lookback requirements (daily bars).
#
# The configured indicator suite needs, at minimum, as many bars as its
# LARGEST window. The 'ta' library's ATR implementation does positional
# indexing (atr[window-1] = ...) and RAISES IndexError when the series is
# shorter than the window — the production "index 13 is out of bounds for
# axis 0 with size 5" crash. The fix is structural: always request (and
# validate) enough bars for the full suite, never patch the indexing.
#
#   RSI(14)        -> 15 bars      SMA(20) -> 20     EMA(20) -> ~60 (convergence)
#   SMA(50)        -> 50 bars      MACD(26/9) -> ~35 ATR(14) -> 15
#   SMA(200)       -> 200 bars     <== maximum
# ---------------------------------------------------------------------------
REQUIRED_BARS = 200
# Calendar days that reliably contain REQUIRED_BARS trading sessions
# (weekends ≈ 2/7 of days, plus holidays) with a safety buffer.
REQUIRED_CALENDAR_DAYS = int(REQUIRED_BARS * 1.6) + 14


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


def _map_data_error(exc: Exception, context: str) -> str:
    """Maps Alpaca data errors to honest, structured reasons.

    A feed-subscription failure (e.g. the free/paper account requesting SIP)
    becomes DATA_UNAVAILABLE (SUBSCRIPTION_FEED_UNAVAILABLE) naming the
    configured feed — the system NEVER silently switches feeds and NEVER
    pretends limited data is full-market data.
    """
    message = str(exc)
    if "subscription does not permit" in message.lower() or "not entitled" in message.lower():
        return (
            f"DATA_UNAVAILABLE (SUBSCRIPTION_FEED_UNAVAILABLE): the configured "
            f"ALPACA_DATA_FEED={settings.ALPACA_DATA_FEED} is not permitted by this "
            f"Alpaca subscription for {context}. Set ALPACA_DATA_FEED to a permitted "
            f"feed (the free/paper subscription supports iex). Original: {message}"
        )
    return message


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
    does not exist on the installed SDK and always failed).

    Honesty rules (the chart must NEVER draw a fake line from $0):
      - equity values the API reports as null/missing are dropped;
      - non-positive equity values and epoch-0 timestamps are treated as
        API artifacts and dropped (a long-only paper account cannot have
        $0 equity, and timestamp 0 = 1970-01-01 is a placeholder, not a
        real sample) — the dropped count is reported, never hidden;
      - nothing is interpolated, zero-filled or fabricated. If fewer than
        two valid points remain the frontend shows an explicit empty
        state instead of a synthetic curve.
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
        points = []
        dropped = 0
        for ts, eq in zip(timestamps, equity):
            if eq is None or ts is None:
                dropped += 1
                continue
            try:
                eq_f, ts_f = float(eq), float(ts)
            except (TypeError, ValueError):
                dropped += 1
                continue
            if eq_f <= 0 or ts_f <= 0:
                # Missing/placeholder artifacts must never render as $0.
                dropped += 1
                continue
            points.append({"timestamp": ts_f, "equity": eq_f})
        if dropped:
            logger.info(
                f"get_portfolio_history({period}): dropped {dropped} invalid "
                f"point(s) (null/zero equity or epoch-0 timestamps) — never "
                f"rendered as $0."
            )
        return {"points": points, "error": None, "dropped_invalid_points": dropped}
    except Exception as e:
        logger.error(f"get_portfolio_history failed: {e}")
        return {"points": [], "error": str(e), "dropped_invalid_points": 0}


def get_indicators(symbol: str, lookback_days: int = 250) -> dict:
    """
    Fetches daily bars for `symbol` from the CONFIGURED feed and computes
    ALL technical analytics deterministically in Python (never via an LLM):

    RSI(14), SMA20/50/200, EMA20, MACD(+signal), ATR(14), annualized
    volatility, max drawdown, 1d/5d/20d returns, plus a transparent
    rule-based technical_signal summary. The AI reasoning layer receives
    these numbers as evidence; it never computes them.

    Feed honesty: the response carries the configured feed name. If the
    configured feed is not permitted by the subscription, the error is
    DATA_UNAVAILABLE (SUBSCRIPTION_FEED_UNAVAILABLE) — never a silent
    feed switch, never fabricated bars.
    """
    if not TA_AVAILABLE:
        return {"error": "The 'ta' library is not installed.", "symbol": symbol}

    # Always request enough history for the FULL indicator suite (the
    # largest window is SMA200). A small caller-supplied lookback (e.g.
    # the health probe's 5 days) never results in a short series again.
    lookback_days = max(int(lookback_days or 0), REQUIRED_BARS)

    try:
        data_client = _get_data_client()
        end = datetime.utcnow()
        # Buffer for weekends/holidays; never below the suite's requirement.
        calendar_days = max(lookback_days * 1.6, REQUIRED_CALENDAR_DAYS)
        start = end - timedelta(days=calendar_days)

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
        highs = df["high"] if "high" in df else closes
        lows = df["low"] if "low" in df else closes

        # Validate the returned bar count BEFORE any calculation: with too
        # few bars the ta library raises IndexError (e.g. ATR positional
        # indexing) — that must surface as a structured DATA_UNAVAILABLE,
        # never as an exception, and never as fabricated indicator values.
        available_bars = int(len(closes))
        if available_bars < REQUIRED_BARS:
            detail = (
                f"DATA_UNAVAILABLE (INSUFFICIENT_BARS): received {available_bars} "
                f"daily bar(s) for {symbol} on feed={settings.ALPACA_DATA_FEED}; "
                f"the configured indicator suite requires at least {REQUIRED_BARS} "
                f"bars (SMA200). No indicator values are fabricated."
            )
            logger.warning(f"get_indicators({symbol}): {detail}")
            return {
                "symbol": symbol,
                "status": "DATA_UNAVAILABLE",
                "reason": "insufficient_bars",
                "available_bars": available_bars,
                "required_bars": REQUIRED_BARS,
                "error": detail,
                "feed": settings.ALPACA_DATA_FEED,
            }

        # --- deterministic indicators (ta + pandas) ---
        rsi = ta.momentum.RSIIndicator(close=closes, window=14).rsi()
        sma20 = ta.trend.SMAIndicator(close=closes, window=20).sma_indicator()
        sma50 = ta.trend.SMAIndicator(close=closes, window=50).sma_indicator()
        sma200 = ta.trend.SMAIndicator(close=closes, window=200).sma_indicator()
        ema20 = ta.trend.EMAIndicator(close=closes, window=20).ema_indicator()
        macd_ind = ta.trend.MACD(close=closes)
        macd_line = macd_ind.macd()
        macd_signal = macd_ind.macd_signal()
        atr14 = ta.volatility.AverageTrueRange(
            high=highs, low=lows, close=closes, window=14
        ).average_true_range()

        latest_close = float(closes.iloc[-1])

        def _safe_last(series):
            val = series.dropna()
            return float(val.iloc[-1]) if not val.empty else None

        # Returns / volatility / drawdown (pure pandas)
        daily_returns = closes.pct_change().dropna()
        volatility_annual = (
            float(daily_returns.std() * (252 ** 0.5)) if len(daily_returns) >= 2 else None
        )

        def _ret_over(n):
            if len(closes) > n and closes.iloc[-1 - n] and closes.iloc[-1 - n] != 0:
                return round(float(closes.iloc[-1] / closes.iloc[-1 - n] - 1.0), 6)
            return None

        running_max = closes.cummax()
        drawdown = closes / running_max - 1.0
        max_drawdown = round(float(drawdown.min()), 6) if len(drawdown) else None

        # --- transparent rule-based signal (evidence for the AI, not a
        # replacement for it; components are exposed for auditability) ---
        rsi_v = _safe_last(rsi)
        sma50_v = _safe_last(sma50)
        sma200_v = _safe_last(sma200)
        macd_v = _safe_last(macd_line)
        macd_sig_v = _safe_last(macd_signal)

        if sma50_v is not None and sma200_v is not None:
            if latest_close > sma50_v > sma200_v:
                trend = "BULLISH"
            elif latest_close < sma50_v < sma200_v:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"
        else:
            trend = "NEUTRAL"
        if macd_v is not None and macd_sig_v is not None:
            momentum = "BULLISH" if macd_v > macd_sig_v else "BEARISH"
        else:
            momentum = "NEUTRAL"
        if rsi_v is not None:
            rsi_flag = "OVERBOUGHT" if rsi_v > 70 else ("OVERSOLD" if rsi_v < 30 else "NEUTRAL")
        else:
            rsi_flag = "NEUTRAL"

        if trend == "BULLISH" and momentum == "BULLISH" and rsi_flag != "OVERBOUGHT":
            technical_signal = "BULLISH"
        elif trend == "BEARISH" and momentum == "BEARISH" and rsi_flag != "OVERSOLD":
            technical_signal = "BEARISH"
        else:
            technical_signal = "NEUTRAL"

        return {
            "symbol": symbol,
            "feed": settings.ALPACA_DATA_FEED,
            "latest_close": latest_close,
            "bars_available": available_bars,
            "last_bar_date": str(closes.index[-1]) if len(closes) else None,
            "rsi_14": _safe_last(rsi),
            "sma_20": _safe_last(sma20),
            "sma_50": sma50_v,
            "sma_200": sma200_v,
            "ema_20": _safe_last(ema20),
            "macd": macd_v,
            "macd_signal": macd_sig_v,
            "atr_14": _safe_last(atr14),
            "volatility_annualized": volatility_annual,
            "max_drawdown": max_drawdown,
            "return_1d": _ret_over(1),
            "return_5d": _ret_over(5),
            "return_20d": _ret_over(20),
            "technical_signal": technical_signal,
            "technical_components": {
                "trend": trend, "momentum": momentum, "rsi_flag": rsi_flag,
            },
            "recent_closes": [round(float(c), 2) for c in closes.tail(10).tolist()],
            "error": None,
        }
    except IndexError as e:
        # Belt-and-braces: with the pre-validation above this should be
        # unreachable, but a short-series IndexError from the indicator
        # library is still an INSUFFICIENT_BARS condition — structured,
        # never an exception into the trading cycle, never fabricated.
        detail = (
            f"DATA_UNAVAILABLE (INSUFFICIENT_BARS): indicator library ran out "
            f"of data for {symbol} ({e}); the suite requires at least "
            f"{REQUIRED_BARS} daily bars."
        )
        logger.warning(f"get_indicators({symbol}): {detail}")
        return {
            "symbol": symbol,
            "status": "DATA_UNAVAILABLE",
            "reason": "insufficient_bars",
            "available_bars": None,
            "required_bars": REQUIRED_BARS,
            "error": detail,
            "feed": settings.ALPACA_DATA_FEED,
        }
    except Exception as e:
        logger.error(f"get_indicators failed for {symbol}: {e}")
        return {"error": _map_data_error(e, f"bars for {symbol}"), "symbol": symbol}


def get_snapshot(symbol: str) -> dict:
    """
    Latest trade, latest quote, and current minute/daily bars for `symbol`
    from the CONFIGURED feed (Alpaca snapshot API). Fields the feed cannot
    supply are null — never fabricated. Feed-subscription failures map to
    DATA_UNAVAILABLE (SUBSCRIPTION_FEED_UNAVAILABLE).
    """
    try:
        data_client = _get_data_client()
        req = StockSnapshotRequest(symbol_or_symbols=symbol, feed=_resolve_feed())
        snapshots = data_client.get_stock_snapshot(req)

        if isinstance(snapshots, dict):
            snap = snapshots.get(symbol)
        else:
            snap = snapshots
        if snap is None:
            return {"symbol": symbol, "error": f"No snapshot returned for {symbol}"}

        def _trade(t):
            return {
                "price": float(t.price) if t and t.price is not None else None,
                "size": int(t.size) if t and t.size is not None else None,
                "timestamp": t.timestamp.isoformat() if t and t.timestamp else None,
            } if t else None

        def _quote(q):
            return {
                "bid_price": float(q.bid_price) if q and q.bid_price is not None else None,
                "ask_price": float(q.ask_price) if q and q.ask_price is not None else None,
                "bid_size": int(q.bid_size) if q and q.bid_size is not None else None,
                "ask_size": int(q.ask_size) if q and q.ask_size is not None else None,
                "timestamp": q.timestamp.isoformat() if q and q.timestamp else None,
            } if q else None

        def _bar(b):
            return {
                "open": float(b.open) if b and b.open is not None else None,
                "high": float(b.high) if b and b.high is not None else None,
                "low": float(b.low) if b and b.low is not None else None,
                "close": float(b.close) if b and b.close is not None else None,
                "volume": int(b.volume) if b and b.volume is not None else None,
                "timestamp": b.timestamp.isoformat() if b and b.timestamp else None,
            } if b else None

        return {
            "symbol": symbol,
            "feed": settings.ALPACA_DATA_FEED,
            "latest_trade": _trade(getattr(snap, "latest_trade", None)),
            "latest_quote": _quote(getattr(snap, "latest_quote", None)),
            "minute_bar": _bar(getattr(snap, "minute_bar", None)),
            "daily_bar": _bar(getattr(snap, "daily_bar", None)),
            "previous_daily_bar": _bar(getattr(snap, "previous_daily_bar", None)),
            "error": None,
        }
    except Exception as e:
        logger.error(f"get_snapshot failed for {symbol}: {e}")
        return {"symbol": symbol, "error": _map_data_error(e, f"snapshot for {symbol}")}


def get_clock() -> dict:
    """Market clock: is the market open, and the next open/close times."""
    try:
        client = _get_trading_client()
        clock = client.get_clock()
        if isinstance(clock, dict):
            return {
                "is_open": bool(clock.get("is_open")),
                "timestamp": str(clock.get("timestamp")),
                "next_open": str(clock.get("next_open")),
                "next_close": str(clock.get("next_close")),
                "error": None,
            }
        return {
            "is_open": bool(clock.is_open),
            "timestamp": clock.timestamp.isoformat() if clock.timestamp else None,
            "next_open": clock.next_open.isoformat() if clock.next_open else None,
            "next_close": clock.next_close.isoformat() if clock.next_close else None,
            "error": None,
        }
    except Exception as e:
        logger.error(f"get_clock failed: {e}")
        return {"is_open": None, "timestamp": None, "next_open": None,
                "next_close": None, "error": str(e)}


def get_corporate_actions(symbol: str, days_back: int = 90) -> dict:
    """Recent corporate-action announcements (dividends, splits, mergers,
    spinoffs) for `symbol` via the Alpaca Trading API. Available on demand
    for the normalized data layer; not part of every cycle."""
    try:
        client = _get_trading_client()
        # The SDK requires exact dates (midnight) for since/until.
        until = datetime.utcnow().date()
        since = until - timedelta(days=days_back)
        req = GetCorporateAnnouncementsRequest(
            ca_types=[
                CorporateActionType.DIVIDEND,
                CorporateActionType.MERGER,
                CorporateActionType.SPINOFF,
                CorporateActionType.SPLIT,
            ],
            since=since,
            until=until,
            symbol=symbol,
        )
        announcements = client.get_corporate_announcements(req)
        items = []
        for a in announcements or []:
            items.append({
                "id": str(a.id),
                "type": str(a.ca_type.value if hasattr(a.ca_type, "value") else a.ca_type),
                "sub_type": str(a.ca_sub_type) if a.ca_sub_type else None,
                "target_symbol": a.target_symbol,
                "declaration_date": a.declaration_date.isoformat() if a.declaration_date else None,
                "ex_date": a.ex_date.isoformat() if a.ex_date else None,
                "payable_date": a.payable_date.isoformat() if a.payable_date else None,
            })
        return {"symbol": symbol, "actions": items, "error": None}
    except Exception as e:
        logger.error(f"get_corporate_actions failed for {symbol}: {e}")
        return {"symbol": symbol, "actions": [], "error": _map_data_error(e, f"corporate actions for {symbol}")}


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
