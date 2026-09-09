"""
services/realtime_service.py

Real-time market data via Alpaca WebSockets (the official SDK streams):

  - latest trades        (StockDataStream.subscribe_trades)
  - latest quotes        (StockDataStream.subscribe_quotes)
  - minute bars          (StockDataStream.subscribe_bars)
  - news stream          (NewsDataStream.subscribe_news)

The service maintains a small normalized in-memory state for the trade
universe, exposed via /api/realtime for the dashboard. Yahoo Finance is
never polled for real-time prices.

Hard rules:
  - A market tick NEVER triggers an LLM call. The AI cycle stays on its
    scheduled interval (CYCLE_INTERVAL_MINUTES).
  - Fully optional: without Alpaca keys (or with REALTIME_ENABLED=false)
    the service stays down and the bot is unaffected.
  - Connection failures back off and reconnect; they never crash the app.
"""

import asyncio
import logging
import time
from collections import deque

from config import settings
from services import alpaca_service

logger = logging.getLogger("realtime_service")

# ---------------------------------------------------------------------------
# Normalized state (single process, single asyncio loop)
# ---------------------------------------------------------------------------

_state: dict = {}            # symbol -> {"last_trade": {...}, "last_quote": {...}, "minute_bar": {...}, "updated_at": ...}
_recent_news = deque(maxlen=50)
_status = {
    "enabled": False,
    "connected": False,
    "started_at": None,
    "last_event_at": None,
    "last_error": None,
    "reconnects": 0,
    "symbols": [],
    "feed": None,
}


def _get(obj, attr):
    """Reads an attribute from a stream payload (model or dict)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)


def _iso(ts) -> str:
    return ts.isoformat() if hasattr(ts, "isoformat") else (str(ts) if ts else None)


async def _on_trade(trade) -> None:
    symbol = _get(trade, "symbol")
    if not symbol:
        return
    slot = _state.setdefault(symbol, {})
    slot["last_trade"] = {
        "price": _get(trade, "price"),
        "size": _get(trade, "size"),
        "timestamp": _iso(_get(trade, "timestamp")),
    }
    slot["updated_at"] = time.time()
    _status["last_event_at"] = time.time()


async def _on_quote(quote) -> None:
    symbol = _get(quote, "symbol")
    if not symbol:
        return
    slot = _state.setdefault(symbol, {})
    slot["last_quote"] = {
        "bid_price": _get(quote, "bid_price"),
        "ask_price": _get(quote, "ask_price"),
        "bid_size": _get(quote, "bid_size"),
        "ask_size": _get(quote, "ask_size"),
        "timestamp": _iso(_get(quote, "timestamp")),
    }
    slot["updated_at"] = time.time()
    _status["last_event_at"] = time.time()


async def _on_bar(bar) -> None:
    symbol = _get(bar, "symbol")
    if not symbol:
        return
    slot = _state.setdefault(symbol, {})
    slot["minute_bar"] = {
        "open": _get(bar, "open"),
        "high": _get(bar, "high"),
        "low": _get(bar, "low"),
        "close": _get(bar, "close"),
        "volume": _get(bar, "volume"),
        "timestamp": _iso(_get(bar, "timestamp")),
    }
    slot["updated_at"] = time.time()
    _status["last_event_at"] = time.time()


async def _on_news(news) -> None:
    _recent_news.appendleft({
        "headline": _get(news, "headline"),
        "summary": _get(news, "summary"),
        "source": _get(news, "source"),
        "symbols": _get(news, "symbols") or [],
        "created_at": _iso(_get(news, "created_at")),
    })
    _status["last_event_at"] = time.time()


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------


def _build_streams():
    """Constructs the Alpaca SDK stream objects. Tests may monkeypatch this."""
    from alpaca.data.live import StockDataStream, NewsDataStream

    feed = alpaca_service._resolve_feed()
    symbols = [s for s in settings.TRADE_UNIVERSE]

    stock = StockDataStream(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_SECRET_KEY,
        feed=feed,
    )
    stock.subscribe_trades(_on_trade, *symbols)
    stock.subscribe_quotes(_on_quote, *symbols)
    stock.subscribe_bars(_on_bar, *symbols)

    news = NewsDataStream(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_SECRET_KEY,
    )
    news.subscribe_news(_on_news, *symbols)

    return stock, news, feed, symbols


async def run_forever() -> None:
    """Supervisor: connect, run both streams, reconnect with backoff on
    failure. Cancel-safe (the app cancels this task on shutdown)."""
    _status["enabled"] = True
    if not settings.REALTIME_ENABLED:
        _status["enabled"] = False
        _status["last_error"] = "disabled via REALTIME_ENABLED"
        logger.info("Real-time layer disabled (REALTIME_ENABLED=false).")
        return
    if not settings.ALPACA_API_KEY or not settings.ALPACA_SECRET_KEY:
        _status["last_error"] = "Alpaca API keys not configured"
        logger.info("Real-time layer idle: Alpaca API keys not configured.")
        return

    _status["started_at"] = time.time()
    while True:
        try:
            stock, news, feed, symbols = _build_streams()
            _status["symbols"] = symbols
            _status["feed"] = getattr(feed, "value", str(feed))
            _status["connected"] = True
            _status["last_error"] = None
            logger.info(
                f"Real-time layer connected (feed={_status['feed']}, "
                f"symbols={symbols}). Ticks never trigger LLM calls."
            )
            # Both streams run for the lifetime of the connection; an error
            # in either propagates to the supervisor and reconnects.
            await asyncio.gather(stock.run(), news.run())
        except asyncio.CancelledError:
            _status["connected"] = False
            logger.info("Real-time layer stopped (shutdown).")
            raise
        except Exception as exc:  # noqa: BLE001 — reconnect on any failure
            _status["connected"] = False
            _status["last_error"] = str(exc)
            _status["reconnects"] += 1
            delay = max(5.0, float(settings.REALTIME_RECONNECT_SECONDS))
            logger.warning(
                f"Real-time stream error: {exc}. Reconnecting in {delay:.0f}s "
                f"(attempt #{_status['reconnects']})."
            )
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Read API (dashboard + health)
# ---------------------------------------------------------------------------


def get_state() -> dict:
    """Normalized real-time snapshot for /api/realtime."""
    return {
        "status": (
            "DISABLED" if not settings.REALTIME_ENABLED
            else "NO_KEYS" if not settings.ALPACA_API_KEY
            else "CONNECTED" if _status.get("connected")
            else "DISCONNECTED"
        ),
        "feed": _status.get("feed"),
        "symbols": _status.get("symbols") or list(settings.TRADE_UNIVERSE),
        "data": dict(_state),
        "news": list(_recent_news),
        "last_event_at": _status.get("last_event_at"),
        "last_error": _status.get("last_error"),
        "reconnects": _status.get("reconnects", 0),
    }


def reset_state() -> None:
    """Test hook."""
    _state.clear()
    _recent_news.clear()
    _status.update({
        "enabled": False, "connected": False, "started_at": None,
        "last_event_at": None, "last_error": None, "reconnects": 0,
        "symbols": [], "feed": None,
    })
