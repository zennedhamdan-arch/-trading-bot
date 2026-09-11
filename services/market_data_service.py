"""
services/market_data_service.py

Normalized market-data layer. ALL agents consume data through this service —
never by calling external APIs directly.

    MarketDataService
      ├── PriceData          (latest price: snapshot trade → quote mid → bar close)
      ├── HistoricalBars     (Alpaca bars + deterministic indicator analytics)
      ├── Quotes             (Alpaca snapshots: latest trade/quote/minute bar)
      ├── News               (Alpaca News API)
      ├── Fundamentals       (FundamentalsProvider abstraction, default "none")
      └── CorporateActions   (Alpaca corporate announcements, on demand)

Alpaca is the PRIMARY market-data source (bars, quotes, trades, snapshots,
account, orders, market status, news). yfinance is gone from the critical
path — fundamentals come from the pluggable FundamentalsProvider and are
explicitly DATA_UNAVAILABLE when no provider is configured.

Every bundle carries a data_quality map so downstream stages (and the
dashboard) know exactly what is real, what is unavailable, and why:

    {
      "symbol": "NVDA",
      "price": 172.40,
      "quote": {...}, "snapshot": {...},
      "indicators": {... deterministic analytics ...},
      "news": [...],
      "fundamentals": {... normalized or DATA_UNAVAILABLE ...},
      "data_quality": {
        "price": "OK",
        "bars": "OK",
        "news": "OK",
        "fundamentals": "DATA_UNAVAILABLE"
      }
    }

Feed honesty: the configured ALPACA_DATA_FEED (default iex) is carried in
every bars/snapshot response. A feed the subscription does not permit
produces DATA_UNAVAILABLE (SUBSCRIPTION_FEED_UNAVAILABLE) — never a silent
feed switch.
"""

import logging

from config import settings
from services import alpaca_service, fundamentals_service

logger = logging.getLogger("market_data_service")


# ---------------------------------------------------------------------------
# Individual accessors (delegating to the Alpaca transport or providers)
# ---------------------------------------------------------------------------


def get_bars_and_indicators(symbol: str) -> dict:
    """Historical daily bars + deterministic technical analytics (RSI, SMAs,
    EMA, MACD, ATR, volatility, drawdown, returns, rule-based signal)."""
    return alpaca_service.get_indicators(symbol)


def get_quote(symbol: str) -> dict:
    """Latest trade/quote/minute bar via the Alpaca snapshot API."""
    return alpaca_service.get_snapshot(symbol)


def _price_from_snapshot(snapshot: dict):
    """(price, source) from a snapshot response, or (None, None)."""
    if snapshot.get("error"):
        return None, None
    trade = snapshot.get("latest_trade") or {}
    if trade.get("price") is not None:
        return trade["price"], "snapshot.latest_trade"
    quote = snapshot.get("latest_quote") or {}
    if quote.get("bid_price") is not None and quote.get("ask_price") is not None:
        mid = round((quote["bid_price"] + quote["ask_price"]) / 2.0, 4)
        return mid, "snapshot.quote_mid"
    minute_bar = snapshot.get("minute_bar") or {}
    if minute_bar.get("close") is not None:
        return minute_bar["close"], "snapshot.minute_bar"
    return None, None


def get_price(symbol: str, indicators: dict = None) -> dict:
    """Latest price, best-effort from the snapshot chain:
    latest trade → quote midpoint → latest bar close. Returns
    {price, source, error}; never fabricates a number."""
    snapshot = alpaca_service.get_snapshot(symbol)
    if not snapshot.get("error"):
        trade = snapshot.get("latest_trade") or {}
        if trade.get("price") is not None:
            return {"price": trade["price"], "source": "snapshot.latest_trade", "error": None}
        quote = snapshot.get("latest_quote") or {}
        if quote.get("bid_price") is not None and quote.get("ask_price") is not None:
            mid = round((quote["bid_price"] + quote["ask_price"]) / 2.0, 4)
            return {"price": mid, "source": "snapshot.quote_mid", "error": None}
        minute_bar = snapshot.get("minute_bar") or {}
        if minute_bar.get("close") is not None:
            return {"price": minute_bar["close"], "source": "snapshot.minute_bar", "error": None}
    # Snapshot unavailable (e.g. outside feed support) — fall back to the
    # daily-bar close, labeled as such.
    indicators = indicators if indicators is not None else alpaca_service.get_indicators(symbol)
    if indicators.get("latest_close") is not None:
        return {"price": indicators["latest_close"], "source": "daily_bar_close", "error": None}
    err = snapshot.get("error") or indicators.get("error") or "no price source available"
    return {"price": None, "source": None, "error": err}


def get_news(symbol: str, limit: int = 10) -> dict:
    """Recent news headlines for a symbol via Alpaca's News API.
    Returns {"headlines": [...], "error": None} — an empty list on failure,
    never fabricated headlines."""
    result = get_news_articles(symbol, limit=limit)
    return {"headlines": [a["headline"] for a in result["articles"]],
            "error": result["error"]}


def get_news_articles(symbol: str, limit: int = 20) -> dict:
    """Full article objects for a symbol via Alpaca's News API — the raw
    material of the News Intelligence system (id, headline, summary,
    source, url, published_at, symbols). Returns {"articles": [...],
    "error": None}; an empty list on failure, never fabricated articles."""
    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest

        client = NewsClient(api_key=settings.ALPACA_API_KEY, secret_key=settings.ALPACA_SECRET_KEY)
        req = NewsRequest(symbols=symbol, limit=limit)
        news_set = client.get_news(req)
        raw = []
        if hasattr(news_set, "data"):
            raw = news_set.data.get("news", []) if isinstance(news_set.data, dict) else []
        if not raw and hasattr(news_set, "news"):
            raw = news_set.news
        articles = []
        for item in raw:
            articles.append({
                "id": str(getattr(item, "id", "") or ""),
                "headline": str(getattr(item, "headline", "") or ""),
                "summary": str(getattr(item, "summary", "") or ""),
                "source": (getattr(getattr(item, "source", None), "name", None)
                           if not isinstance(getattr(item, "source", None), str)
                           else getattr(item, "source", "")) or "",
                "url": str(getattr(item, "url", "") or ""),
                "author": str(getattr(item, "author", "") or ""),
                "published_at": str(getattr(item, "created_at", "") or ""),
                "symbols": [str(s) for s in (getattr(item, "symbols", None) or [])],
            })
        return {"articles": articles, "error": None}
    except Exception as exc:  # noqa: BLE001 — news failure never crashes a cycle
        logger.warning(f"Could not fetch news for {symbol}: {exc}")
        return {"articles": [], "error": str(exc)}


def get_fundamentals(symbol: str) -> dict:
    """Normalized fundamentals via the FundamentalsProvider abstraction
    (primary + optional fallback). DATA_UNAVAILABLE when no provider is
    configured — never fabricated, never scraped."""
    return fundamentals_service.get_fundamentals(symbol)


def get_market_status() -> dict:
    """Market open/closed + next open/close (Alpaca clock)."""
    return alpaca_service.get_clock()


def get_corporate_actions(symbol: str) -> dict:
    """Corporate-action announcements for a symbol (Alpaca Trading API)."""
    return alpaca_service.get_corporate_actions(symbol)


# ---------------------------------------------------------------------------
# The per-symbol normalized bundle the cycle consumes
# ---------------------------------------------------------------------------


def get_symbol_data(symbol: str) -> dict:
    """Collects and normalizes everything available for one symbol in one
    call: bars+analytics, price, quote/snapshot, news, fundamentals, plus a
    data_quality map. Individual failures are recorded per field — one
    unavailable source NEVER aborts the bundle."""
    indicators = get_bars_and_indicators(symbol)
    bars_ok = not indicators.get("error")

    snapshot = alpaca_service.get_snapshot(symbol)
    price, price_source = _price_from_snapshot(snapshot)
    price_error = None
    if price is None:
        # Snapshot unavailable (e.g. feed limitation) — fall back to the
        # daily-bar close, labeled as such.
        if indicators.get("latest_close") is not None:
            price, price_source = indicators["latest_close"], "daily_bar_close"
        else:
            price_error = snapshot.get("error") or indicators.get("error") or "no price source available"

    fundamentals = get_fundamentals(symbol)
    fundamentals_status = fundamentals.get("status", "DATA_UNAVAILABLE")

    bundle = {
        "symbol": symbol,
        "price": price,
        "price_source": price_source,
        "quote": (snapshot.get("latest_quote") if not snapshot.get("error") else None),
        "snapshot_error": snapshot.get("error"),
        "indicators": indicators,
        # News is OWNED by the independent news worker (services/news_worker.py)
        # — a trading cycle never waits on the News API. The cycle reads the
        # persistent intelligence cache instead (agents/news_agent.py).
        "news": [],
        "fundamentals": fundamentals,
        "data_quality": {
            "price": "OK" if price is not None else "DATA_UNAVAILABLE",
            "bars": "OK" if bars_ok else "DATA_UNAVAILABLE",
            "news": "DELEGATED_TO_WORKER",
            "fundamentals": fundamentals_status,
        },
        "feed": settings.ALPACA_DATA_FEED,
    }
    if price_error:
        bundle["price_error"] = price_error
    if not bars_ok:
        bundle["bars_error"] = indicators["error"]
    return bundle


def feed_config() -> dict:
    """Configured feed info (for /api/config and health; honest labeling)."""
    return {
        "configured_feed": settings.ALPACA_DATA_FEED,
        "note": "IEX is the free/paper-subscription feed; SIP requires a paid "
                "subscription. A non-permitted feed fails honestly with "
                "SUBSCRIPTION_FEED_UNAVAILABLE — never switched silently.",
    }
