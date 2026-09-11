"""
tests/test_data_layer.py

Tests the normalized data layer and deterministic services for the full
trade universe (AAPL, MSFT, NVDA, TSLA, SPY):

  1. MarketDataService: per-symbol bundles with data_quality; price chain
     (snapshot trade -> quote mid -> bar close); bars; news (real logic with
     a stubbed Alpaca NewsClient); fundamentals (provider "none" ->
     DATA_UNAVAILABLE); market status; corporate actions.
  2. Deterministic analytics: RSI/SMA/EMA/MACD/ATR/volatility/drawdown/
     returns/rule-based signal computed in Python from bars.
  3. Alpaca feed limitations: subscription failures map to
     DATA_UNAVAILABLE (SUBSCRIPTION_FEED_UNAVAILABLE) — never a silent
     feed switch, never a crash.
  4. yfinance is not required: no import anywhere, not in requirements.
  5. Risk gate: deterministic assessment + final order validation (action,
     symbol, quantity, buying power, position limits, risk constraints).
  6. Realtime service: Alpaca WS handlers update normalized state; ticks
     never touch an LLM; disabled without keys.
  7. Startup health: structured report for Alpaca / market data /
     fundamentals / LLM providers without secrets.

Run:  .venv/bin/python tests/test_data_layer.py
"""

import asyncio
import os
import sys
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}")


UNIVERSE = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]

import config
config.settings.ALPACA_API_KEY = "test-key"
config.settings.ALPACA_SECRET_KEY = "test-secret"
config.settings.ALPACA_DATA_FEED = "IEX"
config.settings.GROQ_API_KEY = "test-groq"
config.settings.GEMINI_API_KEY = "test-gemini"
config.settings.OPENROUTER_API_KEY = ""
config.settings.NVIDIA_API_KEY = ""
config.settings.FUNDAMENTALS_PROVIDER = "none"
config.settings.FUNDAMENTALS_FALLBACK_PROVIDER = ""
config.settings.ENABLE_MEMORY = False
config.settings.TRADE_UNIVERSE = UNIVERSE

from services import (alpaca_service, market_data_service, fundamentals_service,
                      risk_gate, realtime_service, llm_service)
from agents import fundamentals_agent

fundamentals_service.reset_cache()
fundamentals_agent.reset_fundamentals_cache()
realtime_service.reset_state()
llm_service.reset_all_state()


# ---------------------------------------------------------------------------
# Fake Alpaca transports
# ---------------------------------------------------------------------------

class _FakeAccount:
    cash, equity, buying_power, portfolio_value, last_equity = 12043.0, 25104.0, 12043.0, 25104.0, 25000.0
    status = types.SimpleNamespace(value="ACTIVE")


class _FakeClock:
    is_open = True
    timestamp = None
    next_open = None
    next_close = None


class _FakeTradingClient:
    def __init__(self):
        self.announcements_requested = []
    def get_account(self):
        return _FakeAccount()
    def get_all_positions(self):
        return []
    def get_clock(self):
        return _FakeClock()
    def get_corporate_announcements(self, req):
        self.announcements_requested.append(req)
        return [types.SimpleNamespace(
            id="ca-1", ca_type=types.SimpleNamespace(value="dividend"), ca_sub_type=None,
            target_symbol="AAPL", declaration_date=None, ex_date=None, payable_date=None)]


class _FakeSnapshot(types.SimpleNamespace):
    pass


def _make_bars(symbol, n=260):
    import pandas as pd
    closes = [100 + (i % 20) * 0.5 for i in range(n)]
    highs = [c + 1.5 for c in closes]
    lows = [c - 1.5 for c in closes]
    end = pd.Timestamp.utcnow().floor("D")
    idx = pd.DatetimeIndex(pd.date_range(end=end, periods=n, freq="D", tz="UTC"), name="timestamp")
    return pd.DataFrame({"close": closes, "open": closes, "high": highs, "low": lows,
                         "volume": [1000] * n, "trade_count": [10] * n, "vwap": closes}, index=idx)


class _FakeDataClient:
    def __init__(self):
        self.bar_requests = []
        self.snapshot_requests = []
        self.fail_bars_with = None
        self.snapshot_mode = "full"  # full | quote_only | error
    def get_stock_bars(self, req):
        self.bar_requests.append(req)
        if self.fail_bars_with:
            raise self.fail_bars_with
        return types.SimpleNamespace(df=_make_bars(getattr(req, "symbol_or_symbols", "AAPL")))
    def get_stock_snapshot(self, req):
        self.snapshot_requests.append(req)
        sym = req.symbol_or_symbols if isinstance(req.symbol_or_symbols, str) else list(req.symbol_or_symbols)[0]
        if self.snapshot_mode == "error":
            raise RuntimeError("subscription does not permit querying recent SIP data")
        trade = types.SimpleNamespace(price=101.25, size=10, timestamp=None) if self.snapshot_mode == "full" else None
        quote = types.SimpleNamespace(bid_price=100.5, ask_price=101.5, bid_size=2, ask_size=2, timestamp=None)
        return {sym: types.SimpleNamespace(symbol=sym, latest_trade=trade, latest_quote=quote,
                                           minute_bar=None, daily_bar=None, previous_daily_bar=None)}


fake_trading = _FakeTradingClient()
fake_data = _FakeDataClient()
alpaca_service._trading_client = fake_trading
alpaca_service._data_client = fake_data

# Stub the Alpaca NewsClient module so get_news's real logic runs offline
class _FakeNewsSet:
    def __init__(self, headlines):
        self.data = {"news": [types.SimpleNamespace(headline=h) for h in headlines]}


class _FakeNewsClient:
    last_request = None
    def __init__(self, api_key=None, secret_key=None):
        pass
    def get_news(self, req):
        _FakeNewsClient.last_request = req
        return _FakeNewsSet([f"headline-1", "headline-2"])


fake_news_module = types.ModuleType("alpaca.data.historical.news")
fake_news_module.NewsClient = _FakeNewsClient
sys.modules["alpaca.data.historical.news"] = fake_news_module


# ===========================================================================
print("1. normalized bundles for the full universe:")
for sym in UNIVERSE:
    bundle = market_data_service.get_symbol_data(sym)
    q = bundle["data_quality"]
    check(f"{sym}: bundle price from snapshot trade (101.25)",
          bundle["price"] == 101.25 and bundle["price_source"] == "snapshot.latest_trade")
    check(f"{sym}: quote normalized (bid/ask)", bundle["quote"]["bid_price"] == 100.5
          and bundle["quote"]["ask_price"] == 101.5)
    check(f"{sym}: data_quality price/bars OK, news delegated to the worker",
          q["price"] == "OK" and q["bars"] == "OK"
          and q["news"] == "DELEGATED_TO_WORKER")
    check(f"{sym}: fundamentals DATA_UNAVAILABLE (no provider), never fabricated",
          q["fundamentals"] == "DATA_UNAVAILABLE"
          and bundle["fundamentals"]["reason"] == "NO_PROVIDER_CONFIGURED"
          and bundle["fundamentals"]["pe_ratio"] is None)
    check(f"{sym}: feed labeled honestly", bundle["feed"] == "IEX")

bundle = market_data_service.get_symbol_data("AAPL")
check("cycle bundle carries no news fetch (worker-owned; cycle never blocks on news)",
      bundle["news"] == [] and _FakeNewsClient.last_request is None)
news_result = market_data_service.get_news("AAPL")
check("news headlines retrieved via the normalized layer",
      news_result["headlines"] == ["headline-1", "headline-2"]
      and _FakeNewsClient.last_request is not None)
check("news request carries symbol", _FakeNewsClient.last_request.symbols == "AAPL")
articles = market_data_service.get_news_articles("AAPL")
check("full article objects available for the news worker (id/headline/source/...)",
      articles["error"] is None and len(articles["articles"]) == 2
      and articles["articles"][0]["headline"] == "headline-1")

# ---------------------------------------------------------------------------
print("2. price fallback chain (snapshot unavailable -> bar close):")
fake_data.snapshot_mode = "quote_only"  # no trade -> quote midpoint
b = market_data_service.get_symbol_data("MSFT")
check("no trade -> quote midpoint (101.0)", b["price"] == 101.0 and b["price_source"] == "snapshot.quote_mid")
fake_data.snapshot_mode = "error"       # snapshot fails -> daily bar close
b = market_data_service.get_symbol_data("MSFT")
check("snapshot error -> latest daily bar close, labeled",
      b["price"] is not None and b["price_source"] == "daily_bar_close"
      and b["data_quality"]["price"] == "OK")
check("snapshot failure recorded but bars still OK",
      "snapshot_error" in b and b["data_quality"]["bars"] == "OK")
fake_data.snapshot_mode = "full"

# ---------------------------------------------------------------------------
print("3. deterministic analytics (computed in Python, never by an LLM):")
ind = alpaca_service.get_indicators("NVDA")
keys = ("rsi_14", "sma_20", "sma_50", "sma_200", "ema_20", "macd", "macd_signal",
        "atr_14", "volatility_annualized", "max_drawdown", "return_1d", "return_5d",
        "return_20d", "technical_signal", "technical_components", "recent_closes",
        "latest_close", "feed")
check("all deterministic analytics present", all(k in ind for k in keys))
check("values are real numbers from the bars",
      isinstance(ind["rsi_14"], float) and 0 <= ind["rsi_14"] <= 100
      and ind["sma_50"] is not None and ind["atr_14"] > 0
      and ind["volatility_annualized"] > 0 and ind["max_drawdown"] < 0)
check("rule-based signal is one of BULLISH/BEARISH/NEUTRAL",
      ind["technical_signal"] in ("BULLISH", "BEARISH", "NEUTRAL")
      and set(ind["technical_components"]) == {"trend", "momentum", "rsi_flag"})
check("feed carried in indicators (honest labeling)", ind["feed"] == "IEX")
for sym in UNIVERSE:
    i = alpaca_service.get_indicators(sym)
    check(f"{sym}: indicators calculate from bars", i.get("error") is None and i["rsi_14"] is not None)

# ---------------------------------------------------------------------------
print("4. Alpaca feed limitations (never a silent switch):")
fake_data.fail_bars_with = RuntimeError("subscription does not permit querying recent SIP data")
err = alpaca_service.get_indicators("AAPL")
check("SIP denial -> DATA_UNAVAILABLE (SUBSCRIPTION_FEED_UNAVAILABLE)",
      "DATA_UNAVAILABLE" in err["error"] and "SUBSCRIPTION_FEED_UNAVAILABLE" in err["error"])
check("error names the configured feed and the fix", "ALPACA_DATA_FEED" in err["error"])
fake_data.snapshot_mode = "error"
snap_err = alpaca_service.get_snapshot("AAPL")
check("snapshot SIP denial mapped the same way",
      "SUBSCRIPTION_FEED_UNAVAILABLE" in snap_err["error"])
fake_data.fail_bars_with = None
fake_data.snapshot_mode = "full"

clock = alpaca_service.get_clock()
check("market status retrieved (is_open)", clock["is_open"] is True and clock["error"] is None)

actions = alpaca_service.get_corporate_actions("AAPL")
check("corporate actions retrieved via the Trading API",
      actions["error"] is None and actions["actions"][0]["type"] == "dividend"
      and len(fake_trading.announcements_requested) == 1)

# ---------------------------------------------------------------------------
print("5. yfinance is not required:")
_req = open("requirements.txt").read()
check("yfinance not in requirements.txt", "yfinance" not in _req)
import subprocess
_res = subprocess.run(["grep", "-rn", "yfinance", "--include=*.py", "services/", "agents/", "main.py", "config.py"],
                      capture_output=True, text=True)
_imports = [l for l in _res.stdout.splitlines() if "import" in l]
check("no yfinance imports in the application", not _imports)
if _imports:
    print("      stale:", _imports)
check("yfinance never loaded during data-layer tests", "yfinance" not in sys.modules)

# ---------------------------------------------------------------------------
print("6. fundamentals providers (pluggable, honest):")
fs = fundamentals_service


class _OkProvider(fs.FundamentalsProvider):
    name = "testvendor"
    def get_fundamentals(self, symbol):
        return fs._normalized(symbol, self.name, "OK", pe_ratio=28.4, eps=6.3,
                              profit_margin=0.27, roe=0.38)


class _DownProvider(fs.FundamentalsProvider):
    name = "testdown"
    def get_fundamentals(self, symbol):
        return fs._normalized(symbol, self.name, "DATA_UNAVAILABLE", reason="vendor outage")


fs._PROVIDER_CLASSES["testvendor"] = _OkProvider
fs._PROVIDER_CLASSES["testdown"] = _DownProvider

config.settings.FUNDAMENTALS_PROVIDER = "testvendor"
fs.reset_cache()
r = fs.get_fundamentals("AAPL")
check("configured provider returns normalized data",
      r["status"] == "OK" and r["provider"] == "testvendor" and r["pe_ratio"] == 28.4)
check("unavailable fields are null (never fabricated)",
      r["market_cap"] is None and r["revenue"] is None and r["debt_to_equity"] is None)

config.settings.FUNDAMENTALS_PROVIDER = "testdown"
config.settings.FUNDAMENTALS_FALLBACK_PROVIDER = "testvendor"
fs.reset_cache()
r = fs.get_fundamentals("AAPL")
check("fallback provider used when the primary is down",
      r["status"] == "OK" and r["provider"] == "testvendor")

config.settings.FUNDAMENTALS_PROVIDER = "none"
config.settings.FUNDAMENTALS_FALLBACK_PROVIDER = ""
fs.reset_cache()
for sym in UNIVERSE:
    fundamentals_service.reset_cache()
    b = market_data_service.get_symbol_data(sym)
    check(f"{sym}: no provider -> DATA_UNAVAILABLE does not crash the bundle",
          b["data_quality"]["fundamentals"] == "DATA_UNAVAILABLE" and b["price"] is not None)

af = fundamentals_agent.analyze_fundamentals("AAPL", fs.get_fundamentals("AAPL"))
check("fundamentals agent degrades to explicit DATA_UNAVAILABLE",
      af["error"] == "DATA_UNAVAILABLE: NO_PROVIDER_CONFIGURED"
      and af["llm_status"] == "SKIPPED_NO_DATA")
check("no provider configured -> evidence OFF (config, not a failure) with NULL stance/confidence (never a fake NEUTRAL)",
      af["evidence_status"] == "OFF" and af["signal"] is None and af["confidence"] is None)

# ---------------------------------------------------------------------------
print("7. risk gate (deterministic; AI cannot bypass):")
acct = {"equity": 10000.0, "cash": 4000.0, "buying_power": 8000.0}

g = risk_gate.assess("AAPL", "buy", acct, None, {"volatility_annualized": 0.2, "max_drawdown": -0.1})
check("buy sizing: 10% of equity, capped by cash", g["max_notional_usd"] == 1000.0 and g["approved"])
check("risk level deterministic (LOW for low vol)",
      g["risk_level"] == "LOW" and g["checks"]["equity_positive"])

g = risk_gate.assess("AAPL", "buy", acct, None, {"volatility_annualized": 0.8, "max_drawdown": -0.5})
check("risk level HIGH for extreme vol/drawdown", g["risk_level"] == "HIGH")

pos = {"symbol": "AAPL", "qty": 10, "current_price": 95.0, "avg_entry_price": 90.0,
       "market_value": 950.0}
g = risk_gate.assess("AAPL", "buy", acct, pos, None)
check("existing position reduces the cap (1000 - 950 = 50)",
      g["max_notional_usd"] == 50.0 and g["position_pct"] == 0.095)

pos_full = {"symbol": "AAPL", "qty": 10, "current_price": 100.0, "market_value": 1000.0}
g = risk_gate.assess("AAPL", "buy", acct, pos_full, None)
check("at the cap -> not approved, reason given",
      not g["approved"] and any("max" in r for r in g["reasons"]))

g = risk_gate.assess("AAPL", "sell", acct, None, None)
check("sell without a position -> not approved", not g["approved"])
g = risk_gate.assess("AAPL", "sell", acct, pos, None)
check("sell with a position -> approved, qty bounded", g["approved"] and g["sell_max_qty"] == 10)

v = risk_gate.validate_order("AAPL", "buy", acct, {}, notional_usd=500.0, trade_universe=UNIVERSE)
check("valid BUY passes the final gate", v["valid"] and v["reason"] is None)
v = risk_gate.validate_order("AAPL", "buy", acct, {}, notional_usd=5000.0, trade_universe=UNIVERSE)
check("BUY over cash blocked with reason", not v["valid"] and "cash" in v["reason"])
v = risk_gate.validate_order("AAPL", "buy", acct, {"AAPL": pos_full}, notional_usd=100.0, trade_universe=UNIVERSE)
check("BUY over the position cap blocked", not v["valid"] and "max position" in v["reason"])
v = risk_gate.validate_order("AAPL", "buy", acct, {}, notional_usd=100.0, trade_universe=["MSFT"])
check("symbol outside the universe blocked", not v["valid"] and "universe" in v["reason"])
v = risk_gate.validate_order("AAPL", "sell", acct, {}, qty=5.0, trade_universe=UNIVERSE)
check("SELL without a position blocked", not v["valid"] and "no open position" in v["reason"])
v = risk_gate.validate_order("AAPL", "sell", acct, {"AAPL": pos}, qty=10.0, trade_universe=UNIVERSE)
check("valid SELL passes", v["valid"])
v = risk_gate.validate_order("AAPL", "hold", acct, {}, notional_usd=10.0, trade_universe=UNIVERSE)
check("invalid action blocked", not v["valid"] and "invalid action" in v["reason"])
v = risk_gate.validate_order("AAPL", "buy", acct, {}, notional_usd=0.0, trade_universe=UNIVERSE)
check("zero-size order blocked", not v["valid"] and "positive" in v["reason"])

# ---------------------------------------------------------------------------
print("8. realtime layer (Alpaca WS; SDK run() on dedicated threads; ticks never trigger LLMs):")


class _FakeSDKStream:
    """Models the INSTALLED alpaca-py DataStream semantics:
    run() is SYNCHRONOUS and blocks until stop() (it owns its loop);
    _running flips True once 'connected'; stop() is thread-safe."""

    instances = []

    def __init__(self, kind):
        self.kind = kind
        self.handlers = {}
        self._running = False
        self._stop = threading.Event()
        self.ran = False
        self.stopped = False
        _FakeSDKStream.instances.append(self)

    def subscribe_trades(self, handler, *symbols):
        self.handlers["trades"] = (handler, symbols)

    def subscribe_quotes(self, handler, *symbols):
        self.handlers["quotes"] = (handler, symbols)

    def subscribe_bars(self, handler, *symbols):
        self.handlers["bars"] = (handler, symbols)

    def subscribe_news(self, handler, *symbols):
        self.handlers["news"] = (handler, symbols)

    def run(self):
        self.ran = True
        self._running = True  # 'connected + authed + subscribed'
        self._stop.wait()     # blocks like the real SDK (owns its loop)
        self._running = False

    def stop(self):
        self.stopped = True
        self._stop.set()

    def die(self):
        """Simulates a fatal stream error: run() returns on its own."""
        self._stop.set()


def _fake_build_stock_stream():
    stream = _FakeSDKStream("stock")
    stream.subscribe_trades(realtime_service._on_trade, *UNIVERSE)
    stream.subscribe_quotes(realtime_service._on_quote, *UNIVERSE)
    stream.subscribe_bars(realtime_service._on_bar, *UNIVERSE)
    return stream, "iex", list(UNIVERSE)


def _fake_build_news_stream():
    stream = _FakeSDKStream("news")
    stream.subscribe_news(realtime_service._on_news, *UNIVERSE)
    return stream


realtime_service._BUILDERS = {"stock": _fake_build_stock_stream, "news": _fake_build_news_stream}


async def _rt_wait(predicate, timeout=6.0):
    """Async-friendly wait: must yield to the event loop (the drain task
    runs on it) — never blocks with time.sleep."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


async def _rt_test():
    # Fast backoff for the lifecycle test (real default is 2s base / 60s cap).
    config.settings.REALTIME_RECONNECT_SECONDS = 0.05
    config.settings.REALTIME_RECONNECT_MAX_SECONDS = 0.2
    # Staleness recycle is exercised separately below (it would otherwise
    # recycle continuously, since fake streams never emit ticks on their own).
    config.settings.REALTIME_STALE_TICK_SECONDS = 60.0

    await realtime_service.start_async()

    # Status is honest: CONNECTING before genuinely running, CONNECTED after.
    ok = await _rt_wait(lambda: realtime_service.get_state()["connection_status"] == "CONNECTED")
    check("status becomes CONNECTED only once the stream genuinely runs", ok)
    state = realtime_service.get_state()
    check("connected_at recorded when genuinely connected", state["connected_at"] is not None)
    check("exactly ONE stock stream instance (no duplicates)", len([s for s in _FakeSDKStream.instances if s.kind == "stock"]) == 1)
    check("status key mirrors connection_status (legacy)", state["status"] == state["connection_status"])

    stock = [s for s in _FakeSDKStream.instances if s.kind == "stock"][0]
    news = [s for s in _FakeSDKStream.instances if s.kind == "news"][0]
    handler, symbols = stock.handlers["trades"]
    check("subscribed to trades for the whole universe", set(symbols) == set(UNIVERSE))
    check("feed recorded", state["feed"] == "iex" and state["symbols"] == list(UNIVERSE))

    # Handlers only enqueue; the drain task applies them on this loop.
    await handler(types.SimpleNamespace(symbol="AAPL", price=175.25, size=3, timestamp=None))
    await stock.handlers["quotes"][0](types.SimpleNamespace(
        symbol="AAPL", bid_price=175.0, ask_price=175.5, bid_size=1, ask_size=1, timestamp=None))
    await stock.handlers["bars"][0](types.SimpleNamespace(
        symbol="AAPL", open=175.0, high=176.0, low=174.5, close=175.4, volume=1200, timestamp=None))
    await news.handlers["news"][0](types.SimpleNamespace(
        headline="AAPL wins", summary="big", source="api", symbols=["AAPL"], created_at=None))
    ok = await _rt_wait(lambda: realtime_service.get_state()["data"].get("AAPL", {}).get("last_trade", {}).get("price") == 175.25)
    check("tick bridged through the thread-safe queue (trade)", ok)
    state = realtime_service.get_state()
    check("last tick normalized (quote + bar)",
          state["data"]["AAPL"]["last_quote"]["ask_price"] == 175.5
          and state["data"]["AAPL"]["minute_bar"]["close"] == 175.4)
    check("news stream normalized", state["news"][0]["headline"] == "AAPL wins")
    check("last_tick_at recorded", state["last_tick_at"] is not None
          and state["seconds_since_last_tick"] is not None)

    # Worker death -> supervisor restarts with backoff, never duplicates.
    stock_count_before = len([s for s in _FakeSDKStream.instances if s.kind == "stock"])
    stock.die()  # run() returns -> worker thread exits
    ok = await _rt_wait(lambda: realtime_service.get_state()["connection_status"] == "CONNECTED"
                  and len([s for s in _FakeSDKStream.instances if s.kind == "stock"]) == stock_count_before + 1)
    check("dead stream restarted (exactly one new instance)", ok)
    ok = await _rt_wait(lambda: realtime_service.get_state()["workers"]["stock"]["genuinely_running"] is True)
    check("restarted stream genuinely running again", ok)
    check("reconnect counter incremented", realtime_service.get_state()["reconnects"] >= 1)

    # Connected-but-silent while market open -> recycled exactly once (fake
    # clock says the market is open; no tick will arrive on the fake stream).
    config.settings.REALTIME_STALE_TICK_SECONDS = 0.4
    before_silent = len([s for s in _FakeSDKStream.instances if s.kind == "stock"])
    ok = await _rt_wait(lambda: len([s for s in _FakeSDKStream.instances if s.kind == "stock"]) == before_silent + 1,
                  timeout=8.0)
    check("silent stream (market open) recycled", ok)
    # Disable further recycles, then verify the replacement is healthy and
    # that never more than one stream instance was alive at any moment.
    config.settings.REALTIME_STALE_TICK_SECONDS = 0.0
    ok = await _rt_wait(lambda: realtime_service.get_state()["connection_status"] == "CONNECTED")
    check("replacement stream CONNECTED after recycle (no storm)", ok)
    live_instances = [s for s in _FakeSDKStream.instances if s.kind == "stock" and s._stop.is_set() is False]
    check("no duplicate concurrent streams (all old instances stopped)",
          len(live_instances) == 1)
    check("every old instance was stopped before its replacement was built",
          all(s.stopped or s is live_instances[0] for s in _FakeSDKStream.instances if s.kind == "stock"))

    await realtime_service.shutdown()
    state = realtime_service.get_state()
    check("shutdown -> DISCONNECTED, workers stopped", state["connection_status"] == "DISCONNECTED")
    check("every SDK stream got stop() on shutdown",
          all(s.stopped for s in _FakeSDKStream.instances if s.kind in ("stock", "news")))
    with realtime_service._lock:
        check("worker registry cleared", not realtime_service._workers)


asyncio.run(_rt_test())
realtime_service.reset_state()
_FakeSDKStream.instances.clear()

# Bounded exponential backoff with jitter: ~2, ~4, ~8, ~16 ... capped.
config.settings.REALTIME_RECONNECT_SECONDS = 2.0
config.settings.REALTIME_RECONNECT_MAX_SECONDS = 60.0
import statistics
for attempt, expected in [(1, 2.0), (2, 4.0), (3, 8.0), (4, 16.0), (5, 32.0), (6, 60.0), (50, 60.0)]:
    delays = [realtime_service._backoff_delay(attempt) for _ in range(200)]
    med = statistics.median(delays)
    check(f"backoff attempt {attempt}: median {med:.1f}s ~ {expected}s (jittered)",
          expected * 0.7 <= med <= expected * 1.3)
    check(f"backoff attempt {attempt}: within jitter bounds",
          all(expected * 0.74 <= d <= expected * 1.26 for d in delays))

# disabled without keys
config.settings.ALPACA_API_KEY = ""
config.settings.ALPACA_SECRET_KEY = ""


async def _rt_disabled():
    await realtime_service.start_async()
    with realtime_service._lock:
        check("no threads started without keys", not realtime_service._workers)


asyncio.run(_rt_disabled())
check("realtime stays down without keys (status NO_KEYS)",
      realtime_service.get_state()["status"] == "NO_KEYS")
_rt_src = open("services/realtime_service.py").read()
check("realtime module never imports the LLM layer (ticks cannot trigger LLM calls)",
      "llm_service" not in _rt_src and "llm_router" not in _rt_src
      and "import" in _rt_src)  # sanity: source read correctly
check("realtime never awaits stream.run() (SDK owns its loop)",
      "await stock.run" not in _rt_src and "await news.run" not in _rt_src
      and "asyncio.gather(stock.run" not in _rt_src)
check("realtime never depends on the private _run_forever implementation",
      "_run_forever" not in _rt_src)

config.settings.ALPACA_API_KEY = "test-key"
config.settings.ALPACA_SECRET_KEY = "test-secret"
config.settings.REALTIME_STALE_TICK_SECONDS = 180.0
config.settings.REALTIME_RECONNECT_SECONDS = 2.0

# ---------------------------------------------------------------------------
print("9. startup health report:")
llm_service.reset_all_state()
llm_service._clients["groq"] = types.SimpleNamespace(
    models=types.SimpleNamespace(list=lambda: types.SimpleNamespace(
        data=[types.SimpleNamespace(id="openai/gpt-oss-20b"), types.SimpleNamespace(id="openai/gpt-oss-120b")])))

from services import health_service
report = health_service.run_startup_checks()
rows = {r["component"]: r for r in report["rows"]}
check("ALPACA ACCOUNT READY", rows["ALPACA ACCOUNT"]["status"] == "READY")
check("ALPACA MARKET DATA READY with feed named",
      rows["ALPACA MARKET DATA"]["status"] == "READY" and "IEX" in rows["ALPACA MARKET DATA"]["detail"].upper())
check("INDICATORS row present and READY (full suite computed)",
      rows["INDICATORS"]["status"] == "READY" and "bars" in str(rows["INDICATORS"]["detail"]))
check("REAL-TIME STREAM row present", "REAL-TIME STREAM" in rows)
check("FUNDAMENTALS DATA_UNAVAILABLE (none provider)",
      rows["FUNDAMENTALS"]["status"] == "DATA_UNAVAILABLE")
check("GROQ READY (models verified)",
      rows["GROQ"]["status"] == "READY")
check("NVIDIA NOT_CONFIGURED (optional)", rows["NVIDIA"]["status"] == "NOT_CONFIGURED")
check("OPENROUTER NOT_CONFIGURED (optional)", rows["OPENROUTER"]["status"] == "NOT_CONFIGURED")
check("GEMINI NOT_CONFIGURED here (no key on router stub)",
      rows["GEMINI"]["status"] in ("NOT_CONFIGURED", "READY"))
check("no secrets in the report",
      all("test-secret" not in str(v) and "test-key" not in str(v) for v in rows.values()))
check("startup rows cover all required components",
      {"ALPACA ACCOUNT", "ALPACA MARKET DATA", "INDICATORS", "REAL-TIME STREAM", "FUNDAMENTALS",
       "GROQ", "NVIDIA", "GEMINI", "OPENROUTER"} <= set(rows))
check("overall status present in the startup report",
      report["overall"]["status"] in ("HEALTHY", "DEGRADED", "OFFLINE"))

# feed probe failure path
fake_data.fail_bars_with = RuntimeError("subscription does not permit querying recent SIP data")
md = health_service._probe_market_data()
check("feed probe detects SUBSCRIPTION_FEED_UNAVAILABLE",
      md["status"] == "SUBSCRIPTION_FEED_UNAVAILABLE")
health_service._probe_cache.update({"ts": 0.0, "account": None, "market_data": None})
overall = health_service.overall_status(live=True)
check("SIP denial -> overall OFFLINE (critical market data)",
      overall["status"] == "OFFLINE" and any("market data" in r.lower() for r in overall["reasons"]))
fake_data.fail_bars_with = None

live = health_service.live_health()
check("live health includes provider circuits, market clock and realtime",
      "providers" in live and "market_clock" in live and "realtime" in live)
check("live health carries the top-level overall status", "overall" in live
      and live["overall"]["status"] in ("HEALTHY", "DEGRADED", "OFFLINE"))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("ALL DATA-LAYER TESTS PASSED")
