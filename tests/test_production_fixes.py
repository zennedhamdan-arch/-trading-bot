"""
tests/test_production_fixes.py

Verifies the five Render production fixes without live external services:
  1. openai/groq clients construct under httpx>=0.28 (the 'proxies' bug)
  2. Alpaca market data requests the configured feed (IEX by default)
  3. Portfolio history uses TradingClient.get_portfolio_history + request object
  4. yfinance fundamentals: caching, pacing, 429 -> DATA_UNAVAILABLE + backoff
  5. Cycle integrity: per-symbol/per-agent status, OK/PARTIAL_ERROR/ERROR

Run:  .venv/bin/python tests/test_production_fixes.py
Exits non-zero on any failure. No external test dependencies.
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}")


# ---------------------------------------------------------------------------
# 1. HTTP client / proxies fix — the exact production failure was at client
#    construction, so constructing the clients IS the regression test.
# ---------------------------------------------------------------------------
print("1. httpx / proxies compatibility:")
import httpx
import openai
import groq

check(f"httpx version >= 0.28 ({httpx.__version__})", httpx.__version__.split(".")[:2] >= ["0", "28"])
try:
    from openai import OpenAI
    OpenAI(api_key="dummy", base_url="https://openrouter.ai/api/v1")
    check(f"openai {openai.__version__} client constructs (risk agent)", True)
except TypeError as e:
    check(f"openai client constructs — STILL BROKEN: {e}", False)

try:
    from groq import Groq
    Groq(api_key="dummy")
    check(f"groq {groq.__version__} client constructs (debate + CIO agents)", True)
except TypeError as e:
    check(f"groq client constructs — STILL BROKEN: {e}", False)

# The invalid kwarg must be gone from the installed SDK internals
import inspect
from openai import _base_client as _obc
_src = inspect.getsource(_obc)
check("openai no longer passes proxies= to httpx", "proxies=proxies" not in _src)

# ---------------------------------------------------------------------------
# 2. Alpaca market data feed (IEX vs SIP)
# ---------------------------------------------------------------------------
print("2. Alpaca market-data feed:")
import config
from services import alpaca_service

check("default feed is IEX", config.settings.ALPACA_DATA_FEED == "IEX")

feed = alpaca_service._resolve_feed()
check("resolve_feed returns DataFeed.IEX by default", getattr(feed, "value", None) == "iex")

config.settings.ALPACA_DATA_FEED = "SIP"
check("ALPACA_DATA_FEED=SIP honored", alpaca_service._resolve_feed().value == "sip")
config.settings.ALPACA_DATA_FEED = "IEX"

# get_indicators must pass the feed into the bars request
class _CapturingDataClient:
    def __init__(self, response=None, error=None):
        self.captured = None
        self._response = response
        self._error = error
    def get_stock_bars(self, req):
        self.captured = req
        if self._error:
            raise self._error
        return self._response

class _FakeBars:
    def __init__(self, df):
        self.df = df

import pandas as pd
n = 260
closes = [100 + i * 0.5 for i in range(n)]
df = pd.DataFrame(
    {"close": closes, "open": closes, "high": closes, "low": closes, "volume": [1000] * n, "trade_count": [10] * n, "vwap": closes},
    index=pd.DatetimeIndex(pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC"), name="timestamp"),
)
fake = _CapturingDataClient(response=_FakeBars(df))
alpaca_service._data_client = fake
res = alpaca_service.get_indicators("AAPL")
check("get_indicators succeeds on IEX bars", res.get("error") is None)
check("bars request carried the IEX feed", getattr(fake.captured, "feed", None) is not None and fake.captured.feed.value == "iex")
check("indicators computed from returned bars (RSI/SMA50/SMA200/MACD)",
      res.get("rsi_14") is not None and res.get("sma_50") is not None and res.get("sma_200") is not None and res.get("macd") is not None)
check("latest close comes from real bars", abs(res.get("latest_close") - closes[-1]) < 1e-6)
alpaca_service._data_client = None

# ---------------------------------------------------------------------------
# 3. Portfolio history via the correct SDK API
# ---------------------------------------------------------------------------
print("3. Alpaca portfolio history:")

class _FakeTradingClient:
    def __init__(self, response):
        self.response = response
        self.captured_filter = None
    def get_portfolio_history(self, history_filter=None, **kwargs):
        self.captured_filter = history_filter
        return self.response

class _FakeHistory:
    def __init__(self, timestamps, equities):
        self.timestamp = timestamps
        self.equity = equities

from alpaca.trading.requests import GetPortfolioHistoryRequest

fake_tc = _FakeTradingClient(_FakeHistory([1700000000, 1700086400, 1700172800], [25000.0, None, 25100.0]))
alpaca_service._trading_client = fake_tc
hist = alpaca_service.get_portfolio_history(period="1M")
check("uses TradingClient.get_portfolio_history", fake_tc.captured_filter is not None)
check("request is a GetPortfolioHistoryRequest", isinstance(fake_tc.captured_filter, GetPortfolioHistoryRequest))
check("period passed through", fake_tc.captured_filter.period == "1M")
check("daily timeframe for 1M", fake_tc.captured_filter.timeframe == "1D")
check("null equity points filtered, real points kept",
      hist["points"] == [{"timestamp": 1700000000, "equity": 25000.0}, {"timestamp": 1700172800, "equity": 25100.0}])
check("no error", hist["error"] is None)

hist_1d = alpaca_service.get_portfolio_history(period="1D")
check("intraday timeframe for 1D range", fake_tc.captured_filter.timeframe == "5Min")

# dict-shaped responses also handled
fake_tc2 = _FakeTradingClient({"timestamp": [1, 2], "equity": [10.0, 20.0]})
alpaca_service._trading_client = fake_tc2
hist2 = alpaca_service.get_portfolio_history(period="1W")
check("dict-shaped history response handled", hist2["points"] == [{"timestamp": 1, "equity": 10.0}, {"timestamp": 2, "equity": 20.0}])

# SDK exceptions surface as structured errors (not swallowed silently)
class _BoomClient(_FakeTradingClient):
    def get_portfolio_history(self, history_filter=None, **kwargs):
        raise AttributeError("boom")
alpaca_service._trading_client = _BoomClient(None)
hist3 = alpaca_service.get_portfolio_history(period="1M")
check("failures surface as structured error", hist3["error"] == "boom" and hist3["points"] == [])
alpaca_service._trading_client = None

# ---------------------------------------------------------------------------
# 4. Fundamentals: cache, pacing, 429 handling
# ---------------------------------------------------------------------------
print("4. yfinance fundamentals layer:")
from agents import fundamentals_agent as fa

fa.reset_fundamentals_cache()

# 4a. success path with a stubbed yfinance
class _FakeInfoTicker:
    calls = 0
    @property
    def info(self):
        _FakeInfoTicker.calls += 1
        return {"trailingPE": 30.1, "forwardPE": 28.0, "revenueGrowth": 0.12,
                "profitMargins": 0.25, "debtToEquity": 1.3, "returnOnEquity": 0.4}

class _FakeYF:
    def __init__(self, symbol):
        self.symbol = symbol
    Ticker = staticmethod(lambda s: _FakeInfoTicker())

import types as _types
fake_yf_module = _types.ModuleType("yfinance")
fake_yf_module.Ticker = lambda s: _FakeInfoTicker()
sys.modules["yfinance"] = fake_yf_module

config.settings.FUNDAMENTALS_MIN_INTERVAL_SECONDS = 0  # don't slow the tests
fa.reset_fundamentals_cache()
r1 = fa.get_fundamentals("AAPL")
check("success path returns real fields", r1["pe_ratio"] == 30.1 and r1["error"] is None)
calls_after_first = _FakeInfoTicker.calls
r2 = fa.get_fundamentals("AAPL")
check("cached: second call does not re-request", _FakeInfoTicker.calls == calls_after_first and r2["pe_ratio"] == 30.1)

# 4b. 429 -> DATA_UNAVAILABLE + global backoff
class _RateLimitedTicker:
    calls = 0
    @property
    def info(self):
        _RateLimitedTicker.calls += 1
        raise Exception("429 Client Error: Too Many Requests for url: https://query2.finance.yahoo.com/v10/finance/quoteSummary/AAPL")

fake_yf_module.Ticker = lambda s: _RateLimitedTicker()
fa.reset_fundamentals_cache()
r429 = fa.get_fundamentals("MSFT")
check("429 -> DATA_UNAVAILABLE", r429["error"] is not None and r429["error"].startswith("DATA_UNAVAILABLE"))
check("429 message identifies rate limit", "429" in r429["error"])
check("no fake values on 429", "pe_ratio" not in r429)
r429b = fa.get_fundamentals("NVDA")  # different symbol, backoff must apply process-wide
check("backoff: other symbols short-circuit without hitting Yahoo", _RateLimitedTicker.calls == 1)
check("backoff result is DATA_UNAVAILABLE", r429b["error"] is not None and r429b["error"].startswith("DATA_UNAVAILABLE"))
check("429 error is classified as rate limit", fa._is_rate_limit_error(Exception("429 Client Error: Too Many Requests")))

# 4c. the JSONDecodeError symptom ("Expecting value: line 1 column 1")
json_err = ValueError("Expecting value: line 1 column 1 (char 0)")
check("JSONDecodeError symptom classified as rate limit", fa._is_rate_limit_error(json_err))
class _JsonErrTicker:
    @property
    def info(self):
        raise json_err
fake_yf_module.Ticker = lambda s: _JsonErrTicker()
fa.reset_fundamentals_cache()
rjson = fa.get_fundamentals("TSLA")
check("JSON garbage from provider -> DATA_UNAVAILABLE (not a crash)", rjson["error"].startswith("DATA_UNAVAILABLE"))

# 4d. other provider failures
class _SslTicker:
    @property
    def info(self):
        raise RuntimeError("SSLError: connection closed")
fake_yf_module.Ticker = lambda s: _SslTicker()
fa.reset_fundamentals_cache()
rssl = fa.get_fundamentals("SPY")
check("other provider failure -> DATA_UNAVAILABLE with reason", rssl["error"] == "DATA_UNAVAILABLE: SSLError: connection closed")

# 4e. analyze_fundamentals stays graceful on unavailable data
config.settings.GEMINI_API_KEY = "dummy"
af = fa.analyze_fundamentals("SPY", rssl)
check("analyze_fundamentals degrades gracefully (no crash, NEUTRAL)",
      af["signal"] == "NEUTRAL" and af["error"] is not None)

# 4f. min-interval pacing is applied between live calls
config.settings.FUNDAMENTALS_MIN_INTERVAL_SECONDS = 0.4
fa.reset_fundamentals_cache()
t0 = time.monotonic()
fa.get_fundamentals("AAPL")
fa.get_fundamentals("MSFT")  # different symbol, cache miss -> must respect pacing
elapsed = time.monotonic() - t0
check("pacing enforced between live calls (>=0.4s)", elapsed >= 0.35)
config.settings.FUNDAMENTALS_MIN_INTERVAL_SECONDS = 2.0

del sys.modules["yfinance"]
fa.reset_fundamentals_cache()

# ---------------------------------------------------------------------------
# 5. Cycle integrity — per-symbol/per-agent status + honest cycle status
# ---------------------------------------------------------------------------
print("5. cycle integrity:")
import main
from agents import tech_agent, news_agent, risk_agent, cio_agent, debate_agent

def _stub_everything():
    main.alpaca_service.get_account_summary = lambda: {"cash": 1000.0, "equity": 10000.0, "error": None}
    main.alpaca_service.get_open_positions = lambda: {"positions": [], "error": None}
    main.alpaca_service.get_indicators = lambda s: {"symbol": s, "latest_close": 100.0, "rsi_14": 55.0, "sma_50": 99.0, "sma_200": 95.0, "macd": 1.0, "macd_signal": 0.5, "recent_closes": [100.0], "error": None}
    main.alpaca_service.execute_order = lambda *a, **k: {"success": False, "error": "not in tests"}
    main._placeholder_headlines = lambda s: ["headline"]
    main._get_agent_weights = lambda: {}

def _ok_report(agent, symbol):
    return {"agent": agent, "symbol": symbol, "signal": "BULLISH", "sentiment": "BULLISH",
            "confidence": 0.8, "summary": "ok", "error": None}

def _run_cycle():
    main.agent_logs.clear()
    main.cycle_history.clear()
    main._previous_positions.clear()
    main.bot_state.update({"running": True, "last_cycle_at": None, "last_cycle_status": "NEVER_RUN"})
    return asyncio.run(main.run_trading_cycle(triggered_by="test"))

# 5a. everything healthy -> OK
_stub_everything()
main.tech_agent.analyze_technicals = lambda s, i: _ok_report("technical", s)
main.news_agent.analyze_news = lambda s, h: _ok_report("news", s)
main.fundamentals_agent.get_fundamentals = lambda s: {"symbol": s, "pe_ratio": 10.0, "error": None}
main.fundamentals_agent.analyze_fundamentals = lambda s, f: _ok_report("fundamentals", s)
main.debate_agent.run_debate = lambda s, t, n, f=None: {"agent": "debate", "symbol": s, "bull_strength": 0.7, "bull_summary": "b", "bear_strength": 0.3, "bear_summary": "r", "edge": 0.4, "error": None}
main.risk_agent.assess_risk = lambda s, side, acct, pos: {"agent": "risk", "symbol": s, "approved": True, "max_notional_usd": 500.0, "risk_level": "LOW", "reasoning": "ok", "error": None}
main.cio_agent.make_decision = lambda *a, **k: {"agent": "cio", "symbol": "X", "decision": "HOLD", "confidence": 0.5, "notional_usd": 0.0, "reasoning": "hold", "error": None}
config.settings.ENABLE_MEMORY = False

res = _run_cycle()
rec = main.cycle_history[0]
check("healthy cycle -> status OK", rec["status"] == "OK" and main.bot_state["last_cycle_status"] == "OK")
check("all 5 symbols processed", rec["symbols_processed"] == config.settings.TRADE_UNIVERSE)
check("per-symbol agent_status recorded", set(rec["agent_status"].keys()) == set(config.settings.TRADE_UNIVERSE))
nv = rec["agent_status"]["NVDA"]
check("per-agent stages recorded", nv["market_data"] == "OK" and nv["technical"] == "OK" and nv["news"] == "OK"
      and nv["fundamentals"] == "OK" and nv["debate"] == "OK" and nv["risk"] == "OK" and nv["cio"] == "OK")
check("HOLD -> execution SKIPPED (not ERROR)", nv["execution"] == "SKIPPED")
check("memory disabled -> SKIPPED", nv["memory"] == "SKIPPED")

# 5b. one agent failing -> PARTIAL_ERROR even though the scheduler "succeeded"
main.tech_agent.analyze_technicals = lambda s, i: {"agent": "technical", "symbol": s, "signal": "NEUTRAL", "confidence": 0.0, "summary": "GROQ_API_KEY not configured.", "error": "GROQ_API_KEY not configured."}
res = _run_cycle()
rec = main.cycle_history[0]
check("agent error -> cycle PARTIAL_ERROR", rec["status"] == "PARTIAL_ERROR")
check("failing agent marked ERROR per symbol", rec["agent_status"]["NVDA"]["technical"] == "ERROR")
check("healthy agents still OK", rec["agent_status"]["NVDA"]["news"] == "OK")
err_events = [l for l in main.agent_logs if l["level"] == "ERROR" and l["agent"] == "technical"]
check("agent failure visible in feed at ERROR level", len(err_events) == len(config.settings.TRADE_UNIVERSE))
main.tech_agent.analyze_technicals = lambda s, i: _ok_report("technical", s)

# 5c. fundamentals data unavailable -> UNAVAILABLE + PARTIAL_ERROR
main.fundamentals_agent.get_fundamentals = lambda s: {"symbol": s, "error": "DATA_UNAVAILABLE: Yahoo Finance rate-limited (HTTP 429)."}
res = _run_cycle()
rec = main.cycle_history[0]
check("provider unavailability -> UNAVAILABLE (distinct from ERROR)", rec["agent_status"]["NVDA"]["fundamentals"] == "UNAVAILABLE")
check("unavailable data still degrades cycle to PARTIAL_ERROR", rec["status"] == "PARTIAL_ERROR")
main.fundamentals_agent.get_fundamentals = lambda s: {"symbol": s, "pe_ratio": 10.0, "error": None}

# 5d. market data failure -> market_data ERROR + PARTIAL_ERROR
main.alpaca_service.get_indicators = lambda s: {"symbol": s, "error": "subscription does not permit querying recent SIP data"}
res = _run_cycle()
rec = main.cycle_history[0]
check("market-data failure marked per symbol", rec["agent_status"]["NVDA"]["market_data"] == "ERROR")
check("market-data failure -> PARTIAL_ERROR", rec["status"] == "PARTIAL_ERROR")
main.alpaca_service.get_indicators = lambda s: {"symbol": s, "latest_close": 100.0, "error": None}

# 5e. broker unreachable -> cycle ERROR
main.alpaca_service.get_account_summary = lambda: {"cash": 0.0, "equity": 0.0, "error": "Alpaca API keys are not configured."}
res = _run_cycle()
rec = main.cycle_history[0]
check("broker failure -> cycle ERROR", rec["status"] == "ERROR" and main.bot_state["last_cycle_status"] == "ERROR")
main.alpaca_service.get_account_summary = lambda: {"cash": 1000.0, "equity": 10000.0, "error": None}

# 5f. execution failure on actionable decision -> execution ERROR
main.cio_agent.make_decision = lambda *a, **k: {"agent": "cio", "symbol": "X", "decision": "BUY", "confidence": 0.9, "notional_usd": 400.0, "reasoning": "buy", "error": None}
main.alpaca_service.execute_order = lambda *a, **k: {"success": False, "error": "insufficient buying power"}
res = _run_cycle()
rec = main.cycle_history[0]
check("failed order -> execution ERROR", rec["agent_status"]["NVDA"]["execution"] == "ERROR")
check("failed order -> cycle PARTIAL_ERROR", rec["status"] == "PARTIAL_ERROR")
main.alpaca_service.execute_order = lambda *a, **k: {"success": True, "order_id": "oid-1", "status": "filled"}
config.settings.ENABLE_MEMORY = False  # keep memory SKIPPED in tests
res = _run_cycle()
rec = main.cycle_history[0]
check("successful order -> execution OK", rec["agent_status"]["NVDA"]["execution"] == "OK")
check("order recorded in cycle", len(rec["orders"]) == len(config.settings.TRADE_UNIVERSE))

# 5g. exception mid-pipeline attributed to the right stage
def _boom_for_nvda(symbol, headlines):
    if symbol == "NVDA":
        raise RuntimeError("news exploded")
    return _ok_report("news", symbol)
main.news_agent.analyze_news = _boom_for_nvda
res = _run_cycle()
rec = main.cycle_history[0]
check("exception attributed to failing stage", rec["agent_status"]["NVDA"]["news"] == "ERROR")
check("later stages stay SKIPPED after abort", rec["agent_status"]["NVDA"]["cio"] == "SKIPPED")
check("other symbols unaffected", rec["agent_status"]["AAPL"]["cio"] == "OK")
check("single-symbol exception -> PARTIAL_ERROR", rec["status"] == "PARTIAL_ERROR")

# 5h. every symbol aborts -> ERROR (not a misleading OK/PARTIAL)
def _boom(symbol, headlines):
    raise RuntimeError("news exploded")
main.news_agent.analyze_news = _boom
res = _run_cycle()
rec = main.cycle_history[0]
check("all symbols abort -> cycle ERROR", rec["status"] == "ERROR")
check("scheduler success never implies cycle OK", main.bot_state["last_cycle_status"] == "ERROR")

# ---------------------------------------------------------------------------
# 6. API surface sanity
# ---------------------------------------------------------------------------
print("6. API surface:")
import importlib.util
spec = importlib.util.spec_from_file_location("main_app", "main.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
routes = {r.path for r in m.app.routes if hasattr(r, "path")}
check("all API routes present", {"/api/portfolio", "/api/logs", "/api/cycles", "/api/orders", "/api/history", "/api/config", "/api/agent-accuracy"} <= routes)
check("agent-accuracy untouched", hasattr(m, "_get_agent_weights"))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("ALL PRODUCTION-FIX TESTS PASSED")
