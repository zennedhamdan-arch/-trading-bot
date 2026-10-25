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
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Isolated SQLite for this run (news intelligence + risk engine share it).
os.environ["MEMORY_DB_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="prodfix-"), "test-memory.db")

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

# yfinance must be gone from the dependency tree (no Yahoo scraping)
_req = open("requirements.txt").read()
check("yfinance removed from requirements.txt", "yfinance" not in _req)
_app_sources = ""
import glob as _glob
for _p in _glob.glob("services/*.py") + _glob.glob("agents/*.py") + ["main.py", "config.py"]:
    _app_sources += open(_p).read()
check("no yfinance import anywhere in the app",
      "import yfinance" not in _app_sources and "from yfinance" not in _app_sources)

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
# ---------------------------------------------------------------------------
# 4. Fundamentals provider abstraction (yfinance is gone)
# ---------------------------------------------------------------------------
print("4. fundamentals provider layer:")
from services import fundamentals_service as fs

fs.reset_cache()

# 4a. default "none" provider -> honest DATA_UNAVAILABLE, nothing fabricated
config.settings.FUNDAMENTALS_PROVIDER = "none"
config.settings.FUNDAMENTALS_FALLBACK_PROVIDER = ""
r = fs.get_fundamentals("AAPL")
check("none provider -> DATA_UNAVAILABLE", r["status"] == "DATA_UNAVAILABLE")
check("none provider reason is NO_PROVIDER_CONFIGURED", r.get("reason") == "NO_PROVIDER_CONFIGURED")
check("normalized shape (symbol/provider/timestamp)", r["symbol"] == "AAPL" and r["provider"] == "none" and "timestamp" in r)
check("no fabricated metrics", all(r[f] is None for f in
      ("market_cap", "pe_ratio", "eps", "revenue", "profit_margin", "roe", "debt_to_equity")))

# 4b. pluggable provider: registered, returns normalized data, unavailable fields null
class _FakeProvider(fs.FundamentalsProvider):
    name = "fakevendor"
    calls = 0
    def get_fundamentals(self, symbol):
        _FakeProvider.calls += 1
        return fs._normalized(symbol, self.name, "OK",
                              pe_ratio=30.1, profit_margin=0.25, roe=0.4)

fs._PROVIDER_CLASSES["fakevendor"] = _FakeProvider
config.settings.FUNDAMENTALS_PROVIDER = "fakevendor"
fs.reset_cache()
r1 = fs.get_fundamentals("AAPL")
check("fake provider returns normalized OK", r1["status"] == "OK" and r1["provider"] == "fakevendor")
check("available fields carried through", r1["pe_ratio"] == 30.1 and r1["roe"] == 0.4)
check("unavailable fields are null, never fabricated",
      r1["market_cap"] is None and r1["eps"] is None and r1["debt_to_equity"] is None)
calls_after_first = _FakeProvider.calls
r2 = fs.get_fundamentals("AAPL")
check("result cache: second call does not re-request", _FakeProvider.calls == calls_after_first)

# 4c. fallback provider: used when the primary fails
class _FailingProvider(fs.FundamentalsProvider):
    name = "failing"
    def get_fundamentals(self, symbol):
        return fs._normalized(symbol, self.name, "DATA_UNAVAILABLE", reason="vendor down")

fs._PROVIDER_CLASSES["failing"] = _FailingProvider
config.settings.FUNDAMENTALS_PROVIDER = "failing"
config.settings.FUNDAMENTALS_FALLBACK_PROVIDER = "fakevendor"
fs.reset_cache()
rf = fs.get_fundamentals("MSFT")
check("fallback used when primary unavailable", rf["status"] == "OK" and rf["provider"] == "fakevendor")

# primary OK -> fallback never called
config.settings.FUNDAMENTALS_PROVIDER = "fakevendor"
fs.reset_cache()
rf2 = fs.get_fundamentals("NVDA")
check("primary OK -> fallback untouched", rf2["provider"] == "fakevendor")

# 4d. provider raising -> structured ERROR, never a crash
class _RaisingProvider(fs.FundamentalsProvider):
    name = "raising"
    def get_fundamentals(self, symbol):
        raise RuntimeError("SSLError: connection closed")

fs._PROVIDER_CLASSES["raising"] = _RaisingProvider
config.settings.FUNDAMENTALS_PROVIDER = "raising"
config.settings.FUNDAMENTALS_FALLBACK_PROVIDER = ""
fs.reset_cache()
rr = fs.get_fundamentals("TSLA")
check("raising provider -> DATA_UNAVAILABLE/ERROR with reason, no crash",
      rr["status"] in ("DATA_UNAVAILABLE", "ERROR") and "SSLError" in rr.get("reason", ""))

# 4e. unknown provider name -> honest DATA_UNAVAILABLE
config.settings.FUNDAMENTALS_PROVIDER = "doesnotexist"
fs.reset_cache()
ru = fs.get_fundamentals("SPY")
check("unknown provider -> DATA_UNAVAILABLE with reason",
      ru["status"] == "DATA_UNAVAILABLE" and "UNKNOWN_PROVIDER" in ru.get("reason", ""))

# 4f. analyze_fundamentals degrades gracefully on unavailable data
from agents import fundamentals_agent as fa
fa.reset_fundamentals_cache()
config.settings.FUNDAMENTALS_PROVIDER = "none"
config.settings.FUNDAMENTALS_FALLBACK_PROVIDER = ""
fs.reset_cache()  # drop 4e's cached UNKNOWN_PROVIDER result
af = fa.analyze_fundamentals("SPY", fs.get_fundamentals("SPY"))
check("analyze_fundamentals degrades gracefully (no crash, null signal, DATA_UNAVAILABLE)",
      af["signal"] is None and af["confidence"] is None and af["error"] is not None
      and af["error"].startswith("DATA_UNAVAILABLE")
      and af["evidence_status"] == "OFF")  # provider=none is off-by-config, not a failure

# 4g. provider health check
health = fs.health_check()
check("health check reports none-provider DATA_UNAVAILABLE",
      health["provider"] == "none" and health["status"] == "DATA_UNAVAILABLE")

fs.reset_cache()

# ---------------------------------------------------------------------------
# 5. Cycle integrity — per-symbol/per-agent status + honest cycle status
# ---------------------------------------------------------------------------
print("5. cycle integrity:")
import main
from agents import tech_agent, news_agent, risk_agent, cio_agent, debate_agent

def _stub_everything():
    main.alpaca_service.get_account_summary = lambda: {"cash": 1000.0, "equity": 10000.0, "buying_power": 1000.0, "error": None}
    main.alpaca_service.get_open_positions = lambda: {"positions": [], "error": None}
    main.alpaca_service.get_clock = lambda: {"is_open": True, "error": None}
    main.alpaca_service.get_recent_orders = lambda limit=20: {"orders": [], "error": None}
    main.alpaca_service.get_indicators = lambda s, lookback_days=250: {"symbol": s, "latest_close": 100.0, "rsi_14": 55.0, "sma_20": 98.0, "sma_50": 99.0, "sma_200": 95.0, "ema_20": 99.5, "macd": 1.0, "macd_signal": 0.5, "volatility_annualized": 0.25, "max_drawdown": -0.1, "technical_signal": "BULLISH", "recent_closes": [100.0], "error": None}
    main.alpaca_service.get_snapshot = lambda s: {"symbol": s, "error": "snapshots not exercised in this section"}
    main.market_data_service.get_news = lambda s, limit=10: {"headlines": ["headline"], "error": None}
    main.fundamentals_service.get_fundamentals = lambda s: {
        "status": "OK", "symbol": s, "provider": "fake", "pe_ratio": 10.0, "eps": 2.0,
        "market_cap": None, "revenue": None, "profit_margin": 0.2, "roe": 0.15,
        "debt_to_equity": None, "timestamp": 0.0}
    main.alpaca_service.execute_order = lambda *a, **k: {"success": False, "error": "not in tests"}
    main._get_agent_weights = lambda: {}

def _ok_report(agent, symbol):
    return {"agent": agent, "symbol": symbol, "signal": "BULLISH", "sentiment": "BULLISH",
            "confidence": 0.8, "summary": "ok", "error": None}

def _run_cycle():
    main.agent_logs.clear()
    main.cycle_history.clear()
    main._previous_positions.clear()
    main.risk_engine.begin_cycle()   # per-cycle duplicate-order guard
    main.risk_engine.init_db()       # persisted daily trade counter
    with main.risk_engine._conn() as conn:
        conn.execute("DELETE FROM risk_engine_trades")
    main.bot_state.update({"running": True, "last_cycle_at": None, "last_cycle_status": "NEVER_RUN"})
    return asyncio.run(main.run_trading_cycle(triggered_by="test"))

# 5a. everything healthy -> OK
_stub_everything()
_real_analyze_news = main.news_agent.analyze_news
main.tech_agent.analyze_technicals = lambda s, i: _ok_report("technical", s)
main.news_agent.analyze_news = lambda s, h=None: _ok_report("news", s)
main.fundamentals_agent.get_fundamentals = lambda s: {"symbol": s, "pe_ratio": 10.0, "error": None}
main.fundamentals_agent.analyze_fundamentals = lambda s, f: _ok_report("fundamentals", s)
main.debate_agent.run_debate = lambda s, t, n, f=None, evidence=None: {"agent": "debate", "symbol": s, "bull_strength": 0.7, "bull_summary": "b", "bear_strength": 0.3, "bear_summary": "r", "edge": 0.4, "error": None}
main.risk_agent.assess_risk = lambda s, side, acct, pos, indicators=None, evidence=None: {"agent": "risk", "symbol": s, "approved": True, "max_notional_usd": 500.0, "risk_level": "LOW", "reasoning": "ok", "error": None, "llm_status": "OK"}
main.cio_agent.make_decision = lambda *a, **k: {"agent": "cio", "symbol": "X", "decision": "HOLD", "confidence": 0.5, "notional_usd": 0.0, "reasoning": "hold", "error": None}
config.settings.ENABLE_MEMORY = False

res = _run_cycle()
rec = main.cycle_history[0]
check("healthy cycle -> status OK", rec["status"] == "OK" and main.bot_state["last_cycle_status"] == "OK")
check("all 5 symbols processed", rec["symbols_processed"] == config.settings.TRADE_UNIVERSE)
check("per-symbol agent_status recorded", set(rec["agent_status"].keys()) == set(config.settings.TRADE_UNIVERSE))
nv = rec["agent_status"]["NVDA"]
check("per-agent stages recorded", nv["market_data"] == "OK" and nv["technical"] == "OK" and nv["news"] == "OK"
      and nv["fundamentals"] == "OK" and nv["risk"] == "OK" and nv["cio"] == "OK")
check("debate disabled by default -> SKIPPED (V2 default; opt-in feature)",
      nv["debate"] == "SKIPPED")
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
check("agent_results show technical 0/5", rec["agent_results"]["technical"]["ok"] == 0
      and rec["agent_results"]["technical"]["total"] == len(config.settings.TRADE_UNIVERSE))
tech_errors = [e for e in rec["errors"] if e["agent"] == "technical"]
check("structured errors present (no more PARTIAL_ERROR with Errors=[])",
      len(tech_errors) == len(config.settings.TRADE_UNIVERSE)
      and all(set(e) >= {"provider", "type", "agent", "symbol"} for e in tech_errors))
main.tech_agent.analyze_technicals = lambda s, i: _ok_report("technical", s)

# 5c. fundamentals provider failure -> UNAVAILABLE + PARTIAL_ERROR + structured errors
main.fundamentals_service.get_fundamentals = lambda s: {
    "status": "DATA_UNAVAILABLE", "symbol": s, "provider": "fakevendor",
    "reason": "vendor down", "timestamp": 0.0}
res = _run_cycle()
rec = main.cycle_history[0]
check("provider unavailability -> UNAVAILABLE (distinct from ERROR)", rec["agent_status"]["NVDA"]["fundamentals"] == "UNAVAILABLE")
check("unavailable data still degrades cycle to PARTIAL_ERROR", rec["status"] == "PARTIAL_ERROR")
fund_errors = [e for e in rec["errors"] if e["agent"] == "fundamentals"]
check("fundamentals data failure in structured errors",
      len(fund_errors) == len(config.settings.TRADE_UNIVERSE)
      and fund_errors[0]["type"] == "DATA_UNAVAILABLE"
      and fund_errors[0]["provider"] == "fundamentals:fakevendor")
main.fundamentals_service.get_fundamentals = lambda s: {
    "status": "OK", "symbol": s, "provider": "fake", "pe_ratio": 10.0, "eps": 2.0,
    "market_cap": None, "revenue": None, "profit_margin": 0.2, "roe": 0.15,
    "debt_to_equity": None, "timestamp": 0.0}

# 5c-2. news intelligence cache empty -> news UNAVAILABLE (recorded, cycle
#       continues; the worker populates the cache on its own schedule)
main.news_agent.analyze_news = _real_analyze_news
res = _run_cycle()
rec = main.cycle_history[0]
check("no cached news intelligence -> news UNAVAILABLE",
      rec["agent_status"]["NVDA"]["news"] == "UNAVAILABLE")
check("news unavailability in structured errors",
      any(e["agent"] == "news" and e["type"] == "DATA_UNAVAILABLE" for e in rec["errors"]))
check("trading continued safely (risk + cio still ran)",
      rec["agent_status"]["NVDA"]["risk"] == "OK" and rec["agent_status"]["NVDA"]["cio"] == "OK")
main.news_agent.analyze_news = lambda s, h=None: _ok_report("news", s)

# 5c-3. fundamentals with NO provider configured -> SKIPPED (visible, not an error)
main.fundamentals_service.get_fundamentals = lambda s: {
    "status": "DATA_UNAVAILABLE", "symbol": s, "provider": "none",
    "reason": "NO_PROVIDER_CONFIGURED", "timestamp": 0.0}
res = _run_cycle()
rec = main.cycle_history[0]
check("no fundamentals provider -> stage SKIPPED", rec["agent_status"]["NVDA"]["fundamentals"] == "SKIPPED")
check("no-provider fundamentals does NOT degrade the cycle", rec["status"] == "OK")
check("provider_results report fundamentals provider none + unavailable count",
      rec["provider_results"]["fundamentals"]["provider"] == "none"
      and rec["provider_results"]["fundamentals"]["symbols_unavailable"] == len(config.settings.TRADE_UNIVERSE))
main.fundamentals_service.get_fundamentals = lambda s: {
    "status": "OK", "symbol": s, "provider": "fake", "pe_ratio": 10.0, "eps": 2.0,
    "market_cap": None, "revenue": None, "profit_margin": 0.2, "roe": 0.15,
    "debt_to_equity": None, "timestamp": 0.0}

# 5d. market data failure -> market_data ERROR + PARTIAL_ERROR; subscription
#     feed failures specifically -> UNAVAILABLE with SUBSCRIPTION_FEED_UNAVAILABLE
main.alpaca_service.get_indicators = lambda s, lookback_days=250: {"symbol": s, "error": "boom: data API down"}
res = _run_cycle()
rec = main.cycle_history[0]
check("market-data failure marked per symbol", rec["agent_status"]["NVDA"]["market_data"] == "ERROR")
check("market-data failure -> PARTIAL_ERROR", rec["status"] == "PARTIAL_ERROR")
check("market-data failure in structured errors",
      any(e["agent"] == "market_data" and e["provider"] == "alpaca" for e in rec["errors"]))

main.alpaca_service.get_indicators = lambda s, lookback_days=250: {"symbol": s, "error": "DATA_UNAVAILABLE (SUBSCRIPTION_FEED_UNAVAILABLE): the configured ALPACA_DATA_FEED=SIP is not permitted by this Alpaca subscription for bars for " + s}
res = _run_cycle()
rec = main.cycle_history[0]
check("subscription feed failure -> UNAVAILABLE (not a crash, not silent)",
      rec["agent_status"]["NVDA"]["market_data"] == "UNAVAILABLE")
check("subscription failure typed SUBSCRIPTION_FEED_UNAVAILABLE",
      any(e["type"] == "SUBSCRIPTION_FEED_UNAVAILABLE" and e["provider"] == "alpaca"
          for e in rec["errors"]))
# V2: restore a stub WITH a tradeable deterministic setup (the V2 setup
# gate holds everything when technical_signal is absent/NEUTRAL).
main.alpaca_service.get_indicators = lambda s, lookback_days=250: {"symbol": s, "latest_close": 100.0, "rsi_14": 55.0, "sma_20": 98.0, "sma_50": 99.0, "sma_200": 95.0, "ema_20": 99.5, "macd": 1.0, "macd_signal": 0.5, "volatility_annualized": 0.25, "max_drawdown": -0.1, "technical_signal": "BULLISH", "recent_closes": [100.0], "error": None}

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
def _boom_for_nvda(symbol, headlines=None):
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
def _boom(symbol, headlines=None):
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
check("bot control endpoints present", {"/api/bot/start", "/api/bot/stop", "/api/bot/run-now"} <= routes)
check("health + realtime endpoints present", {"/api/health", "/api/realtime"} <= routes)
check("V2 endpoints present (providers/news)",
      {"/api/providers/health", "/api/news/status", "/api/news/{symbol}",
       "/api/news/{symbol}/articles"} <= routes)
check("agent-accuracy untouched", hasattr(m, "_get_agent_weights"))
_main_src = open("main.py").read()
check("cycle record carries agent_results/provider_results/llm_usage",
      all(f'"{k}"' in _main_src for k in ("agent_results", "provider_results", "llm_usage")))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("ALL PRODUCTION-FIX TESTS PASSED")
