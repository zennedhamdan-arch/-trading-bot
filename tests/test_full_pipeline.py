"""
tests/test_full_pipeline.py

Full trading-cycle integration test: runs main.run_trading_cycle() with the
REAL agent implementations and the REAL cycle orchestration — only the
external transports are stubbed (Groq, OpenAI/OpenRouter, Gemini, Alpaca,
yfinance). Verifies the complete pipeline for AAPL, MSFT, NVDA, TSLA, SPY:
market data -> technical (Groq) -> news (Gemini) -> fundamentals (yfinance
+ Gemini) -> debate (Groq, ONE call) -> risk (OpenRouter) -> CIO (Groq)
-> execution -> per-symbol/per-agent status -> LLM usage accounting ->
honest cycle status.

Two consecutive cycles are run with unchanged inputs to prove the Gemini
call reduction: cycle 1 sends 10 Gemini requests (5 news + 5 fundamentals),
cycle 2 reuses the cached analyses and sends ZERO.

Run:  .venv/bin/python tests/test_full_pipeline.py
"""

import asyncio
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}")


# ---------------------------------------------------------------------------
# Transport stubs (no network)
# ---------------------------------------------------------------------------
UNIVERSE = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]

GROQ_TECH = json.dumps({"signal": "BULLISH", "confidence": 0.72,
                        "summary": "Price above SMA50 and SMA200; MACD positive."})
GROQ_CIO = json.dumps({"decision": "BUY", "confidence": 0.8, "notional_usd": 250.0,
                       "reasoning": "Momentum intact and debate edge positive."})
GROQ_DEBATE = json.dumps({"bull_strength": 0.7, "bull_summary": "Trend and sentiment align.",
                          "bear_strength": 0.3, "bear_summary": "Valuation is full."})
OPENROUTER_RISK = json.dumps({"approved": True, "max_notional_usd": 500.0, "risk_level": "LOW",
                              "reasoning": "Position within limits."})
GEMINI_NEWS = json.dumps({"sentiment": "BULLISH", "confidence": 0.7,
                          "summary": "Coverage is positive.", "key_headline": "Beats earnings"})
GEMINI_FUND = json.dumps({"signal": "BULLISH", "confidence": 0.65,
                          "summary": "Growth strong, valuation reasonable."})


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Completion:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, responder):
        self._responder = responder
        self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self._responder(kwargs)
        if isinstance(out, Exception):
            raise out
        return _Completion(out)


class _FakeGroq:
    """Stands in for groq.Groq; routes CIO/tech/debate calls by system prompt."""
    def __init__(self, api_key=None):
        self.chat = types.SimpleNamespace()
    def _respond(self, kwargs):
        sys_prompt = str(kwargs.get("messages", [{}])[0].get("content", ""))
        if "Chief Investment Officer" in sys_prompt:
            return GROQ_CIO
        if "bull vs. bear debate" in sys_prompt:
            return GROQ_DEBATE
        return GROQ_TECH


def _make_groq(api_key=None):
    g = _FakeGroq(api_key)
    g.chat = types.SimpleNamespace(completions=_Completions(g._respond))
    return g


class _FakeOpenAI:
    def __init__(self, api_key=None, base_url=None):
        self.chat = types.SimpleNamespace(completions=_Completions(lambda kw: OPENROUTER_RISK))


class _FakeInteraction:
    def __init__(self, output_text):
        self.output_text = output_text
        self.errors = None


class _FakeGeminiInteractions:
    def __init__(self):
        self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        sys_prompt = str(kwargs.get("system_instruction", ""))
        if "financial news sentiment analyst" in sys_prompt:
            return _FakeInteraction(GEMINI_NEWS)
        return _FakeInteraction(GEMINI_FUND)


class _FakeGeminiClient:
    def __init__(self):
        self.interactions = _FakeGeminiInteractions()
        self.models = types.SimpleNamespace(list=lambda: types.SimpleNamespace(
            data=[types.SimpleNamespace(id="gemini-3.6-flash")]))


# Alpaca fakes
class _FakeAccount:
    cash, equity, buying_power, portfolio_value, last_equity = 12043.0, 25104.0, 12043.0, 25104.0, 25000.0
    status = types.SimpleNamespace(value="ACTIVE")


class _FakeTradingClient:
    def __init__(self):
        self.captured = []
    def get_account(self):
        return _FakeAccount()
    def get_all_positions(self):
        return []
    def get_portfolio_history(self, history_filter=None):
        return types.SimpleNamespace(timestamp=[1700000000, 1700086400], equity=[25000.0, 25104.0])


class _FakeDataClient:
    def __init__(self):
        self.captured = []
    def get_stock_bars(self, req):
        import pandas as pd
        n = 260
        closes = [100 + (i % 20) * 0.5 for i in range(n)]
        idx = pd.DatetimeIndex(pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC"), name="timestamp")
        df = pd.DataFrame({"close": closes, "open": closes, "high": closes, "low": closes,
                           "volume": [1000] * n, "trade_count": [10] * n, "vwap": closes}, index=idx)
        return types.SimpleNamespace(df=df)


# yfinance fake
class _FakeYfTicker:
    def __init__(self, symbol):
        self.symbol = symbol
    @property
    def info(self):
        return {"trailingPE": 28.4, "forwardPE": 26.1, "revenueGrowth": 0.14,
                "profitMargins": 0.27, "debtToEquity": 1.2, "returnOnEquity": 0.38}


# ---------------------------------------------------------------------------
# Install stubs, run TWO cycles with unchanged inputs
# ---------------------------------------------------------------------------
print("full pipeline (real agents, stubbed transports):")

import groq as groq_module
import openai as openai_module
groq_module.Groq = _make_groq
openai_module.OpenAI = _FakeOpenAI

import config
config.settings.GROQ_API_KEY = "dummy-groq"
config.settings.GEMINI_API_KEY = "dummy-gemini"
config.settings.OPENROUTER_API_KEY = "dummy-openrouter"
config.settings.ALPACA_API_KEY = "dummy-alpaca"
config.settings.ALPACA_SECRET_KEY = "dummy-alpaca-secret"
config.settings.ALPACA_DATA_FEED = "IEX"
config.settings.ENABLE_MEMORY = False
config.settings.TRADE_UNIVERSE = UNIVERSE

from services import gemini_service, alpaca_service, llm_service
gemini_service._client = _FakeGeminiClient()
alpaca_service._trading_client = _FakeTradingClient()
alpaca_service._data_client = _FakeDataClient()
alpaca_service.execute_order = lambda symbol, side, notional_usd=None, qty=None: {
    "success": True, "order_id": f"test-{symbol}", "symbol": symbol, "side": side,
    "qty": qty, "notional_usd": notional_usd, "status": "filled", "error": None}

# Provider-layer state: clean quotas/backoff/usage; verify the fake catalogs
llm_service.reset_all_state()
llm_service._verified_models["groq"] = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
llm_service._verified_models["openrouter"] = {"openai/gpt-oss-20b:free"}
llm_service._verified_models["gemini"] = {"gemini-3.6-flash"}

fake_yf = types.ModuleType("yfinance")
fake_yf.Ticker = _FakeYfTicker
sys.modules["yfinance"] = fake_yf

from agents import fundamentals_agent, news_agent
fundamentals_agent.reset_fundamentals_cache()
news_agent.reset_news_analysis_cache()
config.settings.FUNDAMENTALS_MIN_INTERVAL_SECONDS = 0

import main
main._placeholder_headlines = lambda s: [f"{s} beats earnings expectations", f"{s} announces buyback"]
main._get_agent_weights = lambda: {}

main.agent_logs.clear()
main.cycle_history.clear()
main._previous_positions.clear()


def _groq_calls():
    return sum(len(c.chat.completions.calls) for c in llm_service._clients.values()
               if hasattr(c, "chat") and hasattr(c.chat, "completions"))


async def _run(tag):
    summary = await main.run_trading_cycle(triggered_by=tag)
    return main.cycle_history[0]


record1 = asyncio.run(_run("test-cycle-1"))
check("cycle 1: all 5 symbols processed", sorted(record1["symbols_processed"]) == sorted(UNIVERSE))

# --- LLM usage accounting (the core call-count contract)
u1 = record1["llm_usage"] or {}
g1 = len(gemini_service._client.interactions.calls)
check("cycle 1: exactly 10 Gemini requests (5 news + 5 fundamentals), 2/symbol",
      u1.get("gemini", {}).get("requests") == 10 and g1 == 10)
check("cycle 1: Gemini per-agent breakdown 5 news + 5 fundamentals",
      u1["gemini"]["by_agent"]["news"]["requests"] == 5
      and u1["gemini"]["by_agent"]["fundamentals"]["requests"] == 5)
check("cycle 1: per-symbol Gemini usage 2 per symbol",
      all(u1["gemini"]["by_symbol"][s]["requests"] == 2 for s in UNIVERSE))
check("cycle 1: exactly 15 Groq requests (tech + debate + cio, 3/symbol)",
      u1.get("groq", {}).get("requests") == 15)
check("cycle 1: debate = ONE Groq call per symbol",
      u1["groq"]["by_agent"]["debate"]["requests"] == 5)
check("cycle 1: exactly 5 OpenRouter requests (risk, 1/symbol)",
      u1.get("openrouter", {}).get("requests") == 5)
check("cycle 1: all LLM requests OK",
      u1["gemini"]["ok"] == 10 and u1["groq"]["ok"] == 15 and u1["openrouter"]["ok"] == 5)

# --- per-agent provider/model/status in reports
info_logs = [l for l in main.agent_logs if l.get("data") and l["agent"] in
             ("technical", "news", "fundamentals", "debate", "risk", "cio")]
check("every agent report carries provider/model/llm_status",
      all(all(k in l["data"] for k in ("provider", "model", "llm_status")) for l in info_logs))
check("reports name the configured models",
      {l["data"]["model"] for l in info_logs if l["agent"] == "technical"} == {"openai/gpt-oss-20b"}
      and {l["data"]["model"] for l in info_logs if l["agent"] == "cio"} == {"openai/gpt-oss-120b"}
      and {l["data"]["model"] for l in info_logs if l["agent"] == "risk"} == {"openai/gpt-oss-20b:free"}
      and {l["data"]["model"] for l in info_logs if l["agent"] == "news"} == {"gemini-3.6-flash"})
check("providers correct per agent",
      {l["data"]["provider"] for l in info_logs if l["agent"] in ("technical", "debate", "cio")} == {"groq"}
      and {l["data"]["provider"] for l in info_logs if l["agent"] in ("news", "fundamentals")} == {"gemini"}
      and {l["data"]["provider"] for l in info_logs if l["agent"] == "risk"} == {"openrouter"})

# --- pipeline stages
check("market data OK for all symbols", all(record1["agent_status"][s]["market_data"] == "OK" for s in UNIVERSE))
check("technical (Groq) OK for all symbols", all(record1["agent_status"][s]["technical"] == "OK" for s in UNIVERSE))
check("news (Gemini) OK for all symbols", all(record1["agent_status"][s]["news"] == "OK" for s in UNIVERSE))
check("fundamentals OK for all symbols", all(record1["agent_status"][s]["fundamentals"] == "OK" for s in UNIVERSE))
check("debate (Groq) OK for all symbols", all(record1["agent_status"][s]["debate"] == "OK" for s in UNIVERSE))
check("risk (OpenRouter) OK for all symbols", all(record1["agent_status"][s]["risk"] == "OK" for s in UNIVERSE))
check("CIO (Groq) OK for all symbols", all(record1["agent_status"][s]["cio"] == "OK" for s in UNIVERSE))
check("decisions produced for all symbols", len(record1["decisions"]) == len(UNIVERSE))
check("risk verdict honored (approved BUYs only)",
      all(d["decision"] in ("BUY", "HOLD") for d in record1["decisions"]))
check("orders recorded", len(record1["orders"]) == len([d for d in record1["decisions"] if d["decision"] == "BUY"]))
check("cycle 1 status OK", record1["status"] == "OK" and record1["errors"] == [])

# --- portfolio history (correct alpaca-py API)
hist = alpaca_service.get_portfolio_history(period="1M")
check("portfolio history returns real points", hist["points"] == [
    {"timestamp": 1700000000, "equity": 25000.0}, {"timestamp": 1700086400, "equity": 25104.0}])

# ---------------------------------------------------------------------------
print("second cycle with unchanged inputs (Gemini call reuse):")
record2 = asyncio.run(_run("test-cycle-2"))
u2 = record2["llm_usage"] or {}
g2 = len(gemini_service._client.interactions.calls)
check("cycle 2: ZERO new Gemini requests (analyses reused)",
      u2.get("gemini", {}).get("requests", 0) == 0 and g2 == 10)
check("cycle 2: Gemini provider absent from usage (untouched)",
      "gemini" not in u2)
check("cycle 2: Groq still runs (15 requests — tech/debate/cio are per-cycle)",
      u2.get("groq", {}).get("requests") == 15)
check("cycle 2: OpenRouter still runs (5 risk requests)",
      u2.get("openrouter", {}).get("requests") == 5)
news_logs = [l for l in main.agent_logs if l["agent"] == "news" and l.get("data")]
cached_news = [l for l in news_logs if l["data"].get("cached") is True]
check("cycle 2: news analyses flagged cached", len(cached_news) >= len(UNIVERSE))
check("cycle 2: all stages still OK (cache reuse is not an error)",
      all(record2["agent_status"][s][st] == "OK" for s in UNIVERSE
          for st in ("technical", "news", "fundamentals", "debate", "risk", "cio")))
check("cycle 2 status OK", record2["status"] == "OK")

# --- feed events recorded at correct levels
info_events = [l for l in main.agent_logs if l["level"] == "INFO"]
check("agent events in feed", len(info_events) >= len(UNIVERSE) * 6)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("FULL PIPELINE TEST PASSED — 2 cycles, all 5 symbols, call-count contract verified")
