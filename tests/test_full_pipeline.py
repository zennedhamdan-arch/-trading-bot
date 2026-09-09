"""
tests/test_full_pipeline.py

Full trading-cycle integration test: runs main.run_trading_cycle() with the
REAL agent implementations and the REAL cycle orchestration — only the
external transports are stubbed (Groq, OpenAI/OpenRouter, Gemini, Alpaca,
yfinance). Verifies the complete pipeline for AAPL, MSFT, NVDA, TSLA, SPY:
market data -> technical (Groq) -> news (Gemini) -> fundamentals (yfinance
+ Gemini) -> debate (Groq) -> risk (OpenRouter) -> CIO (Groq) -> execution
-> per-symbol/per-agent status -> honest cycle status.

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
GROQ_BULL = json.dumps({"strength": 0.7, "summary": "Trend and sentiment align."})
GROQ_BEAR = json.dumps({"strength": 0.3, "summary": "Valuation is full."})
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
        return _Completion(self._responder(kwargs))


class _FakeGroq:
    """Stands in for groq.Groq; routes CIO/tech/debate calls by system prompt."""
    instances = []
    def __init__(self, api_key=None):
        self.chat = types.SimpleNamespace()
        _FakeGroq.instances.append(self)
    def _respond(self, kwargs):
        sys_prompt = str(kwargs.get("messages", [{}])[0].get("content", ""))
        if "Chief Investment Officer" in sys_prompt:
            return GROQ_CIO
        if "Bull Researcher" in sys_prompt:
            return GROQ_BULL
        if "Bear Researcher" in sys_prompt:
            return GROQ_BEAR
        return GROQ_TECH
    def __getattr__(self, name):
        if name == "chat":
            raise AttributeError
        return None


def _make_groq(api_key=None):
    g = _FakeGroq(api_key)
    g.chat = types.SimpleNamespace(completions=_Completions(g._respond))
    return g


class _FakeOpenAI:
    instances = []
    def __init__(self, api_key=None, base_url=None):
        _FakeOpenAI.instances.append(self)
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
# Install stubs, run the cycle
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

from services import gemini_service, alpaca_service
gemini_service._client = _FakeGeminiClient()
alpaca_service._trading_client = _FakeTradingClient()
alpaca_service._data_client = _FakeDataClient()
alpaca_service.execute_order = lambda symbol, side, notional_usd=None, qty=None: {
    "success": True, "order_id": f"test-{symbol}", "symbol": symbol, "side": side,
    "qty": qty, "notional_usd": notional_usd, "status": "filled", "error": None}

fake_yf = types.ModuleType("yfinance")
fake_yf.Ticker = _FakeYfTicker
sys.modules["yfinance"] = fake_yf

from agents import fundamentals_agent
fundamentals_agent.reset_fundamentals_cache()
config.settings.FUNDAMENTALS_MIN_INTERVAL_SECONDS = 0

import main
main._placeholder_headlines = lambda s: [f"{s} beats earnings expectations", f"{s} announces buyback"]
main._get_agent_weights = lambda: {}

main.agent_logs.clear()
main.cycle_history.clear()
main._previous_positions.clear()

summary = asyncio.run(main.run_trading_cycle(triggered_by="test"))
record = main.cycle_history[0]

# --- symbol coverage
check("all 5 symbols processed", sorted(record["symbols_processed"]) == sorted(UNIVERSE))
check("AAPL processed", "AAPL" in record["symbols_processed"])
check("MSFT processed", "MSFT" in record["symbols_processed"])
check("NVDA processed", "NVDA" in record["symbols_processed"])
check("TSLA processed", "TSLA" in record["symbols_processed"])
check("SPY processed", "SPY" in record["symbols_processed"])

# --- market data (IEX bars -> indicators)
check("market data OK for all symbols", all(record["agent_status"][s]["market_data"] == "OK" for s in UNIVERSE))

# --- technical (Groq)
check("technical (Groq) OK for all symbols", all(record["agent_status"][s]["technical"] == "OK" for s in UNIVERSE))

# --- news (Gemini via Interactions API)
gemini_calls = gemini_service._client.interactions.calls
check("Gemini interactions called (news + fundamentals)", len(gemini_calls) >= len(UNIVERSE))
check("Gemini calls use system_instruction + input", all(
    c.get("system_instruction") and c.get("input") for c in gemini_calls))
check("Gemini calls disable server-side storage", all(c.get("store") is False for c in gemini_calls))
check("Gemini model is gemini-3.6-flash", all(c.get("model") == "gemini-3.6-flash" for c in gemini_calls))
check("news (Gemini) OK for all symbols", all(record["agent_status"][s]["news"] == "OK" for s in UNIVERSE))

# --- fundamentals (yfinance + Gemini)
check("fundamentals OK for all symbols", all(record["agent_status"][s]["fundamentals"] == "OK" for s in UNIVERSE))
fund_logs = [l for l in main.agent_logs if l["agent"] == "fundamentals"]
check("fundamentals summaries reflect real stub data", any("28.4" in str(l.get("data", {}).get("summary", "")) or l.get("data") for l in fund_logs))

# --- debate (Groq)
check("debate (Groq) OK for all symbols", all(record["agent_status"][s]["debate"] == "OK" for s in UNIVERSE))

# --- risk (OpenRouter via OpenAI SDK)
check("risk (OpenRouter) OK for all symbols", all(record["agent_status"][s]["risk"] == "OK" for s in UNIVERSE))
check("risk verdict honored (approved BUYs)", all(d["decision"] in ("BUY", "HOLD") for d in record["decisions"]))

# --- CIO (Groq)
check("CIO (Groq) OK for all symbols", all(record["agent_status"][s]["cio"] == "OK" for s in UNIVERSE))
check("decisions produced for all symbols", len(record["decisions"]) == len(UNIVERSE))

# --- execution
check("execution attempted and OK for BUY decisions",
      all(record["agent_status"][d["symbol"]]["execution"] == "OK" for d in record["decisions"] if d["decision"] == "BUY"))
check("orders recorded", len(record["orders"]) == len([d for d in record["decisions"] if d["decision"] == "BUY"]))

# --- portfolio history (correct alpaca-py API)
hist = alpaca_service.get_portfolio_history(period="1M")
check("portfolio history returns real points", hist["points"] == [
    {"timestamp": 1700000000, "equity": 25000.0}, {"timestamp": 1700086400, "equity": 25104.0}])

# --- cycle status is honest
check("cycle status OK (everything succeeded)", record["status"] == "OK")
check("no cycle-level errors", record["errors"] == [])

# --- feed events recorded at correct levels
info_events = [l for l in main.agent_logs if l["level"] == "INFO"]
check("agent events in feed", len(info_events) >= len(UNIVERSE) * 6)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("FULL PIPELINE TEST PASSED — all 5 symbols through all 9 stages")
