"""
tests/test_full_pipeline.py

Full trading-cycle integration test: runs main.run_trading_cycle() with the
REAL agents, the REAL normalized data layer, the REAL deterministic
analytics and the REAL risk/validation gates — only the external transports
are stubbed (Groq/NVIDIA/OpenRouter/Gemini LLM clients, Alpaca REST,
news). Exercises the complete architecture for AAPL, MSFT, NVDA, TSLA, SPY:

    MARKET DATA (Alpaca stub) -> NORMALIZED DATA LAYER (data_quality)
    -> DETERMINISTIC ANALYTICS (RSI/SMA/EMA/MACD/ATR/vol/drawdown/returns)
    -> AI REASONING (tech/news [Gemini] /fundamentals [SKIPPED, no provider]
       /debate/risk/cio [Groq]) -> VALIDATION + RISK GATE -> paper execution

Three cycles are run:
  A. everything healthy  -> OK, exact LLM call-count contract
  B. unchanged inputs    -> Gemini 0 calls (news analysis cached)
  C. Gemini quota death  -> news UNAVAILABLE x5, PARTIAL_ERROR with 5
     structured errors, agent_results news 0/5, provider circuit open.

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
RISK_LLM = json.dumps({"approved": True, "max_notional_usd": 500.0, "risk_level": "LOW",
                       "reasoning": "Position within limits."})
GEMINI_NEWS = json.dumps({"sentiment": "BULLISH", "confidence": 0.7,
                          "summary": "Coverage is positive.", "key_headline": "Beats earnings"})


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


def _groq_responder(kwargs):
    sys_prompt = str(kwargs.get("messages", [{}])[0].get("content", ""))
    if "Chief Investment Officer" in sys_prompt:
        return GROQ_CIO
    if "bull vs. bear debate" in sys_prompt:
        return GROQ_DEBATE
    if "risk manager" in sys_prompt:
        return RISK_LLM
    return GROQ_TECH


class _ModelsList:
    def __init__(self, ids):
        self.data = [types.SimpleNamespace(id=i) for i in ids]


def _llm_client(responder, model_ids):
    client = types.SimpleNamespace()
    client.chat = types.SimpleNamespace(completions=_Completions(responder))
    client.models = types.SimpleNamespace(list=lambda: _ModelsList(list(model_ids)))
    return client


class _FakeInteraction:
    def __init__(self, output_text):
        self.output_text = output_text
        self.errors = None


class _FakeGeminiInteractions:
    def __init__(self):
        self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeInteraction(GEMINI_NEWS)


class _FakeGeminiClient:
    def __init__(self):
        self.interactions = _FakeGeminiInteractions()
        self.models = types.SimpleNamespace(list=lambda: _ModelsList(["gemini-3.6-flash"]))


# Alpaca fakes
class _FakeAccount:
    cash, equity, buying_power, portfolio_value, last_equity = 12043.0, 25104.0, 12043.0, 25104.0, 25000.0
    status = types.SimpleNamespace(value="ACTIVE")


class _FakeTradingClient:
    def __init__(self):
        self.submitted = []
    def get_account(self):
        return _FakeAccount()
    def get_all_positions(self):
        return []
    def get_portfolio_history(self, history_filter=None):
        return types.SimpleNamespace(timestamp=[1700000000, 1700086400], equity=[25000.0, 25104.0])
    def get_clock(self):
        return types.SimpleNamespace(is_open=True, timestamp=None, next_open=None, next_close=None)
    def submit_order(self, req):
        self.submitted.append(req)
        return types.SimpleNamespace(id="test-order", symbol=req.symbol, qty=req.qty,
                                     notional=getattr(req, "notional", None),
                                     status=types.SimpleNamespace(value="accepted"))


class _FakeDataClient:
    def __init__(self):
        self.bar_requests = []
        self.snapshot_requests = []
    def get_stock_bars(self, req):
        self.bar_requests.append(req)
        import pandas as pd
        n = 260
        closes = [100 + (i % 20) * 0.5 for i in range(n)]
        highs = [c + 1.5 for c in closes]
        lows = [c - 1.5 for c in closes]
        idx = pd.DatetimeIndex(pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC"), name="timestamp")
        df = pd.DataFrame({"close": closes, "open": closes, "high": highs, "low": lows,
                           "volume": [1000] * n, "trade_count": [10] * n, "vwap": closes}, index=idx)
        return types.SimpleNamespace(df=df)
    def get_stock_snapshot(self, req):
        self.snapshot_requests.append(req)
        sym = req.symbol_or_symbols if isinstance(req.symbol_or_symbols, str) else list(req.symbol_or_symbols)[0]
        return {sym: types.SimpleNamespace(
            symbol=sym,
            latest_trade=types.SimpleNamespace(price=101.25, size=10, timestamp=None),
            latest_quote=types.SimpleNamespace(bid_price=100.5, ask_price=101.5, bid_size=2, ask_size=2, timestamp=None),
            minute_bar=None, daily_bar=None, previous_daily_bar=None)}


# ---------------------------------------------------------------------------
# Install stubs
# ---------------------------------------------------------------------------
print("full pipeline (real agents/data layer, stubbed transports):")

import config
config.settings.GROQ_API_KEY = "dummy-groq"
config.settings.GEMINI_API_KEY = "dummy-gemini"
config.settings.OPENROUTER_API_KEY = "dummy-openrouter"
config.settings.NVIDIA_API_KEY = ""
config.settings.ALPACA_API_KEY = "dummy-alpaca"
config.settings.ALPACA_SECRET_KEY = "dummy-alpaca-secret"
config.settings.ALPACA_DATA_FEED = "IEX"
config.settings.ENABLE_MEMORY = False
config.settings.TRADE_UNIVERSE = UNIVERSE
config.settings.FUNDAMENTALS_PROVIDER = "none"       # no provider by default
config.settings.LLM_FALLBACK_PROVIDER = ""
config.settings.LLM_FALLBACK_MODEL = ""

from services import gemini_service, alpaca_service, llm_service, market_data_service
from agents import fundamentals_agent, news_agent

GROQ_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")

def _install_llm_stubs():
    llm_service.reset_all_state()
    gemini_service._client = _FakeGeminiClient()
    llm_service._clients["groq"] = _llm_client(_groq_responder, GROQ_MODELS)
    llm_service._verified_models["groq"] = set(GROQ_MODELS)
    llm_service._verified_models["gemini"] = {"gemini-3.6-flash"}

_install_llm_stubs()

fake_trading = _FakeTradingClient()
fake_data = _FakeDataClient()
alpaca_service._trading_client = fake_trading
alpaca_service._data_client = fake_data

# Paper orders: use the real execute_order path via the fake trading client.
alpaca_service.execute_order = alpaca_service.execute_order  # real function, fake client

# News transport stub (Alpaca News API shape is covered by test_data_layer)
market_data_service.get_news = lambda s, limit=10: {
    "headlines": [f"{s} beats earnings expectations", f"{s} announces buyback"], "error": None}

fundamentals_agent.reset_fundamentals_cache()
news_agent.reset_news_analysis_cache()

import main
main._get_agent_weights = lambda: {}
main.agent_logs.clear()
main.cycle_history.clear()
main._previous_positions.clear()


async def _run(tag):
    await main.run_trading_cycle(triggered_by=tag)
    return main.cycle_history[0]


def _ind_has(record, symbol):
    """The deterministic evidence reached the agents (checked via the LLM
    request payloads the stub captured)."""
    groq_client = llm_service._clients.get("groq")
    tech_calls = [c for c in groq_client.chat.completions.calls
                  if c["model"] == "openai/gpt-oss-20b"
                  and "technical analysis expert" in c["messages"][0]["content"]
                  and symbol in c["messages"][1]["content"]]
    if not tech_calls:
        return False
    evidence = tech_calls[-1]["messages"][1]["content"]
    return all(k in evidence for k in (
        "RSI(14)", "SMA(20)", "EMA(20)", "ATR(14)", "Annualized volatility",
        "Max drawdown", "Rule-based signal"))


# ===========================================================================
print("cycle A — everything healthy (exact call-count contract):")
recordA = asyncio.run(_run("test-cycle-A"))

uA = recordA["llm_usage"] or {}
gA = len(gemini_service._client.interactions.calls)
check("A: all 5 symbols processed", sorted(recordA["symbols_processed"]) == sorted(UNIVERSE))
check("A: market data OK for all symbols (bars + snapshot price)",
      all(recordA["agent_status"][s]["market_data"] == "OK" for s in UNIVERSE))
check("A: snapshots used for prices (Alpaca primary)",
      len(fake_data.snapshot_requests) >= len(UNIVERSE))
check("A: deterministic analytics computed (RSI/SMA/EMA/ATR/vol/drawdown/returns/signal)",
      all(_ind_has(recordA, s) for s in UNIVERSE))
check("A: technical (Groq) OK for all", all(recordA["agent_status"][s]["technical"] == "OK" for s in UNIVERSE))
check("A: news (Gemini) OK for all", all(recordA["agent_status"][s]["news"] == "OK" for s in UNIVERSE))
check("A: fundamentals SKIPPED (no provider configured — visible, not an error)",
      all(recordA["agent_status"][s]["fundamentals"] == "SKIPPED" for s in UNIVERSE))
check("A: debate (Groq, ONE call/symbol) OK for all", all(recordA["agent_status"][s]["debate"] == "OK" for s in UNIVERSE))
check("A: risk (deterministic gate + Groq reasoning) OK for all", all(recordA["agent_status"][s]["risk"] == "OK" for s in UNIVERSE))
check("A: CIO (Groq) OK for all", all(recordA["agent_status"][s]["cio"] == "OK" for s in UNIVERSE))
check("A: cycle status OK (SKIPPED fundamentals does not degrade)",
      recordA["status"] == "OK" and recordA["errors"] == [])

# --- exact LLM call-count contract (the BEFORE/AFTER measurement)
check("A: exactly 20 Groq requests (tech+debate+risk+cio = 4/symbol)",
      uA.get("groq", {}).get("requests") == 20 and uA["groq"]["ok"] == 20)
check("A: exactly 5 Gemini requests (news only, 1/symbol)",
      uA.get("gemini", {}).get("requests") == 5 and gA == 5)
check("A: ZERO OpenRouter requests (optional, unrouted by default)",
      "openrouter" not in uA and uA.get("openrouter", {}).get("requests", 0) == 0)
check("A: ZERO NVIDIA requests (not configured)",
      uA.get("nvidia", {}).get("requests", 0) == 0)
check("A: per-agent breakdown correct",
      uA["groq"]["by_agent"]["technical"]["requests"] == 5
      and uA["groq"]["by_agent"]["debate"]["requests"] == 5
      and uA["groq"]["by_agent"]["risk"]["requests"] == 5
      and uA["groq"]["by_agent"]["cio"]["requests"] == 5
      and uA["gemini"]["by_agent"]["news"]["requests"] == 5)

# --- reports carry provider/model/status
info_logs = [l for l in main.agent_logs if l.get("data") and l["agent"] in
             ("technical", "news", "debate", "risk", "cio")]
check("A: every agent report carries provider/model/llm_status",
      all(all(k in l["data"] for k in ("provider", "model", "llm_status")) for l in info_logs))
check("A: models reported per agent",
      {l["data"]["model"] for l in info_logs if l["agent"] == "technical"} == {"openai/gpt-oss-20b"}
      and {l["data"]["model"] for l in info_logs if l["agent"] == "cio"} == {"openai/gpt-oss-120b"}
      and {l["data"]["model"] for l in info_logs if l["agent"] == "news"} == {"gemini-3.6-flash"})

# --- decisions + validated paper execution
check("A: decisions produced for all symbols", len(recordA["decisions"]) == len(UNIVERSE))
buys = [d for d in recordA["decisions"] if d["decision"] == "BUY"]
check("A: BUY decisions carry reasoning", all(d.get("reasoning") for d in recordA["decisions"]))
check("A: orders recorded for BUYs", len(recordA["orders"]) == len(buys))
check("A: notional clamped to deterministic cap (10% of 25104 = 2510.40)",
      all(o["notional_usd"] <= 2510.41 for o in recordA["orders"] if o["notional_usd"]))

# --- agent_results + provider_results
ar = recordA["agent_results"]
check("A: agent_results summary (technical 5/5, news 5/5, fundamentals 0/5 skipped)",
      ar["technical"]["ok"] == 5 and ar["news"]["ok"] == 5
      and ar["fundamentals"]["ok"] == 0 and ar["fundamentals"]["skipped"] == 5)
pr = recordA["provider_results"]
check("A: provider_results include market data (feed + symbols_ok)",
      pr["market_data"]["feed"] == "IEX" and pr["market_data"]["symbols_ok"] == 5)
check("A: provider_results include fundamentals provider none",
      pr["fundamentals"]["provider"] == "none" and pr["fundamentals"]["symbols_unavailable"] == 5)
check("A: provider_results include LLM states (groq READY)",
      pr["llm_states"]["groq"]["state"] == "READY")

# ===========================================================================
print("cycle B — unchanged inputs (LLM call reduction):")
recordB = asyncio.run(_run("test-cycle-B"))
uB = recordB["llm_usage"] or {}
gB = len(gemini_service._client.interactions.calls)
check("B: ZERO new Gemini requests (news analysis cached)",
      uB.get("gemini", {}).get("requests", 0) == 0 and gB == 5)
check("B: Groq still runs (20 requests — reasoning is per-cycle)",
      uB.get("groq", {}).get("requests") == 20)
check("B: news analyses flagged cached",
      all(l["data"].get("cached") is True for l in main.agent_logs
          if l["agent"] == "news" and l.get("data") and l["timestamp"] > recordB["started_at"]))
check("B: all stages still OK", recordB["status"] == "OK")

# ===========================================================================
print("cycle C — Gemini quota exhausted mid-cycle (circuit breaker + honest errors):")

class _QuotaErr(Exception):
    def __init__(self):
        super().__init__("Quota exceeded for metric generate_content_free_tier_requests, "
                         "limit: 20. You have exhausted your daily quota on this model.")
        self.status_code = 429

class _QuotaGeminiInteractions:
    def __init__(self):
        self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        raise _QuotaErr()

class _QuotaGeminiClient:
    def __init__(self):
        self.interactions = _QuotaGeminiInteractions()
        self.models = types.SimpleNamespace(list=lambda: _ModelsList(["gemini-3.6-flash"]))

gemini_service._client = _QuotaGeminiClient()
news_agent.reset_news_analysis_cache()
recordC = asyncio.run(_run("test-cycle-C"))
uC = recordC["llm_usage"] or {}

check("C: exactly ONE real Gemini request before the circuit opens (no per-symbol storm)",
      len(gemini_service._client.interactions.calls) == 1)
check("C: all 5 news attempts recorded as QUOTA_EXCEEDED (1 real 429 + 4 short-circuited)",
      uC.get("gemini", {}).get("errors", {}).get("PROVIDER_QUOTA_EXCEEDED") == 5
      and uC["gemini"]["requests"] == 5 and uC["gemini"]["ok"] == 0)
check("C: news UNAVAILABLE for all symbols",
      all(recordC["agent_status"][s]["news"] == "UNAVAILABLE" for s in UNIVERSE))
check("C: cycle status PARTIAL_ERROR (not OK, not silent)",
      recordC["status"] == "PARTIAL_ERROR")
news_errors = [e for e in recordC["errors"] if e["agent"] == "news"]
check("C: structured errors for every symbol (provider=gemini, type=QUOTA_EXCEEDED)",
      len(news_errors) == len(UNIVERSE)
      and all(e["provider"] == "gemini" and e["type"] == "QUOTA_EXCEEDED" for e in news_errors))
check("C: agent_results news 0/5", recordC["agent_results"]["news"]["ok"] == 0
      and recordC["agent_results"]["news"]["unavailable"] == 5)
check("C: provider_results show gemini circuit QUOTA_EXHAUSTED",
      recordC["provider_results"]["llm_states"]["gemini"]["state"] == "QUOTA_EXHAUSTED")
check("C: technical/debate/risk/cio still ran (Groq independent)",
      recordC["agent_results"]["technical"]["ok"] == 5
      and recordC["agent_results"]["debate"]["ok"] == 5
      and recordC["agent_results"]["risk"]["ok"] == 5
      and recordC["agent_results"]["cio"]["ok"] == 5)
check("C: no fabricated news decisions — news reports carry the quota error",
      all("QUOTA" in str(l["data"].get("error", "")) for l in main.agent_logs
          if l["agent"] == "news" and l.get("data") and l["level"] == "ERROR"))

# ===========================================================================
print("yfinance is not required:")
check("yfinance absent from sys.modules after full cycles", "yfinance" not in sys.modules)
check("yfinance absent from requirements.txt",
      "yfinance" not in open("requirements.txt").read())

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("FULL PIPELINE TEST PASSED — 3 cycles, call-count contract, quota circuit, honest errors")
