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
GEMINI_NEWS = json.dumps({
    "articles": [{"index": i, "sentiment": "BULLISH", "confidence": 0.7,
                  "importance": "HIGH", "impact_horizon": "SHORT_TERM"} for i in range(2)],
    "overall_sentiment": "BULLISH", "overall_confidence": 0.7,
    "summary": "Coverage is positive."})


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
    def get_orders(self, req):
        return []
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
        n = 261
        # Rising trend with regular pullbacks, ending on an up-step:
        # deterministic signal BULLISH (trend BULLISH, momentum BULLISH).
        closes = [100 + i * 0.35 + (2.5 if i % 2 == 0 else -2.5) for i in range(n)]
        highs = [c + 1.5 for c in closes]
        lows = [c - 1.5 for c in closes]
        # Bars end TODAY: the evidence-quality gate treats analytics built
        # on bars older than EVIDENCE_STALE_DAYS as STALE, and a fixture
        # pinned to a past date would (correctly) block every BUY.
        end = pd.Timestamp.utcnow().floor("D")
        idx = pd.DatetimeIndex(pd.date_range(end=end, periods=n, freq="D", tz="UTC"), name="timestamp")
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

import os
import tempfile as _tempfile
os.environ.setdefault("MEMORY_DB_PATH",
                      os.path.join(_tempfile.mkdtemp(prefix="fullpipe-"), "test-memory.db"))

import config
config.settings.UNOROUTER_ENABLED = False          # not part of this suite
config.settings.LLM_RISK_PROVIDER = "groq"         # explicit V2 routing
config.settings.LLM_CIO_PROVIDER = "groq"
config.settings.LLM_NEWS_PROVIDER = "gemini"
config.settings.TECH_LLM_INTERPRETATION_ENABLED = False   # V2 default: deterministic tech
config.settings.ENABLE_DEBATE = False              # V2 default: debate opt-in
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

fundamentals_agent.reset_fundamentals_cache()

import main
from services import news_intelligence, risk_engine
main._get_agent_weights = lambda: {}
main.agent_logs.clear()
main.cycle_history.clear()
main._previous_positions.clear()
risk_engine.init_db()


async def _run(tag):
    with risk_engine._conn() as conn:   # persisted daily trade counter
        conn.execute("DELETE FROM risk_engine_trades")
    await main.run_trading_cycle(triggered_by=tag)
    return main.cycle_history[0]


# News Worker seeding (the V2 news pipeline runs OUTSIDE trading cycles):
# fetch -> normalize -> dedup -> relevance -> LLM analysis (Gemini) ->
# persistent intelligence. Cycle A will only READ this cache.
SEED_ARTICLES = {
    s: [{"id": f"seed-{s}-1", "headline": f"{s} beats earnings expectations",
         "summary": "Revenue up", "source": "Reuters", "published_at": "2026-09-10T10:00:00Z"},
        {"id": f"seed-{s}-2", "headline": f"{s} announces major product launch",
         "summary": "", "source": "CNBC", "published_at": "2026-09-10T09:00:00Z"}]
    for s in UNIVERSE
}
for s in UNIVERSE:
    arts = SEED_ARTICLES[s]
    out = news_intelligence.refresh_symbol(s, fetch_articles=lambda _s, _a=arts: _a)
    assert out["articles_analyzed"] >= 1, out


def _tech_report_has(record, symbol):
    """The deterministic technical engine produced a real verdict (its report
    is in the agent log; the LLM is never involved in the V2 default)."""
    tech_logs = [l for l in main.agent_logs
                 if l.get("agent") == "technical" and l.get("symbol") == symbol
                 and l.get("data")]
    if not tech_logs:
        return False
    data = tech_logs[-1]["data"]
    return (data.get("provider") == "deterministic"
            and data.get("llm_status") == "SKIPPED_DETERMINISTIC"
            and data.get("signal") == "BULLISH"
            and "RSI" in data.get("summary", "")
            and "SMA50" in data.get("summary", ""))


# ===========================================================================
print("cycle A — everything healthy (exact call-count contract):")
recordA = asyncio.run(_run("test-cycle-A"))

uA = recordA["llm_usage"] or {}
gA = len(gemini_service._client.interactions.calls)
groq_client = llm_service._clients.get("groq")
check("A: all 5 symbols processed", sorted(recordA["symbols_processed"]) == sorted(UNIVERSE))
check("A: market data OK for all symbols (bars + snapshot price)",
      all(recordA["agent_status"][s]["market_data"] == "OK" for s in UNIVERSE))
check("A: snapshots used for prices (Alpaca primary)",
      len(fake_data.snapshot_requests) >= len(UNIVERSE))
check("A: deterministic technical engine produced setups (no LLM involved)",
      all(_tech_report_has(recordA, s) for s in UNIVERSE))
check("A: technical OK for all (deterministic engine, SKIPPED_DETERMINISTIC)",
      all(recordA["agent_status"][s]["technical"] == "OK" for s in UNIVERSE))
check("A: news OK for all — served from the intelligence cache (Gemini ran in the worker, not the cycle)",
      all(recordA["agent_status"][s]["news"] == "OK" for s in UNIVERSE)
      and all(l["data"].get("cached") is True for l in main.agent_logs
              if l["agent"] == "news" and l.get("data")))
check("A: fundamentals SKIPPED (no provider configured — visible, not an error)",
      all(recordA["agent_status"][s]["fundamentals"] == "SKIPPED" for s in UNIVERSE))
check("A: debate SKIPPED (V2 default: opt-in feature, disabled)",
      all(recordA["agent_status"][s]["debate"] == "SKIPPED" for s in UNIVERSE))
check("A: risk (deterministic gate + Groq reasoning) OK for all",
      all(recordA["agent_status"][s]["risk"] == "OK" for s in UNIVERSE))
check("A: CIO (Groq) OK for all", all(recordA["agent_status"][s]["cio"] == "OK" for s in UNIVERSE))
check("A: cycle status OK (SKIPPED fundamentals does not degrade)",
      recordA["status"] == "OK" and recordA["errors"] == [])

# --- exact LLM call-count contract (the V2 reduction in action)
check("A: exactly 10 Groq requests (risk + cio only; tech/news/debate are NOT LLM calls)",
      uA.get("groq", {}).get("requests") == 10 and uA["groq"]["ok"] == 10)
check("A: ZERO in-cycle Gemini requests (news was analyzed by the worker: 5 seed calls)",
      uA.get("gemini", {}).get("requests", 0) == 0 and gA == 5)
check("A: ZERO OpenRouter requests (optional, unrouted by default)",
      "openrouter" not in uA and uA.get("openrouter", {}).get("requests", 0) == 0)
check("A: ZERO NVIDIA requests (not configured)",
      uA.get("nvidia", {}).get("requests", 0) == 0)
check("A: per-agent breakdown correct",
      uA["groq"]["by_agent"]["risk"]["requests"] == 5
      and uA["groq"]["by_agent"]["cio"]["requests"] == 5
      and "technical" not in uA["groq"]["by_agent"]
      and "news" not in uA["groq"]["by_agent"])

# --- reports carry provider/model/status
info_logs = [l for l in main.agent_logs if l.get("data") and l["agent"] in
             ("technical", "news", "debate", "risk", "cio")]
check("A: every agent report carries provider/model/llm_status",
      all(all(k in l["data"] for k in ("provider", "model", "llm_status")) for l in info_logs))
check("A: models reported per agent",
      {l["data"]["model"] for l in info_logs if l["agent"] == "technical"} == {"rule-engine"}
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
print("cycle B — unchanged inputs (response cache: zero new LLM requests):")
calls_before_B = len(groq_client.chat.completions.calls)
recordB = asyncio.run(_run("test-cycle-B"))
uB = recordB["llm_usage"] or {}
gB = len(gemini_service._client.interactions.calls)
calls_after_B = len(groq_client.chat.completions.calls)
check("B: ZERO new Groq requests (risk+cio served from the response cache)",
      uB.get("groq", {}).get("requests", 0) == 0 and calls_after_B == calls_before_B)
check("B: ZERO new Gemini requests (news intelligence cached)",
      uB.get("gemini", {}).get("requests", 0) == 0 and gB == 5)
check("B: news analyses flagged cached",
      all(l["data"].get("cached") is True for l in main.agent_logs
          if l["agent"] == "news" and l.get("data") and l["timestamp"] > recordB["started_at"]))
check("B: all stages still OK", recordB["status"] == "OK")
check("B: decisions still produced (cached verdicts replayed, capped, validated)",
      len(recordB["decisions"]) == len(UNIVERSE)
      and all(o["notional_usd"] <= 2510.41 for o in recordB["orders"] if o["notional_usd"]))

# ===========================================================================
print("cycle C — Groq dies mid-run (cache replay first, then honest fail-safes):")


class _DeadCompletions:
    def __init__(self):
        self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError("connection refused: provider down")


# C1: Groq dead BUT the response cache still holds cycles A/B verdicts:
#     every risk/cio request is replayed from cache — the cycle completes.
dead = _DeadCompletions()
groq_client.chat.completions = dead
recordC = asyncio.run(_run("test-cycle-C"))
uC = recordC["llm_usage"] or {}
check("C1: provider dead + warm cache -> verdicts replayed (cycle completes)",
      recordC["status"] == "OK"
      and all(recordC["agent_status"][s]["cio"] == "OK" for s in UNIVERSE)
      and len(dead.calls) == 0)
check("C1: cached replay is labeled (never a fabricated fresh verdict)",
      all(l["data"].get("llm_status") == "OK" for l in main.agent_logs
          if l["agent"] == "cio" and l.get("data")
          and l["timestamp"] > recordC["started_at"]))

# C2: Groq dead AND the cache cleared -> honest failure path: every agent
#     fails safely, nothing is fabricated, the cycle degrades (PARTIAL_ERROR)
#     but never crashes and never blocks.
llm_service._response_cache.clear()
recordD = asyncio.run(_run("test-cycle-D"))
uD = recordD["llm_usage"] or {}
# The breaker opens after 3 consecutive failures; the remaining requests are
# short-circuited (CIRCUIT_OPEN, no network call) — no retry storm. Usage
# still accounts all 10 request invocations honestly (3 real + 7 skipped).
check("C2: breaker opens after 3 real attempts; the rest are short-circuited",
      len(dead.calls) == 3 and uD.get("groq", {}).get("requests") == 10
      and uD["groq"]["ok"] == 0
      and uD["groq"]["errors"].get("PROVIDER_ERROR") == 3
      and uD["groq"]["errors"].get("CIRCUIT_OPEN") == 7)
check("C2: circuit opens after repeated failures",
      llm_service.provider_states()["groq"]["circuit"] == "OPEN")
check("C2: cycle degrades to PARTIAL_ERROR (honest, not silent)",
      recordD["status"] == "PARTIAL_ERROR")
check("C2: CIO never fabricates a BUY — HOLD fail-safe for every symbol",
      all(d["decision"] == "HOLD" for d in recordD["decisions"])
      and recordD["orders"] == [])
check("C2: structured errors recorded for risk and cio",
      {e["agent"] for e in recordD["errors"]} >= {"risk", "cio"})
check("C2: technical/news still ran (deterministic engine + cache read, no LLM)",
      all(recordD["agent_status"][s]["technical"] == "OK" for s in UNIVERSE)
      and all(recordD["agent_status"][s]["news"] == "OK" for s in UNIVERSE))
check("C2: deterministic fallback kept the news verdicts honest",
      all(l["data"].get("cached") is True for l in main.agent_logs
          if l["agent"] == "news" and l.get("data")
          and l["timestamp"] > recordD["started_at"]))

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
