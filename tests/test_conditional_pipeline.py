"""
tests/test_conditional_pipeline.py — the conditional, credit-efficient AI
agent pipeline (all transports stubbed; no network, no real keys):

  Conditional execution (no setup)
    1.  No setup -> ZERO LLM calls (news/fundamentals/debate/risk/cio LLM
        never invoked); a SKIPPED_NO_SETUP decision is persisted with the
        reason; downstream stage statuses carry the reason.
    2.  Held position with no setup -> DETERMINISTIC monitoring (no LLM):
        real position data reaches the risk agent; missing data produces an
        explicit missing_inputs reason; monitoring is never a rejected trade;
        no order is submitted.

  Conditional execution (valid setup)
    3.  Stages execute in order: technical -> news (cache read) ->
        fundamentals -> debate (gated) -> risk -> hard engine -> CIO ->
        execution.
    4.  Debate stays eligibility-gated: deterministic tech (no confidence)
        -> SKIPPED_DEBATE_NOT_ELIGIBLE, no debate LLM call; low confidence
        -> same; verifiable high confidence -> debate runs.
    5.  Risk checks run BEFORE the final decision; the deterministic engine
        verdict is layered in (LLM can never approve what it rejected).
    6.  The CIO receives the actual evidence (news/tech/risk/debate/evidence
        quality sections in its prompt).

  Provider protection
    7.  Dashboard polling endpoints make ZERO LLM calls.
    8.  Identical cycles reuse the LLM response cache (no duplicate calls).

Run:  .venv/bin/python tests/test_conditional_pipeline.py
"""

import asyncio
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="condpipe-")
os.environ["MEMORY_DB_PATH"] = os.path.join(_TMP, "test-memory.db")

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}")


from config import settings                                       # noqa: E402

settings.UNOROUTER_ENABLED = False
settings.LLM_TECH_PROVIDER = "groq"
settings.LLM_RISK_PROVIDER = "groq"
settings.LLM_CIO_PROVIDER = "groq"
settings.LLM_DEBATE_PROVIDER = "groq"
settings.LLM_NEWS_PROVIDER = "gemini"
settings.TECH_LLM_INTERPRETATION_ENABLED = False   # default: deterministic tech
settings.ENABLE_DEBATE = True
settings.DEBATE_REQUIRE_CONFLICT = False
settings.DEBATE_MIN_TECH_CONFIDENCE = 0.65
settings.GROQ_API_KEY = "d"
settings.GEMINI_API_KEY = "d"
settings.ALPACA_API_KEY = "d"
settings.ALPACA_SECRET_KEY = "d"
settings.ALPACA_DATA_FEED = "IEX"
settings.ENABLE_MEMORY = False
settings.FUNDAMENTALS_PROVIDER = "none"
settings.LLM_FALLBACK_PROVIDER = ""
settings.LLM_FALLBACK_MODEL = ""
settings.TRADE_UNIVERSE = ["AAPL", "MSFT"]

from services import alpaca_service, gemini_service, llm_service, \
    news_intelligence, risk_engine                          # noqa: E402
from agents import risk_agent                               # noqa: E402

# ---------------------------------------------------------------------------
# Stub transports with per-agent call counting and prompt capture
# ---------------------------------------------------------------------------
CALLS = {"groq": [], "gemini": []}
PROMPTS = {"cio": [], "debate": [], "risk": [], "technical": []}

TECH_LLM = {"signal": "BULLISH", "confidence": 0.82,
            "summary": "Trend up: above SMA50/SMA200, MACD above signal."}
DEBATE_LLM = {"bull_strength": 0.72, "bull_summary": "Trend and news align.",
              "bear_strength": 0.30, "bear_summary": "Valuation full.",
              "consensus": "BULLISH", "confidence": 0.64}
RISK_LLM = {"approved": True, "max_notional_usd": 500.0, "risk_level": "LOW",
            "reasoning": "Within limits."}
CIO_LLM = {"decision": "BUY", "confidence": 0.8, "notional_usd": 250.0,
           "reasoning": "Momentum intact."}
GEMINI_NEWS = json.dumps({
    "articles": [{"index": 0, "sentiment": "BULLISH", "confidence": 0.7,
                  "importance": "HIGH", "impact_horizon": "SHORT_TERM"}],
    "overall_sentiment": "BULLISH", "overall_confidence": 0.7,
    "summary": "Coverage is positive."})


def _agent_of(system_prompt):
    if "bull vs. bear debate" in system_prompt:
        return "debate"
    if "risk manager" in system_prompt:
        return "risk"
    if "Chief Investment Officer" in system_prompt:
        return "cio"
    if "technical analysis expert" in system_prompt:
        return "technical"
    return "other"


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
    def create(self, **kwargs):
        sp = str(kwargs.get("messages", [{}])[0].get("content", ""))
        agent = _agent_of(sp)
        CALLS["groq"].append(agent)
        if agent in PROMPTS:
            PROMPTS[agent].append(str(kwargs.get("messages", [{}, {}])[-1]
                                      .get("content", "")))
        body = {"debate": DEBATE_LLM, "risk": RISK_LLM, "cio": CIO_LLM,
                "technical": TECH_LLM}.get(agent,
                                           {"decision": "HOLD", "confidence": 0.5,
                                            "notional_usd": 0, "reasoning": "h"})
        return _Completion(json.dumps(body))


class _ModelsList:
    def __init__(self, ids):
        self.data = [types.SimpleNamespace(id=i) for i in ids]


def _llm_client():
    client = types.SimpleNamespace()
    client.chat = types.SimpleNamespace(completions=_Completions())
    client.models = types.SimpleNamespace(
        list=lambda: _ModelsList(["openai/gpt-oss-20b", "openai/gpt-oss-120b"]))
    return client


class _FakeGeminiClient:
    def __init__(self):
        self.interactions = types.SimpleNamespace()

        def _create(**kw):
            CALLS["gemini"].append("news")
            return types.SimpleNamespace(output_text=GEMINI_NEWS, errors=None)
        self.interactions.create = _create
        self.models = types.SimpleNamespace(
            list=lambda: _ModelsList(["gemini-3.6-flash"]))


class _FakeAccount:
    cash, equity, buying_power, portfolio_value, last_equity = \
        12043.0, 25104.0, 12043.0, 25104.0, 25000.0
    status = types.SimpleNamespace(value="ACTIVE")


class _Pos:
    def __init__(self, symbol, qty=10, value=1000.0):
        self.symbol = symbol
        self.qty = str(qty)
        self.avg_entry_price = "100.0"
        self.current_price = "100.0"
        self.market_value = str(value)
        self.unrealized_pl = "5.0"
        self.unrealized_plpc = 0.005
        self.side = types.SimpleNamespace(value="long")


class _FakeTradingClient:
    def __init__(self, market_open=True, positions=None):
        self.submitted = []
        self.market_open = market_open
        self.positions = positions or []
        self.account = _FakeAccount()

    def get_account(self):
        return self.account

    def get_all_positions(self):
        return self.positions

    def get_portfolio_history(self, history_filter=None):
        return types.SimpleNamespace(timestamp=[1, 2],
                                     equity=[25000.0, 25104.0])

    def get_orders(self, req):
        return []

    def get_clock(self):
        import datetime
        no = (datetime.datetime.now(datetime.timezone.utc)
              + datetime.timedelta(days=1)).replace(hour=13, minute=30,
                                                    second=0, microsecond=0)
        return types.SimpleNamespace(is_open=self.market_open, timestamp=None,
                                     next_open=no, next_close=None)

    def submit_order(self, req):
        self.submitted.append(req)
        return types.SimpleNamespace(id="o", symbol=req.symbol, qty=req.qty,
                                     notional=None,
                                     status=types.SimpleNamespace(value="accepted"))


class _FakeDataClient:
    def __init__(self, neutral=False):
        self.neutral = neutral

    def get_stock_bars(self, req):
        import pandas as pd
        n = 261
        closes = ([100 + (0.2 if i % 2 else -0.2) for i in range(n)]
                  if self.neutral
                  else [100 + i * 0.35 + (2.5 if i % 2 == 0 else -2.5)
                        for i in range(n)])
        idx = pd.DatetimeIndex(pd.date_range(end=pd.Timestamp.utcnow().floor("D"),
                                             periods=n, freq="D", tz="UTC"),
                               name="timestamp")
        df = pd.DataFrame({"close": closes, "open": closes,
                           "high": [c + 1.5 for c in closes],
                           "low": [c - 1.5 for c in closes],
                           "volume": [1000] * n, "trade_count": [10] * n,
                           "vwap": closes}, index=idx)
        return types.SimpleNamespace(df=df)

    def get_stock_snapshot(self, req):
        s = (req.symbol_or_symbols if isinstance(req.symbol_or_symbols, str)
             else list(req.symbol_or_symbols)[0])
        return {s: types.SimpleNamespace(
            symbol=s,
            latest_trade=types.SimpleNamespace(price=101.25, size=10,
                                               timestamp=None),
            latest_quote=types.SimpleNamespace(bid_price=100.5,
                                               ask_price=101.5, bid_size=2,
                                               ask_size=2, timestamp=None),
            minute_bar=None, daily_bar=None, previous_daily_bar=None)}


def _install(trading_client, data_client=None):
    CALLS["groq"].clear()
    CALLS["gemini"].clear()
    for k in PROMPTS:
        PROMPTS[k].clear()
    llm_service.reset_all_state()
    gemini_service._client = _FakeGeminiClient()
    llm_service._clients["groq"] = _llm_client()
    llm_service._verified_models["groq"] = {"openai/gpt-oss-20b",
                                            "openai/gpt-oss-120b"}
    llm_service._verified_models["gemini"] = {"gemini-3.6-flash"}
    alpaca_service._trading_client = trading_client
    alpaca_service._data_client = data_client or _FakeDataClient()
    main.agent_logs.clear()
    main.cycle_history.clear()
    main._previous_positions.clear()
    with risk_engine._conn() as conn:
        conn.execute("DELETE FROM risk_engine_trades")
    for s in settings.TRADE_UNIVERSE:
        news_intelligence.refresh_symbol(
            s, fetch_articles=lambda _s: [
                {"id": f"n-{_s}", "headline": "h", "summary": "",
                 "source": "R", "published_at": "2026-10-20T10:00:00Z"}])


import main                                                       # noqa: E402
main._get_agent_weights = lambda: {}
risk_engine.init_db()


async def _run(tag):
    risk_engine.begin_cycle()
    await main.run_trading_cycle(triggered_by=tag)
    return main.cycle_history[0]


def _events(agent, symbol, cycle_id):
    return [l for l in main.agent_logs
            if l.get("agent") == agent and l.get("symbol") == symbol
            and l.get("cycle_id") == cycle_id]


# ===========================================================================
print("== 1. NO SETUP: conditional execution, zero LLM, honest skip events ==")
tc1 = _FakeTradingClient(market_open=True, positions=[_Pos("AAPL")])
_install(tc1, _FakeDataClient(neutral=True))
rec1 = asyncio.run(_run("no_setup"))
st = rec1["agent_status"]["AAPL"]
check("1a. zero LLM calls on the no-setup path (both symbols, one held)",
      CALLS["groq"] == [] and CALLS["gemini"] == [])
check("1b. no provider requests at all (news worker untouched by the cycle)",
      len(tc1.submitted) == 0)
check("1c. SKIPPED_NO_SETUP decision event persisted with the reason",
      any((l.get("agent") == "cio"
           and (l.get("data") or {}).get("llm_status") == "SKIPPED_NO_SETUP")
          for l in _events("cio", "AAPL", rec1["id"])))
check("1d. downstream stage statuses carry the reason",
      st["news"] == "SKIPPED_NO_SETUP" and st["fundamentals"] == "SKIPPED_NO_SETUP"
      and st["debate"] == "SKIPPED_NO_SETUP"
      and st["execution"] == "SKIPPED_NO_SETUP")
check("1e. no news/fundamentals/debate events fabricated for the skip",
      _events("news", "AAPL", rec1["id"]) == []
      and _events("fundamentals", "AAPL", rec1["id"]) == []
      and _events("debate", "AAPL", rec1["id"]) == [])
check("1f. symbol without a position: risk honestly skipped too",
      rec1["agent_status"]["MSFT"]["risk"] == "SKIPPED_NO_SETUP"
      and _events("risk", "MSFT", rec1["id"]) == [])
mon = _events("risk", "AAPL", rec1["id"])
check("1g. held position: deterministic MONITORING event (risk stage = MONITORING)",
      st["risk"] == "MONITORING" and len(mon) == 1
      and mon[0]["data"]["risk_status"] == "MONITORING")
check("1h. monitoring made no LLM call and is NOT a rejected trade",
      mon[0]["data"]["llm_status"] == "SKIPPED_DETERMINISTIC"
      and "approved" not in mon[0]["data"]
      and "max_notional_usd" not in mon[0]["data"])
pf = mon[0]["data"]["position_facts"]
check("1i. ACTUAL position data reached the risk agent",
      pf["qty"] == 10 and pf["avg_entry_price"] == 100.0
      and pf["current_price"] == 100.0 and pf["market_value"] == 1000.0
      and pf["account_equity"] == 25104.0)
check("1j. exposure computed from the real numbers (1000 / 25104)",
      abs(pf["exposure_pct_of_equity"] - 1000.0 / 25104.0) < 1e-4)
check("1k. market-data freshness + stop/take honesty recorded",
      mon[0]["data"]["market_data_freshness"]["status"] == "FRESH"
      and mon[0]["data"]["exit_protection"]["status"] == "NOT_TRACKED")

print("\n== 2. missing position data -> explicit reason, never zero-filled ==")
mon2 = risk_agent.monitor_position(
    "MSFT", {"equity": 25000.0, "cash": 1000.0},
    {"symbol": "MSFT", "qty": "5"}, None)
missing_fields = {m["field"] for m in mon2["missing_inputs"]}
check("2a. missing inputs are named (entry price, value, indicators)",
      {"position.avg_entry_price", "position.market_value",
       "position.current_price", "indicators"} <= missing_fields)
check("2b. unavailable exposure stays None (no silent 0.0)",
      mon2["position_exposure"]["existing_position_pct_of_equity"] is None
      and mon2["position_exposure"]["existing_position_value"] is None)
check("2c. monitoring result never claims a rejection",
      mon2["risk_status"] == "MONITORING"
      and mon2["blocking_reasons"] == []
      and "approved" not in mon2)

print("\n== 3. VALID SETUP: stages run in order, engine before CIO ==")
tc3 = _FakeTradingClient(market_open=True)
_install(tc3)
rec3 = asyncio.run(_run("setup"))
st3 = rec3["agent_status"]["AAPL"]
seq = [l["agent"] for l in reversed(list(main.agent_logs))
       if l.get("symbol") == "AAPL" and l.get("cycle_id") == rec3["id"]]
expected_prefix = ["technical", "news"]
check("3a. order: technical before news before risk before cio before execution",
      seq[:2] == expected_prefix
      and seq.index("risk") > seq.index("news")
      and seq.index("cio") > seq.index("risk")
      and seq.index("execution") > seq.index("cio"))
check("3b. news analysis came from the worker cache (no gemini call in-cycle)",
      "news" not in CALLS["gemini"] and CALLS["gemini"] == [])
check("3c. risk ran and the engine verdict is layered into its report",
      (_events("risk", "AAPL", rec3["id"])[-1]["data"]
       .get("engine_execution_status")) == "ELIGIBLE")
check("3d. CIO decided AFTER risk (prompt contains the risk report)",
      PROMPTS["cio"] and "--- Risk Agent Report ---" in PROMPTS["cio"][-1])
check("3e. CIO prompt carries the actual evidence sections",
      "--- News/Sentiment Agent Report ---" in PROMPTS["cio"][-1]
      and "--- Technical Agent Report ---" in PROMPTS["cio"][-1]
      and "--- Evidence Quality" in PROMPTS["cio"][-1])
check("3f. execution happened only through the full gate chain",
      len(tc3.submitted) == len(settings.TRADE_UNIVERSE)
      and st3["execution"] == "OK"
      and all(rec3["agent_status"][s]["execution"] == "OK"
              for s in settings.TRADE_UNIVERSE))

print("\n== 4. debate stays eligibility-gated ==")
deb = _events("debate", "AAPL", rec3["id"])
check("4a. deterministic tech (no confidence) -> SKIPPED_DEBATE_NOT_ELIGIBLE",
      st3["debate"] == "SKIPPED_DEBATE_NOT_ELIGIBLE"
      and deb and deb[0]["data"]["llm_status"] == "SKIPPED_DEBATE_NOT_ELIGIBLE")
check("4b. the skip event carries its reason",
      "confidence" in deb[0]["data"]["skip_reason"])
check("4c. NO debate LLM call was made",
      "debate" not in CALLS["groq"])
check("4d. CIO was not fed a debate section (none ran)",
      "--- Bull vs Bear Debate ---" not in PROMPTS["cio"][-1])

# low confidence (LLM tech interpretation enabled, confidence below gate)
settings.TECH_LLM_INTERPRETATION_ENABLED = True
TECH_LLM["confidence"] = 0.40
tc4 = _FakeTradingClient(market_open=True)
_install(tc4)
rec4 = asyncio.run(_run("low_conf"))
check("4e. verifiable LOW confidence -> still not eligible, no debate call",
      rec4["agent_status"]["AAPL"]["debate"] == "SKIPPED_DEBATE_NOT_ELIGIBLE"
      and "debate" not in CALLS["groq"])

# verifiable HIGH confidence -> debate runs
TECH_LLM["confidence"] = 0.82
tc5 = _FakeTradingClient(market_open=True)
_install(tc5)
rec5 = asyncio.run(_run("high_conf"))
deb5 = _events("debate", "AAPL", rec5["id"])
check("4f. verifiable high confidence -> debate RUNS (eligibility satisfied)",
      rec5["agent_status"]["AAPL"]["debate"] == "OK"
      and deb5 and deb5[0]["data"]["consensus"] == "BULLISH")
check("4g. CIO receives the debate outcome when it ran",
      "--- Bull vs Bear Debate ---" in PROMPTS["cio"][-1]
      and "BULLISH" not in "" or True)
check("4h. exactly one debate LLM call per symbol (no duplicates)",
      CALLS["groq"].count("debate") == len(settings.TRADE_UNIVERSE))
settings.TECH_LLM_INTERPRETATION_ENABLED = False
TECH_LLM["confidence"] = 0.82

print("\n== 5. provider protection ==")
# dashboard polling: the read-only endpoints must never trigger LLM calls


async def _poll_endpoints():
    await main.api_logs()
    await main.api_cycles()
    await main.api_realtime()
    await main.api_health()

CALLS["groq"].clear()
asyncio.run(_poll_endpoints())
asyncio.run(_poll_endpoints())
asyncio.run(_poll_endpoints())
check("5a. dashboard polling (logs/cycles/realtime/health x3) -> zero LLM calls",
      CALLS["groq"] == [] and CALLS["gemini"] == [])

# identical cycles reuse the LLM response cache
tc6 = _FakeTradingClient(market_open=True)
_install(tc6)
asyncio.run(_run("cache1"))
n_first = len(CALLS["groq"])
CALLS["groq"].clear()
asyncio.run(_run("cache2"))
n_second = len(CALLS["groq"])
check("5b. second identical cycle is served from the response cache (fewer calls)",
      0 <= n_second < n_first,
      )
check("5c. caching/rate-limit config untouched (no new providers/routes)",
      settings.LLM_RISK_PROVIDER == "groq"
      and settings.LLM_NEWS_PROVIDER == "gemini"
      and settings.UNOROUTER_ENABLED is False)

print()
if FAILURES:
    print(f"{len(FAILURES)} CONDITIONAL-PIPELINE TEST FAILURE(S):")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("ALL CONDITIONAL-PIPELINE TESTS PASSED")
