"""
tests/test_debate_risk_pipeline.py — Bull/Bear debate + Risk Agent execution
and visibility proofs (all transports stubbed; no network, no real keys):

  Orchestration / visibility (the reported bugs)
    1.  Debate and Risk Agent run BEFORE the hard risk engine — their
        records exist in the live buffer EVEN WHEN the engine rejects.
    2.  Hard rejection: deterministic CIO HOLD with the exact failed
        checks + sizing detail; no LLM CIO review (cannot override).
    3.  MARKET_CLOSED does not silently suppress analysis: debate, risk
        agent and the CIO review all run; only EXECUTION defers
        (DEFERRED_MARKET_CLOSED, next session recorded, no order).
    4.  Existing position with NO setup still gets risk MONITORING
        (exposure + exit risks) — also when the market is closed.
    5.  Debate skip is RECORDED with its reason (never a silent hole).
    6.  Normal approved path still executes (regression).

  Position-size semantics
    7.  Existing-position exposure is separated from proposed-order
        sizing (existing_position_ok stays true; proposed_order_sizing_ok
        false; binding limit + exact values recorded).
    8.  Cash-bound and cap-bound rejections name the responsible limit.

  Debate engine (direct)
    9.  Consensus BULLISH preserved; omitted consensus derived from edge.
    10. Low-strength / single-source evidence -> INSUFFICIENT_EVIDENCE
        (never a forced conclusion).
    11. Evidence lists, data sources and timestamps recorded.

  Risk agent (direct)
    12. Structured output (risk_status/score/factors/exposures/warnings/
        blocking_reasons/recommendations/evidence_sources/timestamp).
    13. LLM failure -> deterministic stands, error surfaced.

  Safety
    14. LLM "approved" can NEVER override a hard rejection (fail-closed).
    15. Duplicate-cycle guard + engine duplicate-order check.
    16. Stale market data -> STALE_DATA hard block (analysis still ran).

Run:  .venv/bin/python tests/test_debate_risk_pipeline.py
"""

import asyncio
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="debaterisk-")
os.environ["MEMORY_DB_PATH"] = os.path.join(_TMP, "test-memory.db")

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}")


import config                                                     # noqa: E402
from config import settings                                       # noqa: E402

settings.UNOROUTER_ENABLED = False
settings.LLM_TECH_PROVIDER = "groq"
settings.LLM_RISK_PROVIDER = "groq"
settings.LLM_CIO_PROVIDER = "groq"
settings.LLM_DEBATE_PROVIDER = "groq"
settings.LLM_NEWS_PROVIDER = "gemini"
settings.TECH_LLM_INTERPRETATION_ENABLED = False
settings.ENABLE_DEBATE = True
settings.DEBATE_REQUIRE_CONFLICT = False          # debate runs in these tests
settings.DEBATE_MIN_TECH_CONFIDENCE = 0.65
settings.GROQ_API_KEY = "dummy-groq"
settings.GEMINI_API_KEY = "dummy-gemini"
settings.ALPACA_API_KEY = "dummy-alpaca"
settings.ALPACA_SECRET_KEY = "dummy-alpaca-secret"
settings.ALPACA_DATA_FEED = "IEX"
settings.ENABLE_MEMORY = False
settings.FUNDAMENTALS_PROVIDER = "none"
settings.LLM_FALLBACK_PROVIDER = ""
settings.LLM_FALLBACK_MODEL = ""
settings.TRADE_UNIVERSE = ["AAPL", "MSFT"]

from services import alpaca_service, gemini_service, llm_service, \
    news_intelligence, risk_engine                          # noqa: E402
from agents import debate_agent, risk_agent                 # noqa: E402

# ---------------------------------------------------------------------------
# Transport stubs (no network)
# ---------------------------------------------------------------------------
DEBATE_LLM = {
    "bull_strength": 0.72, "bull_summary": "Trend and sentiment align.",
    "bear_strength": 0.30, "bear_summary": "Valuation is full.",
    "consensus": "BULLISH", "confidence": 0.64,
    "supporting_evidence": ["Price above SMA50/SMA200", "News sentiment positive"],
    "contradicting_evidence": ["P/E above 5-year average"],
    "unresolved_uncertainty": ["Next quarter margin trajectory"],
}
RISK_LLM = {
    "approved": True, "max_notional_usd": 500.0, "risk_level": "LOW",
    "reasoning": "Position within limits.",
    "risk_factors": ["moderate volatility"],
    "warnings": ["concentration building"],
    "recommendations": ["keep the stop tight"],
}
CIO_BUY = json.dumps({"decision": "BUY", "confidence": 0.8,
                      "notional_usd": 250.0, "reasoning": "Momentum intact."})
GEMINI_NEWS = json.dumps({
    "articles": [{"index": 0, "sentiment": "BULLISH", "confidence": 0.7,
                  "importance": "HIGH", "impact_horizon": "SHORT_TERM"}],
    "overall_sentiment": "BULLISH", "overall_confidence": 0.7,
    "summary": "Coverage is positive."})

ANSWERS = {"debate": json.dumps(DEBATE_LLM), "risk": json.dumps(RISK_LLM),
           "cio": CIO_BUY}
FAIL_NEXT = {}          # agent -> Exception to raise on the next call


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
    def __init__(self):
        self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        sys_prompt = str(kwargs.get("messages", [{}])[0].get("content", ""))
        if "bull vs. bear debate" in sys_prompt:
            if "debate" in FAIL_NEXT:
                raise FAIL_NEXT.pop("debate")
            return _Completion(ANSWERS["debate"])
        if "risk manager" in sys_prompt:
            if "risk" in FAIL_NEXT:
                raise FAIL_NEXT.pop("risk")
            return _Completion(ANSWERS["risk"])
        if "Chief Investment Officer" in sys_prompt:
            return _Completion(ANSWERS["cio"])
        return _Completion(CIO_BUY)


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
        self.interactions.create = lambda **kw: types.SimpleNamespace(
            output_text=GEMINI_NEWS, errors=None)
        self.models = types.SimpleNamespace(
            list=lambda: _ModelsList(["gemini-3.6-flash"]))


class _FakeAccount:
    cash, equity, buying_power, portfolio_value, last_equity = \
        12043.0, 25104.0, 12043.0, 25104.0, 25000.0
    status = types.SimpleNamespace(value="ACTIVE")


class _Pos:
    def __init__(self, symbol, qty, value):
        self.symbol = symbol
        self.qty = str(qty)
        self.avg_entry_price = "100.0"
        self.current_price = "100.0"
        self.market_value = str(value)
        self.unrealized_pl = "0.0"
        self.unrealized_plpc = 0.0
        self.side = types.SimpleNamespace(value="long")


def _next_open_dt():
    """Next session as a tz-aware datetime (the real SDK returns datetimes;
    get_clock calls .isoformat() on them)."""
    from datetime import datetime, timedelta, timezone
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1))
    return tomorrow.replace(hour=13, minute=30, second=0, microsecond=0)


class _FakeTradingClient:
    def __init__(self, market_open=True, positions=None, cash=None,
                 next_open=None):
        self.submitted = []
        self.market_open = market_open
        self.positions = positions or []
        self.account = _FakeAccount()
        if cash is not None:
            self.account.cash = cash
            self.account.buying_power = cash
        self.next_open = next_open or _next_open_dt()
    def get_account(self):
        return self.account
    def get_all_positions(self):
        return self.positions
    def get_portfolio_history(self, history_filter=None):
        return types.SimpleNamespace(timestamp=[1700000000, 1700086400],
                                     equity=[25000.0, 25104.0])
    def get_orders(self, req):
        return []
    def get_clock(self):
        return types.SimpleNamespace(is_open=self.market_open, timestamp=None,
                                     next_open=self.next_open, next_close=None)
    def submit_order(self, req):
        self.submitted.append(req)
        return types.SimpleNamespace(id="test-order", symbol=req.symbol,
                                     qty=req.qty,
                                     notional=getattr(req, "notional", None),
                                     status=types.SimpleNamespace(value="accepted"))


class _FakeDataClient:
    def __init__(self, stale=False, neutral=False):
        self.stale = stale
        self.neutral = neutral
    def get_stock_bars(self, req):
        import pandas as pd
        n = 261
        if self.neutral:
            closes = [100 + (0.2 if i % 2 else -0.2) for i in range(n)]
        else:
            closes = [100 + i * 0.35 + (2.5 if i % 2 == 0 else -2.5)
                      for i in range(n)]
        highs = [c + 1.5 for c in closes]
        lows = [c - 1.5 for c in closes]
        end = pd.Timestamp.utcnow().floor("D")
        if self.stale:
            end = end - pd.Timedelta(days=90)
        idx = pd.DatetimeIndex(pd.date_range(end=end, periods=n, freq="D",
                                             tz="UTC"), name="timestamp")
        df = pd.DataFrame({"close": closes, "open": closes, "high": highs,
                           "low": lows, "volume": [1000] * n,
                           "trade_count": [10] * n, "vwap": closes}, index=idx)
        return types.SimpleNamespace(df=df)
    def get_stock_snapshot(self, req):
        sym = req.symbol_or_symbols if isinstance(req.symbol_or_symbols, str) \
            else list(req.symbol_or_symbols)[0]
        return {sym: types.SimpleNamespace(
            symbol=sym,
            latest_trade=types.SimpleNamespace(price=101.25, size=10,
                                               timestamp=None),
            latest_quote=types.SimpleNamespace(bid_price=100.5, ask_price=101.5,
                                               bid_size=2, ask_size=2,
                                               timestamp=None),
            minute_bar=None, daily_bar=None, previous_daily_bar=None)}


def _install(trading_client, data_client=None):
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
                {"id": f"seed-{_s}-1", "headline": f"{_s} beats earnings",
                 "summary": "Revenue up", "source": "Reuters",
                 "published_at": "2026-10-20T10:00:00Z"}])


import main                                                       # noqa: E402
main._get_agent_weights = lambda: {}
risk_engine.init_db()


async def _run(tag):
    risk_engine.begin_cycle()
    await main.run_trading_cycle(triggered_by=tag)
    return main.cycle_history[0]


def _logs(agent, symbol=None, since=None):
    return [l for l in main.agent_logs
            if l.get("agent") == agent
            and (symbol is None or l.get("symbol") == symbol)
            and (since is None or l.get("timestamp", "") > since)]


def _last(agent, symbol, since):
    entries = _logs(agent, symbol, since)
    return entries[-1] if entries else None


print("== orchestration: analysis runs even when the engine rejects ==")

# --- Scenario A: market OPEN, setup exists, debate eligible -----------------
tc = _FakeTradingClient(market_open=True)
_install(tc)
recA = asyncio.run(_run("A"))
startedA = recA["started_at"]
for sym in settings.TRADE_UNIVERSE:
    dlog = _last("debate", sym, startedA)
    rlog = _last("risk", sym, startedA)
    check(f"1a. debate event in live buffer for {sym} (engine-approved path)",
          bool(dlog) and dlog["data"].get("evidence_status") == "AVAILABLE")
    check(f"1b. risk event in live buffer for {sym}",
          bool(rlog) and rlog["data"].get("risk_status") in
          ("APPROVED", "VETOED", "BLOCKED"))
check("1c. debate consensus recorded (BULLISH)",
      _last("debate", "AAPL", startedA)["data"].get("consensus") == "BULLISH")
check("1d. debate evidence lists recorded",
      bool(_last("debate", "AAPL", startedA)["data"].get("supporting_evidence")))
check("1e. risk structured fields present",
      all(k in _last("risk", "AAPL", startedA)["data"] for k in (
          "risk_status", "risk_score", "risk_factors", "position_exposure",
          "proposed_order_exposure", "warnings", "blocking_reasons",
          "recommendations", "timestamp", "evidence_sources")))
check("6. approved path still executes (order submitted)",
      len(tc.submitted) >= 1
      and all(recA["agent_status"][s]["execution"] == "OK"
              for s in settings.TRADE_UNIVERSE))
check("6b. CIO ran on the approved path",
      all(recA["agent_status"][s]["cio"] == "OK"
          for s in settings.TRADE_UNIVERSE))

print("\n== hard rejection: analysis visible, deterministic HOLD ==")

# --- Scenario B: cash=0 -> engine BLOCKS; debate/risk still ran -------------
tcB = _FakeTradingClient(market_open=True, cash=0.0)
_install(tcB)
recB = asyncio.run(_run("B"))
stB = recB["agent_status"]["AAPL"]
db = _last("debate", "AAPL", recB["started_at"])
rb = _last("risk", "AAPL", recB["started_at"])
check("2a. debate ran BEFORE the engine rejection (event exists)",
      bool(db) and db["data"].get("evidence_status") == "AVAILABLE")
check("2b. risk agent ran (event exists, structured)",
      bool(rb) and rb["data"].get("risk_status") == "BLOCKED")
check("2c. risk stage status from the risk AGENT (not the engine)",
      stB["risk"] == "OK")
check("2d. CIO deterministic HOLD after a hard block",
      stB["cio"] == "SKIPPED" and stB["execution"] == "BLOCKED")
ciob = _last("cio", "AAPL", recB["started_at"])
check("2e. CIO reasoning cites failed checks + sizing (exact values)",
      "failed checks" in ciob["data"]["reasoning"]
      and "sizing" in ciob["data"]["reasoning"]
      and ciob["data"]["llm_status"] == "SKIPPED_RISK_REJECTED")
check("2f. no order submitted on a hard block", tcB.submitted == [])
check("2g. no LLM CIO call after a hard block (cannot override)",
      ciob["data"]["provider"] == "deterministic"
      and ciob["data"]["model"] == "risk-engine")

print("\n== MARKET_CLOSED: analysis runs, only execution defers ==")

# --- Scenario C: market closed, setup exists --------------------------------
tcC = _FakeTradingClient(market_open=False)
_install(tcC)
recC = asyncio.run(_run("C"))
stC = recC["agent_status"]["AAPL"]
check("3a. market closed does NOT suppress analysis (debate ran)",
      bool(_last("debate", "AAPL", recC["started_at"])))
check("3b. risk agent ran with the market closed",
      bool(_last("risk", "AAPL", recC["started_at"])))
check("3c. CIO review ran with the market closed",
      stC["cio"] == "OK"
      and _last("cio", "AAPL", recC["started_at"])["data"]["provider"] != "deterministic")
check("3d. execution DEFERRED_MARKET_CLOSED (not silent, not submitted)",
      stC["execution"] == "DEFERRED_MARKET_CLOSED" and tcC.submitted == [])
exC = _last("execution", "AAPL", recC["started_at"])
check("3e. deferral event records the next eligible session",
      bool(exC) and exC["data"].get("execution_status") == "DEFERRED_MARKET_CLOSED"
      and exC["data"].get("next_session"))
decC = [d for d in recC["decisions"] if d["symbol"] == "AAPL"][0]
check("3f. decision record separates ANALYSIS/RISK/MARKET/EXECUTION status",
      decC.get("execution_status") == "DEFERRED_MARKET_CLOSED"
      and decC.get("market_status") == "CLOSED"
      and decC.get("next_session"))
check("3g. deferred cycle is NOT silently degraded to an error cycle",
      recC["status"] in ("OK", "PARTIAL_ERROR"))
check("3h. CIO reasoning notes the deferral",
      "DEFERRED_MARKET_CLOSED" in _last("cio", "AAPL", recC["started_at"])["data"]["reasoning"])

print("\n== existing-position monitoring (no setup, market closed) ==")

# --- Scenario D: flat bars (no setup) + held position + market closed -------
tcD = _FakeTradingClient(market_open=False,
                         positions=[_Pos("AAPL", 10, 3000.0)])
_install(tcD, _FakeDataClient(neutral=True))
recD = asyncio.run(_run("D"))
stD = recD["agent_status"]["AAPL"]
rd = _last("risk", "AAPL", recD["started_at"])
check("4a. held position gets risk MONITORING with no setup",
      bool(rd) and rd["data"].get("risk_status") == "MONITORING")
check("4b. monitoring records exposure facts (not a proposed order)",
      rd["data"]["position_exposure"]["monitoring_only"] is True
      and rd["data"]["proposed_order_exposure"] is None)
check("4c. monitoring is analysis, never execution (no order)",
      tcD.submitted == [] and stD["execution"] == "SKIPPED")
check("4d. no-setup decision is the deterministic HOLD",
      stD["cio"] == "SKIPPED"
      and "No technical setup" in _last("cio", "AAPL", recD["started_at"])["data"]["reasoning"])
ms = _last("risk", "MSFT", recD["started_at"])
check("4e. symbols WITHOUT a position are not monitored (honest skip)",
      ms is None and recD["agent_status"]["MSFT"]["risk"] == "SKIPPED")

print("\n== position-size semantics ==")

# --- Scenario E: position at cap -> proposed-order sizing separated ---------
tcE = _FakeTradingClient(market_open=True, positions=[_Pos("AAPL", 30, 3000.0)])
_install(tcE)
recE = asyncio.run(_run("E"))
stE = recE["agent_status"]["AAPL"]
exE = _last("execution", "AAPL", recE["started_at"])
sizing = (exE["data"] or {}).get("sizing_detail") or {}
checksE = (exE["data"] or {}).get("failed_checks") or {}
full_checks = ((exE["data"] or {}).get("engine") or {}).get("checks") or {}
check("7a. proposed-order sizing rejected at the cap",
      checksE.get("proposed_order_sizing_ok") is False
      and checksE.get("position_size_ok") is False)
check("7b. existing position NOT classified invalid by a new-order rejection",
      full_checks.get("existing_position_ok") is True)
check("7c. existing-position exposure recorded (informational)",
      isinstance(full_checks.get("existing_position_exposure_pct"),
                 (int, float)))
check("7d. binding limit named with exact values",
      sizing.get("binding_limit") == "POSITION_CAP"
      and sizing.get("existing_position_value") == 3000.0
      and sizing.get("max_position_pct") == 0.10
      and sizing.get("room_under_cap") == 0.0)
# cash-bound rejection names CASH (separate scenario from the cap-bound one)
tcX = _FakeTradingClient(market_open=True, cash=0.0)
_install(tcX)
recX = asyncio.run(_run("X"))
exX = _last("execution", "AAPL", recX["started_at"])
check("8. cash-bound rejection names CASH",
      exX and (exX["data"] or {}).get("sizing_detail", {}).get("binding_limit") == "CASH")

print("\n== debate engine (direct) ==")


def _tech(available=True, conf=0.8):
    return ({"evidence_status": "AVAILABLE", "signal": "BULLISH",
             "confidence": conf, "summary": "Trend up.",
             "provider": "deterministic", "model": "rule-engine",
             "llm_status": "SKIPPED_DETERMINISTIC", "error": None}
            if available else
            {"evidence_status": "UNAVAILABLE", "signal": None,
             "confidence": None, "summary": "no data", "error": "x"})


def _news(available=True):
    return ({"evidence_status": "AVAILABLE", "sentiment": "BULLISH",
             "confidence": 0.7, "summary": "positive", "error": None}
            if available else
            {"evidence_status": "UNAVAILABLE", "sentiment": None,
             "confidence": None, "summary": "", "error": "no cache"})


llm_service.reset_all_state()
llm_service._clients["groq"] = _llm_client()
llm_service._verified_models["groq"] = {"openai/gpt-oss-20b"}


def _clear_llm_cache():
    """The router caches identical prompts; direct-call scenarios vary the
    ANSWERS, so the cache must not replay a stale reply."""
    llm_service._response_cache.clear()


d = debate_agent.run_debate("AAPL", _tech(), _news())
check("9a. consensus BULLISH preserved with 2 available sources",
      d["consensus"] == "BULLISH" and d["confidence"] == 0.64)
check("9b. timestamp + data sources recorded",
      bool(d["timestamp"]) and {s["source"] for s in d["data_sources"]}
      == {"technical", "news"})

_clear_llm_cache()
ANSWERS["debate"] = json.dumps({k: v for k, v in DEBATE_LLM.items()
                                if k != "consensus"})
d2 = debate_agent.run_debate("AAPL", _tech(), _news())
check("9c. omitted consensus derived from the measured edge",
      d2["consensus"] == "BULLISH")

_clear_llm_cache()
ANSWERS["debate"] = json.dumps({**DEBATE_LLM, "bull_strength": 0.2,
                                "bear_strength": 0.2,
                                "consensus": "BULLISH", "confidence": 0.9})
d3 = debate_agent.run_debate("AAPL", _tech(), _news())
check("10a. weak both sides -> INSUFFICIENT_EVIDENCE (never forced)",
      d3["consensus"] == "INSUFFICIENT_EVIDENCE" and d3["confidence"] <= 0.3)

_clear_llm_cache()
d4 = debate_agent.run_debate("AAPL", _tech(), _news(available=False))
check("10b. single available source -> INSUFFICIENT_EVIDENCE",
      d4["consensus"] == "INSUFFICIENT_EVIDENCE"
      and d4["data_sources"][1]["state"] == "UNAVAILABLE")
ANSWERS["debate"] = json.dumps(DEBATE_LLM)

print("\n== risk agent (direct) ==")

llm_service.reset_all_state()
llm_service._clients["groq"] = _llm_client()
llm_service._verified_models["groq"] = {"openai/gpt-oss-20b"}
r = risk_agent.assess_risk(
    "AAPL", "buy", {"equity": 25000.0, "cash": 5000.0, "buying_power": 5000.0},
    {"symbol": "AAPL", "qty": 10, "market_value": 2000.0,
     "avg_entry_price": 200.0, "current_price": 200.0},
    {"volatility_annualized": 0.42, "max_drawdown": -0.25})
check("12a. structured risk output present",
      all(k in r for k in ("risk_status", "risk_score", "risk_factors",
                           "position_exposure", "proposed_order_exposure",
                           "warnings", "blocking_reasons", "recommendations",
                           "timestamp", "evidence_sources")))
check("12b. LLM factors/warnings/recommendations merged",
      "moderate volatility" in r["risk_factors"]
      and r["warnings"] and r["recommendations"])
check("12c. exposures computed (existing vs proposed separated)",
      r["position_exposure"]["existing_position_value"] == 2000.0
      and r["proposed_order_exposure"]["proposed_notional_usd"] == 500.0)

_clear_llm_cache()
FAIL_NEXT["risk"] = RuntimeError("provider down")
r2 = risk_agent.assess_risk(
    "AAPL", "buy", {"equity": 25000.0, "cash": 5000.0, "buying_power": 5000.0},
    None, None)
check("13. LLM failure -> deterministic stands, error surfaced",
      r2["error"] and r2["approved"] is True
      and r2["max_notional_usd"] == 2500.0   # the deterministic cap, NOT the
      # LLM's 500 — the LLM layer failing never invents a smaller/bigger one
      )

print("\n== safety ==")

# 14. fail-closed: LLM approves everything, engine still blocks (cash=0)
tcF = _FakeTradingClient(market_open=True, cash=0.0)
_install(tcF)
recF = asyncio.run(_run("F"))
check("14. LLM approval can NEVER override a hard rejection (fail-closed)",
      tcF.submitted == []
      and all(d["decision"] == "HOLD" for d in recF["decisions"])
      and recF["agent_status"]["AAPL"]["execution"] == "BLOCKED")

# 15a. duplicate-cycle guard
main._cycle_running = True
before = len(main.cycle_history)
out = asyncio.run(main.run_trading_cycle(triggered_by="dup"))
main._cycle_running = False
check("15a. duplicate cycle prevented (single-flight guard)",
      len(main.cycle_history) == before
      and out.get("status") in ("SKIPPED_OVERLAP", "OK")
      and (out.get("status") != "OK" or True))

# 15b. engine duplicate-order check (same cycle, same symbol+side+size)
risk_engine.begin_cycle()
risk_engine.mark_executed("AAPL", "buy", 250.0)
dup = risk_engine.assess_trade(
    "AAPL", "buy", {"equity": 25000.0, "cash": 5000.0, "buying_power": 5000.0},
    {}, indicators={"volatility_annualized": 0.2},
    proposed_notional_usd=250.0)
check("15b. engine duplicate-order check blocks a same-cycle repeat",
      "DUPLICATE_ORDER" in dup["blocked_reasons"] and not dup["approved"])

# 16. stale market data -> STALE_DATA hard block, analysis still ran
tcG = _FakeTradingClient(market_open=True)
_install(tcG, _FakeDataClient(stale=True))
recG = asyncio.run(_run("G"))
exG = _last("execution", "AAPL", recG["started_at"])
check("16. stale market data -> STALE_DATA hard block (analysis still ran)",
      exG and exG["data"].get("blocked_reason") == "STALE_DATA"
      and bool(_last("debate", "AAPL", recG["started_at"]))
      and bool(_last("risk", "AAPL", recG["started_at"])))

print("\n== live-buffer event shape ==")

ok_events = True
for l in main.agent_logs:
    if l.get("agent") in ("debate", "risk", "cio", "execution", "technical"):
        if not (l.get("timestamp") and l.get("agent")
                and isinstance(l.get("data"), dict)):
            ok_events = False
            break
check("17. every persisted agent event carries agent/timestamp/data",
      ok_events)
risk_events = [l for l in main.agent_logs if l.get("agent") == "risk"]
check("18. risk events never fabricate an approval over the engine",
      all(l["data"].get("approved") is not True
          or l["data"].get("engine_reason") == "TRADE_APPROVED"
          or l["data"].get("risk_status") == "MONITORING"
          or l["data"].get("llm_status") != "OK"
          for l in risk_events
          if l["data"].get("engine_reason")))

# _log_event enrichment: cycle_id on every event; duration/execution_status
# mirrored top-level for flat consumers (nothing invented when absent).
cyc_ids = {l.get("cycle_id") for l in main.agent_logs if l.get("agent") != "system"}
check("19. every agent event carries the cycle_id it belongs to",
      cyc_ids and all(isinstance(c, int) for c in cyc_ids))
timed = [l for l in main.agent_logs
         if l.get("agent") in ("debate", "risk", "cio", "execution")]
check("20. LLM/agent events carry duration_ms",
      bool(timed) and all(isinstance(l.get("duration_ms"), int) for l in timed))
ex_events = [l for l in main.agent_logs if l.get("agent") == "execution"]
check("21. execution events surface execution_status top-level",
      bool(ex_events)
      and all(l.get("execution_status") in ("SUBMITTED", "FAILED", "DEFERRED_MARKET_CLOSED",
                                            "BLOCKED")
              for l in ex_events))

print()
if FAILURES:
    print(f"{len(FAILURES)} DEBATE/RISK PIPELINE TEST FAILURE(S):")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("ALL DEBATE/RISK PIPELINE TESTS PASSED")
