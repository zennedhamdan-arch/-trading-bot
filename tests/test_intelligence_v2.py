"""
tests/test_intelligence_v2.py

AI-Trader Intelligence Infrastructure V2 — focused test suite for the 19
specified behaviors:

 1.  Duplicate article is not analyzed twice
 2.  Article fingerprints work
 3.  Cache prevents unnecessary LLM requests
 4.  HIGH relevance reaches LLM
 5.  LOW relevance does not
 6.  Primary model failure triggers fallback
 7.  First fallback failure triggers second fallback
 8.  UnoRouter failure falls back to Groq
 9.  All providers failing returns deterministic fallback
 10. Empty LLM responses do not crash
 11. 429 does not block trading
 12. Circuit breaker opens
 13. Circuit breaker prevents repeated requests
 14. Risk Engine works without LLM
 15. LLM cannot override risk rejection
 16. CIO does not run without valid setup
 17. CIO does not run after risk rejection
 18. Debate Agent does not run when disabled
 19. No secrets appear in logs

Run:  .venv/bin/python tests/test_intelligence_v2.py
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Isolated DB BEFORE importing config (news intelligence + risk engine share
# memory_service's DB file).
_TMPDIR = tempfile.mkdtemp(prefix="itv2-")
os.environ["MEMORY_DB_PATH"] = os.path.join(_TMPDIR, "test-memory.db")

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}")


# ---------------------------------------------------------------------------
# Fakes (OpenAI-compatible clients, Alpaca trading/data clients)
# ---------------------------------------------------------------------------

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


class _HTTPError(Exception):
    def __init__(self, status_code, message=""):
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


class _ModelsList:
    def __init__(self, ids):
        self.data = [types.SimpleNamespace(id=i) for i in ids]


def _client(responder, model_ids):
    client = types.SimpleNamespace()
    client.chat = types.SimpleNamespace(completions=_Completions(responder))
    client.models = types.SimpleNamespace(list=lambda: _ModelsList(list(model_ids)))
    return client


UNO_MODELS = ("glm-5.3-flash-thinking:free", "qwen3.8-flash-next:free", "glm-5.3-flash:free")
GROQ_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")

NEWS_ANALYSIS_JSON = json.dumps({
    "articles": [
        {"index": 0, "sentiment": "BULLISH", "confidence": 0.7,
         "importance": "HIGH", "impact_horizon": "SHORT_TERM"},
    ],
    "overall_sentiment": "BULLISH", "overall_confidence": 0.7,
    "summary": "Coverage is positive across the board.",
})


class _FakeAccount:
    cash, equity, buying_power, portfolio_value, last_equity = 12043.0, 25104.0, 12043.0, 25104.0, 25000.0
    status = types.SimpleNamespace(value="ACTIVE")


class _FakeTradingClient:
    def __init__(self, day_pl_pct=0.0, positions=None, orders=None):
        self.submitted = []
        self._day_pl_pct = day_pl_pct
        self._positions = positions or []
        self._orders = orders or []

    def get_account(self):
        acct = _FakeAccount()
        # day_pl_pct is derived by alpaca_service; expose equity/last_equity
        # delta instead for realism.
        acct.equity = 25104.0
        acct.last_equity = 25104.0 * (1.0 - self._day_pl_pct)
        return acct

    def get_all_positions(self):
        return self._positions

    def get_orders(self, req):
        return self._orders

    def get_portfolio_history(self, history_filter=None):
        return types.SimpleNamespace(timestamp=[1700000000, 1700086400],
                                     equity=[25000.0, 25104.0])

    def get_clock(self):
        return types.SimpleNamespace(is_open=True, timestamp=None,
                                     next_open=None, next_close=None)

    def submit_order(self, req):
        self.submitted.append(req)
        return types.SimpleNamespace(id="test-order", symbol=req.symbol, qty=req.qty,
                                     notional=getattr(req, "notional", None),
                                     status=types.SimpleNamespace(value="accepted"))


def _bars_client(closes):
    """Alpaca data client stub with a fixture close series (ends today)."""
    import pandas as pd

    class _FakeDataClient:
        def __init__(self):
            self.bar_requests = []

        def get_stock_bars(self, req):
            self.bar_requests.append(req)
            n = len(closes)
            highs = [c + 1.5 for c in closes]
            lows = [c - 1.5 for c in closes]
            end = pd.Timestamp.utcnow().floor("D")
            idx = pd.DatetimeIndex(pd.date_range(end=end, periods=n, freq="D", tz="UTC"),
                                   name="timestamp")
            df = pd.DataFrame({"close": closes, "open": closes, "high": highs,
                               "low": lows, "volume": [1000] * n,
                               "trade_count": [10] * n, "vwap": closes}, index=idx)
            return types.SimpleNamespace(df=df)

        def get_stock_snapshot(self, req):
            sym = req.symbol_or_symbols
            sym = sym if isinstance(sym, str) else list(sym)[0]
            return {sym: types.SimpleNamespace(
                symbol=sym,
                latest_trade=types.SimpleNamespace(price=closes[-1], size=10, timestamp=None),
                latest_quote=types.SimpleNamespace(bid_price=closes[-1] - 0.5,
                                                   ask_price=closes[-1] + 0.5,
                                                   bid_size=2, ask_size=2, timestamp=None),
                minute_bar=None, daily_bar=None, previous_daily_bar=None)}

    return _FakeDataClient()


def _rising_series(n=261):
    # Rising trend with regular pullbacks, ending on an up-step:
    # trend BULLISH, momentum BULLISH, RSI below 70.
    return [100 + i * 0.35 + (2.5 if i % 2 == 0 else -2.5) for i in range(n)]


def _flat_series(n=261):
    # Perfectly flat: trend NEUTRAL -> no tradeable setup.
    return [100.0 for _ in range(n)]


# ---------------------------------------------------------------------------
# Config + imports
# ---------------------------------------------------------------------------

import config  # noqa: E402

config.settings.UNOROUTER_ENABLED = True
config.settings.UNOROUTER_API_KEY = "sk-test-unorouter"
config.settings.GROQ_ENABLED = True
config.settings.GROQ_API_KEY = "sk-test-groq"
config.settings.GEMINI_API_KEY = ""
config.settings.OPENROUTER_API_KEY = ""
config.settings.NVIDIA_API_KEY = ""
config.settings.ALPACA_API_KEY = "test-alpaca-key"
config.settings.ALPACA_SECRET_KEY = "test-alpaca-secret"
config.settings.ALPACA_DATA_FEED = "IEX"
config.settings.ENABLE_MEMORY = False
config.settings.TRADE_UNIVERSE = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]
config.settings.FUNDAMENTALS_PROVIDER = "none"
config.settings.LLM_FALLBACK_PROVIDER = ""
config.settings.LLM_FALLBACK_MODEL = ""
config.settings.LLM_RATE_LIMIT_MAX_RETRIES = 0

from services import llm_service, news_intelligence, news_worker, risk_engine  # noqa: E402
from services import alpaca_service  # noqa: E402


def _reset(reset_breaker_threshold=3):
    llm_service.reset_all_state()
    config.settings.LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD = reset_breaker_threshold
    config.settings.LLM_CIRCUIT_BREAKER_ENABLED = True
    llm_service._clients["unorouter"] = _client(lambda kw: NEWS_ANALYSIS_JSON, UNO_MODELS)
    llm_service._clients["groq"] = _client(lambda kw: NEWS_ANALYSIS_JSON, GROQ_MODELS)
    llm_service._verified_models["unorouter"] = set(UNO_MODELS)
    llm_service._verified_models["groq"] = set(GROQ_MODELS)


# ===========================================================================
print("1-2. article normalization, fingerprints, deduplication:")

art = news_intelligence.normalize_article("AAPL", {
    "id": "",  # no provider id -> fingerprint
    "headline": "Apple   BEATS  earnings",
    "source": "Reuters",
    "published_at": "2026-09-10T13:00:00Z",
})
check("fingerprint = SHA256(normalized headline + source + published_at)",
      art["article_id"] == news_intelligence.fingerprint("Apple   BEATS  earnings",
                                                         "Reuters", "2026-09-10T13:00:00Z"))
check("headline normalized (lowercase, trimmed, single spaces)",
      art["normalized_headline"] == "apple beats earnings")
same_again = news_intelligence.normalize_article("AAPL", {
    "headline": "apple beats earnings", "source": "Reuters",
    "published_at": "2026-09-10T13:00:00Z"})
check("same content (any casing/spacing) -> same fingerprint",
      same_again["article_id"] == art["article_id"])
with_id = news_intelligence.normalize_article("AAPL", {"id": 12345, "headline": "x"})
check("provider article id is used when present", with_id["article_id"] == "12345")

llm_calls = {"n": 0}


def _counting_responder(kwargs):
    llm_calls["n"] += 1
    return NEWS_ANALYSIS_JSON


llm_service.reset_all_state()
llm_service._clients["unorouter"] = _client(_counting_responder, UNO_MODELS)
llm_service._clients["groq"] = _client(_counting_responder, GROQ_MODELS)
llm_service._verified_models["unorouter"] = set(UNO_MODELS)

ARTICLES = [
    {"id": "a-1", "headline": "Apple beats earnings expectations",
     "summary": "Revenue up 12%", "source": "Reuters",
     "published_at": "2026-09-10T13:00:00Z"},
    {"id": "a-2", "headline": "Microsoft announces Azure growth",
     "summary": "", "source": "CNBC", "published_at": "2026-09-10T12:00:00Z"},
]

s1 = news_intelligence.refresh_symbol("AAPL", fetch_articles=lambda s: ARTICLES)
check("first refresh analyzes the new important article (LLM called)",
      s1["articles_analyzed"] >= 1 and llm_calls["n"] >= 1)
calls_after_first = llm_calls["n"]
s2 = news_intelligence.refresh_symbol("AAPL", fetch_articles=lambda s: ARTICLES)
check("duplicate article is NOT analyzed twice (0 new LLM calls)",
      s2["duplicates_ignored"] == 2 and llm_calls["n"] == calls_after_first
      and s2["articles_analyzed"] == 0)
check("dedup survives process state (persistent SQLite fingerprints)",
      news_intelligence.get_recent_articles("AAPL", limit=50).__len__() == 2)

# ===========================================================================
print("4-5. deterministic relevance filter:")

high, _ = news_intelligence.score_relevance("AAPL", "Apple beats earnings, raises guidance", "")
check("HIGH relevance: symbol/company + material event", high == "HIGH")
medium, _ = news_intelligence.score_relevance("AAPL", "Apple product review published", "")
check("MEDIUM relevance: company mention without material event", medium == "MEDIUM")
low, _ = news_intelligence.score_relevance("AAPL", "Top stock market trading ideas this week", "")
check("LOW relevance: generic finance vocabulary only", low == "LOW")
irrelevant, _ = news_intelligence.score_relevance("AAPL", "Soccer match ends in a draw", "")
check("IRRELEVANT: no signal at all", irrelevant == "IRRELEVANT")
macro, _ = news_intelligence.score_relevance("SPY", "Fed signals interest rate cut", "")
check("macro event is HIGH relevance for the index (SPY)", macro == "HIGH")

check("HIGH reaches the LLM at threshold MEDIUM",
      news_intelligence.article_reaches_llm("HIGH"))
check("MEDIUM reaches the LLM at threshold MEDIUM",
      news_intelligence.article_reaches_llm("MEDIUM"))
check("LOW does NOT reach the LLM at threshold MEDIUM",
      not news_intelligence.article_reaches_llm("LOW"))
check("IRRELEVANT never reaches the LLM",
      not news_intelligence.article_reaches_llm("IRRELEVANT"))

# LOW relevance never produces an LLM call
llm_calls["n"] = 0
news_intelligence.refresh_symbol("TSLA", fetch_articles=lambda s: [
    {"id": "low-1", "headline": "General stock market trading ideas",
     "summary": "", "source": "Blog", "published_at": "2026-09-10T10:00:00Z"}])
check("LOW-relevance article stored but never sent to an LLM",
      llm_calls["n"] == 0
      and any(a["article_id"] == "low-1" for a in news_intelligence.get_recent_articles("TSLA")))

# ===========================================================================
print("6-8. model + provider fallback chain:")

# 6: primary model 500s -> fallback model answers
_reset()


def _primary_fails(kwargs):
    if kwargs["model"] == "glm-5.3-flash-thinking:free":
        raise _HTTPError(500, "upstream exploded")
    return NEWS_ANALYSIS_JSON


llm_service._clients["unorouter"] = _client(_primary_fails, UNO_MODELS)
r = llm_service.call("news", system="s", user="prompt-6", symbol="AAPL", expect_json=True)
uno_client = llm_service._clients["unorouter"]
models_tried = [c["model"] for c in uno_client.chat.completions.calls]
check("primary model failure triggers the fallback model",
      r.ok and r.fallback_used and models_tried[0] == "glm-5.3-flash-thinking:free"
      and r.model != "glm-5.3-flash-thinking:free")

# 7: first fallback 404s (the REAL catalog situation for qwen3.8-flash-next:free)
_reset()


def _fb1_404(kwargs):
    if kwargs["model"] == "qwen3.8-flash-next:free":
        raise _HTTPError(404, "model not found")
    if kwargs["model"] == "glm-5.3-flash-thinking:free":
        raise _HTTPError(500, "down")
    return NEWS_ANALYSIS_JSON


llm_service._clients["unorouter"] = _client(_fb1_404, UNO_MODELS)
r = llm_service.call("news", system="s", user="prompt-7", symbol="AAPL", expect_json=True)
check("first fallback failure (404) triggers the second fallback",
      r.ok and r.model == "glm-5.3-flash:free")
check("dead model circuit short-circuits it (LLM_MODEL_UNAVAILABLE)",
      llm_service.provider_states()["unorouter"]["state"] == "MODEL_UNAVAILABLE"
      or ("qwen3.8-flash-next:free", ) and
      ("unorouter", "qwen3.8-flash-next:free") in llm_service._model_unavailable_until)

# 8: ALL UnoRouter models fail -> Groq answers
_reset()


def _all_uno_fail(kwargs):
    raise _HTTPError(500, "unorouter down")


llm_service._clients["unorouter"] = _client(_all_uno_fail, UNO_MODELS)
r = llm_service.call("news", system="s", user="prompt-8", symbol="AAPL", expect_json=True)
check("UnoRouter total failure falls back to Groq",
      r.ok and r.provider == "groq" and r.fallback_used)

# ===========================================================================
print("9-10. deterministic fallback + empty responses:")

# 9: every provider fails -> news intelligence returns the deterministic fallback
_reset()


def _everything_fails(kwargs):
    raise _HTTPError(500, "total outage")


llm_service._clients["unorouter"] = _client(_everything_fails, UNO_MODELS)
llm_service._clients["groq"] = _client(_everything_fails, GROQ_MODELS)
det_articles = [
    {"id": "det-1", "headline": "Microsoft beats earnings, record revenue growth",
     "summary": "", "source": "Reuters", "published_at": "2026-09-10T09:00:00Z"},
    {"id": "det-2", "headline": "Microsoft faces lawsuit investigation after miss",
     "summary": "", "source": "Bloomberg", "published_at": "2026-09-10T08:00:00Z"},
]
out = news_intelligence.refresh_symbol("MSFT", fetch_articles=lambda s: det_articles)
check("all providers failing still analyzes (deterministic keyword fallback)",
      out["articles_analyzed"] == 2 and out["provider_failures"] == 1)
intel = news_intelligence.get_cached_news_intelligence("MSFT")
check("deterministic fallback is honestly labeled (source, low confidence)",
      intel["source"] == "deterministic_fallback" and intel["confidence"] is not None
      and intel["confidence"] <= 0.35)
arts = {a["article_id"]: a for a in news_intelligence.get_recent_articles("MSFT")}
check("deterministic sentiment uses the keyword lists (pos headline BULLISH)",
      arts["det-1"]["analysis"]["sentiment"] == "BULLISH")
check("deterministic sentiment uses the keyword lists (neg headline BEARISH)",
      arts["det-2"]["analysis"]["sentiment"] == "BEARISH")

# Router-level: failure result uses the standardized shape
r = llm_service.call("news", system="s", user="prompt-9", symbol="MSFT")
std = r.standard()
check("total failure -> standardized failure shape (success=false, error code)",
      std["success"] is False and std["provider"] is None and std["model"] is None
      and std["content"] is None and std["error"])

# 10: empty responses never crash and move down the chain
_reset()


def _empty_then_ok(kwargs):
    if kwargs["model"] == "glm-5.3-flash-thinking:free":
        return "   "  # empty/whitespace response
    return NEWS_ANALYSIS_JSON


llm_service._clients["unorouter"] = _client(_empty_then_ok, UNO_MODELS)
r = llm_service.call_json("news", system="s", user="prompt-10", symbol="AAPL")
check("empty LLM response does not crash; next model answers",
      r.ok and r.model != "glm-5.3-flash-thinking:free")


def _garbage(kwargs):
    return "not json at all {{{"


llm_service.reset_all_state()
llm_service._clients["unorouter"] = _client(_garbage, UNO_MODELS)
llm_service._clients["groq"] = _client(_garbage, GROQ_MODELS)
r = llm_service.call_json("news", system="s", user="prompt-10b", symbol="AAPL")
check("malformed JSON from every model -> INVALID_RESPONSE, never a crash",
      not r.ok and r.status == "INVALID_RESPONSE")

# ===========================================================================
print("11. 429 / long retries never block trading:")

_reset()
config.settings.GROQ_ENABLED = False  # isolate: uno chain only


def _rate_limited(kwargs):
    raise _HTTPError(429, "Rate limit exceeded — daily quota exhausted. Please retry in 3600s")


llm_service._clients["unorouter"] = _client(_rate_limited, UNO_MODELS)
t0 = time.monotonic()
r = llm_service.call("news", system="s", user="prompt-11", symbol="AAPL")
elapsed = time.monotonic() - t0
check("429 quota fails FAST (no 30-60s retries; < 2s here)",
      not r.ok and r.status == "PROVIDER_QUOTA_EXCEEDED" and elapsed < 2.0)
check("quota exhaustion opens the provider circuit",
      llm_service.provider_states()["unorouter"]["state"] == "QUOTA_EXHAUSTED")
config.settings.GROQ_ENABLED = True

# ===========================================================================
print("12-13. classic circuit breaker (CLOSED -> OPEN -> HALF_OPEN):"

      )
_reset()
config.settings.LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3


def _network_down(kwargs):
    raise ConnectionError("connection refused")


llm_service._clients["unorouter"] = _client(_network_down, UNO_MODELS)
llm_service._clients["groq"] = _client(_network_down, GROQ_MODELS)
config.settings.UNOROUTER_FALLBACK_MODELS = []  # single model -> clean counts
config.settings.LLM_MAX_MODEL_ATTEMPTS = 1
llm_service.call("news", system="s", user="b1", symbol="AAPL")
llm_service.call("news", system="s", user="b2", symbol="AAPL")
state = llm_service.provider_states()["unorouter"]
check("breaker stays CLOSED below the failure threshold",
      state["circuit"] == "CLOSED" and state["failure_count"] == 2)
llm_service.call("news", system="s", user="b3", symbol="AAPL")
state = llm_service.provider_states()["unorouter"]
check("breaker OPENS after 3 consecutive failures",
      state["circuit"] == "OPEN" and state["failure_count"] >= 3)

uno_client = llm_service._clients["unorouter"]
sends_before = len(uno_client.chat.completions.calls)
r = llm_service.call("news", system="s", user="b4", symbol="AAPL")
sends_after = len(uno_client.chat.completions.calls)
check("OPEN breaker prevents repeated requests (no network call)",
      sends_after == sends_before and not r.ok and r.status == "CIRCUIT_OPEN")

# HALF_OPEN probe after the backoff window expires
llm_service._breaker["unorouter"]["open_until"] = time.monotonic() - 1.0
check("expired OPEN breaker reports HALF_OPEN (probe allowed)",
      llm_service.provider_states()["unorouter"]["circuit"] == "HALF_OPEN")
llm_service._clients["unorouter"] = _client(lambda kw: NEWS_ANALYSIS_JSON, UNO_MODELS)
r = llm_service.call("news", system="s", user="b5", symbol="AAPL")
check("successful HALF_OPEN probe closes the breaker",
      r.ok and llm_service.provider_states()["unorouter"]["circuit"] == "CLOSED")
config.settings.UNOROUTER_FALLBACK_MODELS = ["qwen3.8-flash-next:free", "glm-5.3-flash:free"]
config.settings.LLM_MAX_MODEL_ATTEMPTS = 3

# ===========================================================================
print("3. response cache prevents unnecessary LLM requests:")

_reset()
cached_client = _client(lambda kw: NEWS_ANALYSIS_JSON, UNO_MODELS)
llm_service._clients["unorouter"] = cached_client
r1 = llm_service.call("news", system="s", user="identical prompt", symbol="AAPL")
sends1 = len(cached_client.chat.completions.calls)
r2 = llm_service.call("news", system="s", user="identical prompt", symbol="AAPL")
sends2 = len(cached_client.chat.completions.calls)
check("identical prompt served from cache (no second network call)",
      r1.ok and r2.ok and r2.cached and sends2 == sends1)

# cache replay after a total chain failure: expire the entry so the fresh
# cache no longer short-circuits, then fail every provider.
llm_service._clients["groq"].chat.completions._responder = (
    lambda kw: (_ for _ in ()).throw(_HTTPError(500, "down")))
cached_client.chat.completions._responder = lambda kw: (_ for _ in ()).throw(_HTTPError(500, "down"))
for entry in llm_service._response_cache.values():
    entry["expires"] = 0.0
r3 = llm_service.call("news", system="s", user="identical prompt", symbol="AAPL")
check("total failure after a success -> cached result replay (fallback of last resort)",
      r3.ok and r3.cache_replay and r3.fallback_used)

# ===========================================================================
print("14. risk engine works without any LLM:")

risk_engine.init_db()
verdict = risk_engine.assess_trade(
    "AAPL", "buy",
    {"equity": 100000.0, "cash": 50000.0, "buying_power": 100000.0,
     "day_pl_pct": 0.0, "error": None},
    {}, indicators={"volatility_annualized": 0.2, "last_bar_date": None},
    market_clock={"is_open": True}, news_intelligence=None,
    proposed_notional_usd=2000.0,
)
check("healthy proposal approved with TRADE_APPROVED",
      verdict["approved"] and verdict["reason"] == "TRADE_APPROVED")
check("risk engine verdict is the standardized shape",
      set(verdict.keys()) >= {"approved", "reason", "checks"})

verdict = risk_engine.assess_trade(
    "AAPL", "buy",
    {"equity": 100000.0, "cash": 50000.0, "buying_power": 100000.0,
     "day_pl_pct": -5.0, "error": None},
    {}, indicators=None, market_clock={"is_open": True},
    proposed_notional_usd=2000.0,
)
check("MAX_DAILY_LOSS_REACHED rejects deterministically",
      not verdict["approved"] and verdict["reason"] == "MAX_DAILY_LOSS_REACHED")

verdict = risk_engine.assess_trade(
    "AAPL", "buy",
    {"equity": 100000.0, "cash": 50000.0, "buying_power": 100000.0,
     "day_pl_pct": 0.0, "error": None},
    {}, indicators=None, market_clock={"is_open": False},
    proposed_notional_usd=2000.0,
)
check("verifiably closed market rejects new BUYs",
      not verdict["approved"] and verdict["reason"] == "MARKET_CLOSED")

verdict = risk_engine.assess_trade(
    "AAPL", "buy",
    {"equity": 100000.0, "cash": 50000.0, "buying_power": 100000.0,
     "day_pl_pct": 0.0, "error": None},
    {}, indicators=None, market_clock={"is_open": True},
    news_intelligence={"event_risk": True, "event_type": "EARNINGS",
                       "event_risk_level": "HIGH"},
    proposed_notional_usd=2000.0,
)
check("HIGH event risk reduces the notional (configurable action)",
      verdict["approved"] and verdict["notional_reduced_by_event_risk"]
      and verdict["max_notional_usd"] < 10000.0)

ok_sl, _ = risk_engine.validate_stop_take_profit("buy", 100.0, stop_loss=95.0, take_profit=120.0)
bad_sl, _ = risk_engine.validate_stop_take_profit("buy", 100.0, stop_loss=105.0)
check("stop/take-profit validation is deterministic",
      ok_sl and not bad_sl)

# daily trade cap is persisted
risk_engine._cycle_orders.clear()
with risk_engine._conn() as conn:
    conn.execute("DELETE FROM risk_engine_trades")
for _ in range(int(config.settings.RISK_MAX_TRADES_PER_DAY)):
    risk_engine.mark_executed("AAPL", "buy", 100.0)
verdict = risk_engine.assess_trade(
    "AAPL", "buy",
    {"equity": 100000.0, "cash": 50000.0, "buying_power": 100000.0,
     "day_pl_pct": 0.0, "error": None},
    {}, indicators=None, market_clock={"is_open": True}, proposed_notional_usd=100.0,
)
check("MAX_TRADES_PER_DAY_REACHED rejects (persisted counter)",
      not verdict["approved"] and verdict["reason"] == "MAX_TRADES_PER_DAY_REACHED")
with risk_engine._conn() as conn:
    conn.execute("DELETE FROM risk_engine_trades")

# ===========================================================================
print("15-18. cycle integration: setup gate, risk-rejection gate, debate off:")

from agents import cio_agent  # noqa: E402
import main  # noqa: E402

RISING = _rising_series()
FLAT = _flat_series()


def _install_cycle_stubs(closes, day_pl_pct=0.0, cio_says_buy=True):
    """Installs Alpaca + LLM stubs for a full cycle run. Returns handles."""
    llm_service.reset_all_state()
    config.settings.LLM_CIRCUIT_BREAKER_ENABLED = True

    cio_calls = {"n": 0}
    debate_calls = {"n": 0}
    risk_llm_calls = {"n": 0}

    def responder(kwargs):
        sys_prompt = str(kwargs.get("messages", [{}])[0].get("content", ""))
        if "Chief Investment Officer" in sys_prompt:
            cio_calls["n"] += 1
            return json.dumps({"decision": "BUY" if cio_says_buy else "HOLD",
                               "confidence": 0.9,
                               "notional_usd": 9000.0,
                               "reasoning": "stubbed CIO decision"})
        if "bull vs. bear debate" in sys_prompt:
            debate_calls["n"] += 1
            return json.dumps({"bull_strength": 0.8, "bull_summary": "b",
                               "bear_strength": 0.2, "bear_summary": "s"})
        if "risk manager" in sys_prompt:
            risk_llm_calls["n"] += 1
            return json.dumps({"approved": True, "max_notional_usd": 99999.0,
                               "risk_level": "LOW", "reasoning": "ok"})
        return NEWS_ANALYSIS_JSON

    llm_service._clients["unorouter"] = _client(responder, UNO_MODELS)
    llm_service._clients["groq"] = _client(responder, GROQ_MODELS)
    llm_service._verified_models["unorouter"] = set(UNO_MODELS)
    llm_service._verified_models["groq"] = set(GROQ_MODELS)

    trading = _FakeTradingClient(day_pl_pct=day_pl_pct)
    alpaca_service._trading_client = trading
    alpaca_service._data_client = _bars_client(closes)
    main._get_agent_weights = lambda: {}
    main.agent_logs.clear()
    main.cycle_history.clear()
    main._previous_positions.clear()
    return {"cio": cio_calls, "debate": debate_calls, "risk_llm": risk_llm_calls,
            "trading": trading}


# Sanity: the rising fixture really is a BULLISH setup; flat is not.
alpaca_service._data_client = _bars_client(RISING)
ind_rising = alpaca_service.get_indicators("AAPL")
check("fixture sanity: rising series -> deterministic BULLISH setup",
      ind_rising.get("technical_signal") == "BULLISH")
alpaca_service._data_client = _bars_client(FLAT)
ind_flat = alpaca_service.get_indicators("AAPL")
check("fixture sanity: flat series -> no setup (NEUTRAL)",
      ind_flat.get("technical_signal") == "NEUTRAL")

# --- 16: CIO does not run without a valid setup ---------------------------
config.settings.CIO_AGENT_ENABLED = True
config.settings.ENABLE_DEBATE = False
config.settings.TECH_LLM_INTERPRETATION_ENABLED = False
h = _install_cycle_stubs(FLAT)
payload = asyncio.run(main.run_trading_cycle(triggered_by="test-no-setup"))
record = main.cycle_history[0]
decisions = {d["symbol"]: d for d in record["decisions"]}
check("no setup -> every decision HOLD with NO_TECHNICAL_SETUP",
      all(d["decision"] == "HOLD" and d.get("blocked_reason") == "NO_TECHNICAL_SETUP"
          for d in decisions.values()))
check("no setup -> ZERO CIO LLM calls (deterministic HOLD, no AI review)",
      h["cio"]["n"] == 0 and h["risk_llm"]["n"] == 0)
st = record["agent_status"]["AAPL"]
check("no setup: technical OK, news/risk/cio/debate SKIPPED",
      st["technical"] == "OK" and st["news"] == "SKIPPED"
      and st["risk"] == "SKIPPED" and st["cio"] == "SKIPPED"
      and st["debate"] == "SKIPPED")
check("no-setup cycle is a clean OK (not an error, not partial)",
      record["status"] == "OK")

# --- 17 + 15: CIO does not run after risk rejection; LLM cannot override ---
h = _install_cycle_stubs(RISING, day_pl_pct=0.0)
# Force a hard risk rejection via the persisted daily trade cap.
with risk_engine._conn() as conn:
    conn.execute("DELETE FROM risk_engine_trades")
for _ in range(int(config.settings.RISK_MAX_TRADES_PER_DAY)):
    risk_engine.mark_executed("AAPL", "buy", 100.0)

asyncio.run(main.run_trading_cycle(triggered_by="test-risk-reject"))
record = main.cycle_history[0]
decisions = {d["symbol"]: d for d in record["decisions"]}
check("risk engine rejection -> BUY held for every symbol (MAX_TRADES_PER_DAY_REACHED)",
      all(d["decision"] == "HOLD"
          and d.get("blocked_reason") == "MAX_TRADES_PER_DAY_REACHED"
          for d in decisions.values()))
check("after a risk rejection the CIO LLM is NEVER consulted",
      h["cio"]["n"] == 0)
check("after a risk rejection NO order is submitted",
      len(h["trading"].submitted) == 0 and record["orders"] == [])
st = record["agent_status"]["AAPL"]
check("risk-rejection path: risk OK (engine decided), cio SKIPPED",
      st["risk"] == "OK" and st["cio"] == "SKIPPED")
with risk_engine._conn() as conn:
    conn.execute("DELETE FROM risk_engine_trades")

# LLM-override test with the cap lifted but a CIO that wants to overspend:
# the CIO's notional is clamped to the engine/gate cap (existing hard rule),
# and a rejection still cannot be overridden because the CIO never runs.
h = _install_cycle_stubs(RISING, day_pl_pct=0.0, cio_says_buy=True)
asyncio.run(main.run_trading_cycle(triggered_by="test-override"))
record = main.cycle_history[0]
buys = [d for d in record["decisions"] if d["decision"] == "BUY"]
check("healthy setup + approval -> CIO runs and BUYs are capped by the gate",
      h["cio"]["n"] > 0 and all(d["notional_usd"] <= 2510.4 + 1e-6 for d in buys))

# --- 18: debate does not run when disabled ---------------------------------
config.settings.ENABLE_DEBATE = False
h = _install_cycle_stubs(RISING)
asyncio.run(main.run_trading_cycle(triggered_by="test-debate-off"))
record = main.cycle_history[0]
check("debate disabled (default) -> ZERO debate LLM calls",
      h["debate"]["n"] == 0
      and all(rec["debate"] == "SKIPPED"
              for sym in config.settings.TRADE_UNIVERSE
              for rec in [record["agent_status"][sym]]))
check("system works normally with debate disabled (cycle completed)",
      record["status"] in ("OK", "PARTIAL_ERROR") and len(record["decisions"]) == 5)

# --- debate eligibility: runs only with conflict + healthy provider --------
config.settings.ENABLE_DEBATE = True
h = _install_cycle_stubs(RISING)
asyncio.run(main.run_trading_cycle(triggered_by="test-debate-elig"))
# news intelligence for these symbols was seeded BULLISH earlier or is the
# no-intelligence fallback -> no conflicting evidence -> debate still skipped
check("debate enabled but no conflicting evidence -> still skipped",
      h["debate"]["n"] == 0)
config.settings.ENABLE_DEBATE = False

# ===========================================================================
print("19. no secrets in logs / API payloads:")

import io  # noqa: E402

SECRET = "sk-test-unorouter"
log_capture = io.StringIO()
handler = logging.StreamHandler(log_capture)
logging.getLogger().addHandler(handler)

_reset()
llm_service.call("news", system="s", user="secret-test-prompt", symbol="AAPL")
states = llm_service.provider_states()
payload = json.dumps(states) + json.dumps(news_worker.stats())
check("provider states + news status expose no API keys",
      SECRET not in payload and "sk-test-groq" not in payload)
check("API keys never appear in log output",
      SECRET not in log_capture.getvalue() and "sk-test-groq" not in log_capture.getvalue())
logging.getLogger().removeHandler(handler)

# providers/health shape (no secrets) via the service layer
health_states = llm_service.provider_states()
uno_health = health_states.get("unorouter", {})
check("provider health exposes circuit/failure/last-success fields",
      {"enabled", "state", "circuit", "failure_count",
       "last_success", "last_failure"} <= set(uno_health.keys()))
check("paper trading is enforced (no live option)",
      config.settings.ALPACA_PAPER is True
      and config.settings.ALPACA_BASE_URL == "https://paper-api.alpaca.markets"
      and config.settings.PAPER_TRADING_ONLY is True)

# ===========================================================================
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("ALL INTELLIGENCE-V2 TESTS PASSED")
