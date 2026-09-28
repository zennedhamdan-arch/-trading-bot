"""
tests/test_llm_quota_management.py

LLM request VOLUME / quota management — the required proofs:

  1. Duplicate article        -> 0 additional LLM calls
  2. Cached news intelligence -> 0 additional LLM calls
  3. Health checks            -> 0 chat-completion calls (catalog only)
  4. 429                      -> circuit opens
  5. Calls during cooldown    -> 0 network calls
  6. Cooldown expires         -> exactly ONE controlled retry
  7. Multiple agents consume the same cached analysis (no re-analysis)
  8. Overlapping cycles       -> prevented (single-flight guard)

Plus: per-model 60s minimum interval (fail-over, never blocking), global
per-cycle send budget, /api/llm/usage payload, per-request logging with
provider/model/agent/symbol/reason and never an API key.

Run:  .venv/bin/python tests/test_llm_quota_management.py
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

# Isolated SQLite (request log + news intelligence share the memory DB).
_TMP = tempfile.mkdtemp(prefix="quotafix-")
os.environ["MEMORY_DB_PATH"] = os.path.join(_TMP, "test-memory.db")

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}")


# ---------------------------------------------------------------------------
# Fakes: OpenAI-compatible clients counting EVERY network call
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


class _HTTPError(Exception):
    def __init__(self, status_code, message=""):
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


class _Completions:
    def __init__(self, responder):
        self._responder = responder
        self.calls = []          # every chat.completions.create (network)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self._responder(kwargs)
        if isinstance(out, Exception):
            raise out
        return _Completion(out)


class _ModelsList:
    def __init__(self, ids, delay=0.0):
        self._ids = list(ids)
        self._delay = delay
        self.calls = 0

    def list(self):
        self.calls += 1
        if self._delay:
            time.sleep(self._delay)
        return types.SimpleNamespace(
            data=[types.SimpleNamespace(id=i) for i in self._ids])


def _client(model_ids, responder=None):
    client = types.SimpleNamespace()
    client.chat = types.SimpleNamespace(
        completions=_Completions(responder or (lambda kw: json.dumps({"ok": True}))))
    client.models = _ModelsList(model_ids)
    return client


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def _capture():
    cap = _LogCapture()
    logging.getLogger("llm_service").addHandler(cap)
    return cap


# ---------------------------------------------------------------------------
# Config + imports
# ---------------------------------------------------------------------------

import config  # noqa: E402

config.settings.UNOROUTER_ENABLED = True
config.settings.UNOROUTER_API_KEY = "sk-test-unorouter-SECRET"
config.settings.GROQ_ENABLED = True
config.settings.GROQ_API_KEY = ""            # keyed later per-section
config.settings.GEMINI_API_KEY = ""
config.settings.OPENROUTER_API_KEY = ""
config.settings.NVIDIA_API_KEY = ""
config.settings.ALPACA_API_KEY = ""
config.settings.ALPACA_SECRET_KEY = ""
config.settings.ENABLE_MEMORY = False
config.settings.FUNDAMENTALS_PROVIDER = "none"
config.settings.LLM_FALLBACK_PROVIDER = ""
config.settings.LLM_FALLBACK_MODEL = ""
config.settings.LLM_RATE_LIMIT_MAX_RETRIES = 0

from services import llm_service, news_intelligence, health_service  # noqa: E402
from agents import news_agent, risk_agent, cio_agent  # noqa: E402

PRIMARY = "glm-5.3-flash-thinking:free"
FALLBACKS = ["glm-5.3-flash:free", "glm-5.3-flash-think-search:free"]
UNO_IDS = [PRIMARY] + FALLBACKS + ["qwen3.8-27b:free"]

NEWS_BATCH = json.dumps({
    "articles": [{"index": 0, "sentiment": "BULLISH", "confidence": 0.7,
                  "importance": "HIGH", "impact_horizon": "SHORT_TERM"}],
    "overall_sentiment": "BULLISH", "overall_confidence": 0.7,
    "summary": "Coverage positive.",
})
CIO_BUY = json.dumps({"decision": "BUY", "confidence": 0.9,
                      "notional_usd": 500.0, "reasoning": "stub"})


def _install_uno(responder=None, model_ids=UNO_IDS):
    llm_service.reset_all_state()
    client = _client(model_ids, responder)
    llm_service._clients["unorouter"] = client
    return client


def _send_models(client):
    return [c["model"] for c in client.chat.completions.calls]


# ===========================================================================
print("1. duplicate article -> 0 additional LLM calls:")

config.settings.LLM_MODEL_MIN_INTERVAL_SECONDS_UNOROUTER = 0   # isolate dedup
uno = _install_uno(lambda kw: NEWS_BATCH)
ARTS = [{"id": "qa-1", "headline": "AAPL beats earnings expectations",
         "summary": "", "source": "Reuters", "published_at": "2026-09-20T10:00:00Z"}]
r1 = news_intelligence.refresh_symbol("AAPL", fetch_articles=lambda s: ARTS)
sends_after_first = len(uno.chat.completions.calls)
check("first refresh: the new important article is analyzed once",
      r1["articles_analyzed"] == 1 and sends_after_first == 1)
r2 = news_intelligence.refresh_symbol("AAPL", fetch_articles=lambda s: ARTS)
check("duplicate article: 0 additional LLM calls (persistent dedup)",
      r2["duplicates_ignored"] == 1 and r2["articles_analyzed"] == 0
      and len(uno.chat.completions.calls) == sends_after_first)

# ===========================================================================
print("2. cached news intelligence -> 0 additional LLM calls:")

for _ in range(3):
    n = news_agent.analyze_news("AAPL")
check("3 agent reads of the cached intelligence: 0 additional LLM calls",
      len(uno.chat.completions.calls) == sends_after_first
      and n["cached"] is True and n["sentiment"] == "BULLISH")

# ===========================================================================
print("7. multiple agents consume the SAME cached analysis:")

uno = _install_uno(lambda kw: NEWS_BATCH if "news" in str(kw.get("model")) else CIO_BUY)
# Actually: distinguish by system prompt (the router passes agent prompts).
RISK_OK = json.dumps({"approved": True, "max_notional_usd": 500.0,
                      "risk_level": "LOW", "reasoning": "within limits"})


def _agent_responder(kwargs):
    system = str(kwargs.get("messages", [{}])[0].get("content", ""))
    if "financial news analyst" in system:
        return NEWS_BATCH
    if "risk manager" in system:
        return RISK_OK
    return CIO_BUY


uno = _install_uno(_agent_responder)
news_intelligence.refresh_symbol("MSFT", fetch_articles=lambda s: [
    {"id": "qb-1", "headline": "Microsoft beats earnings expectations",
     "summary": "", "source": "Reuters", "published_at": "2026-09-20T10:00:00Z"}])


def _news_sends():
    return len([c for c in uno.chat.completions.calls
                if "financial news analyst" in str(c["messages"][0]["content"])])


check("news analyzed exactly once for the symbol", _news_sends() == 1)

# risk agent (deterministic-first) consumes the cached NEWS REPORT; its own
# reasoning request is a separate task — the news itself is never re-analyzed
news_report = news_agent.analyze_news("MSFT")
risk_report = risk_agent.assess_risk(
    "MSFT", "buy",
    {"equity": 100000.0, "cash": 50000.0, "buying_power": 100000.0,
     "day_pl_pct": 0.0, "error": None},
    None, indicators={"volatility_annualized": 0.2, "latest_close": 100.0,
                      "error": None},
)
check("risk agent consumed the cached news report without re-analyzing it",
      _news_sends() == 1 and risk_report["approved"] is True)

# CIO consumes the SAME cached news report (its decision call is a different,
# legitimate request — the news itself is still never re-analyzed)
cio1 = cio_agent.make_decision(
    "MSFT", news_report,
    {"agent": "technical", "symbol": "MSFT", "evidence_status": "AVAILABLE",
     "signal": "BULLISH", "confidence": 0.7, "summary": "up", "error": None},
    risk_report,
)
check("CIO consumed the same cached news analysis (0 extra news calls)",
      _news_sends() == 1 and cio1["decision"] == "BUY")
total_after_cio1 = len(uno.chat.completions.calls)
cio2 = cio_agent.make_decision(
    "MSFT", news_report,
    {"agent": "technical", "symbol": "MSFT", "evidence_status": "AVAILABLE",
     "signal": "BULLISH", "confidence": 0.7, "summary": "up", "error": None},
    risk_report,
)
check("identical CIO prompt -> served from the response cache (0 extra sends)",
      len(uno.chat.completions.calls) == total_after_cio1
      and cio2["decision"] == cio1["decision"])
check("news task total across all consumers: exactly ONE call",
      _news_sends() == 1)

# ===========================================================================
print("per-model minimum interval (60s) — fail-over, never blocking:")

config.settings.LLM_MODEL_MIN_INTERVAL_SECONDS_UNOROUTER = 60.0
uno = _install_uno(lambda kw: NEWS_BATCH)
llm_service.reset_cycle_usage()
r1 = llm_service.call("news", system="s", user="prompt-A", symbol="AAPL")
t0 = time.monotonic()
r2 = llm_service.call("news", system="s", user="prompt-B", symbol="AAPL")
elapsed = time.monotonic() - t0
check("first call -> primary model; second within 60s -> NOT sent to it",
      r1.ok and _send_models(uno).count(PRIMARY) == 1)
check("second call fails over to the next model instantly (never blocks)",
      r2.ok and r2.fallback_used and elapsed < 2.0
      and _send_models(uno)[-1] != PRIMARY)
check("no model received two requests inside the interval",
      all(_send_models(uno).count(m) == 1 for m in set(_send_models(uno))))
config.settings.LLM_MODEL_MIN_INTERVAL_SECONDS_UNOROUTER = 0

# ===========================================================================
print("4-6. 429 -> circuit opens; cooldown = 0 calls; expiry -> ONE retry:")

uno = _install_uno(lambda kw: _HTTPError(
    429, "Rate limit exceeded — daily quota exhausted. Please retry in 3600s"))
llm_service.reset_cycle_usage()
r = llm_service.call("news", system="s", user="q1", symbol="AAPL")
check("429 -> PROVIDER_QUOTA_EXCEEDED, one network call made",
      not r.ok and r.status == "PROVIDER_QUOTA_EXCEEDED"
      and len(uno.chat.completions.calls) == 1)
state = llm_service.provider_states()["unorouter"]
check("quota circuit OPEN with a cooldown remaining",
      state["state"] == "QUOTA_EXHAUSTED"
      and state["quota_cooldown_remaining_s"] > 0)

for i in range(5):
    llm_service.call("news", system="s", user=f"cooldown-{i}", symbol="AAPL")
check("5 calls during cooldown -> 0 additional network calls (no hammering)",
      len(uno.chat.completions.calls) == 1)

# cooldown expiry: exactly ONE controlled retry (which may 429 again)
llm_service._quota_backoff_until["unorouter"] = time.monotonic() - 1.0
llm_service.call("news", system="s", user="after-cooldown", symbol="AAPL")
check("cooldown expired -> exactly ONE controlled retry (not a storm)",
      len(uno.chat.completions.calls) == 2)
check("a second 429 re-opens the circuit (window re-armed)",
      llm_service.provider_states()["unorouter"]["state"] == "QUOTA_EXHAUSTED")
for i in range(3):
    llm_service.call("news", system="s", user=f"cooldown2-{i}", symbol="AAPL")
check("still protected after the retry failed (0 further calls)",
      len(uno.chat.completions.calls) == 2)

# ===========================================================================
print("global per-cycle send budget:")

uno = _install_uno(lambda kw: NEWS_BATCH)
config.settings.LLM_CYCLE_MAX_REQUESTS = 3
llm_service.reset_cycle_usage()
for i in range(3):
    r = llm_service.call("news", system="s", user=f"budget-{i}", symbol="AAPL")
check("budget respected: exactly 3 network sends",
      r.ok and len(uno.chat.completions.calls) == 3)
r4 = llm_service.call("news", system="s", user="budget-4", symbol="AAPL")
check("beyond the budget -> CYCLE_BUDGET_EXCEEDED, 0 further sends",
      not r4.ok and r4.status == "CYCLE_BUDGET_EXCEEDED"
      and len(uno.chat.completions.calls) == 3)
config.settings.LLM_CYCLE_MAX_REQUESTS = 12

# ===========================================================================
print("3. health checks -> 0 chat calls (catalog /v1/models only):")

uno = _install_uno(lambda kw: NEWS_BATCH)
llm_service.reset_cycle_usage()
health_service._startup_report = None          # force a fresh startup check
startup = health_service.run_startup_checks()
live = health_service.live_health()
providers_validation = llm_service.validate_models(refresh_if_stale=True)
usage = llm_service.usage_status()
check("startup + live health + validation + usage: 0 chat-completion calls",
      len(uno.chat.completions.calls) == 0)
check("model catalog (/v1/models) WAS consulted (allowed)",
      uno.models.calls >= 1)

# ===========================================================================
print("8. overlapping cycles are prevented:")

import main  # noqa: E402

for attr in ("get_account_summary", "get_open_positions", "get_clock",
             "get_recent_orders"):
    pass  # stubbed below

main.alpaca_service.get_account_summary = lambda: {
    "cash": 50000.0, "equity": 100000.0, "buying_power": 100000.0,
    "day_pl_pct": 0.0, "error": None}
main.alpaca_service.get_open_positions = lambda: {"positions": [], "error": None}
main.alpaca_service.get_clock = lambda: {"is_open": True, "error": None}
main.alpaca_service.get_recent_orders = lambda limit=50: {"orders": [], "error": None}
main.alpaca_service.get_indicators = lambda s, lookback_days=250: {
    "symbol": s, "latest_close": 100.0, "volatility_annualized": 0.25,
    "max_drawdown": -0.1, "technical_signal": "NEUTRAL",
    "technical_components": {}, "recent_closes": [100.0], "error": None}
main.market_data_service.get_symbol_data = lambda s: {
    "symbol": s, "price": 100.0, "price_source": "stub",
    "indicators": main.alpaca_service.get_indicators(s),
    "news": [], "fundamentals": {"status": "DATA_UNAVAILABLE",
                                 "reason": "NO_PROVIDER_CONFIGURED"},
    "data_quality": {"price": "OK", "bars": "OK", "news": "DELEGATED_TO_WORKER",
                     "fundamentals": "DATA_UNAVAILABLE"},
    "feed": "IEX"}
main._get_agent_weights = lambda: {}
main.agent_logs.clear()
main.cycle_history.clear()
main._previous_positions.clear()

# 8a. guard semantics: a second entry while a cycle is in flight is skipped
main._cycle_running = True
before = len(main.cycle_history)
skipped = asyncio.run(main.run_trading_cycle(triggered_by="scheduler"))
check("second entry while a cycle runs -> SKIPPED_OVERLAP (no execution)",
      skipped["status"] == "SKIPPED_OVERLAP" and len(main.cycle_history) == before)
main._cycle_running = False

# 8b. REAL concurrency: a slow in-flight cycle + a concurrent trigger
_real_inner = main._run_trading_cycle_inner


async def _slow_inner(triggered_by="scheduler"):
    await asyncio.sleep(0.15)     # a genuinely slow cycle stage
    return await _real_inner(triggered_by)


main._run_trading_cycle_inner = _slow_inner
main.cycle_history.clear()


async def _concurrent():
    t1 = asyncio.create_task(main.run_trading_cycle(triggered_by="scheduler"))
    await asyncio.sleep(0.05)     # t1 is now in-flight (flag set)
    t2 = asyncio.create_task(main.run_trading_cycle(triggered_by="run-now"))
    return await asyncio.gather(t1, t2)


first, second = asyncio.run(_concurrent())
main._run_trading_cycle_inner = _real_inner
check("concurrent trigger while in-flight -> skipped, not overlapped",
      first["status"] != "SKIPPED_OVERLAP"
      and second["status"] == "SKIPPED_OVERLAP")
check("exactly ONE cycle record was produced (no double accounting)",
      len(main.cycle_history) == 1)

# scheduler job config: single-instance, coalesced (source-level contract)
_src = open("main.py").read()
check("scheduled cycle job configured max_instances=1 + coalesce",
      'id="trading_cycle"' in _src and "max_instances=1" in _src
      and "coalesce=True" in _src)

# ===========================================================================
print("usage endpoint payload + request logging (no secrets):")

cap = _capture()
uno = _install_uno(lambda kw: NEWS_BATCH)
llm_service.reset_cycle_usage()
llm_service.call("news", system="s", user="usage-1", symbol="AAPL",
                 reason="news_batch_analysis")
llm_service.call("news", system="s", user="usage-1", symbol="AAPL",
                 reason="news_batch_analysis")   # cache hit
usage = llm_service.usage_status()
required = {"requests_this_cycle", "requests_last_hour", "requests_today",
            "429_count", "cache_hits", "cache_misses",
            "current_circuit_state", "calls_by_agent", "calls_by_model",
            "cooldown_remaining"}
check("/api/llm/usage payload carries every required field",
      required <= set(usage.keys()))
check("requests_last_hour / requests_today counted from the persisted log",
      usage["requests_last_hour"] >= 1 and usage["requests_today"] >= 1)
check("cache_hits and cache_misses counted (hit rate available)",
      usage["cache_hits"] >= 1 and usage["cache_misses"] >= 1
      and usage["cache_hit_rate"] is not None)
check("calls_by_agent / calls_by_model populated",
      usage["calls_by_agent"].get("news", 0) >= 1
      and any("glm-5.3" in m for m in usage["calls_by_model"]))
check("current_circuit_state carries per-provider state + cooldown",
      all({"state", "cooldown_remaining_s"} <= set(v.keys())
          for v in usage["current_circuit_state"].values()))

req_lines = [m for m in cap.records if m.startswith("LLM_REQUEST")]
check("every request logged with provider/model/agent/symbol/reason",
      any("provider=unorouter" in m and f"model={PRIMARY}" in m
          and "agent=news" in m and "symbol=AAPL" in m
          and "reason=news_batch_analysis" in m for m in req_lines))
check("no API key ever appears in logs or usage payload",
      "sk-test-unorouter-SECRET" not in "\n".join(cap.records)
      and "sk-test-unorouter-SECRET" not in json.dumps(usage))
logging.getLogger("llm_service").removeHandler(cap)

# the endpoint exists on the app
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("main_routes", "main.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
routes = {r.path for r in m.app.routes if hasattr(r, "path")}
check("/api/llm/usage route registered", "/api/llm/usage" in routes)

# ===========================================================================
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("ALL LLM QUOTA MANAGEMENT TESTS PASSED")
