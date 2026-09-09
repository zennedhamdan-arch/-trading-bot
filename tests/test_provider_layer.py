"""
tests/test_provider_layer.py

Tests the common LLM provider layer (services/llm_service.py) and the
agent integrations built on it:

  1. Routing: every agent resolves (provider, model, fallback) from
     environment-driven settings at call time; no model ids live in agent
     source files.
  2. Request lifecycle: OK results carry provider/model/status/latency;
     provider failures map to MODEL_NOT_FOUND / PROVIDER_QUOTA_EXCEEDED /
     RATE_LIMITED / PROVIDER_ERROR / INVALID_RESPONSE / NOT_CONFIGURED.
  3. Quota handling: local rolling-24h budgets short-circuit requests;
     server quota-exhausted 429s set a backoff and are NEVER retried;
     transient 429s get exactly one bounded retry; unparseable responses
     are INVALID_RESPONSE, never fabricated.
  4. Fallbacks: only explicitly-configured AND startup-verified fallbacks
     are used; never for quota errors.
  5. Usage accounting: per-cycle counters per provider/agent/symbol.
  6. Startup validation: configured ids checked against live model lists.
  7. Agents: fail-safes (no fake AI decisions), Gemini call reuse for
     unchanged inputs, single debate call, risk blocking on provider error.
  8. Repo hygiene: no hardcoded model ids or SDK client construction
     inside agent files.

Run:  .venv/bin/python tests/test_provider_layer.py
"""

import json
import os
import subprocess
import sys
import time
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
# Stub transports
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


class _ErrWithCode(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


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


class _ModelsList:
    def __init__(self, ids):
        self.data = [types.SimpleNamespace(id=i) for i in ids]


def _make_client(responder, model_ids=("some-model",)):
    client = types.SimpleNamespace()
    client.chat = types.SimpleNamespace(completions=_Completions(responder))
    client.models = types.SimpleNamespace(list=lambda: _ModelsList(list(model_ids)))
    return client


TECH_JSON = json.dumps({"signal": "BULLISH", "confidence": 0.7, "summary": "Uptrend."})
DEBATE_JSON = json.dumps({"bull_strength": 0.7, "bull_summary": "b",
                          "bear_strength": 0.3, "bear_summary": "r"})
CIO_JSON = json.dumps({"decision": "HOLD", "confidence": 0.5, "notional_usd": 0,
                       "reasoning": "wait"})
RISK_JSON = json.dumps({"approved": True, "max_notional_usd": 500.0,
                        "risk_level": "LOW", "reasoning": "fine"})
NEWS_JSON = json.dumps({"sentiment": "BULLISH", "confidence": 0.6,
                        "summary": "positive", "key_headline": "beats"})
FUND_JSON = json.dumps({"signal": "NEUTRAL", "confidence": 0.5, "summary": "mixed"})


class _FakeInteraction:
    def __init__(self, output_text):
        self.output_text = output_text
        self.errors = None


class _FakeGeminiInteractions:
    def __init__(self, responder=None):
        self.calls = []
        self._responder = responder or (lambda kw: NEWS_JSON)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self._responder(kwargs)
        if isinstance(out, Exception):
            raise out
        return _FakeInteraction(out)


class _FakeGeminiClient:
    def __init__(self, responder=None):
        self.interactions = _FakeGeminiInteractions(responder)
        self.models = types.SimpleNamespace(list=lambda: _ModelsList(
            ["gemini-3.6-flash", "gemini-3.6-flash-lite"]))


# ---------------------------------------------------------------------------
import config
from services import llm_service, gemini_service
from agents import tech_agent, news_agent, fundamentals_agent, debate_agent, risk_agent, cio_agent

# Default settings for this suite
config.settings.GROQ_API_KEY = "test-groq"
config.settings.GEMINI_API_KEY = "test-gemini"
config.settings.OPENROUTER_API_KEY = "test-openrouter"
config.settings.GROQ_TECH_MODEL = "openai/gpt-oss-20b"
config.settings.GROQ_DEBATE_MODEL = "openai/gpt-oss-20b"
config.settings.GROQ_CIO_MODEL = "openai/gpt-oss-120b"
config.settings.OPENROUTER_RISK_MODEL = "openai/gpt-oss-20b:free"
config.settings.GEMINI_MODEL = "gemini-3.6-flash"
config.settings.GROQ_TECH_FALLBACK_MODEL = ""
config.settings.GROQ_DEBATE_FALLBACK_MODEL = ""
config.settings.GROQ_CIO_FALLBACK_MODEL = ""
config.settings.OPENROUTER_RISK_FALLBACK_MODEL = ""
config.settings.GEMINI_FALLBACK_MODEL = ""
config.settings.GEMINI_DAILY_REQUEST_LIMIT = 20
config.settings.GROQ_DAILY_REQUEST_LIMIT = 0
config.settings.OPENROUTER_DAILY_REQUEST_LIMIT = 0
config.settings.LLM_RATE_LIMIT_MAX_RETRIES = 1
config.settings.LLM_RATE_LIMIT_MAX_WAIT_SECONDS = 30
config.settings.ENABLE_DEBATE = True
config.settings.ENABLE_FUNDAMENTALS_AGENT = True
config.settings.NEWS_ANALYSIS_CACHE_TTL_MINUTES = 240
config.settings.FUNDAMENTALS_ANALYSIS_CACHE_TTL_MINUTES = 480

# Speed tests up: never actually sleep for retries
_sleeps = []
llm_service.time.sleep = lambda s: _sleeps.append(s)


def _fresh(responder, model_ids=("openai/gpt-oss-20b", "openai/gpt-oss-120b")):
    """Resets provider-layer state and installs a stub Groq client."""
    llm_service.reset_all_state()
    gemini_service._client = _FakeGeminiClient()
    client = _make_client(responder, model_ids)
    llm_service._clients["groq"] = client
    llm_service._clients["openrouter"] = _make_client(
        lambda kw: RISK_JSON, ("openai/gpt-oss-20b:free",))
    llm_service.reset_cycle_usage()
    return client


# ---------------------------------------------------------------------------
print("1. routing & configuration:")

routes = llm_service.llm_routes()
check("all 6 agents routed", set(routes) == {"technical", "debate", "cio", "risk", "news", "fundamentals"})
check("technical -> groq + env model", routes["technical"]["provider"] == "groq"
      and routes["technical"]["model"] == "openai/gpt-oss-20b")
check("cio -> groq 120b", routes["cio"]["model"] == "openai/gpt-oss-120b")
check("risk -> openrouter free model", routes["risk"]["provider"] == "openrouter"
      and routes["risk"]["model"] == "openai/gpt-oss-20b:free")
check("news/fundamentals -> gemini", routes["news"]["provider"] == "gemini"
      and routes["fundamentals"]["model"] == "gemini-3.6-flash")
check("defaults: no fallbacks configured", all(not r["fallback"] for r in routes.values()))

config.settings.GROQ_TECH_MODEL = "custom/model-x"
check("model ids resolved at call time from settings",
      llm_service.route_info("technical")["model"] == "custom/model-x")
config.settings.GROQ_TECH_MODEL = "openai/gpt-oss-20b"

res = subprocess.run(
    ["grep", "-rn", "-E", "llama-3|deepseek|gpt-oss|gemini-3", "--include=*.py", "agents/"],
    capture_output=True, text=True)
check("no model ids hardcoded in agent files", res.stdout.strip() == "")
if res.stdout.strip():
    print("      stale:", res.stdout.strip()[:400])

res = subprocess.run(
    ["grep", "-rn", "-E", "from groq import|Groq\\(|from openai import|OpenAI\\(|chat\\.completions\\.create",
     "--include=*.py", "agents/"],
    capture_output=True, text=True)
check("agents construct no SDK clients (all through llm_service)", res.stdout.strip() == "")
if res.stdout.strip():
    print("      stale:", res.stdout.strip()[:400])

check("llm_service owns the chat-completions call",
      "chat.completions.create" in open("services/llm_service.py").read())

# ---------------------------------------------------------------------------
print("2. request lifecycle:")

client = _fresh(lambda kw: TECH_JSON)
r = llm_service.call_json("technical", "SYS", "USER", temperature=0.2, max_tokens=400, symbol="AAPL")
check("OK result", r.ok and r.parsed["signal"] == "BULLISH")
check("result carries provider/model", r.provider == "groq" and r.model == "openai/gpt-oss-20b")
check("result carries latency", isinstance(r.latency_ms, (int, float)) and r.latency_ms >= 0)
check("request sent with system+user+model+params",
      client.chat.completions.calls[0]["model"] == "openai/gpt-oss-20b"
      and client.chat.completions.calls[0]["messages"][0]["content"] == "SYS"
      and client.chat.completions.calls[0]["messages"][1]["content"] == "USER"
      and client.chat.completions.calls[0]["temperature"] == 0.2
      and client.chat.completions.calls[0]["max_tokens"] == 400)

usage = llm_service.cycle_usage()
check("usage counted per provider", usage["groq"]["requests"] == 1 and usage["groq"]["ok"] == 1)
check("usage counted per agent", usage["groq"]["by_agent"]["technical"]["requests"] == 1)
check("usage counted per symbol", usage["groq"]["by_symbol"]["AAPL"]["requests"] == 1)

# model_not_found
client = _fresh(lambda kw: _ErrWithCode(
    "The model `openai/gpt-oss-20b` does not exist or you do not have access to it.", 404))
r = llm_service.call("technical", "SYS", "USER")
check("404 -> MODEL_NOT_FOUND", r.status == "MODEL_NOT_FOUND")
check("model_not_found is not retried", len(client.chat.completions.calls) == 1)
check("model error surfaces in message", "does not exist" in (r.error or ""))

# quota exhausted (daily metric, no retry delay)
client = _fresh(lambda kw: _ErrWithCode(
    "Quota exceeded for metric generate_content_free_tier_requests, limit: 20. "
    "You have exhausted your daily quota.", 429))
r = llm_service.call("technical", "SYS", "USER")
check("daily-quota 429 -> PROVIDER_QUOTA_EXCEEDED", r.status == "PROVIDER_QUOTA_EXCEEDED")
check("quota exhaustion never retried", len(client.chat.completions.calls) == 1)
r2 = llm_service.call("technical", "SYS", "USER")
check("backoff short-circuits later calls (no request sent)",
      len(client.chat.completions.calls) == 1 and r2.status == "PROVIDER_QUOTA_EXCEEDED"
      and "no request sent" in r2.error)
q = llm_service.quota_state()["groq"]
check("quota_state reports active backoff", q["backoff_active"] is True and q["backoff_seconds_remaining"] > 0)

# transient rate limit with short explicit delay -> exactly one retry
def _flaky(kwargs):
    if len(client.chat.completions.calls) == 1:
        return _ErrWithCode("Rate limit reached. Please retry in 2s.", 429)
    return TECH_JSON
client = _fresh(lambda kw: TECH_JSON)
client.chat.completions._responder = _flaky
_sleeps.clear()
r = llm_service.call("technical", "SYS", "USER")
check("transient 429 -> retried once, then OK",
      r.ok and len(client.chat.completions.calls) == 2 and _sleeps == [2.0])

# 429 with a retry delay beyond the configured max wait -> treated as quota pause
client = _fresh(lambda kw: _ErrWithCode("Rate limit reached. Please retry in 99999s.", 429))
r = llm_service.call("technical", "SYS", "USER")
check("429 with huge delay -> quota pause, not retried",
      r.status == "PROVIDER_QUOTA_EXCEEDED" and len(client.chat.completions.calls) == 1)

# local rolling-24h budget
llm_service.reset_all_state()
client = _fresh(lambda kw: TECH_JSON)
config.settings.GROQ_DAILY_REQUEST_LIMIT = 2
llm_service.call("technical", "SYS", "USER1")
llm_service.call("technical", "SYS", "USER2")
r = llm_service.call("technical", "SYS", "USER3")
check("local daily budget blocks the next request",
      r.status == "PROVIDER_QUOTA_EXCEEDED" and "budget" in r.error
      and len(client.chat.completions.calls) == 2)
config.settings.GROQ_DAILY_REQUEST_LIMIT = 0

# unparseable / empty responses
client = _fresh(lambda kw: "not json at all {{{")
r = llm_service.call_json("technical", "SYS", "USER")
check("unparseable response -> INVALID_RESPONSE", r.status == "INVALID_RESPONSE" and r.parsed is None)
usage = llm_service.cycle_usage()["groq"]
check("parse failure counted as error, not ok", usage["requests"] == 1 and usage["ok"] == 0
      and usage["errors"].get("INVALID_RESPONSE") == 1)

client = _fresh(lambda kw: "   ")
r = llm_service.call_json("technical", "SYS", "USER")
check("empty response -> INVALID_RESPONSE", r.status == "INVALID_RESPONSE")

# missing key
llm_service.reset_all_state()
llm_service.reset_cycle_usage()
saved_key = config.settings.GROQ_API_KEY
config.settings.GROQ_API_KEY = ""
r = llm_service.call("technical", "SYS", "USER")
check("missing key -> NOT_CONFIGURED, no request",
      r.status == "NOT_CONFIGURED" and "GROQ_API_KEY" in r.error)
config.settings.GROQ_API_KEY = saved_key

# generic exceptions are NOT misclassified as quota events
client = _fresh(lambda kw: TypeError("unsupported keyword 'proxies'"))
r = llm_service.call("technical", "SYS", "USER")
check("non-HTTP exception -> PROVIDER_ERROR (not quota)",
      r.status == "PROVIDER_ERROR" and "proxies" in r.error)

# ---------------------------------------------------------------------------
print("3. gemini transport through the provider layer:")

llm_service.reset_all_state()
llm_service.reset_cycle_usage()
gemini_service._client = _FakeGeminiClient(lambda kw: NEWS_JSON)
r = llm_service.call_json("news", "SYS-N", "USER-N", symbol="MSFT")
check("gemini route OK", r.ok and r.provider == "gemini" and r.model == "gemini-3.6-flash")
call_kwargs = gemini_service._client.interactions.calls[0]
check("Interactions API contract preserved",
      call_kwargs["system_instruction"] == "SYS-N" and call_kwargs["input"] == "USER-N"
      and call_kwargs["store"] is False and call_kwargs["model"] == "gemini-3.6-flash")

gemini_service._client = _FakeGeminiClient(
    lambda kw: _ErrWithCode("Quota exceeded for metric generate_content_free_tier_requests, "
                            "limit: 20", 429))
r = llm_service.call("news", "SYS", "USER")
check("gemini 429 -> PROVIDER_QUOTA_EXCEEDED", r.status == "PROVIDER_QUOTA_EXCEEDED")
r2 = llm_service.call("news", "SYS", "USER")
check("gemini backoff active -> next call not sent", r2.status == "PROVIDER_QUOTA_EXCEEDED")

config.settings.GEMINI_MODEL = "models/gemini-3.6-flash"
llm_service.reset_all_state()
gemini_service._client = _FakeGeminiClient(lambda kw: NEWS_JSON)
r = llm_service.call("news", "SYS", "USER")
check("models/ prefix stripped for the Interactions API",
      gemini_service._client.interactions.calls[0]["model"] == "gemini-3.6-flash")
config.settings.GEMINI_MODEL = "gemini-3.6-flash"

# local Gemini daily budget (free tier: 20/day)
llm_service.reset_all_state()
config.settings.GEMINI_DAILY_REQUEST_LIMIT = 1
gemini_service._client = _FakeGeminiClient(lambda kw: NEWS_JSON)
llm_service.call("news", "SYS", "U1")
r = llm_service.call("fundamentals", "SYS", "U2")
check("gemini local 20/day-style budget enforced",
      r.status == "PROVIDER_QUOTA_EXCEEDED"
      and len(gemini_service._client.interactions.calls) == 1)
config.settings.GEMINI_DAILY_REQUEST_LIMIT = 20

# ---------------------------------------------------------------------------
print("4. fallbacks (explicit + verified only):")

llm_service.reset_all_state()
llm_service.reset_cycle_usage()
client = _fresh(lambda kw: _ErrWithCode("model does not exist", 404),
                model_ids=("openai/gpt-oss-20b",))
config.settings.GROQ_TECH_FALLBACK_MODEL = "openai/gpt-oss-120b"
r = llm_service.call("technical", "SYS", "USER")
check("unverified fallback is NOT used",
      r.status == "MODEL_NOT_FOUND" and len(client.chat.completions.calls) == 1)

llm_service._verified_models["groq"] = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
r = llm_service.call("technical", "SYS", "USER")
check("verified fallback: still MODEL_NOT_FOUND (both stubbed models 404)",
      r.status == "MODEL_NOT_FOUND")
# call #1 (unverified) sent 1 request; call #2 (verified) sent primary + fallback
check("fallback attempt actually made", len(client.chat.completions.calls) == 3
      and client.chat.completions.calls[2]["model"] == "openai/gpt-oss-120b")

client = _fresh(lambda kw: TECH_JSON, model_ids=("openai/gpt-oss-20b", "openai/gpt-oss-120b"))
llm_service._verified_models["groq"] = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
r = llm_service.call("technical", "SYS", "USER")
check("primary OK -> fallback never touched",
      r.ok and r.model == "openai/gpt-oss-20b" and len(client.chat.completions.calls) == 1)

client = _fresh(lambda kw: _ErrWithCode("daily quota exhausted", 429),
                model_ids=("openai/gpt-oss-20b", "openai/gpt-oss-120b"))
llm_service._verified_models["groq"] = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
r = llm_service.call("technical", "SYS", "USER")
check("quota error -> NO fallback (same provider shares quota)",
      r.status == "PROVIDER_QUOTA_EXCEEDED" and len(client.chat.completions.calls) == 1)
config.settings.GROQ_TECH_FALLBACK_MODEL = ""

# ---------------------------------------------------------------------------
print("5. startup validation:")

llm_service.reset_all_state()
client = _fresh(lambda kw: TECH_JSON,
                model_ids=("openai/gpt-oss-20b", "openai/gpt-oss-120b"))
llm_service._clients["openrouter"] = _make_client(lambda kw: RISK_JSON, ("openai/gpt-oss-20b:free",))
report = llm_service.validate_models()
check("groq catalog fetched", report["providers"]["groq"]["ok"] is True
      and report["providers"]["groq"]["models_found"] == 2)
check("all configured groq models verified",
      report["providers"]["groq"]["checked"] == {"technical.model": True, "debate.model": True, "cio.model": True}
      and report["providers"]["groq"]["missing"] == [])
check("openrouter risk model verified",
      report["providers"]["openrouter"]["checked"].get("risk.model") is True)
check("gemini model verified against its catalog",
      report["providers"]["gemini"]["checked"].get("news.model") is True
      and report["providers"]["gemini"]["checked"].get("fundamentals.model") is True)

config.settings.GROQ_CIO_MODEL = "made-up/model"
report = llm_service.validate_models()
check("missing model reported", "cio.model: made-up/model" in report["providers"]["groq"]["missing"]
      and report["providers"]["groq"]["checked"]["cio.model"] is False)
config.settings.GROQ_CIO_MODEL = "openai/gpt-oss-120b"

saved_key = config.settings.OPENROUTER_API_KEY
config.settings.OPENROUTER_API_KEY = ""
report = llm_service.validate_models()
check("provider without key skipped cleanly",
      report["providers"]["openrouter"]["ok"] is False
      and "not configured" in report["providers"]["openrouter"]["error"])
config.settings.OPENROUTER_API_KEY = saved_key

# ---------------------------------------------------------------------------
print("6. agents on the provider layer:")

llm_service.reset_all_state()
llm_service.reset_cycle_usage()
fundamentals_agent.reset_fundamentals_cache()
news_agent.reset_news_analysis_cache()

# -- tech agent
client = _fresh(lambda kw: TECH_JSON)
tr = tech_agent.analyze_technicals("AAPL", {"rsi_14": 55, "sma_50": 100, "sma_200": 90,
                                            "macd": 1, "macd_signal": 0.5,
                                            "latest_close": 101, "recent_closes": [100], "error": None})
check("tech: fields mapped", tr["signal"] == "BULLISH" and tr["error"] is None)
check("tech: carries provider/model/llm_status/latency",
      tr["provider"] == "groq" and tr["model"] == "openai/gpt-oss-20b"
      and tr["llm_status"] == "OK" and tr["latency_ms"] is not None)
tr = tech_agent.analyze_technicals("AAPL", {"error": "DATA_UNAVAILABLE: no bars"})
check("tech: no data -> SKIPPED_NO_DATA, no LLM call",
      tr["llm_status"] == "SKIPPED_NO_DATA" and tr["error"] == "DATA_UNAVAILABLE: no bars"
      and len(client.chat.completions.calls) == 1)

# -- news agent + Gemini call reuse
gemini_service._client = _FakeGeminiClient(lambda kw: NEWS_JSON)
headlines = ["AAPL beats earnings", "AAPL announces buyback"]
n1 = news_agent.analyze_news("AAPL", headlines)
calls_after_first = len(gemini_service._client.interactions.calls)
check("news: fields mapped", n1["sentiment"] == "BULLISH" and n1["llm_status"] == "OK"
      and n1["cached"] is False)
n2 = news_agent.analyze_news("AAPL", list(headlines))  # same content, new list object
check("news: unchanged headlines -> cached analysis, NO new Gemini call",
      n2["cached"] is True and n2["llm_status"] == "OK"
      and len(gemini_service._client.interactions.calls) == calls_after_first)
n3 = news_agent.analyze_news("AAPL", headlines + ["Fresh headline changes everything"])
check("news: changed headlines -> new Gemini call",
      n3["cached"] is False and len(gemini_service._client.interactions.calls) == calls_after_first + 1)
n4 = news_agent.analyze_news("MSFT", [])
check("news: no headlines -> no LLM call, NEUTRAL",
      n4["llm_status"] == "SKIPPED_NO_DATA" and n4["sentiment"] == "NEUTRAL"
      and len(gemini_service._client.interactions.calls) == calls_after_first + 1)

# -- fundamentals agent + analysis reuse
fundamentals_agent.reset_fundamentals_cache()
metrics = {"symbol": "MSFT", "pe_ratio": 30.0, "forward_pe": 28.0, "revenue_growth": 0.1,
           "profit_margin": 0.3, "debt_to_equity": 0.5, "return_on_equity": 0.2, "error": None}
gemini_service._client = _FakeGeminiClient(lambda kw: FUND_JSON)
f1 = fundamentals_agent.analyze_fundamentals("MSFT", dict(metrics))
calls_f1 = len(gemini_service._client.interactions.calls)
f2 = fundamentals_agent.analyze_fundamentals("MSFT", dict(metrics))
check("fundamentals: unchanged metrics -> cached analysis, NO new call",
      f1["llm_status"] == "OK" and f2["cached"] is True
      and len(gemini_service._client.interactions.calls) == calls_f1)
f3 = fundamentals_agent.analyze_fundamentals("MSFT", {**metrics, "pe_ratio": 99.0})
check("fundamentals: changed metrics -> new call",
      f3["cached"] is False and len(gemini_service._client.interactions.calls) == calls_f1 + 1)
f4 = fundamentals_agent.analyze_fundamentals("MSFT", {"symbol": "MSFT", "error": "DATA_UNAVAILABLE: 429"})
check("fundamentals: data error -> SKIPPED_NO_DATA", f4["llm_status"] == "SKIPPED_NO_DATA"
      and f4["error"].startswith("DATA_UNAVAILABLE"))

# -- debate agent: ONE call per run, both sides from one response
client = _fresh(lambda kw: DEBATE_JSON)
d = debate_agent.run_debate("NVDA",
                            {"signal": "BULLISH", "summary": "up"},
                            {"sentiment": "BULLISH", "summary": "good"},
                            {"signal": "NEUTRAL", "summary": "mixed"})
check("debate: both sides mapped from ONE request",
      d["bull_strength"] == 0.7 and d["bear_strength"] == 0.3 and d["edge"] == 0.4
      and d["llm_status"] == "OK" and len(client.chat.completions.calls) == 1)
check("debate: request asks for both sides in one JSON",
      "bull_strength" in client.chat.completions.calls[0]["messages"][0]["content"]
      and "bear_strength" in client.chat.completions.calls[0]["messages"][0]["content"])

# -- risk agent: provider failure blocks the trade, no fake verdict
llm_service.reset_all_state()
llm_service._clients["openrouter"] = _make_client(
    lambda kw: _ErrWithCode("No allowed providers are available for `openai/gpt-oss-20b:free`.", 404),
    ("openai/gpt-oss-20b:free",))
acct = {"equity": 10000.0, "cash": 5000.0}
rr = risk_agent.assess_risk("AAPL", "buy", acct, None)
check("risk: model unavailable -> MODEL_NOT_FOUND + trade blocked",
      rr["llm_status"] == "MODEL_NOT_FOUND" and rr["approved"] is False
      and rr["max_notional_usd"] == 0.0 and rr["error"] is not None)
llm_service.reset_all_state()
llm_service._clients["openrouter"] = _make_client(
    lambda kw: _ErrWithCode("free tier daily quota exhausted", 429),
    ("openai/gpt-oss-20b:free",))
rr = risk_agent.assess_risk("AAPL", "buy", acct, None)
check("risk: quota exceeded -> blocked, clean status",
      rr["llm_status"] == "PROVIDER_QUOTA_EXCEEDED" and rr["approved"] is False)
llm_service.reset_all_state()
llm_service._clients["openrouter"] = _make_client(lambda kw: RISK_JSON, ("openai/gpt-oss-20b:free",))
rr = risk_agent.assess_risk("AAPL", "buy", {"equity": 1000.0, "cash": 200.0}, None)
check("risk: deterministic clamp still enforced (LLM said 500)",
      rr["approved"] is True and rr["max_notional_usd"] == 100.0)  # 10% of 1000, capped by cash 200

# -- CIO: LLM failure -> HOLD fail-safe; BUY clamped by risk
client = _fresh(lambda kw: _ErrWithCode("boom", 500))
cr = cio_agent.make_decision("AAPL", {"sentiment": "BULLISH"}, {"signal": "BULLISH"},
                             {"approved": True, "max_notional_usd": 500.0})
check("cio: provider error -> HOLD fail-safe, explicit error",
      cr["decision"] == "HOLD" and cr["notional_usd"] == 0.0 and cr["llm_status"] == "PROVIDER_ERROR"
      and cr["error"] is not None)
client = _fresh(lambda kw: json.dumps({"decision": "BUY", "confidence": 0.9, "notional_usd": 99999.0, "reasoning": "go"}))
cr = cio_agent.make_decision("AAPL", {"sentiment": "BULLISH"}, {"signal": "BULLISH"},
                             {"approved": True, "max_notional_usd": 500.0})
check("cio: BUY clamped to risk-approved notional", cr["decision"] == "BUY" and cr["notional_usd"] == 500.0)
client = _fresh(lambda kw: json.dumps({"decision": "BUY", "confidence": 0.9, "notional_usd": 100.0, "reasoning": "go"}))
cr = cio_agent.make_decision("AAPL", {"sentiment": "BULLISH"}, {"signal": "BULLISH"},
                             {"approved": False, "max_notional_usd": 0.0})
check("cio: BUY blocked when risk did not approve", cr["decision"] == "HOLD")

# ---------------------------------------------------------------------------
print("7. usage accounting across a simulated cycle:")

llm_service.reset_all_state()
llm_service.reset_cycle_usage()
fundamentals_agent.reset_fundamentals_cache()
news_agent.reset_news_analysis_cache()
client = _fresh(lambda kw: TECH_JSON)
gemini_service._client = _FakeGeminiClient(lambda kw: NEWS_JSON)

symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]
for sym in symbols:
    tech_agent.analyze_technicals(sym, {"rsi_14": 50, "error": None})
    news_agent.analyze_news(sym, [f"{sym} headline one", f"{sym} headline two"])
    fundamentals_agent.analyze_fundamentals(sym, {"symbol": sym, "pe_ratio": 20.0, "error": None})
    debate_agent.run_debate(sym, {"signal": "BULLISH", "summary": "u"}, {"sentiment": "BULLISH", "summary": "g"}, None)
    risk_agent.assess_risk(sym, "buy", {"equity": 10000.0, "cash": 5000.0}, None)
    cio_agent.make_decision(sym, {"sentiment": "BULLISH"}, {"signal": "BULLISH"},
                            {"approved": True, "max_notional_usd": 100.0})

usage = llm_service.cycle_usage()
check("groq: 3 calls/symbol (tech+debate+cio) = 15",
      usage["groq"]["requests"] == 15 and usage["groq"]["ok"] == 15)
check("gemini: 2 calls/symbol (news+fundamentals) = 10",
      usage["gemini"]["requests"] == 10 and usage["gemini"]["ok"] == 10)
check("openrouter: 1 call/symbol (risk) = 5",
      usage["openrouter"]["requests"] == 5 and usage["openrouter"]["ok"] == 5)
check("per-agent breakdown present",
      usage["groq"]["by_agent"]["cio"]["requests"] == 5
      and usage["gemini"]["by_agent"]["news"]["requests"] == 5
      and usage["openrouter"]["by_agent"]["risk"]["requests"] == 5)
check("per-symbol breakdown present", usage["groq"]["by_symbol"]["AAPL"]["requests"] == 3)

# ---------------------------------------------------------------------------
print("8. honest statuses map through main._stage_status:")
import main

check("OK report -> OK", main._stage_status({"error": None}) == "OK")
check("quota report -> UNAVAILABLE",
      main._stage_status({"error": "PROVIDER_QUOTA_EXCEEDED: ...", "llm_status": "PROVIDER_QUOTA_EXCEEDED"}) == "UNAVAILABLE")
check("model-not-found report -> UNAVAILABLE",
      main._stage_status({"error": "MODEL_NOT_FOUND: ...", "llm_status": "MODEL_NOT_FOUND"}) == "UNAVAILABLE")
check("not-configured report -> UNAVAILABLE",
      main._stage_status({"error": "GROQ_API_KEY not configured.", "llm_status": "NOT_CONFIGURED"}) == "UNAVAILABLE")
check("other provider error -> ERROR",
      main._stage_status({"error": "PROVIDER_ERROR: boom", "llm_status": "PROVIDER_ERROR"}) == "ERROR")
check("legacy report without llm_status -> ERROR",
      main._stage_status({"error": "something failed"}) == "ERROR")
check("None report -> SKIPPED", main._stage_status(None) == "SKIPPED")

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("ALL PROVIDER-LAYER TESTS PASSED")
