"""
tests/test_provider_layer.py

Tests the centralized LLM router (services/llm_service.py) — the single
place where every agent's LLM request is routed, quota-managed,
circuit-broken and accounted:

  1. Routing: task -> (provider, model) resolved from env settings at call
     time; no model ids or SDK clients in agent source files; NVIDIA and
     OpenRouter are optional providers; complete(task=..., messages=...).
  2. Request lifecycle: OK results carry provider/model/status/latency;
     failures map to MODEL_NOT_FOUND / PROVIDER_QUOTA_EXCEEDED /
     RATE_LIMITED / PROVIDER_ERROR / AUTH_ERROR / NETWORK_ERROR /
     INVALID_RESPONSE / NOT_CONFIGURED.
  3. Circuit breakers: 429-quota opens a provider circuit (no retries, no
     per-symbol storms); 404 opens a per-model circuit (same request never
     re-sent); 401/403 opens an auth circuit; local rolling-24h budgets
     short-circuit before sending; transient 429s get one bounded retry.
  4. Global fallback route: only when explicitly configured AND verified
     live; never on quota errors; never silently.
  5. Startup validation against live model catalogs; provider_states().
  6. Usage accounting per provider/agent/symbol.
  7. Agent fail-safes on the router (no fabricated AI decisions).
  8. main._stage_status maps router statuses to honest pipeline statuses.

Run:  .venv/bin/python tests/test_provider_layer.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Isolated SQLite for this run (news intelligence shares the memory DB; a
# shared DB would poison duplicate-detection across runs).
os.environ["MEMORY_DB_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="provlayer-"), "test-memory.db")

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
config.settings.NVIDIA_API_KEY = "test-nvidia"
config.settings.GROQ_TECH_MODEL = "openai/gpt-oss-20b"
config.settings.GROQ_DEBATE_MODEL = "openai/gpt-oss-20b"
config.settings.GROQ_RISK_MODEL = "openai/gpt-oss-20b"
config.settings.GROQ_CIO_MODEL = "openai/gpt-oss-120b"
config.settings.OPENROUTER_MODEL = ""
config.settings.NVIDIA_MODEL = "meta/llama-test"
config.settings.NVIDIA_BASE_URL = "https://integrate.test.nvidia.example/v1"
config.settings.GEMINI_MODEL = "gemini-3.6-flash"
config.settings.LLM_FALLBACK_PROVIDER = ""
config.settings.LLM_FALLBACK_MODEL = ""
config.settings.LLM_TECH_PROVIDER = "groq"
config.settings.LLM_DEBATE_PROVIDER = "groq"
config.settings.LLM_RISK_PROVIDER = "groq"
config.settings.LLM_CIO_PROVIDER = "groq"
config.settings.LLM_NEWS_PROVIDER = "gemini"
config.settings.LLM_FUNDAMENTALS_PROVIDER = "gemini"
config.settings.GEMINI_DAILY_REQUEST_LIMIT = 20
config.settings.GROQ_DAILY_REQUEST_LIMIT = 0
config.settings.OPENROUTER_DAILY_REQUEST_LIMIT = 0
config.settings.NVIDIA_DAILY_REQUEST_LIMIT = 0
config.settings.LLM_RATE_LIMIT_MAX_RETRIES = 1
config.settings.LLM_RATE_LIMIT_MAX_WAIT_SECONDS = 30
config.settings.MODEL_UNAVAILABLE_COOLDOWN_MINUTES = 30
config.settings.AUTH_COOLDOWN_MINUTES = 60
config.settings.ENABLE_DEBATE = True
config.settings.ENABLE_FUNDAMENTALS_AGENT = True
config.settings.NEWS_ANALYSIS_CACHE_TTL_MINUTES = 240
config.settings.FUNDAMENTALS_ANALYSIS_CACHE_TTL_MINUTES = 480

# Never actually sleep for retries in tests
_sleeps = []
llm_service.time.sleep = lambda s: _sleeps.append(s)

GROQ_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")
NVIDIA_MODELS = ("meta/llama-test",)
OPENROUTER_MODELS = ("openai/gpt-oss-20b:free",)


def _fresh(responder=None, model_ids=GROQ_MODELS):
    """Resets router state and installs stub clients for all providers."""
    llm_service.reset_all_state()
    gemini_service._client = _FakeGeminiClient()
    client = _make_client(responder or (lambda kw: TECH_JSON), model_ids)
    llm_service._clients["groq"] = client
    llm_service._clients["nvidia"] = _make_client(lambda kw: TECH_JSON, NVIDIA_MODELS)
    llm_service._clients["openrouter"] = _make_client(lambda kw: RISK_JSON, OPENROUTER_MODELS)
    llm_service.reset_cycle_usage()
    return client


# ---------------------------------------------------------------------------
print("1. routing & configuration:")

routes = llm_service.llm_routes()
check("all 6 tasks routed", set(routes) == {"technical", "debate", "cio", "risk", "news", "fundamentals"})
check("technical/debate/risk/cio -> groq by default",
      all(routes[t]["provider"] == "groq" for t in ("technical", "debate", "risk", "cio")))
check("tech+debate+risk use the verified 20b model, cio the 120b model",
      routes["technical"]["model"] == "openai/gpt-oss-20b"
      and routes["debate"]["model"] == "openai/gpt-oss-20b"
      and routes["risk"]["model"] == "openai/gpt-oss-20b"
      and routes["cio"]["model"] == "openai/gpt-oss-120b")
check("news/fundamentals -> gemini by default",
      routes["news"]["provider"] == "gemini" and routes["fundamentals"]["provider"] == "gemini")
check("openrouter optional: no model routed by default", config.settings.OPENROUTER_MODEL == "")

# provider overrides resolved at call time
config.settings.LLM_RISK_PROVIDER = "nvidia"
check("risk can be routed to NVIDIA via env", llm_service.route_info("risk") ==
      {"provider": "nvidia", "model": "meta/llama-test",
       "fallback_provider": "", "fallback_model": ""})
config.settings.LLM_RISK_PROVIDER = "groq"

config.settings.GROQ_TECH_MODEL = "custom/model-x"
check("model ids resolved at call time from settings",
      llm_service.route_info("technical")["model"] == "custom/model-x")
config.settings.GROQ_TECH_MODEL = "openai/gpt-oss-20b"

# unknown provider -> honest PROVIDER_ERROR (no silent normalization)
config.settings.LLM_TECH_PROVIDER = "mistral"
r = llm_service.call("technical", "SYS", "USER")
check("unknown provider -> PROVIDER_ERROR with config hint",
      r.status == "PROVIDER_ERROR" and "unknown LLM provider 'mistral'" in r.error)
config.settings.LLM_TECH_PROVIDER = "groq"

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
check("agents construct no SDK clients (all through the router)", res.stdout.strip() == "")
if res.stdout.strip():
    print("      stale:", res.stdout.strip()[:400])

check("router owns the chat-completions call",
      "chat.completions.create" in open("services/llm_service.py").read())

# ---------------------------------------------------------------------------
print("2. request lifecycle:")

client = _fresh()
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

# complete(): messages-style API
client = _fresh()
r = llm_service.complete(task="debate", messages=[
    {"role": "system", "content": "DEB-SYS"},
    {"role": "user", "content": "DEB-USER"},
])
check("complete(task, messages) works and maps roles",
      r.ok and client.chat.completions.calls[0]["messages"][0]["content"] == "DEB-SYS"
      and client.chat.completions.calls[0]["messages"][1]["content"] == "DEB-USER")

# model_not_found
client = _fresh(lambda kw: _ErrWithCode(
    "The model `openai/gpt-oss-20b` does not exist or you do not have access to it.", 404))
r = llm_service.call("technical", "SYS", "USER")
check("404 -> MODEL_NOT_FOUND", r.status == "MODEL_NOT_FOUND")
check("model error surfaces in message", "does not exist" in (r.error or ""))

# quota exhausted (daily metric, no retry delay)
client = _fresh(lambda kw: _ErrWithCode(
    "Quota exceeded for metric generate_content_free_tier_requests, limit: 20. "
    "You have exhausted your daily quota.", 429))
r = llm_service.call("technical", "SYS", "USER")
check("daily-quota 429 -> PROVIDER_QUOTA_EXCEEDED", r.status == "PROVIDER_QUOTA_EXCEEDED")
check("quota exhaustion never retried", len(client.chat.completions.calls) == 1)
r2 = llm_service.call("technical", "SYS", "USER")
check("quota circuit short-circuits later calls (no request sent)",
      len(client.chat.completions.calls) == 1 and r2.status == "PROVIDER_QUOTA_EXCEEDED"
      and "no request sent" in r2.error)
check("provider_states reports QUOTA_EXHAUSTED",
      llm_service.provider_states()["groq"]["state"] == "QUOTA_EXHAUSTED")

# 404 circuit: same model never re-sent within the cooldown
llm_service.reset_all_state()
client = _fresh(lambda kw: _ErrWithCode(
    "This model is unavailable for free. The paid version is available now.", 404))
r1 = llm_service.call("technical", "SYS", "USER")
r2 = llm_service.call("technical", "SYS", "USER2")
check("404 circuit: second call short-circuits without a request",
      r1.status == "MODEL_NOT_FOUND" and r2.status == "MODEL_NOT_FOUND"
      and len(client.chat.completions.calls) == 1)
check("provider_states reports MODEL_UNAVAILABLE",
      llm_service.provider_states()["groq"]["state"] == "MODEL_UNAVAILABLE")
other = llm_service.call("cio", "SYS", "USER")  # different model on same provider
check("404 circuit is per-model (cio model still attempted)",
      len(client.chat.completions.calls) == 2 and other.model == "openai/gpt-oss-120b")

# auth circuit
llm_service.reset_all_state()
client = _fresh(lambda kw: _ErrWithCode("invalid api key", 401))
r1 = llm_service.call("technical", "SYS", "USER")
r2 = llm_service.call("cio", "SYS", "USER")
check("401 -> AUTH_ERROR, provider circuit opens, no repeated auth",
      r1.status == "AUTH_ERROR" and r2.status == "AUTH_ERROR"
      and len(client.chat.completions.calls) == 1)
check("provider_states reports AUTH_ERROR",
      llm_service.provider_states()["groq"]["state"] == "AUTH_ERROR")

# network errors classified (real httpx transport types)
import httpx as _httpx
client = _fresh(lambda kw: _httpx.ConnectError("connection closed"))
r = llm_service.call("technical", "SYS", "USER")
check("transport error -> NETWORK_ERROR", r.status == "NETWORK_ERROR")

# transient rate limit with short explicit delay -> exactly one retry
client = _fresh(lambda kw: TECH_JSON)
def _flaky(kwargs):
    if len(client.chat.completions.calls) == 1:
        return _ErrWithCode("Rate limit reached. Please retry in 2s.", 429)
    return TECH_JSON
client.chat.completions._responder = _flaky
_sleeps.clear()
r = llm_service.call("technical", "SYS", "USER")
check("transient 429 -> retried once, then OK",
      r.ok and len(client.chat.completions.calls) == 2 and _sleeps == [2.0])

# 429 with a huge retry delay -> quota pause, not retried
client = _fresh(lambda kw: _ErrWithCode("Rate limit reached. Please retry in 99999s.", 429))
r = llm_service.call("technical", "SYS", "USER")
check("429 with huge delay -> quota pause, not retried",
      r.status == "PROVIDER_QUOTA_EXCEEDED" and len(client.chat.completions.calls) == 1)

# local rolling-24h budget
llm_service.reset_all_state()
client = _fresh()
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
check("missing key -> NOT_CONFIGURED, no request, legacy message shape",
      r.status == "NOT_CONFIGURED" and r.error == "GROQ_API_KEY not configured.")
config.settings.GROQ_API_KEY = saved_key

# empty model (optional provider not configured for a task)
config.settings.LLM_RISK_PROVIDER = "openrouter"
r = llm_service.call("risk", "SYS", "USER")
check("routed provider without a model -> NOT_CONFIGURED naming the env var",
      r.status == "NOT_CONFIGURED" and "OPENROUTER_MODEL" in r.error)
config.settings.LLM_RISK_PROVIDER = "groq"

# generic exceptions are NOT misclassified as quota events
client = _fresh(lambda kw: TypeError("unsupported keyword 'proxies'"))
r = llm_service.call("technical", "SYS", "USER")
check("non-HTTP exception -> PROVIDER_ERROR (not quota)",
      r.status == "PROVIDER_ERROR" and "proxies" in r.error)

# ---------------------------------------------------------------------------
print("3. gemini transport through the router:")

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
r3 = llm_service.call("fundamentals", "SYS", "USER")
check("gemini quota circuit: later agents/symbols short-circuit (no storm)",
      r2.status == "PROVIDER_QUOTA_EXCEEDED" and r3.status == "PROVIDER_QUOTA_EXCEEDED"
      and len(gemini_service._client.interactions.calls) == 1)
check("gemini provider state QUOTA_EXHAUSTED",
      llm_service.provider_states()["gemini"]["state"] == "QUOTA_EXHAUSTED")

# server retry_after honored for the circuit window
llm_service.reset_all_state()
gemini_service._client = _FakeGeminiClient(
    lambda kw: _ErrWithCode("Quota exceeded ... daily quota. Please retry in 3600s.", 429))
llm_service.call("news", "SYS", "USER")
q = llm_service.provider_states()["gemini"]
check("server retry_after honored (cooldown ~3600s)",
      3500 <= q["quota_cooldown_remaining_s"] <= 3600)

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
print("4. nvidia (optional secondary provider):")

llm_service.reset_all_state()
llm_service.reset_cycle_usage()
nvidia_client = llm_service._clients["nvidia"] = _make_client(
    lambda kw: TECH_JSON, NVIDIA_MODELS)
config.settings.LLM_CIO_PROVIDER = "nvidia"
r = llm_service.call("cio", "SYS", "USER")
check("task routed to NVIDIA via env", r.ok and r.provider == "nvidia"
      and r.model == "meta/llama-test")
check("NVIDIA request went through the OpenAI-compatible client",
      nvidia_client.chat.completions.calls[0]["model"] == "meta/llama-test")
config.settings.LLM_CIO_PROVIDER = "groq"

# nvidia not required to start
llm_service.reset_all_state()
saved = config.settings.NVIDIA_API_KEY
config.settings.NVIDIA_API_KEY = ""
states = llm_service.provider_states()
check("nvidia without key -> NOT_CONFIGURED (optional)",
      states["nvidia"]["state"] == "NOT_CONFIGURED")
config.settings.NVIDIA_API_KEY = saved

# ---------------------------------------------------------------------------
print("5. global fallback route (explicit + verified only):")

llm_service.reset_all_state()
llm_service.reset_cycle_usage()
client = _fresh(lambda kw: _ErrWithCode("model does not exist", 404))
config.settings.LLM_FALLBACK_PROVIDER = "nvidia"
config.settings.LLM_FALLBACK_MODEL = "meta/llama-test"

r = llm_service.call("technical", "SYS", "USER")
check("unverified fallback is NOT used",
      r.status == "MODEL_NOT_FOUND" and len(client.chat.completions.calls) == 1)

llm_service._verified_models["nvidia"] = set(NVIDIA_MODELS)
nvidia_client = llm_service._clients["nvidia"]
r = llm_service.call("technical", "SYS", "USER")
check("verified fallback used after primary 404 (cross-provider)",
      r.ok and r.provider == "nvidia" and r.fallback_used is True
      and len(nvidia_client.chat.completions.calls) == 1)
check("fallback attempt actually made on the nvidia client",
      nvidia_client.chat.completions.calls[0]["model"] == "meta/llama-test")

client = _fresh(lambda kw: TECH_JSON)
llm_service._verified_models["nvidia"] = set(NVIDIA_MODELS)
r = llm_service.call("technical", "SYS", "USER")
check("primary OK -> fallback never touched",
      r.ok and r.model == "openai/gpt-oss-20b" and not r.fallback_used)

client = _fresh(lambda kw: _ErrWithCode("daily quota exhausted", 429))
llm_service._verified_models["nvidia"] = set(NVIDIA_MODELS)
r = llm_service.call("technical", "SYS", "USER")
# V2 semantics: quota exhaustion is never RETIED on the same provider (its
# circuit opens), but the chain DOES advance to the next configured member.
check("quota error -> provider circuit opens, chain advances (no same-provider retry)",
      r.ok and r.provider == "nvidia" and r.fallback_used is True
      and llm_service.provider_states()["groq"]["state"] == "QUOTA_EXHAUSTED"
      and len(llm_service._clients["groq"].chat.completions.calls) == 1)
r2 = llm_service.call("technical", "SYS", "USER2")
check("quota circuit short-circuits the SAME provider (no second groq request)",
      len(llm_service._clients["groq"].chat.completions.calls) == 1)
config.settings.LLM_FALLBACK_PROVIDER = ""
config.settings.LLM_FALLBACK_MODEL = ""

# ---------------------------------------------------------------------------
print("6. startup validation + provider states:")

llm_service.reset_all_state()
client = _fresh(model_ids=GROQ_MODELS)
llm_service._clients["nvidia"] = _make_client(lambda kw: TECH_JSON, NVIDIA_MODELS)
llm_service._clients["openrouter"] = _make_client(lambda kw: RISK_JSON, OPENROUTER_MODELS)
report = llm_service.validate_models()
check("groq catalog fetched", report["providers"]["groq"]["ok"] is True
      and report["providers"]["groq"]["models_found"] == 2)
check("all groq models verified (tech/debate/risk/cio)",
      report["providers"]["groq"]["checked"] == {
          "technical.model": True, "debate.model": True,
          "risk.model": True, "cio.model": True}
      and report["providers"]["groq"]["missing"] == [])
check("gemini model verified against its catalog",
      report["providers"]["gemini"]["checked"].get("news.model") is True
      and report["providers"]["gemini"]["checked"].get("fundamentals.model") is True)
check("nvidia catalog fetched", report["providers"]["nvidia"]["ok"] is True)
check("openrouter has no model to check (optional, unset)",
      report["providers"]["openrouter"]["checked"] == {})

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

# fallback model validated too
config.settings.LLM_FALLBACK_PROVIDER = "nvidia"
config.settings.LLM_FALLBACK_MODEL = "meta/llama-test"
llm_service._verified_models.clear()
llm_service._clients["nvidia"] = _make_client(lambda kw: TECH_JSON, NVIDIA_MODELS)
report = llm_service.validate_models()
check("fallback model checked against the fallback provider's catalog",
      report["providers"]["nvidia"]["checked"].get("technical.fallback") is True)
config.settings.LLM_FALLBACK_PROVIDER = ""
config.settings.LLM_FALLBACK_MODEL = ""

states = llm_service.provider_states()
check("provider_states covers all five providers (unorouter added)",
      set(states) == {"unorouter", "groq", "nvidia", "gemini", "openrouter"})
check("unorouter NOT_CONFIGURED without a key (honest, not hidden)",
      states["unorouter"]["state"] == "NOT_CONFIGURED")
check("healthy providers report READY",
      states["groq"]["state"] == "READY" and states["gemini"]["state"] == "READY")

# ---------------------------------------------------------------------------
print("7. agents on the router:")

llm_service.reset_all_state()
llm_service.reset_cycle_usage()
fundamentals_agent.reset_fundamentals_cache()
news_agent.reset_news_analysis_cache()

# tech agent — V2: deterministic by default, LLM interpretation opt-in
config.settings.TECH_LLM_INTERPRETATION_ENABLED = False
client = _fresh()
tr = tech_agent.analyze_technicals("AAPL", {"rsi_14": 55, "sma_50": 100, "sma_200": 90,
                                            "macd": 1, "macd_signal": 0.5, "sma_20": 99,
                                            "volatility_annualized": 0.3, "max_drawdown": -0.1,
                                            "technical_signal": "BULLISH",
                                            "technical_components": {"trend": "BULLISH",
                                                                     "momentum": "BULLISH",
                                                                     "rsi_flag": "NEUTRAL"},
                                            "latest_close": 101, "recent_closes": [100],
                                            "feed": "iex", "error": None})
check("tech (deterministic mode): rule-engine verdict, ZERO LLM calls",
      tr["signal"] == "BULLISH" and tr["evidence_status"] == "AVAILABLE"
      and tr["llm_status"] == "SKIPPED_DETERMINISTIC"
      and tr["provider"] == "deterministic"
      and len(client.chat.completions.calls) == 0)

config.settings.TECH_LLM_INTERPRETATION_ENABLED = True
client = _fresh()
tr = tech_agent.analyze_technicals("AAPL", {"rsi_14": 55, "sma_50": 100, "sma_200": 90,
                                            "macd": 1, "macd_signal": 0.5, "sma_20": 99,
                                            "volatility_annualized": 0.3, "max_drawdown": -0.1,
                                            "technical_signal": "BULLISH",
                                            "latest_close": 101, "recent_closes": [100],
                                            "feed": "iex", "error": None})
check("tech: fields mapped", tr["signal"] == "BULLISH" and tr["error"] is None)
check("tech: carries provider/model/llm_status/latency",
      tr["provider"] == "groq" and tr["model"] == "openai/gpt-oss-20b"
      and tr["llm_status"] == "OK" and tr["latency_ms"] is not None)
check("tech: deterministic evidence sent to the LLM",
      "volatility" in client.chat.completions.calls[0]["messages"][1]["content"]
      and "Rule-based signal" in client.chat.completions.calls[0]["messages"][1]["content"])
tr = tech_agent.analyze_technicals("AAPL", {"error": "DATA_UNAVAILABLE: no bars"})
check("tech: no data -> SKIPPED_NO_DATA, no LLM call",
      tr["llm_status"] == "SKIPPED_NO_DATA" and tr["error"] == "DATA_UNAVAILABLE: no bars"
      and len(client.chat.completions.calls) == 1)
config.settings.TECH_LLM_INTERPRETATION_ENABLED = False  # restore V2 default

# news agent V2: the agent reads the persistent intelligence cache; the
# LLM work happens in news_intelligence (driven by the worker). The stub
# returns the V2 BATCH analysis shape the worker's prompt asks for.
from services import news_intelligence  # noqa: E402
config.settings.LLM_NEWS_PROVIDER = "gemini"
llm_service.reset_all_state()
NEWS_BATCH_JSON = json.dumps({
    "articles": [{"index": 0, "sentiment": "BULLISH", "confidence": 0.7,
                  "importance": "HIGH", "impact_horizon": "SHORT_TERM"},
                 {"index": 1, "sentiment": "BULLISH", "confidence": 0.6,
                  "importance": "MEDIUM", "impact_horizon": "SHORT_TERM"}],
    "overall_sentiment": "BULLISH", "overall_confidence": 0.65,
    "summary": "Coverage is positive.",
})
gemini_service._client = _FakeGeminiClient(lambda kw: NEWS_BATCH_JSON)

ARTICLES_A = [
    {"id": "pl-a1", "headline": "AAPL beats earnings", "summary": "",
     "source": "Reuters", "published_at": "2026-09-10T10:00:00Z"},
    {"id": "pl-a2", "headline": "AAPL announces buyback", "summary": "",
     "source": "CNBC", "published_at": "2026-09-10T09:00:00Z"},
]
news_intelligence.refresh_symbol("AAPL", fetch_articles=lambda s: ARTICLES_A)
calls_after_first = len(gemini_service._client.interactions.calls)
n1 = news_agent.analyze_news("AAPL")
check("news: fields mapped from the intelligence cache",
      n1["sentiment"] == "BULLISH" and n1["llm_status"] == "OK"
      and n1["cached"] is True and n1["headline_count"] == 2)
n2 = news_agent.analyze_news("AAPL")
check("news: repeated reads -> still cached, NO new Gemini call",
      n2["cached"] is True and n2["llm_status"] == "OK"
      and len(gemini_service._client.interactions.calls) == calls_after_first)
news_intelligence.refresh_symbol("AAPL", fetch_articles=lambda s: ARTICLES_A + [
    {"id": "pl-a3", "headline": "AAPL fresh headline changes everything",
     "summary": "", "source": "Bloomberg", "published_at": "2026-09-10T11:00:00Z"}])
check("news: NEW important article -> one new Gemini call (worker path)",
      len(gemini_service._client.interactions.calls) == calls_after_first + 1)
n4 = news_agent.analyze_news("ZZZZ")
check("news: no intelligence -> no LLM call, null stance + UNAVAILABLE evidence",
      n4["llm_status"] == "SKIPPED_NO_DATA" and n4["sentiment"] is None
      and n4["evidence_status"] == "UNAVAILABLE"
      and len(gemini_service._client.interactions.calls) == calls_after_first + 1)

# fundamentals agent consumes the normalized provider payload
fundamentals_agent.reset_fundamentals_cache()
metrics = {"status": "OK", "symbol": "MSFT", "provider": "fake", "pe_ratio": 30.0,
           "eps": 2.5, "revenue": None, "profit_margin": 0.3, "roe": 0.2,
           "debt_to_equity": 0.5, "market_cap": None, "timestamp": 0.0}
gemini_service._client = _FakeGeminiClient(lambda kw: FUND_JSON)
f1 = fundamentals_agent.analyze_fundamentals("MSFT", dict(metrics))
calls_f1 = len(gemini_service._client.interactions.calls)
f2 = fundamentals_agent.analyze_fundamentals("MSFT", dict(metrics))
check("fundamentals: unchanged metrics -> cached analysis, NO new call",
      f1["llm_status"] == "OK" and f2["cached"] is True
      and len(gemini_service._client.interactions.calls) == calls_f1)
check("fundamentals: unavailable fields labeled, not fabricated",
      "not available" in gemini_service._client.interactions.calls[calls_f1 - 1]["input"])
f4 = fundamentals_agent.analyze_fundamentals("MSFT", {"status": "DATA_UNAVAILABLE",
                                                      "symbol": "MSFT", "provider": "none",
                                                      "reason": "NO_PROVIDER_CONFIGURED"})
check("fundamentals: no provider -> DATA_UNAVAILABLE fail-safe",
      f4["llm_status"] == "SKIPPED_NO_DATA"
      and f4["error"] == "DATA_UNAVAILABLE: NO_PROVIDER_CONFIGURED"
      and len(gemini_service._client.interactions.calls) == calls_f1)

# debate: ONE call per run
client = _fresh(lambda kw: DEBATE_JSON)
d = debate_agent.run_debate("NVDA", {"signal": "BULLISH", "summary": "up"},
                            {"sentiment": "BULLISH", "summary": "good"},
                            {"signal": "NEUTRAL", "summary": "mixed"})
check("debate: both sides from ONE request",
      d["bull_strength"] == 0.7 and d["bear_strength"] == 0.3 and d["edge"] == 0.4
      and d["llm_status"] == "OK" and len(client.chat.completions.calls) == 1)

# risk: deterministic gate + LLM veto-only
client = _fresh(lambda kw: RISK_JSON)
acct = {"equity": 10000.0, "cash": 5000.0, "buying_power": 5000.0}
rr = risk_agent.assess_risk("AAPL", "buy", acct, None,
                            {"volatility_annualized": 0.2, "max_drawdown": -0.1})
check("risk: deterministic gate computes the cap (LLM said 500, cash/cap allows 1000)",
      rr["max_notional_usd"] == 500.0 and rr["approved"] is True)
check("risk: deterministic evidence included", rr["deterministic"]["equity"] == 10000.0)

client = _fresh(lambda kw: json.dumps({"approved": True, "max_notional_usd": 99999.0,
                                        "risk_level": "LOW", "reasoning": "go"}))
rr = risk_agent.assess_risk("AAPL", "buy", acct, None)
check("risk: LLM cannot exceed the deterministic cap (99999 -> 1000)",
      rr["max_notional_usd"] == 1000.0 and rr["approved"] is True)

client = _fresh(lambda kw: json.dumps({"approved": False, "max_notional_usd": 0,
                                        "risk_level": "HIGH", "reasoning": "veto"}))
rr = risk_agent.assess_risk("AAPL", "buy", acct, None)
check("risk: LLM veto blocks the trade", rr["approved"] is False and rr["max_notional_usd"] == 0.0)

client = _fresh(lambda kw: _ErrWithCode("provider exploded", 500))
rr = risk_agent.assess_risk("AAPL", "buy", acct, None)
check("risk: LLM failure -> deterministic gate stands, error surfaced",
      rr["deterministic"]["approved"] is True and rr["error"] is not None
      and rr["llm_status"] == "PROVIDER_ERROR")

# pure deterministic mode (no model configured)
config.settings.GROQ_RISK_MODEL = ""
rr = risk_agent.assess_risk("AAPL", "buy", acct, None)
check("risk: no model configured -> deterministic-only, no LLM call",
      rr["llm_status"] == "SKIPPED_DETERMINISTIC" and rr["approved"] is True
      and rr["deterministic"]["max_notional_usd"] == 1000.0)
config.settings.GROQ_RISK_MODEL = "openai/gpt-oss-20b"

# cio: LLM failure -> HOLD fail-safe; BUY clamped
client = _fresh(lambda kw: _ErrWithCode("boom", 500))
cr = cio_agent.make_decision("AAPL", {"sentiment": "BULLISH"}, {"signal": "BULLISH"},
                             {"approved": True, "max_notional_usd": 500.0})
check("cio: provider error -> HOLD fail-safe, explicit error",
      cr["decision"] == "HOLD" and cr["notional_usd"] == 0.0 and cr["llm_status"] == "PROVIDER_ERROR"
      and cr["error"] is not None)
client = _fresh(lambda kw: json.dumps({"decision": "BUY", "confidence": 0.9,
                                       "notional_usd": 99999.0, "reasoning": "go"}))
cr = cio_agent.make_decision("AAPL", {"sentiment": "BULLISH"}, {"signal": "BULLISH"},
                             {"approved": True, "max_notional_usd": 500.0})
check("cio: BUY clamped to risk-approved notional", cr["decision"] == "BUY" and cr["notional_usd"] == 500.0)
client = _fresh(lambda kw: json.dumps({"decision": "BUY", "confidence": 0.9,
                                       "notional_usd": 100.0, "reasoning": "go"}))
cr = cio_agent.make_decision("AAPL", {"sentiment": "BULLISH"}, {"signal": "BULLISH"},
                             {"approved": False, "max_notional_usd": 0.0})
check("cio: BUY blocked when risk did not approve", cr["decision"] == "HOLD")

# ---------------------------------------------------------------------------
print("8. usage accounting + honest stage statuses:")

llm_service.reset_all_state()
llm_service.reset_cycle_usage()
fundamentals_agent.reset_fundamentals_cache()
news_agent.reset_news_analysis_cache()
client = _fresh()
gemini_service._client = _FakeGeminiClient(lambda kw: NEWS_BATCH_JSON)

symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]
# V2: the tech agent's default path is deterministic (no LLM); enable the
# optional interpretation to exercise usage accounting on the LLM path.
config.settings.TECH_LLM_INTERPRETATION_ENABLED = True
for sym in symbols:
    tech_agent.analyze_technicals(sym, {"rsi_14": 50, "error": None,
                                        "technical_signal": "BULLISH",
                                        "technical_components": {}, "latest_close": 100.0,
                                        "sma_50": 99.0, "sma_200": 95.0})
    arts = [{"id": f"pl8-{sym}", "headline": f"{sym} beats earnings expectations",
             "summary": "", "source": "Reuters", "published_at": "2026-09-10T10:00:00Z"}]
    news_intelligence.refresh_symbol(sym, fetch_articles=lambda _s, _a=arts: _a)
config.settings.TECH_LLM_INTERPRETATION_ENABLED = False

usage = llm_service.cycle_usage()
check("groq: 1 tech call/symbol = 5", usage["groq"]["requests"] == 5)
check("gemini: 1 news call/symbol = 5 (via the news worker path)",
      usage["gemini"]["requests"] == 5 and usage["gemini"]["by_agent"]["news"]["requests"] == 5)

import main
check("OK report -> OK", main._stage_status({"error": None}) == "OK")
check("quota report -> UNAVAILABLE",
      main._stage_status({"error": "PROVIDER_QUOTA_EXCEEDED: ...", "llm_status": "PROVIDER_QUOTA_EXCEEDED"}) == "UNAVAILABLE")
check("model-not-found report -> UNAVAILABLE",
      main._stage_status({"error": "MODEL_NOT_FOUND: ...", "llm_status": "MODEL_NOT_FOUND"}) == "UNAVAILABLE")
check("auth report -> UNAVAILABLE",
      main._stage_status({"error": "AUTH_ERROR: ...", "llm_status": "AUTH_ERROR"}) == "UNAVAILABLE")
check("network report -> UNAVAILABLE",
      main._stage_status({"error": "NETWORK_ERROR: ...", "llm_status": "NETWORK_ERROR"}) == "UNAVAILABLE")
check("not-configured report -> UNAVAILABLE",
      main._stage_status({"error": "GROQ_API_KEY not configured.", "llm_status": "NOT_CONFIGURED"}) == "UNAVAILABLE")
check("other provider error -> ERROR",
      main._stage_status({"error": "PROVIDER_ERROR: boom", "llm_status": "PROVIDER_ERROR"}) == "ERROR")
check("legacy report without llm_status -> ERROR",
      main._stage_status({"error": "something failed"}) == "ERROR")
check("None report -> SKIPPED", main._stage_status(None) == "SKIPPED")
check("error type map: quota -> QUOTA_EXCEEDED",
      main._llm_error_type("PROVIDER_QUOTA_EXCEEDED") == "QUOTA_EXCEEDED")

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("ALL PROVIDER-LAYER TESTS PASSED")
