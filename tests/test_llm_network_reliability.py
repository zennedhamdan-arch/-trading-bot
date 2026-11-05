"""
tests/test_llm_network_reliability.py — LLM provider circuit-breaker and
network-reliability regressions (all transports mocked; zero real API quota):

  1.  Obsolete fallback id: recorded as obsolete by a fetched catalog and
      EXCLUDED FROM ROUTING — including after the catalog becomes
      unavailable (it cannot re-enter the chain on a timeout).
  2.  An obsolete/dead model id cannot open or re-open the PROVIDER circuit
      (model-scoped failure), so it cannot poison the models that work.
  3.  NVIDIA-style transport failures preserve the ORIGINAL category:
      CONNECTION / TIMEOUT / TLS / HTTP_5XX / RATE_LIMIT / MODEL_NOT_FOUND /
      EMPTY_RESPONSE / MALFORMED_RESPONSE, with the exception class name.
  4.  A failed NVIDIA request falls back to a VERIFIED working provider
      (Groq) in ONE bounded hop, with no unlimited provider hopping.
  5.  All providers failing yields one honest bounded failure.
  6.  Successful UnoRouter inference updates INFERENCE health independently
      of catalog health (catalog UNAVAILABLE, inference VERIFIED).
  7.  Health distinguishes configured / catalog-matched / selected /
      successfully-tested models, and exposes obsolete ids.
  8.  Bounded retries are charged to the per-cycle send budget (a retry can
      never exceed LLM_CYCLE_MAX_REQUESTS).
  9.  No API key material in reports, diagnostics or last-error evidence.

Run:  python3 tests/test_llm_network_reliability.py
"""

import json
import os
import ssl
import sys
import time
import types

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MEMORY_DB_PATH", "/tmp/llm-network-reliability.db")

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}")


import config                                                     # noqa: E402
from services import llm_service, health_service                  # noqa: E402

_SAVED = {attr: getattr(config.settings, attr) for attr in (
    "UNOROUTER_ENABLED", "UNOROUTER_API_KEY", "UNOROUTER_PRIMARY_MODEL",
    "UNOROUTER_FALLBACK_MODELS", "GROQ_ENABLED", "GROQ_API_KEY",
    "GROQ_FALLBACK_MODEL", "GROQ_CIO_MODEL", "GEMINI_API_KEY",
    "OPENROUTER_API_KEY", "NVIDIA_API_KEY", "NVIDIA_MODEL", "NVIDIA_BASE_URL",
    "LLM_TECH_PROVIDER", "LLM_CIO_PROVIDER", "LLM_NEWS_PROVIDER",
    "LLM_FALLBACK_PROVIDER", "LLM_FALLBACK_MODEL", "LLM_MAX_MODEL_ATTEMPTS",
    "LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "LLM_CIRCUIT_BREAKER_BACKOFF_SECONDS",
    "LLM_CATALOG_TIMEOUT_SECONDS", "LLM_CATALOG_CACHE_TTL_MINUTES",
    "LLM_RATE_LIMIT_MAX_RETRIES", "LLM_RATE_LIMIT_MAX_WAIT_SECONDS",
    "LLM_CYCLE_MAX_REQUESTS", "LLM_MODEL_MIN_INTERVAL_SECONDS_UNOROUTER",
    "MODEL_UNAVAILABLE_COOLDOWN_MINUTES", "ALPACA_API_KEY", "ALPACA_SECRET_KEY")}


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


class _Models:
    def __init__(self, ids=None, error=None):
        self._ids = list(ids or [])
        self._error = error
        self.calls = 0

    def list(self):
        self.calls += 1
        if self._error:
            raise self._error
        return types.SimpleNamespace(
            data=[types.SimpleNamespace(id=i) for i in self._ids])


def _client(model_ids=None, responder=None, models_error=None):
    client = types.SimpleNamespace()
    client.models = _Models(model_ids, error=models_error)
    sends = []

    class _Completions:
        def create(self, **kwargs):
            sends.append(kwargs)
            if responder is not None:
                out = responder(kwargs)
                if isinstance(out, Exception):
                    raise out
                return _Completion(out)
            return _Completion(json.dumps({"ok": True}))

    client.chat = types.SimpleNamespace(completions=_Completions())
    client.sends = sends
    return client


def _reset(**overrides):
    for attr, value in _SAVED.items():
        setattr(config.settings, attr, value)
    config.settings.UNOROUTER_ENABLED = True
    config.settings.UNOROUTER_API_KEY = "sk-test-uno-KEY"
    config.settings.UNOROUTER_PRIMARY_MODEL = "glm-5.3-flash-thinking:free"
    config.settings.UNOROUTER_FALLBACK_MODELS = [
        "glm-5.3-flash:free", "glm-5.3-flash-think-search:free"]
    config.settings.GROQ_API_KEY = ""
    config.settings.GEMINI_API_KEY = ""
    config.settings.OPENROUTER_API_KEY = ""
    config.settings.NVIDIA_API_KEY = ""
    config.settings.LLM_FALLBACK_PROVIDER = ""
    config.settings.LLM_FALLBACK_MODEL = ""
    config.settings.LLM_MAX_MODEL_ATTEMPTS = 3
    config.settings.LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3
    config.settings.LLM_RATE_LIMIT_MAX_RETRIES = 0
    config.settings.LLM_CYCLE_MAX_REQUESTS = 12
    config.settings.LLM_MODEL_MIN_INTERVAL_SECONDS_UNOROUTER = 0
    config.settings.LLM_CATALOG_TIMEOUT_SECONDS = 1.0
    config.settings.LLM_CATALOG_CACHE_TTL_MINUTES = 8
    for attr, value in overrides.items():
        setattr(config.settings, attr, value)
    llm_service.reset_all_state()
    return llm_service


def _catalog_down(llm_service):
    """Forces the next catalog resolution to fail (discovery unavailable)."""
    llm_service._clients["unorouter"].models = _Models(
        error=ConnectionError("model catalog unreachable"))
    with llm_service._catalog_lock:
        llm_service._catalog_cache.clear()


# ===========================================================================
print("== 1. obsolete fallback id is excluded from ROUTING, not just health ==")
llm_service = _reset(
    UNOROUTER_PRIMARY_MODEL="glm-5.3-flash:free",
    UNOROUTER_FALLBACK_MODELS=["qwen/qwen3.8-27b:free",
                               "glm-5.3-flash-think-search:free"])
uno = _client(model_ids=["glm-5.3-flash:free",
                         "glm-5.3-flash-think-search:free"])
llm_service._clients["unorouter"] = uno
report = llm_service.validate_models()
entry = report["providers"]["unorouter"]
check("1a. obsolete id named as missing AND recorded in the registry",
      entry["missing_models"] == ["qwen/qwen3.8-27b:free"]
      and "qwen/qwen3.8-27b:free" in llm_service.obsolete_models("unorouter"))
check("1b. effective chain excludes the obsolete id",
      entry["selected"] == ["glm-5.3-flash:free",
                            "glm-5.3-flash-think-search:free"]
      and entry["routing_excluded"] == ["qwen/qwen3.8-27b:free"])
check("1c. no request is sent to the obsolete id (precheck short-circuit)",
      llm_service._candidate_precheck(
          "unorouter", "qwen/qwen3.8-27b:free")[0] == "MODEL_NOT_FOUND"
      and uno.sends == [])

# catalog goes DOWN: the obsolete id must NOT return to routing
_catalog_down(llm_service)
report = llm_service.validate_models()
entry = report["providers"]["unorouter"]
check("1d. catalog unavailable is reported honestly (discovery only)",
      entry["status"] == "CATALOG_UNAVAILABLE"
      and entry["catalog_status"] == "UNAVAILABLE")
check("1e. routing still excludes the recorded-obsolete id (no timeout "
      "resurrection)",
      entry["selected"] == ["glm-5.3-flash:free",
                            "glm-5.3-flash-think-search:free"]
      and entry["routing_excluded"] == ["qwen/qwen3.8-27b:free"]
      and entry["routing_basis"] == "configured-minus-known-obsolete")

attempted = []


def _primary_down(kwargs):
    attempted.append(kwargs["model"])
    if kwargs["model"] == "glm-5.3-flash:free":
        raise _HTTPError(500, "primary down")
    return json.dumps({"decision": "HOLD"})


llm_service._clients["unorouter"].chat.completions.create = (
    lambda **kw: _Completion(_primary_down(kw)))
r = llm_service.call("cio", system="s", user="1e", reason="test")
check("1f. runtime chain never sends the obsolete id even without a catalog",
      r.ok and r.model == "glm-5.3-flash-think-search:free"
      and attempted == ["glm-5.3-flash:free",
                        "glm-5.3-flash-think-search:free"])

# ===========================================================================
print("\n== 2. an obsolete/dead model cannot poison the PROVIDER circuit ==")
llm_service = _reset(UNOROUTER_PRIMARY_MODEL="dead/model-x",
                     UNOROUTER_FALLBACK_MODELS=[])
uno = _client(models_error=ConnectionError("catalog down"),
              responder=lambda kw: _HTTPError(404, "model not found"))
llm_service._clients["unorouter"] = uno
for i in range(5):
    llm_service.call("cio", system="s", user=f"2.{i}", reason="test")
state = llm_service.provider_states()["unorouter"]
breaker = llm_service._breaker["unorouter"]
check("2a. exactly ONE request for the dead model (per-model circuit)",
      len(uno.sends) == 1)
check("2b. provider breaker untouched by 404s (failures=0, CLOSED)",
      breaker["failures"] == 0 and state["circuit"] == "CLOSED")
check("2c. the failure is attributed to the MODEL, not the provider",
      state["state"] == "MODEL_UNAVAILABLE"
      and state["last_error"]["status"] == "MODEL_NOT_FOUND"
      and state["last_error"]["category"] == "MODEL_NOT_FOUND")

# HALF_OPEN probe on an obsolete id must not re-open the provider circuit
llm_service._breaker["unorouter"].update({
    "failures": 3, "open_until": time.monotonic() - 1.0})
llm_service._model_unavailable_until.clear()
llm_service._clients["unorouter"].chat.completions.create = (
    lambda **kw: (_ for _ in ()).throw(_HTTPError(404, "gone")))
llm_service.call("cio", system="s", user="2d", reason="test")
check("2d. HALF_OPEN probe failing on a dead model leaves the circuit "
      "HALF_OPEN/CLOSED (never OPEN again)",
      llm_service._breaker["unorouter"]["failures"] == 3
      and llm_service.provider_states()["unorouter"]["circuit"] != "OPEN")

# ===========================================================================
print("\n== 3. NVIDIA transport failures keep their ORIGINAL category ==")
CASES = [
    ("connection refused", httpx.ConnectError("connection refused"),
     "NETWORK_ERROR", "CONNECTION", "ConnectError"),
    ("read timeout", httpx.ReadTimeout("read timed out"),
     "NETWORK_ERROR", "TIMEOUT", "ReadTimeout"),
    ("TLS handshake", ssl.SSLError("certificate verify failed"),
     "NETWORK_ERROR", "TLS", "SSLError"),
    ("HTTP 503", _HTTPError(503, "service unavailable"),
     "PROVIDER_ERROR", "HTTP_5XX", "_HTTPError"),
    ("HTTP 404", _HTTPError(404, "model not found"),
     "MODEL_NOT_FOUND", "MODEL_NOT_FOUND", "_HTTPError"),
]
for label, exc, exp_status, exp_cat, exp_type in CASES:
    llm_service = _reset(NVIDIA_API_KEY="sk-test-nvidia",
                         NVIDIA_MODEL="meta/llama-test",
                         LLM_CIO_PROVIDER="nvidia")
    llm_service._clients["nvidia"] = _client(
        model_ids=["meta/llama-test"],
        responder=lambda kw, e=exc: e)
    r = llm_service.call("cio", system="s", user=f"3.{label}", reason="test")
    check(f"3. {label}: {exp_status}/{exp_cat} preserved",
          r.status == exp_status and r.error_category == exp_cat
          and r.exception_type == exp_type)
    if exp_cat == "CONNECTION":
        check("3a. original exception text preserved as the diagnostic",
              "connection refused" in (r.error or ""))

llm_service = _reset(NVIDIA_API_KEY="sk-test-nvidia",
                     NVIDIA_MODEL="meta/llama-test",
                     LLM_CIO_PROVIDER="nvidia")
llm_service._clients["nvidia"] = _client(
    model_ids=["meta/llama-test"],
    responder=lambda kw: _HTTPError(
        429, "Rate limit reached. Please retry in 2s."))
r = llm_service.call("cio", system="s", user="3.rate", reason="test")
check("3b. transient 429 -> RATE_LIMITED/RATE_LIMIT (not quota, not retried "
      "with LLM_RATE_LIMIT_MAX_RETRIES=0)",
      r.status == "RATE_LIMITED" and r.error_category == "RATE_LIMIT"
      and len(llm_service._clients["nvidia"].sends) == 1)

llm_service = _reset(UNOROUTER_FALLBACK_MODELS=[])
llm_service._clients["unorouter"] = _client(
    model_ids=["glm-5.3-flash-thinking:free"], responder=lambda kw: "   ")
r = llm_service.call("cio", system="s", user="3.empty", reason="test")
check("3c. empty completion -> INVALID_RESPONSE/EMPTY_RESPONSE",
      r.status == "INVALID_RESPONSE" and r.error_category == "EMPTY_RESPONSE")

llm_service = _reset(UNOROUTER_FALLBACK_MODELS=[])
llm_service._clients["unorouter"] = _client(
    model_ids=["glm-5.3-flash-thinking:free"],
    responder=lambda kw: "not json at all <<")
r = llm_service.call("cio", system="s", user="3.badjson", expect_json=True,
                     reason="test")
check("3d. unparseable structured reply -> INVALID_RESPONSE/"
      "MALFORMED_RESPONSE",
      r.status == "INVALID_RESPONSE"
      and r.error_category == "MALFORMED_RESPONSE")

# ===========================================================================
print("\n== 4. failed NVIDIA request -> ONE verified fallback (no hopping) ==")
llm_service = _reset(
    NVIDIA_API_KEY="sk-test-nvidia", NVIDIA_MODEL="meta/llama-test",
    LLM_CIO_PROVIDER="nvidia", GROQ_API_KEY="sk-test-groq",
    GROQ_CIO_MODEL="openai/gpt-oss-120b")
nvidia = _client(model_ids=["meta/llama-test"],
                 responder=lambda kw: httpx.ConnectError("clickhouse timeout? no: "
                                                         "connection refused"))
groq = _client(model_ids=["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
               responder=lambda kw: json.dumps({"decision": "HOLD"}))
llm_service._clients["nvidia"] = nvidia
llm_service._clients["groq"] = groq
llm_service.validate_models()     # catalog-verifies Groq before the fallback
r = llm_service.call("cio", system="s", user="4.fallback", reason="test")
check("4a. NVIDIA NETWORK_ERROR -> verified Groq answers",
      r.ok and r.provider == "groq"
      and r.model == "openai/gpt-oss-120b" and r.fallback_used is True
      and r.attempts == 2)
check("4b. one NVIDIA send + one Groq send (bounded, no provider hopping)",
      len(nvidia.sends) == 1 and len(groq.sends) == 1)
nstate = llm_service.provider_states()["nvidia"]
check("4c. NVIDIA keeps its own honest state + typed last error",
      nstate["state"] == "NETWORK_ERROR"
      and nstate["last_error"]["category"] == "CONNECTION"
      and nstate["last_error"]["exception_type"] == "ConnectError"
      and nstate["circuit"] == "CLOSED")

# ===========================================================================
print("\n== 5. all providers failing -> ONE bounded honest failure ==")
llm_service = _reset(
    NVIDIA_API_KEY="sk-test-nvidia", NVIDIA_MODEL="meta/llama-test",
    LLM_CIO_PROVIDER="nvidia", GROQ_API_KEY="sk-test-groq")
nvidia = _client(model_ids=["meta/llama-test"],
                 responder=lambda kw: httpx.ConnectError("down"))
groq = _client(model_ids=["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
               responder=lambda kw: _HTTPError(500, "groq down"))
llm_service._clients["nvidia"] = nvidia
llm_service._clients["groq"] = groq
llm_service.validate_models()
r = llm_service.call("cio", system="s", user="5.all-down", reason="test")
check("5a. honest failure, no cached replay, no fabricated content",
      (not r.ok) and r.status == "PROVIDER_ERROR" and r.text == "")
check("5b. exactly two network sends (2 providers, 1 attempt each)",
      len(nvidia.sends) == 1 and len(groq.sends) == 1)
check("5c. cycle send counter equals the real send count",
      llm_service._cycle_send_count == 2)

# ===========================================================================
print("\n== 6. UnoRouter inference health is INDEPENDENT of catalog health ==")
llm_service = _reset(UNOROUTER_FALLBACK_MODELS=[])
uno = _client(model_ids=["glm-5.3-flash-thinking:free"],
              responder=lambda kw: json.dumps({"decision": "HOLD"}))
uno.models = _Models(error=httpx.ReadTimeout("catalog timed out"))
llm_service._clients["unorouter"] = uno
report = llm_service.validate_models()
entry = report["providers"]["unorouter"]
check("6a. catalog UNAVAILABLE, inference UNKNOWN before any request",
      entry["status"] == "CATALOG_UNAVAILABLE"
      and entry["inference_status"] == "UNKNOWN"
      and entry["inference_tested"] is None)
r = llm_service.call("cio", system="s", user="6.inference", reason="test")
check("6b. inference actually succeeded on UnoRouter", r.ok
      and r.provider == "unorouter"
      and r.model == "glm-5.3-flash-thinking:free")
report = llm_service.validate_models()
entry = report["providers"]["unorouter"]
check("6c. catalog stays UNAVAILABLE while inference reports the success",
      entry["status"] == "CATALOG_UNAVAILABLE"
      and entry["catalog_status"] == "UNAVAILABLE"
      and entry["inference_status"] in ("READY", "VERIFIED")
      and entry["inference_verified"] is True)
check("6d. the successfully-tested model is named (not just a boolean)",
      entry["inference_tested"]["source"] == "inference"
      and entry["inference_tested"]["model"]
      == "glm-5.3-flash-thinking:free"
      and entry["inference_tested"]["ok"] is True)
check("6e. catalog-matched list stays EMPTY (discovery never ran)",
      entry["catalog_matched"] == [])

# ===========================================================================
print("\n== 7. health separates configured / matched / selected / tested ==")
llm_service = _reset(
    UNOROUTER_PRIMARY_MODEL="glm-5.3-flash-thinking:free",
    UNOROUTER_FALLBACK_MODELS=["qwen/qwen3.8-27b:free",
                               "glm-5.3-flash:free"])
uno = _client(model_ids=["glm-5.3-flash-thinking:free",
                         "glm-5.3-flash:free"],
              responder=lambda kw: json.dumps({"decision": "HOLD"}))
llm_service._clients["unorouter"] = uno
report = llm_service.validate_models()
entry = report["providers"]["unorouter"]
check("7a. configured != catalog-matched != selected",
      entry["configured"] == ["glm-5.3-flash-thinking:free",
                              "qwen/qwen3.8-27b:free",
                              "glm-5.3-flash:free"]
      and entry["catalog_matched"] == ["glm-5.3-flash-thinking:free",
                                       "glm-5.3-flash:free"]
      and entry["selected"] == ["glm-5.3-flash-thinking:free",
                                "glm-5.3-flash:free"])
check("7b. obsolete id exposed as routing-excluded",
      entry["routing_excluded"] == ["qwen/qwen3.8-27b:free"])
llm_service.call("cio", system="s", user="7.inference", reason="test")
report = llm_service.validate_models()
entry = report["providers"]["unorouter"]
check("7c. successfully-tested model reported after real inference",
      entry["inference_tested"]["model"] == "glm-5.3-flash-thinking:free"
      and entry["inference_verified"] is True)
row = health_service.run_startup_checks()["llm"]["providers"]["unorouter"]
check("7d. the health row carries all four facts + typed last error field",
      set(row["catalog"]["configured"]) == set(entry["configured"])
      and set(row["catalog"]["catalog_matched"]) == set(entry["catalog_matched"])
      and set(row["catalog"]["selected"]) == set(entry["selected"])
      and row["catalog"]["routing_excluded"] == ["qwen/qwen3.8-27b:free"]
      and row["inference_tested"]["ok"] is True
      and "last_error" in row)

# ===========================================================================
print("\n== 8. bounded retries consume the per-cycle send budget ==")
llm_service = _reset(UNOROUTER_FALLBACK_MODELS=[],
                     LLM_RATE_LIMIT_MAX_RETRIES=2,
                     LLM_CYCLE_MAX_REQUESTS=1)
uno = _client(model_ids=["glm-5.3-flash-thinking:free"],
              responder=lambda kw: _HTTPError(
                  429, "Rate limit reached. Please retry in 1s."))
llm_service._clients["unorouter"] = uno
llm_service.reset_cycle_usage()
r = llm_service.call("cio", system="s", user="8.budget", reason="test")
check("8a. retry would exceed the cycle budget -> NOT sent",
      r.status == "RATE_LIMITED" and len(uno.sends) == 1
      and llm_service._cycle_send_count == 1)
check("8b. the outcome is honest about the local budget stop",
      r.error_category == "RATE_LIMIT")

# ===========================================================================
print("\n== 9. no key material in reports/diagnostics/errors ==")
blob = json.dumps(report) + json.dumps(
    llm_service.startup_diagnostics(log=False)) + json.dumps(
    llm_service.provider_states())
blob += json.dumps(r.as_dict()) if r is not None else ""
check("9a. no API keys or bearer tokens in any payload",
      "sk-test-uno-KEY" not in blob and "sk-test-nvidia" not in blob
      and "sk-test-groq" not in blob and "Bearer " not in blob)
check("9b. last-error evidence is structured, never a raw response body",
      all(not isinstance(v, str) or len(v) < 200
          for v in (llm_service.provider_states()["unorouter"]
                    .get("last_error") or {}).values()))

# restore settings for the process
for attr, value in _SAVED.items():
    setattr(config.settings, attr, value)

print()
if FAILURES:
    print(f"{len(FAILURES)} NETWORK-RELIABILITY TEST FAILURE(S):")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("ALL LLM NETWORK-RELIABILITY TESTS PASSED")
