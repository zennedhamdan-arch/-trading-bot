"""
tests/test_provider_reliability.py — permanent LLM provider reliability
regressions (all provider transports mocked; zero real API quota):

  1.  UnoRouter catalog available but ALL configured ids missing.
  2.  UnoRouter catalog temporarily unavailable (discovery-only).
  3.  UnoRouter disabled while Groq is available.
  4.  Groq primary unavailable and NVIDIA fallback available.
  5.  Primary model exists in the catalog but inference fails.
  6.  Catalog READY but the structured inference response is invalid.
  7.  All providers unavailable.
  8.  Optional provider unavailable while active task routes stay healthy.
  9.  Environment overrides differ from repository defaults (drift).
  10. Health reflects ACTIVE ROUTES, not every optional provider.
  11. Fallback selection respects LLM_MAX_MODEL_ATTEMPTS.
  12. Quota exhaustion: no unlimited retries, no provider-hopping.

  +    Bounded inference probe (opt-in): VERIFIED vs INFERENCE_FAILED.
  +    Startup diagnostics: effective routes, sources, switches (no secrets).

Run:  .venv/bin/python tests/test_provider_reliability.py
"""

import json
import logging
import os
import sys
import time
import types
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MEMORY_DB_PATH", "/tmp/provider-reliability.db")

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}")


import config                                                     # noqa: E402

_SAVED = {attr: getattr(config.settings, attr) for attr in (
    "UNOROUTER_ENABLED", "UNOROUTER_API_KEY", "UNOROUTER_PRIMARY_MODEL",
    "UNOROUTER_FALLBACK_MODELS", "GROQ_ENABLED", "GROQ_API_KEY",
    "GROQ_FALLBACK_MODEL", "GROQ_CIO_MODEL", "GEMINI_API_KEY",
    "OPENROUTER_API_KEY", "NVIDIA_API_KEY", "NVIDIA_MODEL",
    "LLM_CIO_PROVIDER", "LLM_FALLBACK_PROVIDER", "LLM_FALLBACK_MODEL",
    "LLM_MAX_MODEL_ATTEMPTS", "LLM_INFERENCE_PROBE_ENABLED",
    "LLM_CATALOG_TIMEOUT_SECONDS", "LLM_CATALOG_CACHE_TTL_MINUTES",
    "LLM_MAX_RETRIES", "ALPACA_API_KEY", "ALPACA_SECRET_KEY")}


def _reset(**overrides):
    for attr, value in _SAVED.items():
        setattr(config.settings, attr, value)
    # UnoRouter is enabled+keyed in every scenario unless a scenario says
    # otherwise (the other provider keys stay empty unless set explicitly).
    config.settings.UNOROUTER_API_KEY = "sk-test-uno-KEY"
    config.settings.GEMINI_API_KEY = ""
    config.settings.OPENROUTER_API_KEY = ""
    config.settings.NVIDIA_API_KEY = ""
    # _overall_from classifies the account as OFFLINE without Alpaca keys
    config.settings.ALPACA_API_KEY = "test-alpaca"
    config.settings.ALPACA_SECRET_KEY = "test-alpaca-secret"
    config.settings.LLM_CATALOG_TIMEOUT_SECONDS = 2.0
    for attr, value in overrides.items():
        setattr(config.settings, attr, value)
    from services import llm_service
    llm_service.reset_all_state()
    return llm_service


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Completion:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _CountingModels:
    def __init__(self, ids=None, delay=0.0, error=None):
        self._ids = list(ids or [])
        self._delay = delay
        self._error = error
        self.calls = 0

    def list(self):
        self.calls += 1
        if self._delay:
            time.sleep(self._delay)
        if self._error:
            raise self._error
        return types.SimpleNamespace(
            data=[types.SimpleNamespace(id=i) for i in self._ids])


class _HTTPError(Exception):
    def __init__(self, status_code, message=""):
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


def _client(model_ids=None, responder=None, models_error=None, models_delay=0.0):
    """OpenAI-compatible fake: models.list + chat.completions.create."""
    client = types.SimpleNamespace()
    client.models = _CountingModels(model_ids, delay=models_delay,
                                    error=models_error)
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


READY_ACCT = {"error": None}
READY_MARKET = {"status": "READY"}

from services import realtime_service                              # noqa: E402


def _rt_healthy():
    """Mark the observability realtime stream healthy so _overall_from
    reflects ONLY the LLM-provider dimension under test."""
    realtime_service._stopped = False
    realtime_service._status["last_tick_at"] = time.time()
    realtime_service._status["last_error"] = None
    realtime_service._market_open_cache.update({"ts": time.time(),
                                                "is_open": True})
    realtime_service._workers["stock"] = types.SimpleNamespace(
        alive=True, genuinely_running=lambda: True, started_at=time.time(),
        exit_reason=None)


def _overall(llm_service, health_service):
    _rt_healthy()
    return health_service._overall_from(READY_ACCT, READY_MARKET)


# ===========================================================================
print("== 1. UnoRouter catalog available, ALL configured ids missing ==")
llm_service = _reset()
# exactly the reported production configuration: a primary and a fallback
# override that the live catalog no longer serves
config.settings.UNOROUTER_PRIMARY_MODEL = "glm-5.3-flash:free"
config.settings.UNOROUTER_FALLBACK_MODELS = ["qwen/qwen3.8-27b:free"]
uno = _client(model_ids=["meta/some-live-model", "other/live-model"])
llm_service._clients["unorouter"] = uno
groq = _client(model_ids=["openai/gpt-oss-20b", "openai/gpt-oss-120b"])
llm_service._clients["groq"] = groq
config.settings.GROQ_API_KEY = "sk-test-groq"
config.settings.GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"
report = llm_service.validate_models()
e = report["providers"]["unorouter"]
check("1a. provider row NO_USABLE_MODEL (catalog verified, zero matched)",
      e["status"] == "NO_USABLE_MODEL" and e["catalog_status"] == "VERIFIED"
      and e["matched"] == []
      and e["missing_models"] == ["glm-5.3-flash:free",
                                  "qwen/qwen3.8-27b:free"])
check("1b. route_info RECORDED the switch to verified Groq",
      llm_service.route_info("cio")["provider"] == "groq"
      and llm_service.route_info("cio")["model"] == "openai/gpt-oss-120b"
      and llm_service.route_info("cio")["switched"] is True
      and "NO_USABLE_MODEL" in llm_service.route_info("cio")["switch_reason"])
check("1c. unorouter dropped from the required active chain",
      llm_service.active_chain_providers() == ["groq"])
check("1d. requests still succeed through the verified alternate",
      llm_service.call("cio", system="s", user="1. probe",
                       reason="test").status == "OK")
from services import health_service                                # noqa: E402
overall = _overall(llm_service, health_service)
check("1e. overall HEALTHY (active routes served by a verified provider)",
      overall["status"] == "HEALTHY" and overall["reasons"] == [])
check("1f. the unorouter failure is REPORTED as a note, not hidden",
      any("optional provider unorouter NO_USABLE_MODEL" in n
          for n in overall.get("notes", [])))
check("1g. no dead model was ever sent a request",
      all(m in ("openai/gpt-oss-20b", "openai/gpt-oss-120b")
          for m in [kw.get("model") for kw in groq.sends]
          if m) and uno.sends == [])

print("\n== 2. UnoRouter catalog temporarily unavailable (discovery only) ==")
llm_service = _reset()
uno = _client(models_delay=4.0)     # exceeds the 2s catalog timeout
llm_service._clients["unorouter"] = uno
groq = _client(model_ids=["openai/gpt-oss-20b"])
llm_service._clients["groq"] = groq
config.settings.GROQ_API_KEY = "sk-test-groq"
report = llm_service.validate_models()
e = report["providers"]["unorouter"]
check("2a. row CATALOG_UNAVAILABLE with inference reported separately",
      e["status"] == "CATALOG_UNAVAILABLE"
      and e["catalog_status"] == "UNAVAILABLE"
      and e["inference_status"] == "UNKNOWN"
      and e["timed_out"] is True)
check("2b. NO route switch on an UNVERIFIED catalog (never guessed)",
      llm_service.route_info("cio")["switched"] is False
      and llm_service.route_info("cio")["provider"] == "unorouter")
overall = _overall(llm_service, health_service)
check("2c. a catalog timeout alone does not degrade the system",
      overall["status"] == "HEALTHY")

print("\n== 3. UnoRouter disabled while Groq is available ==")
llm_service = _reset(UNOROUTER_ENABLED=False)
groq = _client(model_ids=["openai/gpt-oss-20b", "openai/gpt-oss-120b"])
llm_service._clients["groq"] = groq
config.settings.GROQ_API_KEY = "sk-test-groq"
llm_service.validate_models()
check("3a. a disabled provider is not REQUIRED (nothing to require)",
      llm_service.active_chain_providers() == ["groq"])
overall = _overall(llm_service, health_service)
check("3b. disabled UnoRouter + healthy Groq -> overall HEALTHY",
      overall["status"] == "HEALTHY" and overall["reasons"] == [])
check("3c. the provider row says DISABLED (not NOT_CONFIGURED)",
      llm_service.validate_models()["providers"]["unorouter"]["status"]
      == "DISABLED")

print("\n== 4. Groq primary unavailable, NVIDIA fallback available ==")
llm_service = _reset(LLM_CIO_PROVIDER="groq", GROQ_API_KEY="sk-g",
                     NVIDIA_API_KEY="sk-n",
                     NVIDIA_MODEL="deepseek-ai/deepseek-v4.1-flash",
                     LLM_FALLBACK_PROVIDER="nvidia",
                     LLM_FALLBACK_MODEL="deepseek-ai/deepseek-v4.1-flash")
groq = _client(model_ids=["openai/gpt-oss-120b"],
               responder=lambda kw: _HTTPError(500, "hard down"))
llm_service._clients["groq"] = groq
nvidia = _client(model_ids=["deepseek-ai/deepseek-v4.1-flash"])
llm_service._clients["nvidia"] = nvidia
llm_service.validate_models()
result = llm_service.call("cio", system="s", user="4. groq-down",
                          reason="test")
check("4a. request fails over to the VERIFIED NVIDIA fallback",
      result.status == "OK" and result.provider == "nvidia"
      and result.model == "deepseek-ai/deepseek-v4.1-flash"
      and result.fallback_used is True)
check("4b. the fallback model was catalog-verified before use",
      "deepseek-ai/deepseek-v4.1-flash"
      in llm_service._verified_models.get("nvidia", set()))
check("4c. exactly one Groq send (no retry storm), then NVIDIA",
      len(groq.sends) == 1 and len(nvidia.sends) == 1)

print("\n== 5. model exists in catalog but inference fails ==")
llm_service = _reset()
uno = _client(model_ids=["glm-5.3-flash-thinking:free", "glm-5.3-flash:free",
                         "glm-5.3-flash-search:free"],
              responder=lambda kw: _HTTPError(500, "hard down"))
llm_service._clients["unorouter"] = uno
llm_service.validate_models()
result = llm_service.call("cio", system="s", user="5. dead-inference",
                          reason="test")
check("5a. catalog match is NOT proof of inference: request fails honestly",
      result.ok is False and result.status in ("PROVIDER_ERROR", "DEGRADED"))
check("5b. attempts bounded by the model cap (no unlimited guessing)",
      len(uno.sends) == int(config.settings.LLM_MAX_MODEL_ATTEMPTS))
check("5c. provider circuit reflects the failure",
      llm_service.provider_states()["unorouter"]["state"] in (
          "DEGRADED", "NETWORK_ERROR"))
llm_service.validate_models()
row = health_service.run_startup_checks()["llm"]["providers"]["unorouter"]
check("5d. health row: INFERENCE_FAILED (catalog fine, inference not)",
      row["status"] == "INFERENCE_FAILED"
      and row["catalog_status"] == "VERIFIED")
overall = _overall(llm_service, health_service)
check("5e. overall DEGRADED (required provider failing — never a lying "
      "HEALTHY)", overall["status"] == "DEGRADED")

print("\n== 6. catalog READY but structured inference response invalid ==")
llm_service = _reset(UNOROUTER_PRIMARY_MODEL="glm-5.3-flash-thinking:free",
                     UNOROUTER_FALLBACK_MODELS=["glm-5.3-flash:free"])
uno = _client(model_ids=["glm-5.3-flash-thinking:free",
                         "glm-5.3-flash:free"],
              responder=lambda kw: "this is not JSON at all <<")
llm_service._clients["unorouter"] = uno
llm_service.validate_models()
result = llm_service.call("cio", system="s", user="6. bad-json",
                          expect_json=True, reason="test")
check("6a. invalid structured response is NEVER accepted as a verdict",
      result.ok is False and result.status == "INVALID_RESPONSE")
check("6b. every configured live model got one chance, bounded",
      len(uno.sends) == 2)
check("6c. nothing fabricated into the response cache",
      result.parsed is None)

print("\n== 7. all providers unavailable ==")
llm_service = _reset(UNOROUTER_PRIMARY_MODEL="glm-5.3-flash:free",
                     UNOROUTER_FALLBACK_MODELS=["qwen/qwen3.8-27b:free"])
uno = _client(model_ids=["totally/unrelated"])   # nothing configured is live
llm_service._clients["unorouter"] = uno
groq = _client(model_ids=["groq/live-but-unrelated"])
llm_service._clients["groq"] = groq
config.settings.GROQ_API_KEY = "sk-g"
llm_service.validate_models()
check("7a. no verified alternate -> route NOT switched, failure visible",
      llm_service.route_info("cio")["switched"] is False)
overall = _overall(llm_service, health_service)
check("7b. overall DEGRADED with the exact dead ids in the reason",
      overall["status"] == "DEGRADED"
      and any("NO_USABLE_MODEL" in r for r in overall["reasons"])
      and any("no usable configured LLM model" in r
              for r in overall["reasons"]))
check("7c. zero sends (dead ids never attempted — no 404 round-trips)",
      uno.sends == [] and groq.sends == [])

print("\n== 8. optional provider down, active routes healthy ==")
llm_service = _reset(UNOROUTER_PRIMARY_MODEL="glm-5.3-flash:free",
                     UNOROUTER_FALLBACK_MODELS=["qwen/qwen3.8-27b:free"],
                     GEMINI_API_KEY="sk-gem",
                     GROQ_API_KEY="sk-groq")
uno = _client(model_ids=["unrelated/live"])       # optional: switched away
llm_service._clients["unorouter"] = uno
groq = _client(model_ids=["openai/gpt-oss-20b", "openai/gpt-oss-120b"])
llm_service._clients["groq"] = groq
gemini = _client(models_delay=4.0)                # optional: catalog flaky
llm_service._clients["gemini"] = gemini
llm_service.validate_models()
check("8a. active chain is Groq only",
      llm_service.active_chain_providers() == ["groq"])
overall = _overall(llm_service, health_service)
check("8b. optional UnoRouter NO_USABLE_MODEL + flaky Gemini do NOT "
      "degrade the system",
      overall["status"] == "HEALTHY" and overall["reasons"] == [])
check("8c. both optional failures are visible as notes",
      any("optional provider unorouter NO_USABLE_MODEL" in n
          for n in overall.get("notes", []))
      and any("optional provider gemini catalog unavailable" in n
              for n in overall.get("notes", [])))
check("8d. provider rows still expose the true states",
      llm_service.validate_models()["providers"]["unorouter"]["status"]
      == "NO_USABLE_MODEL"
      and llm_service.validate_models()["providers"]["gemini"]["status"]
      == "CATALOG_UNAVAILABLE")

print("\n== 9. environment overrides differ from repository defaults ==")
llm_service = _reset()
cap = _capture()
with mock.patch.dict(os.environ,
                     {"UNOROUTER_PRIMARY_MODEL": "qwen/qwen3.8-27b:free"}):
    config.settings.UNOROUTER_PRIMARY_MODEL = "qwen/qwen3.8-27b:free"
    config.settings.UNOROUTER_FALLBACK_MODELS = ["glm-5.3-flash:free"]
    uno = _client(model_ids=["something/else"])
    llm_service._clients["unorouter"] = uno
    report = llm_service.validate_models()
    e = report["providers"]["unorouter"]
    check("9a. source reported as env override (never a secret value)",
          e["model_sources"].get("qwen/qwen3.8-27b:free")
          == "env:UNOROUTER_PRIMARY_MODEL"
          and e["model_sources"].get("glm-5.3-flash:free")
          == "default")
    check("9b. drift warning names the environment variable",
          any("LLM_CONFIG_DRIFT" in m
              and "env:UNOROUTER_PRIMARY_MODEL" in m
              for m in cap.records))
    diag = llm_service.startup_diagnostics(log=False)
    check("9c. diagnostics carry the source + catalog match per route",
          diag["routes"]["cio"]["configured_model_source"]
          == "env:UNOROUTER_PRIMARY_MODEL"
          and diag["routes"]["cio"]["configured_model_in_catalog"] is False)
    check("9d. no key material anywhere in the diagnostics payload",
          "sk-test-uno-KEY" not in json.dumps(diag))

print("\n== 10. health reflects ACTIVE ROUTES, not every optional provider ==")
llm_service = _reset(UNOROUTER_PRIMARY_MODEL="glm-5.3-flash:free",
                     UNOROUTER_FALLBACK_MODELS=["qwen/qwen3.8-27b:free"],
                     GROQ_API_KEY="sk-g")
uno = _client(model_ids=["unrelated"])
llm_service._clients["unorouter"] = uno
groq = _client(model_ids=["openai/gpt-oss-20b", "openai/gpt-oss-120b"])
llm_service._clients["groq"] = groq
llm_service.validate_models()
routes = llm_service.validate_models()["routes"]
check("10a. every task route is operational on the effective chain",
      all(routes[a]["operational"] and routes[a]["provider"] == "groq"
          for a in ("technical", "debate", "risk", "cio", "news",
                    "fundamentals")))
check("10b. operator route is EXCLUDED from the required chain",
      "operator" not in llm_service.active_chain_providers())
overall = _overall(llm_service, health_service)
check("10c. overall HEALTHY although an optional provider has no models",
      overall["status"] == "HEALTHY")

print("\n== 11. fallback selection respects LLM_MAX_MODEL_ATTEMPTS ==")
llm_service = _reset(LLM_MAX_MODEL_ATTEMPTS=2,
                     UNOROUTER_PRIMARY_MODEL="m/one",
                     UNOROUTER_FALLBACK_MODELS=["m/two", "m/three",
                                                "m/four"])
uno = _client(model_ids=["m/one", "m/two", "m/three", "m/four"],
              responder=lambda kw: _HTTPError(500, "down"))
llm_service._clients["unorouter"] = uno
llm_service.validate_models()
eff = llm_service.unorouter_effective_chain()
check("11a. effective chain capped at 2 live models (never unlimited)",
      eff["chain"] == ["m/one", "m/two"])
result = llm_service.call("cio", system="s", user="11. cap", reason="test")
check("11b. exactly 2 sends when everything fails (cap enforced)",
      len(uno.sends) == 2 and result.ok is False)

print("\n== 12. quota exhaustion: bounded, no hopping ==")
llm_service = _reset(UNOROUTER_PRIMARY_MODEL="glm-5.3-flash-thinking:free",
                     UNOROUTER_FALLBACK_MODELS=["glm-5.3-flash:free"])
calls = {"n": 0}


def _quota(kwargs):
    calls["n"] += 1
    return _HTTPError(429, "quota exhausted for this month")


uno = _client(model_ids=["glm-5.3-flash-thinking:free", "glm-5.3-flash:free"],
              responder=_quota)
llm_service._clients["unorouter"] = uno
llm_service.validate_models()
r1 = llm_service.call("cio", system="s", user="12. quota", reason="test")
check("12a. quota exhaustion reported honestly (never routed around)",
      r1.ok is False and r1.status == "PROVIDER_QUOTA_EXCEEDED")
sends_after_first = len(uno.sends)
r2 = llm_service.call("cio", system="s", user="12. quota-2", reason="test")
check("12b. quota circuit open: the next call sends NOTHING",
      len(uno.sends) == sends_after_first
      and r2.status == "PROVIDER_QUOTA_EXCEEDED")
check("12c. no unlimited retries (2 models x 1 attempt, then circuit)",
      len(uno.sends) <= 2)
check("12d. provider state QUOTA_EXHAUSTED with the backoff window",
      llm_service.provider_states()["unorouter"]["state"]
      == "QUOTA_EXHAUSTED")

print("\n== +. bounded inference probe (opt-in) ==")
llm_service = _reset(LLM_INFERENCE_PROBE_ENABLED=True,
                     UNOROUTER_PRIMARY_MODEL="glm-5.3-flash-thinking:free",
                     UNOROUTER_FALLBACK_MODELS=[],
                     GROQ_API_KEY="sk-g")
uno = _client(model_ids=["glm-5.3-flash-thinking:free"])
llm_service._clients["unorouter"] = uno
groq = _client(model_ids=["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
               responder=lambda kw: _HTTPError(500, "groq inference down"))
llm_service._clients["groq"] = groq
llm_service.validate_models()
probes = llm_service.probe_active_chain()
check("+a. probe runs once per ACTIVE-CHAIN provider only",
      set(probes) == {"unorouter", "groq"}
      and len(uno.sends) == 1 and len(groq.sends) == 1)
check("+b. successful probe -> inference VERIFIED",
      probes["unorouter"]["ok"] is True
      and llm_service.validate_models()["providers"]["unorouter"]
      ["inference_status"] == "VERIFIED")
check("+c. failed probe -> INFERENCE_FAILED (catalog match means nothing)",
      probes["groq"]["ok"] is False
      and llm_service.validate_models()["providers"]["groq"]
      ["inference_status"] == "INFERENCE_FAILED")
row = health_service.run_startup_checks()["llm"]["providers"]["groq"]
check("+d. health row distinguishes INFERENCE_FAILED",
      row["status"] == "INFERENCE_FAILED")
check("+e. probe is bounded and non-trading (tiny prompt, tiny budget)",
      uno.sends and uno.sends[0].get("max_tokens", 999) <= 16
      and "health probe" in str(uno.sends[0].get("messages", "")))
llm_service = _reset()   # probe back OFF by default
check("+f. probe disabled by default (zero extra provider requests)",
      llm_service.probe_active_chain() == {})

print("\n== +. startup diagnostics ==")
llm_service = _reset(UNOROUTER_PRIMARY_MODEL="glm-5.3-flash:free",
                     UNOROUTER_FALLBACK_MODELS=["qwen/qwen3.8-27b:free"],
                     GROQ_API_KEY="sk-g")
uno = _client(model_ids=["unrelated"])
llm_service._clients["unorouter"] = uno
groq = _client(model_ids=["openai/gpt-oss-20b", "openai/gpt-oss-120b"])
llm_service._clients["groq"] = groq
llm_service.validate_models()
diag = llm_service.startup_diagnostics(log=False)
check("+g. diagnostics show effective route + recorded switch",
      diag["routes"]["risk"]["provider"] == "groq"
      and diag["routes"]["risk"]["switched"] is True
      and diag["active_chain_providers"] == ["groq"])
check("+h. diagnostics expose configured chain vs verified effective chain",
      diag["providers"]["unorouter"]["configured_chain"]
      == ["glm-5.3-flash:free", "qwen/qwen3.8-27b:free"]
      and diag["providers"]["unorouter"]["effective_chain"] == []
      and diag["providers"]["unorouter"]["catalog_available"] is True)
check("+i. per-task chains listed for every route",
      all(diag["routes"][a]["chain"] for a in
          ("technical", "debate", "risk", "cio", "news", "fundamentals")))
check("+j. no secrets in the diagnostics payload",
      "sk-g" not in json.dumps(diag) and "sk-test-uno-KEY" not in json.dumps(diag))

# restore settings for the process
for attr, value in _SAVED.items():
    setattr(config.settings, attr, value)

print()
if FAILURES:
    print(f"{len(FAILURES)} PROVIDER-RELIABILITY TEST FAILURE(S):")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("ALL PROVIDER-RELIABILITY TESTS PASSED")
