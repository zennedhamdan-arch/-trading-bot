"""
tests/test_llm_catalog_fixes.py

LLM/provider layer fixes — the catalog/validation/health contract:

  1.  Valid primary model: primary present in the live catalog -> kept,
      provider READY.
  2.  Missing fallback model: candidate absent from the live catalog ->
      excluded, reported missing, ONE warning.
  3.  Automatic valid fallback selection: the first LIVE candidate after
      the dead one is selected (exact ids only, never invented).
  4.  Duplicate warning suppression: one LLM_MODEL_UNAVAILABLE warning per
      model per catalog/health-check period — never 6 identical lines.
  5.  Provider timeout: a hanging /v1/models fetch is bounded; the provider
      is DEGRADED and validation continues (startup never blocks).
  6.  Zero matched models: endpoint OK but no configured model in the
      catalog -> NO_USABLE_MODEL (never READY).
  7.  Stale realtime stream: STALE with the tick age in seconds, never
      CONNECTED/healthy.
  8.  Overall DEGRADED: no usable configured LLM model, required provider
      unavailable, or stale realtime -> DEGRADED (never a lying HEALTHY).

Plus: /v1/models fetched ONCE per provider per cache window; exact id
normalization; no secrets in logs or diagnostics.

Run:  .venv/bin/python tests/test_llm_catalog_fixes.py
"""

import io
import json
import logging
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Isolated SQLite (news intelligence / risk engine share the memory DB).
os.environ["MEMORY_DB_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="catalogfix-"), "test-memory.db")

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}")


# ---------------------------------------------------------------------------
# Fakes
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


class _CountingModels:
    """models.list() that counts calls (the ONE-FETCH-PER-PERIOD proof)."""

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


def _uno_client(model_ids, responder=None, delay=0.0, models_obj=None):
    client = types.SimpleNamespace()
    client.models = models_obj or _CountingModels(model_ids, delay=delay)

    class _Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if responder is not None:
                out = responder(kwargs)
                if isinstance(out, Exception):
                    raise out
                return _Completion(out)
            return _Completion(json.dumps({"ok": True}))

    client.chat = types.SimpleNamespace(completions=_Completions())
    return client


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def _capture_llm_logs():
    cap = _LogCapture()
    logger = logging.getLogger("llm_service")
    logger.addHandler(cap)
    return cap, logger


# ---------------------------------------------------------------------------
# Config + imports
# ---------------------------------------------------------------------------

import config  # noqa: E402

config.settings.UNOROUTER_ENABLED = True
config.settings.UNOROUTER_API_KEY = "sk-test-unorouter-KEY"
config.settings.GROQ_ENABLED = True
config.settings.GROQ_API_KEY = ""          # enabled later per-section
config.settings.GEMINI_API_KEY = ""
config.settings.OPENROUTER_API_KEY = ""
config.settings.NVIDIA_API_KEY = ""
config.settings.ALPACA_API_KEY = ""
config.settings.ALPACA_SECRET_KEY = ""
config.settings.LLM_CATALOG_CACHE_TTL_MINUTES = 8
config.settings.LLM_CATALOG_TIMEOUT_SECONDS = 8
config.settings.LLM_MAX_MODEL_ATTEMPTS = 3
config.settings.LLM_FALLBACK_PROVIDER = ""
config.settings.LLM_FALLBACK_MODEL = ""
config.settings.REALTIME_STALE_TICK_SECONDS = 180

from services import llm_service, health_service, realtime_service  # noqa: E402

PRIMARY = "glm-5.3-flash-thinking:free"
CANDIDATES = ["glm-5.3-flash:free", "glm-5.3-flash-think-search:free",
              "glm-5.3-flash-search:free", "ling-3.0-flash-fin:free"]
config.settings.UNOROUTER_PRIMARY_MODEL = PRIMARY
config.settings.UNOROUTER_FALLBACK_MODELS = list(CANDIDATES)

# The live-catalog fixture: primary + three valid candidates live;
# glm-5.3-flash:free (candidate #1) is ABSENT (exactly the reported
# production situation).
LIVE_IDS = [PRIMARY, "glm-5.3-flash-think-search:free", "glm-5.3-flash-search:free",
            "ling-3.0-flash-fin:free", "some/unrelated-model", "another/model:free"]


def _install_uno(model_ids=LIVE_IDS, responder=None, delay=0.0, models_obj=None):
    llm_service.reset_all_state()
    client = _uno_client(model_ids, responder=responder, delay=delay,
                         models_obj=models_obj)
    llm_service._clients["unorouter"] = client
    return client


# ===========================================================================
print("1. valid primary model:")

# warning capture spans sections 1+2: the warning fires ONCE per catalog
# period no matter how many validations run inside it.
cap, _logger = _capture_llm_logs()

uno = _install_uno()
report = llm_service.validate_models()
entry = report["providers"]["unorouter"]
check("primary present in the live catalog -> matched, provider READY",
      entry["status"] == "READY" and PRIMARY in entry["matched"]
      and entry["ok"] is True and entry["models_found"] == len(LIVE_IDS))
eff = llm_service.unorouter_effective_chain()
check("effective chain keeps the primary first",
      eff["chain"][0] == PRIMARY and eff["catalog_available"] is True)
check("live model ids parsed from data[].id (exact ids, real count)",
      entry["models_found"] == len(LIVE_IDS))

# ===========================================================================
print("2. missing fallback model:")

report = llm_service.validate_models()          # repeat within the same period
entry = report["providers"]["unorouter"]
check("absent candidate (glm-5.3-flash:free) reported missing exactly once",
      entry["missing_models"] == ["glm-5.3-flash:free"])
check("missing list carries ONE label entry per configured slot (not 6)",
      len([m for m in entry["missing"] if m.startswith("fallback[")]) == 1)
warnings = [m for m in cap.records
            if "LLM_MODEL_UNAVAILABLE" in m and "glm-5.3-flash:free" in m
            and "unorouter" in m]
check("exactly ONE LLM_MODEL_UNAVAILABLE warning across repeated validations",
      len(warnings) == 1)

# ===========================================================================
print("3. automatic valid fallback selection:")

eff = llm_service.unorouter_effective_chain()
check("dead candidate skipped; first LIVE candidate selected as fallback",
      eff["selected_fallback"] == "glm-5.3-flash-think-search:free"
      and "glm-5.3-flash:free" not in eff["chain"]
      and eff["chain"] == [PRIMARY, "glm-5.3-flash-think-search:free",
                           "glm-5.3-flash-search:free"])
check("no model id invented or substituted (only configured ids appear)",
      set(eff["chain"]) <= set([PRIMARY] + CANDIDATES))

# runtime: the dead model is never even attempted (no 404 round-trip)
attempted = []


def _responder(kwargs):
    attempted.append(kwargs["model"])
    if kwargs["model"] == PRIMARY:
        raise _HTTPError(500, "primary down")
    return json.dumps({"decision": "HOLD", "confidence": 0.5, "reasoning": "ok"})


uno = _install_uno(responder=_responder)
r = llm_service.call("cio", system="s", user="prompt-3", symbol="AAPL")
check("runtime chain: primary fails -> live fallback answers, dead id NEVER sent",
      r.ok and r.fallback_used and r.model == "glm-5.3-flash-think-search:free"
      and "glm-5.3-flash:free" not in attempted)
check("attempt cap respected (at most LLM_MAX_MODEL_ATTEMPTS models tried)",
      len(attempted) <= 3)

# ===========================================================================
print("4. duplicate warning suppression (one warning per model per period):")

uno = _install_uno()
cap.records.clear()
for _ in range(3):
    llm_service.validate_models()          # repeated validations
for _ in range(3):
    llm_service.unorouter_effective_chain()  # repeated chain resolutions
warnings = [m for m in cap.records
            if "LLM_MODEL_UNAVAILABLE" in m and "glm-5.3-flash:free" in m]
check("3 validations + 3 chain resolutions -> still ONE warning",
      len(warnings) == 1)

# the runtime 404 circuit path is deduplicated too (simulated circuit expiry
# inside one catalog period)
cap.records.clear()   # count only THIS sub-part's warnings
uno = _install_uno(model_ids=LIVE_IDS + ["glm-5.3-flash:free"])
for _ in range(3):
    llm_service._mark_model_unavailable("unorouter", "glm-5.3-flash:free")
warnings = [m for m in cap.records
            if "LLM_MODEL_UNAVAILABLE" in m and "glm-5.3-flash:free" in m]
check("repeated 404 circuit marks inside one period -> ONE warning",
      len(warnings) == 1)

# a NEW catalog period re-arms the warning (never zero, never six)
llm_service._model_warned.clear()
llm_service._mark_model_unavailable("unorouter", "glm-5.3-flash:free")
warnings = [m for m in cap.records
            if "LLM_MODEL_UNAVAILABLE" in m and "glm-5.3-flash:free" in m]
check("new period -> the warning fires again (exactly once)", len(warnings) == 2)
logging.getLogger("llm_service").removeHandler(cap)

# ===========================================================================
print("catalog fetched ONCE per provider per cache window:")

uno = _install_uno()
llm_service.validate_models()
llm_service.validate_models(refresh_if_stale=True)   # warm -> no refetch
llm_service.unorouter_effective_chain()
r = llm_service.call("cio", system="s", user="warm", symbol="AAPL")
check("validations + chain resolutions + requests -> ONE /v1/models call",
      uno.models.calls == 1)
# TTL expiry -> exactly one refetch
with llm_service._catalog_lock:
    llm_service._catalog_cache["unorouter"]["fetched_at"] = 0.0
llm_service.validate_models(refresh_if_stale=True)
check("cache window expired -> exactly one refetch", uno.models.calls == 2)

# ===========================================================================
print("exact id normalization (no case mangling, prefix/space handling):")

uno = _install_uno()
eff = llm_service.unorouter_effective_chain(catalog_ids={
    "  " + PRIMARY + "  ",                      # whitespace
    "models/gemini-3.6-flash",                  # gemini-style prefix
    "glm-5.3-flash-think-search:free",
})
check("whitespace and models/ prefix normalized on BOTH sides",
      eff["chain"] == [PRIMARY, "glm-5.3-flash-think-search:free"])
check("ids still match EXACTLY (no case/fuzzy matching)",
      "Glm-5.3-Flash-Thinking:Free" not in eff["chain"]
      and "glm-5.3-flash-thinking" not in eff["chain"])

# ===========================================================================
print("5. provider timeout (hanging /v1/models never blocks):")

config.settings.LLM_CATALOG_TIMEOUT_SECONDS = 1
uno = _install_uno()
_hanging_models = _CountingModels(["gemini-3.6-flash"], delay=3.0)


class _HangingGeminiClient:
    models = _hanging_models


# GeminiProvider fetches through gemini_service._get_client() — inject there
# (llm_service._clients has no effect for gemini).
from services import gemini_service  # noqa: E402
gemini_service._client = _HangingGeminiClient()
config.settings.GEMINI_API_KEY = "sk-test-gemini-KEY"
config.settings.LLM_NEWS_PROVIDER = "gemini"     # make gemini a required route
t0 = time.monotonic()
report = llm_service.validate_models()
elapsed = time.monotonic() - t0
gentry = report["providers"]["gemini"]
check("hanging catalog fetch bounded by the short timeout (no startup block)",
      elapsed < 2.5 and _hanging_models.calls >= 1)
check("timed-out provider marked DEGRADED with timed_out=true",
      gentry["status"] == "DEGRADED" and gentry["timed_out"] is True
      and "timed out" in str(gentry["error"]))
check("other providers still validated (chain continues)",
      report["providers"]["unorouter"]["status"] == "READY")
config.settings.LLM_NEWS_PROVIDER = "unorouter"
config.settings.GEMINI_API_KEY = ""
config.settings.LLM_CATALOG_TIMEOUT_SECONDS = 8
gemini_service._client = None
time.sleep(3.2)   # let the abandoned worker thread finish before next reset

# ===========================================================================
print("6. zero matched models -> NO_USABLE_MODEL (never READY):")

uno = _install_uno(model_ids=["totally/unrelated", "nothing/matches-here"])
report = llm_service.validate_models()
entry = report["providers"]["unorouter"]
check("endpoint OK but zero configured models match -> NO_USABLE_MODEL",
      entry["ok"] is True and entry["status"] == "NO_USABLE_MODEL"
      and entry["matched"] == [] and set(entry["missing_models"]) == set([PRIMARY] + CANDIDATES))
check("effective chain is EMPTY (nothing usable on this provider)",
      llm_service.unorouter_effective_chain()["chain"] == [])

# the startup health row reports it honestly
config.settings.ALPACA_API_KEY = ""
config.settings.ALPACA_SECRET_KEY = ""
startup = health_service.run_startup_checks()
uno_row = next(r for r in startup["rows"] if r["component"] == "UNOROUTER")
check("startup health row: NO_USABLE_MODEL with live/missing counts",
      uno_row["status"] == "NO_USABLE_MODEL"
      and "live models" in uno_row["detail"]
      and "glm-5.3-flash-thinking:free" in uno_row["detail"])
uno_provider = startup["llm"]["providers"]["unorouter"]
check("row carries safe catalog diagnostics (configured/matched/missing/selected)",
      uno_provider["catalog"]["live_models"] == 2
      and uno_provider["catalog"]["matched"] == []
      and set(uno_provider["catalog"]["missing"]) == set([PRIMARY] + CANDIDATES)
      and uno_provider["catalog"]["selected_fallback"] is None)

# ===========================================================================
print("8a. overall DEGRADED when no usable configured LLM model:")

READY_ACCT = {"error": None}
READY_MARKET = {"status": "READY"}
config.settings.ALPACA_API_KEY = "test-alpaca"
config.settings.ALPACA_SECRET_KEY = "test-secret"


def _rt_worker():
    return types.SimpleNamespace(alive=True, genuinely_running=lambda: True,
                                 started_at=time.time(), exit_reason=None)


def _rt_healthy():
    realtime_service._stopped = False          # running epoch
    realtime_service._status["last_tick_at"] = time.time()
    realtime_service._status["last_error"] = None
    realtime_service._market_open_cache.update({"ts": time.time(), "is_open": True})
    realtime_service._workers["stock"] = _rt_worker()


def _rt_stale(age_s=999.0):
    realtime_service._stopped = False          # running epoch
    realtime_service._status["last_tick_at"] = time.time() - age_s
    realtime_service._market_open_cache.update({"ts": time.time(), "is_open": True})
    realtime_service._workers["stock"] = _rt_worker()


# unorouter NO_USABLE (fresh validation) AND groq NO_USABLE -> nothing left
groq = _uno_client(["groq/live-but-unrelated"])
llm_service._clients["groq"] = groq
config.settings.GROQ_API_KEY = "sk-test-groq-KEY"
config.settings.GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"
llm_service.validate_models()          # fresh validation for the new state
_rt_healthy()
overall = health_service._overall_from(READY_ACCT, READY_MARKET)
check("all configured models dead on the whole chain -> overall DEGRADED",
      overall["status"] == "DEGRADED"
      and any("NO_USABLE_MODEL" in r for r in overall["reasons"]))
check("'no usable configured LLM model' surfaced as a reason",
      any("no usable configured LLM model" in r for r in overall["reasons"]))

# ===========================================================================
print("3b/8b. valid primary restored + healthy chain -> overall HEALTHY:")

uno = _install_uno(LIVE_IDS)          # primary + live fallbacks
llm_service._clients["groq"] = _uno_client(["openai/gpt-oss-20b"])
llm_service.validate_models()          # healthy catalogs for the control
_rt_healthy()
overall = health_service._overall_from(READY_ACCT, READY_MARKET)
check("healthy catalog + healthy providers + fresh ticks -> HEALTHY (control)",
      overall["status"] == "HEALTHY" and overall["reasons"] == [])

# ===========================================================================
print("8c. required provider unavailable -> overall DEGRADED:")

uno = _install_uno(LIVE_IDS, responder=lambda kw: _HTTPError(500, "hard down"))
llm_service._clients["groq"] = _uno_client(["openai/gpt-oss-20b"])
llm_service.validate_models()          # catalogs valid; the SENDING fails
for _ in range(3):                    # open the breaker
    llm_service.call("cio", system="s", user=f"fail-{_}", symbol="AAPL")
_rt_healthy()
overall = health_service._overall_from(READY_ACCT, READY_MARKET)
check("required provider failing -> overall DEGRADED (never a lying HEALTHY)",
      overall["status"] == "DEGRADED"
      and any(r.startswith("LLM provider unorouter") for r in overall["reasons"]))

# ===========================================================================
print("7. stale realtime stream -> STALE with age, overall DEGRADED:")

uno = _install_uno(LIVE_IDS)
llm_service._clients["groq"] = _uno_client(["openai/gpt-oss-20b"])
llm_service.validate_models()
_rt_stale(age_s=999.0)
state = realtime_service.get_state()
check("stale stream reports connection_status=STALE (not CONNECTED)",
      state["connection_status"] == "STALE" and state["ticks_stale"] is True)
check("tick age exposed in seconds",
      state["seconds_since_last_tick"] is not None
      and state["seconds_since_last_tick"] > 900)
row = health_service._realtime_row()
check("health row: STALE with the age in seconds + threshold",
      row["status"] == "STALE" and "999s ago" in row["detail"]
      and "180s" in row["detail"])
overall = health_service._overall_from(READY_ACCT, READY_MARKET)
check("stale realtime -> overall DEGRADED with the age in the reason",
      overall["status"] == "DEGRADED"
      and any("real-time stream STALE" in r and "s ago" in r
              for r in overall["reasons"]))

# fresh tick -> back to CONNECTED/healthy
_rt_healthy()
row = health_service._realtime_row()
overall = health_service._overall_from(READY_ACCT, READY_MARKET)
check("fresh tick -> CONNECTED again and overall recovers",
      row["status"] == "CONNECTED" and overall["status"] == "HEALTHY")

# ===========================================================================
print("safe diagnostics (never any key/secret):")

cap2, _lg = _capture_llm_logs()
uno = _install_uno()
llm_service.validate_models()
report = llm_service.validate_models()
_rt_healthy()
blob = json.dumps(report) + "\n".join(cap2.records)
blob += json.dumps(health_service._realtime_row())
check("no API keys/secrets in validation reports or logs",
      "sk-test-unorouter-KEY" not in blob and "sk-test-groq-KEY" not in blob
      and "sk-test-gemini-KEY" not in blob and "API_KEY=" not in blob)
check("diagnostics expose provider/configured/live/matched/missing/fallback",
      all(k in report["providers"]["unorouter"]
          for k in ("configured", "matched", "missing_models",
                    "models_found", "selected_fallback", "status")))
logging.getLogger("llm_service").removeHandler(cap2)

# ===========================================================================
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("ALL LLM CATALOG FIX TESTS PASSED")
