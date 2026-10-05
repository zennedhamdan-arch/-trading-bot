"""
tests/test_external_request_hardening.py — production health-log fixes:

ISSUE 1 — Alpha Vantage free-tier protection:
  1.  first fundamentals request -> API call(s)
  2.  second request for the same symbol within the cache TTL -> NO API call
  3.  simultaneous requests for the same symbol -> deduplicated (one fetch)
  4.  vendor rate-limit response -> distinct RATE_LIMITED status/reason
  5.  rate limit -> BOUNDED backoff (held longer than the normal TTL, but
      retried after the backoff window — never immediate, never permanent)
  6.  stale cache + provider rate limit -> clearly-marked unavailability
      (the existing system does not serve stale data — and never fabricates)
  7.  no cache + provider unavailable -> structured DATA_UNAVAILABLE/ERROR
  8.  never fabricate fundamentals
  +   request spacing (vendor 1 req/s) enforced between ALL requests,
      including OVERVIEW -> BALANCE_SHEET
  +   startup/health checks make ZERO Alpha Vantage requests

ISSUE 2 — catalog vs inference status separation:
  1.  catalog fetch success -> catalog_status VERIFIED
  2.  catalog timeout -> provider DEGRADED (not ERROR), catalog UNAVAILABLE,
      inference UNKNOWN — configured models remain usable
  3.  catalog network error -> temporary class -> DEGRADED / UNAVAILABLE
  3b. catalog credentials error -> UNAVAILABLE_AUTH / ERROR (real failure
      is NOT hidden)
  4.  configured model confirmed present -> matched / READY
  5.  configured model confirmed missing -> NO_USABLE_MODEL (validation
      never bypassed)
  6.  inference failure while catalog healthy -> inference_status from the
      live circuit, catalog stays VERIFIED
  7.  catalog failure while inference is available -> catalog UNAVAILABLE +
      inference READY; overall system NOT degraded by discovery alone
  8.  repeated health requests -> no repeated catalog fetches
  9.  concurrent health requests -> no catalog request storm (single-flight)
  10. bounded negative cache: a failed fetch is retried after the short
      failure TTL (not the full success TTL, not immediately)
  11. no secrets cached in catalog entries

ISSUE 3 — health surfaces: startup rows carry the separated views; the
  overall status does not treat a catalog timeout as an inference outage
  (NO_USABLE_MODEL and credentials failures still degrade).

All HTTP is mocked; no real API keys are used.

Run:  .venv/bin/python tests/test_external_request_hardening.py
"""

import json
import os
import sys
import tempfile
import threading
import time
import types
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="hardening-")
os.environ["MEMORY_DB_PATH"] = os.path.join(_TMP, "test-memory.db")

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}")


from config import settings                                      # noqa: E402
from services import fundamentals_service as fs                  # noqa: E402
from services import llm_service                                 # noqa: E402

AV_KEY = "AV-TEST-KEY-SECRET"


def _av_setup(interval=0.0, backoff=120.0, ttl=60.0, http=None):
    """Configures the alphavantage provider with a fresh mock HTTP layer."""
    settings.FUNDAMENTALS_PROVIDER = "alphavantage"
    settings.FUNDAMENTALS_FALLBACK_PROVIDER = ""
    settings.ALPHAVANTAGE_API_KEY = AV_KEY
    settings.ALPHAVANTAGE_MIN_REQUEST_INTERVAL_SECONDS = interval
    settings.ALPHAVANTAGE_RATE_LIMIT_BACKOFF_SECONDS = backoff
    settings.FUNDAMENTALS_RESULT_CACHE_TTL_SECONDS = ttl
    fs.reset_cache()
    fs._av_reset_state()
    fs._http_get_json = http


class _AVHttp:
    """Mock Alpha Vantage HTTP: counts calls, records request-start times,
    returns queued responses (dict or Exception)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []       # (monotonic_start, url)

    def __call__(self, url, timeout):
        start = time.monotonic()
        self.calls.append((start, url))
        if not self.responses:
            raise AssertionError("unexpected extra AV call: " + url[:90])
        out = self.responses.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _overview(**overrides):
    payload = {
        "Symbol": "NVDA", "MarketCapitalization": "3148800000000",
        "PERatio": "48.92", "EPS": "1.71", "RevenueTTM": "96031000000",
        "ProfitMargin": "0.5569", "ReturnOnEquityTTM": "1.0587",
    }
    payload.update(overrides)
    return payload


_BALANCE = {"symbol": "NVDA", "annualReports": [{
    "fiscalDateEnding": "2026-01-25", "shortLongTermDebtTotal": "67154000000",
    "totalShareholderEquity": "32648000000"}]}


# ===========================================================================
print("ISSUE 1 — Alpha Vantage free-tier protection:")

# 1. first request -> API call
mock = _AVHttp([_overview(), _BALANCE])
_av_setup(http=mock)
r = fs.get_fundamentals("NVDA")
check("1. first request -> API calls (OVERVIEW + BALANCE_SHEET)",
      r["status"] == "OK" and len(mock.calls) == 2)

# 2. second request within TTL -> no API call
r2 = fs.get_fundamentals("NVDA")
check("2. second request within cache TTL -> NO API call",
      r2 == r and len(mock.calls) == 2)

# 3. simultaneous requests -> deduplicated
mock = _AVHttp([_overview(), _BALANCE])
_av_setup(http=mock, interval=0.05)
results, errors = [], []
barrier = threading.Barrier(6)

def _worker():
    try:
        barrier.wait(timeout=5)
        results.append(fs.get_fundamentals("AAPL"))
    except Exception as exc:  # noqa: BLE001
        errors.append(exc)

threads = [threading.Thread(target=_worker) for _ in range(6)]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=15)
check("3. simultaneous requests for one symbol -> ONE fetch (deduplicated)",
      not errors and len(mock.calls) == 2 and len(results) == 6
      and all(x == results[0] for x in results))

# 4. rate-limit response -> distinct RATE_LIMITED
mock = _AVHttp([{"Note": "Thank you for using Alpha Vantage! Our standard "
                         "API call frequency is 25/day and 5/min."}])
_av_setup(http=mock)
r = fs.get_fundamentals("MSFT")
check("4. vendor rate-limit response -> distinct RATE_LIMITED status/reason",
      r["status"] == "RATE_LIMITED"
      and r["reason"].startswith("RATE_LIMITED")
      and r["provider"] == "alphavantage")

# 5. bounded backoff: held longer than the normal TTL, retried after
mock = _AVHttp([{"Note": "rate limit"},
                _overview(), _BALANCE])   # retried later -> succeeds
_av_setup(http=mock, backoff=1.5, ttl=0.3)
fs.get_fundamentals("TSLA")                       # rate limited at t0
time.sleep(0.8)                                   # normal TTL (0.3s) expired
fs.get_fundamentals("TSLA")
check("5a. rate limit -> backoff holds beyond the normal TTL (no retry yet)",
      len(mock.calls) == 1)
time.sleep(1.0)                                   # backoff (1.5s) elapsed
r = fs.get_fundamentals("TSLA")
check("5b. rate limit -> BOUNDED: retried once after the backoff window "
      "(not immediate, not permanent)",
      len(mock.calls) == 3 and r["status"] == "OK")

# 6. stale cache + rate limit -> clearly marked, never silently stale
mock = _AVHttp([_overview(), _BALANCE, {"Note": "rate limit"}])
_av_setup(http=mock, ttl=0.3, backoff=60.0)
good = fs.get_fundamentals("SPY")
time.sleep(0.4)                                   # cache went stale
r = fs.get_fundamentals("SPY")
check("6. stale cache + rate limit -> RATE_LIMITED unavailability (existing "
      "system does not serve stale data; fields null, status explicit)",
      r["status"] == "RATE_LIMITED"
      and all(r[f] is None for f in fs._FIELDS)
      and r is not good)

# 7. no cache + provider unavailable -> structured result
mock = _AVHttp([urllib.error.URLError(OSError("connection refused"))])
_av_setup(http=mock)
r = fs.get_fundamentals("AAPL")
check("7. no cache + provider unavailable -> structured ERROR, no crash",
      r["status"] == "ERROR" and r["reason"].startswith("NETWORK_ERROR"))
mock = _AVHttp([TimeoutError("timed out")])
_av_setup(http=mock)
r = fs.get_fundamentals("AAPL")
check("7b. provider timeout -> structured ERROR (TIMEOUT category)",
      r["status"] == "ERROR" and r["reason"].startswith("TIMEOUT"))

# 8. never fabricate
for bad in ({"Note": "rate limit"}, urllib.error.URLError(OSError("down")),
            TimeoutError("t"), {}, {"Error Message": "invalid"}):
    mock = _AVHttp([bad])
    _av_setup(http=mock)
    r = fs.get_fundamentals("QQQ")
    check(f"8. never fabricate ({str(bad)[:28]}…) -> all fields null",
          r["status"] in ("DATA_UNAVAILABLE", "ERROR", "RATE_LIMITED")
          and all(r[f] is None for f in fs._FIELDS))

# + request spacing enforced (different symbols, consecutive fetches)
mock = _AVHttp([_overview(), _BALANCE, _overview(Symbol="AAPL"), _BALANCE])
_av_setup(http=mock, interval=0.25, ttl=0.0)      # ttl=0 -> no result cache
fs.get_fundamentals("NVDA")
fs.get_fundamentals("AAPL")
gaps = [mock.calls[i + 1][0] - mock.calls[i][0] for i in range(len(mock.calls) - 1)]
check("+ request spacing: every consecutive request start >= min interval "
      "(incl. OVERVIEW -> BALANCE_SHEET)",
      len(mock.calls) == 4 and all(g >= 0.24 for g in gaps),
      )
print(f"      (observed gaps: {[round(g, 2) for g in gaps]})")

# + startup/health checks make ZERO AV requests
mock = _AVHttp([])
_av_setup(http=mock)
h = fs.health_check()
check("+ health_check makes ZERO Alpha Vantage requests (no boot hammering)",
      len(mock.calls) == 0 and h["status"] in ("PENDING", "READY", "DATA_UNAVAILABLE", "ERROR"))
from services import health_service                              # noqa: E402
mock = _AVHttp([])
_av_setup(http=mock)
health_service.run_startup_checks()
check("+ startup checks make ZERO Alpha Vantage requests",
      len(mock.calls) == 0)

# + health maps RATE_LIMITED precisely (provider not 'permanently broken')
fs._av_last_fetch.update({"status": "RATE_LIMITED",
                          "reason": "RATE_LIMITED: free-tier call frequency exceeded"})
h = fs.health_check()
check("+ health: DATA_UNAVAILABLE + provider=alphavantage + reason=RATE_LIMITED",
      h["status"] == "DATA_UNAVAILABLE" and h.get("reason") == "RATE_LIMITED"
      and "not marked broken" in h["detail"])


# ===========================================================================
print("ISSUE 2/3 — catalog caching + catalog/inference separation:")

_SAVED = {attr: getattr(settings, attr) for attr in (
    "UNOROUTER_API_KEY", "GROQ_API_KEY", "LLM_CATALOG_TIMEOUT_SECONDS",
    "LLM_CATALOG_FAILURE_TTL_SECONDS", "FUNDAMENTALS_PROVIDER")}


def _llm_reset(**overrides):
    for attr, value in _SAVED.items():
        setattr(settings, attr, value)
    settings.UNOROUTER_API_KEY = "test-unorouter-key"
    settings.LLM_CATALOG_TIMEOUT_SECONDS = 1.0
    settings.LLM_CATALOG_FAILURE_TTL_SECONDS = 60.0
    for attr, value in overrides.items():
        setattr(settings, attr, value)
    llm_service.reset_all_state()


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


def _uno(models_obj):
    client = types.SimpleNamespace()
    client.models = models_obj
    client.chat = types.SimpleNamespace(completions=types.SimpleNamespace())
    llm_service._clients["unorouter"] = client
    return client


# 1. catalog fetch success -> VERIFIED
_llm_reset()
settings.UNOROUTER_PRIMARY_MODEL = "glm-5.3-flash-thinking:free"
uno = _uno(_CountingModels(["glm-5.3-flash-thinking:free", "other-model"]))
v = llm_service.validate_models()
e = v["providers"]["unorouter"]
check("1. catalog success -> catalog_status VERIFIED, provider READY",
      e["catalog_status"] == "VERIFIED" and e["status"] == "READY"
      and e["matched"] == ["glm-5.3-flash-thinking:free"])

# 4. configured model confirmed present
check("4. configured model confirmed present -> matched",
      e["ok"] is True and "glm-5.3-flash-thinking:free" in e["matched"])

# 8. repeated health requests -> no repeated fetches (success: full TTL)
for _ in range(5):
    llm_service.validate_models(refresh_if_stale=True)
check("8. repeated health requests -> ONE catalog fetch (success cached "
      "for the full TTL)", uno.models.calls == 1)

# 2. catalog timeout -> DEGRADED / UNAVAILABLE / inference UNKNOWN
_llm_reset()
uno = _uno(_CountingModels(delay=3.0))     # exceeds the 1s catalog timeout
v = llm_service.validate_models()
e = v["providers"]["unorouter"]
check("2. catalog timeout -> provider DEGRADED (not ERROR), catalog "
      "UNAVAILABLE, inference UNKNOWN",
      e["status"] == "DEGRADED" and e["catalog_status"] == "UNAVAILABLE"
      and e["inference_status"] == "UNKNOWN" and e["timed_out"] is True)

# 7. catalog failure while inference is available -> inference READY
llm_service._request_times["unorouter"].append(time.monotonic())  # a served request
v = llm_service.validate_models()
e = v["providers"]["unorouter"]
check("7. catalog failure + working inference -> catalog UNAVAILABLE, "
      "inference READY (separated views)",
      e["catalog_status"] == "UNAVAILABLE" and e["inference_status"] == "READY"
      and e["status"] == "DEGRADED")

# overall system is NOT degraded by a discovery-only failure
from services.health_service import _overall_from                 # noqa: E402
settings.ALPACA_API_KEY = "test-alpaca-key"        # account/market dicts are
settings.ALPACA_SECRET_KEY = "test-alpaca-secret"  # passed directly (no network)
overall = _overall_from({"equity": 100.0}, {"status": "READY"})
check("7b. catalog timeout alone does NOT degrade the overall system "
      "(inference path intact)",
      overall["status"] in ("HEALTHY", "DEGRADED")
      and not any("model validation" in r for r in overall["reasons"]))

settings.ALPACA_API_KEY = ""
settings.ALPACA_SECRET_KEY = ""

# 3. catalog network error -> temporary class -> DEGRADED
_llm_reset()
uno = _uno(_CountingModels(error=ConnectionError("connection reset by peer")))
v = llm_service.validate_models()
e = v["providers"]["unorouter"]
check("3. catalog network error -> DEGRADED / catalog UNAVAILABLE "
      "(temporary class)",
      e["status"] == "DEGRADED" and e["catalog_status"] == "UNAVAILABLE")

# 3b. catalog credentials error -> real failure NOT hidden
_llm_reset()
uno = _uno(_CountingModels(error=PermissionError(
    "Error code: 401 - invalid API key")))
v = llm_service.validate_models()
e = v["providers"]["unorouter"]
check("3b. catalog credentials error -> ERROR / UNAVAILABLE_AUTH "
      "(real provider failure not hidden)",
      e["status"] == "ERROR" and e["catalog_status"] == "UNAVAILABLE_AUTH")

# 5. configured model confirmed missing -> NO_USABLE_MODEL (never bypassed)
_llm_reset()
uno = _uno(_CountingModels(["some-unrelated-model"]))
v = llm_service.validate_models()
e = v["providers"]["unorouter"]
check("5. configured model confirmed missing -> NO_USABLE_MODEL "
      "(live validation never bypassed)",
      e["status"] == "NO_USABLE_MODEL" and e["catalog_status"] == "VERIFIED"
      and e["missing_models"] and not e["matched"])

# 6. inference failure while catalog healthy
_llm_reset()
uno = _uno(_CountingModels(["glm-5.3-flash-thinking:free"]))
llm_service.validate_models()          # catalog healthy
llm_service._quota_backoff_until["unorouter"] = time.monotonic() + 300
v = llm_service.validate_models()
e = v["providers"]["unorouter"]
check("6. inference failure + healthy catalog -> inference QUOTA_EXHAUSTED, "
      "catalog stays VERIFIED",
      e["catalog_status"] == "VERIFIED"
      and e["inference_status"] == "QUOTA_EXHAUSTED")

# 10. bounded negative cache: retry after the short failure TTL
_llm_reset()
uno = _uno(_CountingModels(error=ConnectionError("temporarily unavailable")))
settings.LLM_CATALOG_FAILURE_TTL_SECONDS = 1.0    # clamped to the 5s floor
llm_service.validate_models()                      # attempt 1 (fails)
llm_service.validate_models(refresh_if_stale=True)  # within failure TTL
check("10a. failed catalog fetch NOT retried within the failure TTL",
      uno.models.calls == 1)
time.sleep(5.3)                                    # past the 5s floor
llm_service.validate_models(refresh_if_stale=True)  # after failure TTL
check("10b. failed catalog fetch retried (bounded) after the failure TTL "
      "— not locked out for the full success TTL",
      uno.models.calls == 2)

# 9. concurrent health requests -> no storm (single-flight)
_llm_reset()
uno = _uno(_CountingModels(["m1"], delay=0.4))
barrier = threading.Barrier(8)
cat_results, cat_errors = [], []

def _cat_worker():
    try:
        barrier.wait(timeout=5)
        cat_results.append(llm_service.get_model_catalog("unorouter", force=True))
    except Exception as exc:  # noqa: BLE001
        cat_errors.append(exc)

threads = [threading.Thread(target=_cat_worker) for _ in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=20)
check("9. concurrent catalog requests -> ONE fetch (single-flight, no storm)",
      not cat_errors and uno.models.calls == 1
      and all(c.get("ids") == {"m1"} for c in cat_results))

# concurrent validate_models (the /api/providers/health shape)
_llm_reset()
uno = _uno(_CountingModels(["m1"], delay=0.4))
barrier = threading.Barrier(6)
val_errors = []

def _val_worker():
    try:
        barrier.wait(timeout=5)
        llm_service.validate_models(refresh_if_stale=True)
    except Exception as exc:  # noqa: BLE001
        val_errors.append(exc)

threads = [threading.Thread(target=_val_worker) for _ in range(6)]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=20)
check("9b. concurrent validate_models -> ONE fetch per provider",
      not val_errors and uno.models.calls == 1)

# 11. no secrets cached in catalog entries
_llm_reset()
uno = _uno(_CountingModels(error=ConnectionError("boom")))
llm_service.validate_models()
blob = json.dumps({p: dict(e) for p, e in llm_service._catalog_cache.items()},
                  default=str) + json.dumps(llm_service.last_validation())
check("11. no credentials/secrets in cached catalog entries or validation",
      "test-unorouter-key" not in blob)

# startup health rows carry the separated views
_llm_reset()
uno = _uno(_CountingModels(delay=3.0))
report = health_service.run_startup_checks()
row = next(r for r in report["rows"] if r["component"] == "UNOROUTER")
check("+ startup row: catalog timeout -> DEGRADED with inference note "
      "(not ERROR)",
      row["status"] == "DEGRADED" and "inference: UNKNOWN" in row["detail"]
      and "remain usable" in row["detail"])
llm_entry = report["llm"]["providers"]["unorouter"]
check("+ startup llm_providers payload carries catalog_status/"
      "inference_status",
      llm_entry.get("catalog_status") == "UNAVAILABLE"
      and llm_entry.get("inference_status") == "UNKNOWN")

# NO_USABLE_MODEL still degrades the overall system (nothing hidden)
_llm_reset()
uno = _uno(_CountingModels(["unrelated"]))
llm_service.validate_models()
settings.ALPACA_API_KEY = "test-alpaca-key"
settings.ALPACA_SECRET_KEY = "test-alpaca-secret"
overall = _overall_from({"equity": 100.0}, {"status": "READY"})
check("+ NO_USABLE_MODEL still degrades the overall system",
      overall["status"] == "DEGRADED"
      and any("NO_USABLE_MODEL" in r for r in overall["reasons"]))

# credentials failure still degrades the overall system
_llm_reset()
uno = _uno(_CountingModels(error=PermissionError("401 invalid API key")))
llm_service.validate_models()
overall = _overall_from({"equity": 100.0}, {"status": "READY"})
settings.ALPACA_API_KEY = ""
settings.ALPACA_SECRET_KEY = ""
check("+ credentials failure still degrades the overall system",
      overall["status"] == "DEGRADED"
      and any("model validation" in r for r in overall["reasons"]))

# restore
for attr, value in _SAVED.items():
    setattr(settings, attr, value)
llm_service.reset_all_state()
fs.reset_cache()
fs._av_reset_state()
settings.FUNDAMENTALS_PROVIDER = "none"
settings.ALPHAVANTAGE_API_KEY = ""

# ===========================================================================
if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("\nALL EXTERNAL-REQUEST-HARDENING TESTS PASSED")
