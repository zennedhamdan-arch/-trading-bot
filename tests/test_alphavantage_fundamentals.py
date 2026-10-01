"""
tests/test_alphavantage_fundamentals.py — Alpha Vantage fundamentals DATA
provider proofs (mocked HTTP; the real API/key are NEVER used):

  1.  Alpha Vantage is registered
  2.  Missing ALPHAVANTAGE_API_KEY -> structured unavailable/config result
  3.  Successful OVERVIEW (+BALANCE_SHEET) normalized correctly
  4.  Missing vendor fields stay null (never 0, never fabricated)
  5.  Vendor error / rate-limit / information / empty -> DATA_UNAVAILABLE
      or ERROR, never a crash
  6.  Network timeout handled (bounded, categorized)
  7.  FUNDAMENTALS_PROVIDER=alphavantage no longer yields UNKNOWN_PROVIDER
  8.  NVIDIA remains the fundamentals LLM route, independent of the data
      provider (and the LLM is never the data provider)
  9.  Fundamentals data collection never calls llm_service
  10. Fundamentals interpretation still goes through the EXISTING
      llm_service.call_json("fundamentals", ...) with the ACTUAL data

Extra hardening proofs: HTTP/malformed-JSON failures categorized; the API
key never appears in any payload, reason or log; BALANCE_SHEET failure
keeps the OVERVIEW data (partial, honest); the existing per-symbol result
cache still prevents repeat requests; the agent marks provider failures
UNAVAILABLE (not OFF) with the precise reason.

Run:  .venv/bin/python tests/test_alphavantage_fundamentals.py
"""

import json
import logging
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="avfund-")
os.environ["MEMORY_DB_PATH"] = os.path.join(_TMP, "test-memory.db")

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}")


import config                                                    # noqa: E402
from config import settings                                      # noqa: E402
from services import fundamentals_service as fs                  # noqa: E402
from services import llm_service                                 # noqa: E402

TEST_KEY = "AV-TEST-KEY-SECRET-DO-NOT-LEAK"      # fake, never real

# --- scenario state save ----------------------------------------------------
_SAVED = {attr: getattr(settings, attr) for attr in (
    "FUNDAMENTALS_PROVIDER", "FUNDAMENTALS_FALLBACK_PROVIDER",
    "ALPHAVANTAGE_API_KEY", "LLM_FUNDAMENTALS_PROVIDER", "NVIDIA_MODEL",
    "ENABLE_FUNDAMENTALS_AGENT")}


def _restore():
    for attr, value in _SAVED.items():
        setattr(settings, attr, value)
    fs.reset_cache()
    fs._av_reset_state()


# ---------------------------------------------------------------------------
# mocked HTTP layer — counts calls, never touches the network
# ---------------------------------------------------------------------------
class _MockHTTP:
    """Replaces fs._http_get_json. responses: list of call -> dict | Exception.
    urls records every requested URL (to prove the key stays out of errors)."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []

    def __call__(self, url, timeout):
        self.urls.append(url)
        if not self.responses:
            raise AssertionError("unexpected extra HTTP call: " + url[:80])
        out = self.responses.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _overview_payload(**overrides):
    payload = {
        "Symbol": "NVDA", "AssetType": "Common Stock", "Name": "NVIDIA Corp",
        "MarketCapitalization": "3148800000000",
        "PERatio": "48.92", "EPS": "1.71",
        "RevenueTTM": "96031000000", "ProfitMargin": "0.5569",
        "ReturnOnEquityTTM": "1.0587",
    }
    payload.update(overrides)
    return payload


_BALANCE_PAYLOAD = {
    "symbol": "NVDA",
    "annualReports": [{
        "fiscalDateEnding": "2026-01-25",
        "shortLongTermDebtTotal": "67154000000",
        "totalShareholderEquity": "32648000000",
    }],
}

# ===========================================================================
print("1-2. registration + missing key:")
# 1. registered
check("1. alphavantage is registered", "alphavantage" in fs.available_providers())
check("1. provider class registered",
      fs._PROVIDER_CLASSES.get("alphavantage") is fs.AlphaVantageProvider)

# 2. missing key -> structured unavailable, no network
_restore()
settings.FUNDAMENTALS_PROVIDER = "alphavantage"
settings.ALPHAVANTAGE_API_KEY = ""
mock = _MockHTTP([])
fs._http_get_json = mock
r = fs.get_fundamentals("NVDA")
check("2. missing key -> DATA_UNAVAILABLE with precise reason",
      r["status"] == "DATA_UNAVAILABLE"
      and r["reason"] == "ALPHAVANTAGE_API_KEY_NOT_SET"
      and r["provider"] == "alphavantage")
check("2. missing key -> zero HTTP calls", len(mock.urls) == 0)
check("2. missing key -> no fabricated metrics",
      all(r[f] is None for f in fs._FIELDS))
check("2. config warning for alphavantage without key",
      any("ALPHAVANTAGE_API_KEY" in w for w in settings.validate()))
check("2. health reports NOT_CONFIGURED",
      fs.health_check()["status"] == "NOT_CONFIGURED")

# ===========================================================================
print("3. successful normalization (OVERVIEW + BALANCE_SHEET):")
_restore()
settings.FUNDAMENTALS_PROVIDER = "alphavantage"
settings.ALPHAVANTAGE_API_KEY = TEST_KEY
mock = _MockHTTP([_overview_payload(), _BALANCE_PAYLOAD])
fs._http_get_json = mock
r = fs.get_fundamentals("NVDA")
check("3. status OK, provider alphavantage",
      r["status"] == "OK" and r["provider"] == "alphavantage")
check("3. market_cap normalized", r["market_cap"] == 3148800000000.0)
check("3. pe_ratio normalized", r["pe_ratio"] == 48.92)
check("3. eps normalized", r["eps"] == 1.71)
check("3. revenue normalized (RevenueTTM)", r["revenue"] == 96031000000.0)
check("3. profit_margin normalized", r["profit_margin"] == 0.5569)
check("3. roe normalized (ReturnOnEquityTTM)", r["roe"] == 1.0587)
check("3. debt_to_equity computed from BALANCE_SHEET "
      "(shortLongTermDebtTotal/totalShareholderEquity)",
      r["debt_to_equity"] is not None
      and abs(r["debt_to_equity"] - 67154000000.0 / 32648000000.0) < 0.001)
check("3. exactly 2 HTTP calls (OVERVIEW + BALANCE_SHEET), no more",
      len(mock.urls) == 2 and len(mock.responses) == 0)
check("3. OVERVIEW requested first, BALANCE_SHEET second",
      "function=OVERVIEW" in mock.urls[0]
      and "function=BALANCE_SHEET" in mock.urls[1])
check("3. health reports READY after a successful fetch",
      fs.health_check()["status"] == "READY")

# ===========================================================================
print("4. missing vendor fields stay null:")
_restore()
settings.FUNDAMENTALS_PROVIDER = "alphavantage"
settings.ALPHAVANTAGE_API_KEY = TEST_KEY
mock = _MockHTTP([
    _overview_payload(PERatio="None", EPS="", RevenueTTM="-",
                      ReturnOnEquityTTM="not-a-number"),
    _BALANCE_PAYLOAD,
])
fs._http_get_json = mock
r = fs.get_fundamentals("NVDA")
check("4. 'None'/''/'-'/unparseable vendor values -> null, never 0",
      r["pe_ratio"] is None and r["eps"] is None and r["revenue"] is None
      and r["roe"] is None)
check("4. present fields still normalized",
      r["market_cap"] == 3148800000000.0
      and r["profit_margin"] == 0.5569)
check("4. status stays OK with partial data (honest partial, not failure)",
      r["status"] == "OK")

# no BALANCE_SHEET data -> debt_to_equity null (never computed from nothing)
mock = _MockHTTP([_overview_payload(), {"symbol": "NVDA", "annualReports": []}])
fs._http_get_json = mock
fs.reset_cache()
r = fs.get_fundamentals("NVDA")
check("4. empty annualReports -> debt_to_equity null",
      r["status"] == "OK" and r["debt_to_equity"] is None)

# ===========================================================================
print("5. vendor refusal payloads -> structured results, no crash:")
cases = [
    ({"Error Message": "Invalid API call. Please retry or visit the "
                       "documentation to learn more."},
     "ERROR", "VENDOR_ERROR"),
    ({"Note": "Thank you for using Alpha Vantage! Our standard API call "
              "frequency is 25 calls per day and 5 calls per minute."},
     "DATA_UNAVAILABLE", "RATE_LIMIT"),
    ({"Information": "Thank you for using Alpha Vantage! Please claim your "
                     "free API key."},
     "DATA_UNAVAILABLE", "VENDOR_INFORMATION"),
    ({}, "DATA_UNAVAILABLE", "EMPTY_RESPONSE"),
    (_overview_payload(MarketCapitalization="None", PERatio="None",
                       EPS="None", RevenueTTM="None", ProfitMargin="None",
                       ReturnOnEquityTTM="None"),
     "DATA_UNAVAILABLE", "NO_USABLE_METRICS"),
]
for payload, want_status, want_reason in cases:
    _restore()
    settings.FUNDAMENTALS_PROVIDER = "alphavantage"
    settings.ALPHAVANTAGE_API_KEY = TEST_KEY
    mock = _MockHTTP([payload])
    fs._http_get_json = mock
    r = fs.get_fundamentals("NVDA")
    label = want_reason
    check(f"5. {label} -> {want_status} with categorized reason, no crash",
          r["status"] == want_status and r.get("reason", "").startswith(want_reason))
    check(f"5. {label} -> no fabricated metrics",
          all(r[f] is None for f in fs._FIELDS))

# BALANCE_SHEET refused (rate limit) while OVERVIEW was OK -> keep OVERVIEW
_restore()
settings.FUNDAMENTALS_PROVIDER = "alphavantage"
settings.ALPHAVANTAGE_API_KEY = TEST_KEY
mock = _MockHTTP([_overview_payload(), {"Note": "rate limit"}])
fs._http_get_json = mock
r = fs.get_fundamentals("NVDA")
check("5. BALANCE_SHEET rate-limited -> OVERVIEW data kept, debt null, OK",
      r["status"] == "OK" and r["pe_ratio"] == 48.92
      and r["debt_to_equity"] is None)

# ===========================================================================
print("6. transport failures categorized:")
transport_cases = [
    (TimeoutError("timed out"), "TIMEOUT"),
    (OSError("connection reset"), "REQUEST_ERROR"),
]
import urllib.error                                            # noqa: E402
transport_cases.append((urllib.error.HTTPError(
    "url", 403, "Forbidden", {}, None), "HTTP_ERROR"))
transport_cases.append((urllib.error.URLError(
    OSError("name resolution failed")), "NETWORK_ERROR"))
for exc, category in transport_cases:
    _restore()
    settings.FUNDAMENTALS_PROVIDER = "alphavantage"
    settings.ALPHAVANTAGE_API_KEY = TEST_KEY
    mock = _MockHTTP([exc])
    fs._http_get_json = mock
    r = fs.get_fundamentals("NVDA")
    check(f"6. {category} -> ERROR with category, no crash",
          r["status"] == "ERROR" and r.get("reason", "").startswith(category))

# malformed JSON
class _BadJSON(ValueError):
    pass
_restore()
settings.FUNDAMENTALS_PROVIDER = "alphavantage"
settings.ALPHAVANTAGE_API_KEY = TEST_KEY
mock = _MockHTTP([_BadJSON("Expecting value: line 1 column 1 (char 0)")])
fs._http_get_json = mock
r = fs.get_fundamentals("NVDA")
check("6. malformed JSON -> ERROR MALFORMED_RESPONSE",
      r["status"] == "ERROR" and r.get("reason", "").startswith("MALFORMED_RESPONSE"))

# ===========================================================================
print("7. provider resolves (no more UNKNOWN_PROVIDER):")
_restore()
settings.FUNDAMENTALS_PROVIDER = "alphavantage"
settings.ALPHAVANTAGE_API_KEY = TEST_KEY
mock = _MockHTTP([_overview_payload(), _BALANCE_PAYLOAD])
fs._http_get_json = mock
r = fs.get_fundamentals("NVDA")
check("7. FUNDAMENTALS_PROVIDER=alphavantage resolves to the provider "
      "(never UNKNOWN_PROVIDER)",
      r["provider"] == "alphavantage" and "UNKNOWN_PROVIDER" not in json.dumps(r)
      and r["status"] == "OK")

# ===========================================================================
print("8. NVIDIA stays the LLM route, independent of the data provider:")
_restore()
settings.LLM_FUNDAMENTALS_PROVIDER = "nvidia"
settings.NVIDIA_MODEL = "deepseek-ai/deepseek-v4-pro"
settings.FUNDAMENTALS_PROVIDER = "alphavantage"
settings.ALPHAVANTAGE_API_KEY = TEST_KEY
route = llm_service.route_info("fundamentals")
check("8. fundamentals LLM route -> nvidia / deepseek-ai/deepseek-v4-pro",
      route["provider"] == "nvidia"
      and route["model"] == "deepseek-ai/deepseek-v4-pro")
pc = fs.provider_config()
check("8. config clearly separates data provider vs LLM interpretation",
      pc["primary"] == "alphavantage"
      and pc["llm_route"]["provider"] == "nvidia"
      and pc["llm_route"]["model"] == "deepseek-ai/deepseek-v4-pro"
      and "never a data source" in pc["llm_route"]["role"])
check("8. the LLM is never registered as a DATA provider",
      "nvidia" not in fs.available_providers()
      and "deepseek-ai/deepseek-v4-pro" not in fs.available_providers())

# ===========================================================================
print("9. data collection never calls llm_service:")
_restore()
settings.FUNDAMENTALS_PROVIDER = "alphavantage"
settings.ALPHAVANTAGE_API_KEY = TEST_KEY
mock = _MockHTTP([_overview_payload(), _BALANCE_PAYLOAD])
fs._http_get_json = mock
llm_calls = {"n": 0}
_orig_call, _orig_call_json = llm_service.call, llm_service.call_json


def _llm_tripwire(*a, **kw):
    llm_calls["n"] += 1
    raise AssertionError("fundamentals DATA collection must not call llm_service")


llm_service.call = _llm_tripwire
llm_service.call_json = _llm_tripwire
try:
    r = fs.get_fundamentals("NVDA")
finally:
    llm_service.call = _orig_call
    llm_service.call_json = _orig_call_json
check("9. get_fundamentals completes without any llm_service call",
      r["status"] == "OK" and llm_calls["n"] == 0)

# ===========================================================================
print("10. interpretation still flows through llm_service.call_json:")
_restore()
settings.ENABLE_FUNDAMENTALS_AGENT = True
settings.LLM_FUNDAMENTALS_PROVIDER = "nvidia"
settings.NVIDIA_MODEL = "deepseek-ai/deepseek-v4-pro"
settings.FUNDAMENTALS_PROVIDER = "alphavantage"
settings.ALPHAVANTAGE_API_KEY = TEST_KEY
mock = _MockHTTP([_overview_payload(), _BALANCE_PAYLOAD])
fs._http_get_json = mock

from agents import fundamentals_agent as fa                        # noqa: E402
fa.reset_fundamentals_cache()

captured = {}


def _fake_call_json(agent, system, user, **kw):
    captured["agent"] = agent
    captured["system"] = system
    captured["user"] = user
    captured["symbol"] = kw.get("symbol")
    captured["reason"] = kw.get("reason")
    return types.SimpleNamespace(
        ok=True, status="OK", provider="nvidia",
        model="deepseek-ai/deepseek-v4-pro", latency_ms=120,
        error=None, error_type=None, text="", attempts=1,
        parsed={"signal": "BULLISH", "confidence": 0.62,
                "summary": "Strong margins and ROE with moderate leverage."})


llm_service.call_json = _fake_call_json
fundamentals = fs.get_fundamentals("NVDA")
report = fa.analyze_fundamentals("NVDA", fundamentals)

check("10. interpretation called llm_service.call_json('fundamentals', ...)",
      captured.get("agent") == "fundamentals")
check("10. the LLM received the ACTUAL Alpha Vantage values",
      captured.get("user") and "48.92" in captured["user"]
      and "1.71" in captured["user"]
      and "96,031,000,000" in captured["user"]
      and "alphavantage" in captured["user"])
check("10. the LLM is told missing data is unknown, never zero",
      "never\nas zero" in captured.get("system", "")
      or "never as zero" in captured.get("system", ""))
check("10. reason='fundamentals_interpretation' + symbol carried",
      captured.get("reason") == "fundamentals_interpretation"
      and captured.get("symbol") == "NVDA")
check("10. agent returns the structured verdict from the LLM",
      report["evidence_status"] == "AVAILABLE" and report["signal"] == "BULLISH"
      and report["llm_status"] == "OK"
      and report["provider"] == "nvidia"
      and report["model"] == "deepseek-ai/deepseek-v4-pro")
check("10. report carries the data provider (alphavantage), not the LLM, "
      "as the data source",
      fundamentals["provider"] == "alphavantage")

# cached analysis: identical metrics are not re-sent
llm_service.call_json = _fake_call_json
report2 = fa.analyze_fundamentals("NVDA", fs.get_fundamentals("NVDA"))
check("10. unchanged metrics reuse the cached interpretation (no re-send)",
      report2.get("cached") is True)

# provider rate-limit -> agent marks UNAVAILABLE with the precise reason
_restore()
settings.ENABLE_FUNDAMENTALS_AGENT = True
settings.FUNDAMENTALS_PROVIDER = "alphavantage"
settings.ALPHAVANTAGE_API_KEY = TEST_KEY
mock = _MockHTTP([{"Note": "25 calls per day limit"}])
fs._http_get_json = mock
fa.reset_fundamentals_cache()
limited = fs.get_fundamentals("NVDA")
report3 = fa.analyze_fundamentals("NVDA", limited)
check("rate-limited data -> evidence UNAVAILABLE (not OFF, not hidden)",
      report3["evidence_status"] == "UNAVAILABLE"
      and report3["error"].startswith("DATA_UNAVAILABLE: RATE_LIMIT")
      and report3["signal"] is None)

# ===========================================================================
print("hardening: key hygiene + free-tier protection:")
_restore()
settings.FUNDAMENTALS_PROVIDER = "alphavantage"
settings.ALPHAVANTAGE_API_KEY = TEST_KEY

class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


cap = _LogCapture()
logging.getLogger("fundamentals_service").addHandler(cap)

scenarios = [
    [_overview_payload(), _BALANCE_PAYLOAD],           # OK
    [TimeoutError("timed out")],                        # timeout
    [{"Note": "rate limit"}],                           # rate limited
    [{"Error Message": "Invalid API call."}],           # vendor error
    [urllib.error.HTTPError("url", 403, "Forbidden", {}, None)],
]
blobs = []
for responses in scenarios:
    fs._http_get_json = _MockHTTP(responses)
    fs.reset_cache()
    blobs.append(json.dumps(fs.get_fundamentals("NVDA")))
fs._http_get_json = mock
logging.getLogger("fundamentals_service").removeHandler(cap)
blobs.append("\n".join(cap.messages))
check("API key never appears in any result, reason or log",
      TEST_KEY not in "\n".join(blobs))

# free-tier: the existing per-symbol result cache still prevents repeats
_restore()
settings.FUNDAMENTALS_PROVIDER = "alphavantage"
settings.ALPHAVANTAGE_API_KEY = TEST_KEY
mock = _MockHTTP([_overview_payload(), _BALANCE_PAYLOAD])
fs._http_get_json = mock
fs.reset_cache()
fs.get_fundamentals("AAPL")
fs.get_fundamentals("AAPL")
fs.get_fundamentals("AAPL")
check("existing result cache: repeated calls within TTL -> one fetch "
      "(2 HTTP calls total for one symbol)",
      len(mock.urls) == 2 and len(mock.responses) == 0)

_restore()

# ===========================================================================
if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("\nALL ALPHA VANTAGE FUNDAMENTALS TESTS PASSED")
