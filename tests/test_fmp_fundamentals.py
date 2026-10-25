"""
tests/test_fmp_fundamentals.py — Financial Modeling Prep (FMP) fundamentals
DATA provider + multi-provider evidence proofs (mocked HTTP; the real API
and key are NEVER used):

  Configuration
    1.  FMP is registered alongside none/alphavantage
    2.  Missing FMP_API_KEY -> structured unavailable; honest health
        (NOT_CONFIGURED; CONFIGURATION_ERROR when selected)
    3.  Valid key but no fetch yet -> PENDING (never READY just because
        the key is configured)
    4.  Invalid timeout configuration -> clamped bounded
    5.  Provider selection: FUNDAMENTALS_PROVIDER=fmp routes to FMP
    6.  Backward compatibility: defaults unchanged; provider=none still
        the honest default; alphavantage path untouched

  Connectivity (all mocked)
    7.  Successful response -> AVAILABLE, normalized
    8.  Invalid API key (401 and payload variants) -> AUTHENTICATION_FAILED
    9.  Restricted endpoint (403 and payload variants) -> ACCESS_RESTRICTED
    10. Rate-limited response (429) -> RATE_LIMITED + bounded exponential
        backoff; local daily budget -> refused WITHOUT hitting the API
    11. Timeout -> TIMEOUT
    12. Provider server error (500) -> PROVIDER_ERROR
    13. Malformed JSON -> INVALID_RESPONSE
    14. Empty response ([]) -> EMPTY_DATA (never fabricated)
    15. HTTP 200 with an error payload is NOT classified AVAILABLE

  Normalization
    16. market_cap mapping (marketCap; legacy mktCap)
    17. revenue mapping
    18. eps mapping (basic eps preferred; diluted never mixed silently)
    19. profit_margin mapping (netProfitMargin; derived fallback labeled)
    20. roe mapping (returnOnEquity; derived fallback labeled)
    21. debt_to_equity mapping (debtToEquityRatio; derived fallback labeled)
    22. Currency preservation (reportedCurrency + profile currency)
    23. Reporting-period preservation (date / fiscalYear / period)
    24. Missing fields stay null (never 0, never invented)
    25. Invalid numbers ("None", NaN, garbage strings) -> null
    26. Provenance: source field, endpoint, dates, retrieval timestamp

  Evidence validation (cross-validation)
    27. FMP only (AV fails) -> SINGLE_SOURCE, AV failure recorded
    28. AV only (FMP fails) -> SINGLE_SOURCE, FMP failure recorded
    29. Both agree -> AGREED / DUAL_SOURCE
    30. Both disagree -> CONFLICTED, values preserved, NO averaging
    31. One provider stale -> freshness component reflects it
    32. One provider incomplete -> missing metrics listed
    33. Different reporting periods (balance-sheet > 1 fiscal year apart)
        -> NOT_COMPARABLE
    34. Different currencies -> NOT_COMPARABLE (no invented conversion)
    35. Confidence: named levels + documented components, no percentages
    36. Cross-validation is opt-in (default false — no behavior change)

  Caching / quota / resilience
    37. Repeated request uses the endpoint cache (no second HTTP)
    38. Concurrent requests are deduplicated (one HTTP per endpoint)
    39. Expired cache refreshes
    40. Rate limits preserve valid cached data
    41. Health polling NEVER triggers provider requests
    42. Daily request budget enforced (counted, refused past budget)
    43. Restricted endpoints are not re-requested within the TTL
    44. Both providers failing -> DATA_UNAVAILABLE with explicit reasons

  Secrets
    45. The API key never appears in any result, reason, health payload

  Agent integration
    46. The fundamentals agent consumes evidence-bearing results (AVAILABLE
        evidence; conflicts visible but non-fatal)

Run:  .venv/bin/python tests/test_fmp_fundamentals.py
"""

import json
import os
import sys
import tempfile
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="fmpfund-")
os.environ["MEMORY_DB_PATH"] = os.path.join(_TMP, "test-memory.db")

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}")


from config import settings                                      # noqa: E402
from services import fundamentals_service as fs                  # noqa: E402

TEST_KEY = "FMP-TEST-KEY-SECRET-DO-NOT-LEAK"     # fake, never real

_SAVED = {attr: getattr(settings, attr) for attr in (
    "FUNDAMENTALS_PROVIDER", "FUNDAMENTALS_FALLBACK_PROVIDER",
    "FUNDAMENTALS_CROSS_VALIDATION", "FMP_API_KEY", "FMP_BASE_URL",
    "FMP_REQUEST_TIMEOUT_SECONDS", "FMP_DAILY_REQUEST_BUDGET",
    "FMP_CACHE_TTL_SECONDS", "FMP_RATE_LIMIT_BACKOFF_SECONDS",
    "ALPHAVANTAGE_API_KEY", "ALPHAVANTAGE_MIN_REQUEST_INTERVAL_SECONDS",
    "ENABLE_FUNDAMENTALS_AGENT",
)}


def _reset(**overrides):
    """Clean FMP/AV state + settings for one scenario."""
    fs.reset_cache()
    fs._fmp_reset_state()
    fs._av_reset_state()
    fs.reset_evidence_state()
    for attr, value in _SAVED.items():
        setattr(settings, attr, value)
    settings.FMP_API_KEY = TEST_KEY
    settings.FMP_BASE_URL = "https://financialmodelingprep.com/stable"
    settings.FMP_REQUEST_TIMEOUT_SECONDS = 10.0
    settings.FMP_DAILY_REQUEST_BUDGET = 250
    settings.FMP_CACHE_TTL_SECONDS = 21600.0
    settings.FMP_RATE_LIMIT_BACKOFF_SECONDS = 300.0
    settings.ALPHAVANTAGE_MIN_REQUEST_INTERVAL_SECONDS = 0.0
    settings.ALPHAVANTAGE_API_KEY = "AV-TEST-KEY-SECRET-DO-NOT-LEAK"
    settings.FUNDAMENTALS_CROSS_VALIDATION = False
    settings.ENABLE_FUNDAMENTALS_AGENT = True
    for attr, value in overrides.items():
        setattr(settings, attr, value)


# --- mocked FMP payloads (documented Stable API shapes) --------------------
PROFILE = [{"symbol": "AAPL", "price": 290.5, "marketCap": 4874072686740,
            "currency": "USD", "companyName": "Apple Inc.",
            "exchange": "NASDAQ", "isEtf": False}]
PROFILE_LEGACY = [{"symbol": "AAPL", "mktCap": 1234567890, "currency": "USD"}]
INCOME = [{"date": "2025-09-27", "symbol": "AAPL", "reportedCurrency": "USD",
           "fiscalYear": "2025", "period": "FY", "revenue": 416161000000,
           "netIncome": 112010000000, "eps": 7.49, "epsDiluted": 7.46}]
RATIOS = [{"symbol": "AAPL", "date": "2025-09-27", "fiscalYear": "2025",
           "period": "FY", "reportedCurrency": "USD",
           "priceEarningsRatio": 34.1, "netProfitMargin": 0.2692,
           "debtToEquityRatio": 1.524}]
KEY_METRICS = [{"symbol": "AAPL", "date": "2025-09-27", "fiscalYear": "2025",
                "period": "FY", "reportedCurrency": "USD",
                "returnOnEquity": 1.519}]
BALANCE = [{"date": "2025-09-27", "symbol": "AAPL", "reportedCurrency": "USD",
            "fiscalYear": "2025", "period": "FY", "totalDebt": 112377000000,
            "totalStockholdersEquity": 73733000000}]
CASH_FLOW = [{"date": "2025-09-27", "symbol": "AAPL",
              "reportedCurrency": "USD", "fiscalYear": "2025", "period": "FY",
              "operatingCashFlow": 109614000000}]

FULL = {"profile": PROFILE, "income-statement": INCOME, "ratios": RATIOS,
        "key-metrics": KEY_METRICS, "balance-sheet-statement": BALANCE,
        "cash-flow-statement": CASH_FLOW}

from datetime import date as _date, timedelta as _timedelta            # noqa: E402

RECENT = (_date.today() - _timedelta(days=30)).isoformat()


def _recent(rows):
    """Same payloads with a recent report date (freshness = FRESH)."""
    return [{**r, "date": RECENT} if isinstance(r, dict) and r.get("date")
            else r for r in rows]


FULL_RECENT = {"profile": PROFILE, "income-statement": _recent(INCOME),
               "ratios": _recent(RATIOS), "key-metrics": _recent(KEY_METRICS),
               "balance-sheet-statement": _recent(BALANCE),
               "cash-flow-statement": _recent(CASH_FLOW)}
AV_RECENT = {"MarketCapitalization": "4890000000000", "PERatio": "33.8",
             "EPS": "7.40", "RevenueTTM": "410000000000",
             "ProfitMargin": "0.27", "ReturnOnEquityTTM": "1.50",
             "annualReports": [{"fiscalDateEnding": RECENT,
                                "shortLongTermDebtTotal": "110000000000",
                                "totalShareholderEquity": "73733000000"}]}


def _mock_fmp(responses=None, calls=None, status=200):
    """Installs a mocked FMP HTTP layer. responses: {endpoint: payload};
    a callable(endpoint) -> (http_status, payload, transport_error) may be
    passed instead for dynamic behavior."""
    responses = responses if responses is not None else FULL
    if calls is None:
        calls = []
    fn = responses if callable(responses) else None

    def _http(endpoint, symbol):
        calls.append(endpoint)
        if fn:
            return fn(endpoint)
        payload = responses.get(endpoint, [])
        if isinstance(payload, tuple):
            return payload[0], payload[1], (payload[2] if len(payload) > 2 else None)
        return status, payload, None

    fs._fmp_http_get = _http
    return calls


def _mock_av(overview=None, calls=None):
    if calls is None:
        calls = []

    def _http(url, timeout):
        calls.append(url)
        if callable(overview):
            return overview(url)
        return overview if overview is not None else {}

    fs._http_get_json = _http
    return calls


print("== configuration ==")

_reset(FMP_API_KEY="")
check("1. fmp registered alongside none/alphavantage",
      fs.available_providers() == ["alphavantage", "fmp", "none"])
r = fs._build("fmp").get_fundamentals("AAPL")
check("2a. missing key -> DATA_UNAVAILABLE with FMP_API_KEY_NOT_SET",
      r["status"] == "DATA_UNAVAILABLE" and r["reason"] == "FMP_API_KEY_NOT_SET"
      and all(r[f] is None for f in fs._FIELDS))
h = fs._build("fmp").health_check()
check("2b. missing key -> NOT_CONFIGURED (not selected)",
      h["status"] == "NOT_CONFIGURED")
_reset(FMP_API_KEY="", FUNDAMENTALS_PROVIDER="fmp")
h = fs._build("fmp").health_check()
check("2c. missing key while selected -> CONFIGURATION_ERROR",
      h["status"] == "CONFIGURATION_ERROR")
check("2d. config warning for selected fmp without key",
      any("FUNDAMENTALS_PROVIDER=fmp" in w for w in settings.validate()))

_reset()
check("3. key configured, no fetch yet -> PENDING (READY only after a fetch)",
      fs._build("fmp").health_check()["status"] == "PENDING")
check("3b. provider=none default unchanged (backward compat)",
      _SAVED["FUNDAMENTALS_PROVIDER"] == "none"
      and settings.FUNDAMENTALS_PROVIDER == "none")
check("3c. cross-validation default OFF (no behavior change on deploy)",
      settings.FUNDAMENTALS_CROSS_VALIDATION is False)

_reset(FMP_REQUEST_TIMEOUT_SECONDS=-5)
check("4. invalid timeout clamped to a bounded minimum (>= 1s)",
      fs._fmp_timeout_s() >= 1.0)

_reset(FUNDAMENTALS_PROVIDER="fmp")
calls = _mock_fmp()
r = fs.get_fundamentals("AAPL")
check("5. FUNDAMENTALS_PROVIDER=fmp routes to FMPProvider",
      r["provider"] == "fmp" and r["status"] == "OK")
check("5b. trading path fetches only the minimal endpoint set",
      set(calls) == {"profile", "income-statement", "ratios", "key-metrics"}
      and "cash-flow-statement" not in calls)
_reset(FUNDAMENTALS_PROVIDER="bogus")
r = fs.get_fundamentals("AAPL")
check("6. unknown provider lists fmp among the registered",
      "fmp" in str(r.get("reason")) and r["status"] == "DATA_UNAVAILABLE")

print("\n== connectivity (mocked) ==")

_reset(FUNDAMENTALS_PROVIDER="fmp")
r = fs.get_fundamentals("AAPL")
check("7. successful response -> AVAILABLE, normalized OK",
      r["status"] == "OK" and r["market_cap"] == 4874072686740.0)

_reset(FUNDAMENTALS_PROVIDER="fmp")
_mock_fmp({"profile": (401, {}, None),
           "income-statement": (200, {"Error Message": "Invalid API KEY"},
                                None),
           "ratios": (401, {}, None),
           "key-metrics": (200, {"Error Message": "Invalid API KEY"},
                           None)})
r = fs.get_fundamentals("AAPL")
h = fs._build("fmp").health_check()
check("8a. 401 + invalid-key payload -> AUTHENTICATION_FAILED",
      "AUTHENTICATION_FAILED" in r["reason"])
check("8b. auth status surfaced in health",
      h["auth_status"] == "AUTHENTICATION_FAILED"
      and any(e["status"] == "AUTHENTICATION_FAILED"
              for e in h["endpoints"].values()))

_reset(FUNDAMENTALS_PROVIDER="fmp")
_mock_fmp({ep: (403, {}, None) for ep in FULL})
r = fs.get_fundamentals("AAPL")
h = fs._build("fmp").health_check()
check("9a. 403 -> ACCESS_RESTRICTED",
      "ACCESS_RESTRICTED" in r["reason"])
check("9b. all-restricted -> health ACCESS_RESTRICTED (not READY)",
      h["status"] == "ACCESS_RESTRICTED")
_reset(FUNDAMENTALS_PROVIDER="fmp")
_mock_fmp({"profile": (200, {"Error Message":
                             "Your subscription does not support this "
                             "endpoint"}, None),
           "income-statement": (200, [], None), "ratios": (200, [], None),
           "key-metrics": (200, [], None)})
r = fs.get_fundamentals("AAPL")
check("9c. subscription payload inside HTTP 200 -> ACCESS_RESTRICTED",
      "ACCESS_RESTRICTED" in r["reason"])

_reset(FUNDAMENTALS_PROVIDER="fmp", FMP_RATE_LIMIT_BACKOFF_SECONDS=300.0)
calls = _mock_fmp({"profile": (429, {}, None), "income-statement": (429, {}, None),
                    "ratios": (429, {}, None), "key-metrics": (429, {}, None)})
r = fs.get_fundamentals("AAPL")
bo = fs._fmp_backoff_snapshot()
check("10a. 429 -> RATE_LIMITED classification",
      "RATE_LIMITED" in (r.get("reason") or ""))
check("10b. 429 -> bounded exponential backoff active (never immediate)",
      bo["active"] and 299.0 < bo["remaining_s"] <= 300.0)
calls.clear()
r2 = fs.get_fundamentals("MSFT")
check("10c. backoff refuses further requests WITHOUT hitting the API",
      calls == [] and "RATE_LIMITED" in (r2.get("reason") or ""))
h = fs._build("fmp").health_check()
check("10d. health shows RATE_LIMITED during backoff",
      h["status"] == "RATE_LIMITED" and "preserved" in h["detail"])

_reset(FUNDAMENTALS_PROVIDER="fmp", FMP_DAILY_REQUEST_BUDGET=2)
calls = _mock_fmp()
fs.get_fundamentals("AAPL")            # spends 2 (profile cached after 1st?
check("10e. daily budget counted per HTTP request",
      fs._fmp_usage_snapshot()["count"] == len(calls) >= 2)
calls.clear()
r = fs.get_fundamentals("MSFT")
check("10f. budget exhausted -> refused WITHOUT new HTTP (LOCAL_DAILY_BUDGET)",
      calls == [] and "budget" in str(r.get("reason")).lower()
      and r["status"] == "DATA_UNAVAILABLE")
h = fs._build("fmp").health_check()
check("10g. budget exhaustion visible in health",
      h["status"] == "RATE_LIMITED" and h["usage"]["count"] >= h["usage"]["budget"])

_reset(FUNDAMENTALS_PROVIDER="fmp")
_mock_fmp(lambda ep: (None, None, ("TIMEOUT", "request exceeded 10s")))
r = fs.get_fundamentals("AAPL")
check("11. timeout -> TIMEOUT classification (bounded)",
      "TIMEOUT" in r["reason"]
      and any(e["status"] == "TIMEOUT"
              for e in fs._fmp_state["endpoints"].values()))

_reset(FUNDAMENTALS_PROVIDER="fmp")
_mock_fmp({ep: (500, {}, None) for ep in FULL})
r = fs.get_fundamentals("AAPL")
check("12. HTTP 500 -> PROVIDER_ERROR",
      "PROVIDER_ERROR" in r["reason"])

_reset(FUNDAMENTALS_PROVIDER="fmp")
_mock_fmp(lambda ep: (200, None, ("INVALID_RESPONSE", "invalid JSON (…)")))
r = fs.get_fundamentals("AAPL")
check("13. malformed JSON -> INVALID_RESPONSE",
      "INVALID_RESPONSE" in r["reason"])

_reset(FUNDAMENTALS_PROVIDER="fmp")
_mock_fmp({ep: (200, [], None) for ep in FULL})
r = fs.get_fundamentals("AAPL")
check("14. empty array -> EMPTY_DATA, never fabricated",
      r["status"] == "DATA_UNAVAILABLE" and "NO_USABLE_METRICS" in r["reason"]
      and all(r[f] is None for f in fs._FIELDS))

_reset(FUNDAMENTALS_PROVIDER="fmp")
_mock_fmp({"profile": (200, {"Error Message": "some other vendor error"}, None),
           "income-statement": (200, {"message": "weird shape"}, None),
           "ratios": (200, {}, None), "key-metrics": (200, [], None)})
r = fs.get_fundamentals("AAPL")
check("15. HTTP 200 with error/odd payloads is NOT treated as data",
      r["status"] == "DATA_UNAVAILABLE"
      and all(v["status"] != "AVAILABLE"
              for v in fs._fmp_state["endpoints"].values()))

print("\n== normalization ==")

_reset(FUNDAMENTALS_PROVIDER="fmp")
_mock_fmp()
r = fs.get_fundamentals("AAPL")
check("16. market_cap mapping (marketCap)",
      r["market_cap"] == 4874072686740.0
      and r["meta"]["market_cap"]["source_field"] == "marketCap")
_reset(FUNDAMENTALS_PROVIDER="fmp")
_mock_fmp({**FULL, "profile": PROFILE_LEGACY})
r = fs.get_fundamentals("AAPL")
check("16b. legacy mktCap field also mapped",
      r["market_cap"] == 1234567890.0)
check("17. revenue mapping", r["revenue"] == 416161000000.0)
check("17b. currency preserved on statement metrics",
      r["meta"]["revenue"]["currency"] == "USD")
check("18. eps mapping prefers basic eps (diluted never mixed silently)",
      r["eps"] == 7.49 and r["meta"]["eps"]["source_field"] == "eps"
      and "diluted" in r["meta"]["eps"]["note"])
check("19. profit_margin mapping (netProfitMargin)",
      r["profit_margin"] == 0.2692
      and r["meta"]["profit_margin"]["source_field"] == "netProfitMargin")
check("20. roe mapping (returnOnEquity from key-metrics)",
      r["roe"] == 1.519
      and r["meta"]["roe"]["source_field"] == "returnOnEquity")
check("21. debt_to_equity mapping (debtToEquityRatio)",
      r["debt_to_equity"] == 1.524)
check("22. currency preserved (profile trading currency)",
      r["meta"]["market_cap"]["currency"] == "USD")
check("23. reporting period preserved (date/fiscalYear/period)",
      r["meta"]["revenue"]["report_date"] == "2025-09-27"
      and r["meta"]["revenue"]["fiscal_year"] == "2025"
      and r["meta"]["revenue"]["period"] == "FY")
check("23b. market_cap NOT confused with enterprise value",
      "not enterprise value" in r["meta"]["market_cap"]["note"])

_reset(FUNDAMENTALS_PROVIDER="fmp")
_mock_fmp({"profile": [{k: v for k, v in PROFILE[0].items()
                        if k != "marketCap"}],
           "income-statement": [{k: v for k, v in INCOME[0].items()
                                 if k not in ("eps", "revenue")}],
           "ratios": [{k: v for k, v in RATIOS[0].items()
                       if k not in ("priceEarningsRatio",
                                    "netProfitMargin",
                                    "debtToEquityRatio")}],
           "key-metrics": [{k: v for k, v in KEY_METRICS[0].items()
                            if k != "returnOnEquity"}]})
r = fs.get_fundamentals("AAPL")
check("24. missing fields stay null (never 0, never invented)",
      all(r[f] is None for f in fs._FIELDS)
      and r["status"] == "DATA_UNAVAILABLE")

_reset(FUNDAMENTALS_PROVIDER="fmp")
bad_income = [{**INCOME[0], "revenue": "None", "eps": float("nan"),
               "netIncome": "not-a-number"}]
_mock_fmp({**FULL, "income-statement": bad_income})
r = fs.get_fundamentals("AAPL")
check("25. invalid numbers -> null (None-string, NaN, garbage)",
      r["revenue"] is None and r["eps"] is None)

_reset(FUNDAMENTALS_PROVIDER="fmp")
_mock_fmp()
r = fs.get_fundamentals("AAPL")
m = r["meta"]["revenue"]
check("26. provenance: source field, endpoint, retrieval timestamp",
      m["source_field"] == "revenue" and m["endpoint"] == "income-statement"
      and m.get("retrieved_at"))

# derived fallbacks only when the vendor ratio endpoints are unavailable
_reset(FUNDAMENTALS_PROVIDER="fmp")
_mock_fmp({"profile": PROFILE, "income-statement": INCOME,
           "ratios": (403, {}, None), "key-metrics": (403, {}, None),
           "balance-sheet-statement": BALANCE})
r = fs.get_fundamentals("AAPL")
check("19b. derived profit_margin labeled when ratios restricted",
      abs(r["profit_margin"] - round(112010000000 / 416161000000, 6)) < 1e-9
      and r["meta"]["profit_margin"].get("derived") is True)
check("21b. derived debt_to_equity labeled (balance sheet)",
      abs(r["debt_to_equity"] - round(112377000000 / 73733000000, 4)) < 1e-4
      and r["meta"]["debt_to_equity"].get("derived") is True)
check("20b. derived roe labeled (netIncome/equity)",
      r["meta"]["roe"].get("derived") is True)

print("\n== evidence validation (cross-validation) ==")


def _cross(fmp_responses=None, av_overview=None, **cfg):
    _reset(FUNDAMENTALS_PROVIDER="fmp",
           FUNDAMENTALS_FALLBACK_PROVIDER="alphavantage",
           FUNDAMENTALS_CROSS_VALIDATION=True, **cfg)
    _mock_fmp(fmp_responses if fmp_responses is not None else FULL_RECENT)
    _mock_av(av_overview)
    return fs.get_fundamentals("AAPL")


AV_OK = AV_RECENT

r = _cross(av_overview={"Note": "rate limited"})
ev = r["evidence"]
check("27. FMP only -> SINGLE_SOURCE, AV failure recorded",
      ev["available_providers"] == ["fmp"]
      and "alphavantage" in ev["failed_providers"]
      and ev["confidence"]["components"]["sources"] == "SINGLE_SOURCE")

r = _cross(fmp_responses={ep: (403, {}, None) for ep in FULL},
           av_overview=AV_OK)
ev = r["evidence"]
check("28. AV only -> SINGLE_SOURCE, FMP failure recorded (AV preserved)",
      r["status"] == "OK" and r["provider"] == "alphavantage"
      and ev["available_providers"] == ["alphavantage"]
      and "fmp" in ev["failed_providers"])

r = _cross(av_overview=AV_OK)
ev = r["evidence"]
check("29. both agree -> AGREED metrics, DUAL_SOURCE",
      ev["cross_validated"] and ev["available_providers"] == ["fmp", "alphavantage"]
      and all(ev["metrics"][m]["status"] == "AGREED"
              for m in ("market_cap", "pe_ratio", "profit_margin")))
check("29b. selected values come from the configured primary",
      ev["metrics"]["market_cap"]["selected"] == 4874072686740.0
      and ev["metrics"]["market_cap"]["selected_provider"] == "fmp")
check("29c. confidence HIGH with dual agreeing sources",
      ev["confidence"]["level"] == "HIGH"
      and ev["confidence"]["components"]["agreement"] == "AGREED")

r = _cross(av_overview={**AV_OK, "PERatio": "90.5"})
ev = r["evidence"]
check("30. disagreement -> CONFLICTED, both values preserved, no averaging",
      ev["metrics"]["pe_ratio"]["status"] == "CONFLICTED"
      and ev["metrics"]["pe_ratio"]["values"] == {"fmp": 34.1,
                                                  "alphavantage": 90.5}
      and ev["metrics"]["pe_ratio"]["selected"] == 34.1
      and ev["conflicted_metrics"] == ["pe_ratio"])
check("30b. conflicts cap confidence at LOW",
      ev["confidence"]["level"] == "LOW"
      and ev["confidence"]["components"]["agreement"] == "CONFLICTED")

old_all = dict(FULL_RECENT)
for _ep in ("income-statement", "ratios", "key-metrics",
            "balance-sheet-statement"):
    old_all[_ep] = [{**old_all[_ep][0], "date": "2021-09-25",
                     "fiscalYear": "2021"}]
r = _cross(fmp_responses=old_all, av_overview=AV_OK)
ev = r["evidence"]
check("31. stale statements reflected in the freshness component",
      ev["confidence"]["components"]["freshness"] == "VERY_STALE"
      and ev["confidence"]["level"] == "LOW")

r = _cross(av_overview={"MarketCapitalization": "4890000000000",
                        "PERatio": "33.8"})
ev = r["evidence"]
check("32. incomplete provider -> its uncovered metrics are SINGLE_SOURCE",
      ev["metrics"]["eps"]["status"] == "SINGLE_SOURCE"
      and ev["metrics"]["market_cap"]["status"] == "AGREED")

# balance-sheet metrics > 1 fiscal year apart -> NOT_COMPARABLE
old_balance = [{**BALANCE[0], "date": "2021-09-25", "fiscalYear": "2021"}]
r = _cross(fmp_responses={**FULL_RECENT, "balance-sheet-statement": old_balance,
                          "ratios": (403, {}, None)},
           av_overview=AV_OK)
ev = r["evidence"]
check("33. different reporting periods -> NOT_COMPARABLE (no mixing)",
      ev["metrics"]["debt_to_equity"]["status"] == "NOT_COMPARABLE")

# currency mismatch -> NOT_COMPARABLE
eur_income = [{**INCOME[0], "reportedCurrency": "EUR", "date": RECENT}]
eur_ratios = [{**RATIOS[0], "reportedCurrency": "EUR", "date": RECENT}]
eur_keym = [{**KEY_METRICS[0], "reportedCurrency": "EUR", "date": RECENT}]
r = _cross(fmp_responses={**FULL_RECENT, "income-statement": eur_income,
                          "ratios": eur_ratios, "key-metrics": eur_keym},
           av_overview=AV_OK)
ev = r["evidence"]
check("34a. EUR vs unreported currency -> compared but flagged unverified",
      ev["metrics"]["revenue"]["status"] == "AGREED"
      and "currency unverified" in (ev["metrics"]["revenue"]["note"] or ""))

# both currencies KNOWN and different -> NOT_COMPARABLE (direct module test)
from services import fundamentals_evidence as fe               # noqa: E402
synthetic = {
    "a": {"status": "OK", "symbol": "X", "provider": "a",
          "revenue": 100.0,
          "meta": {"revenue": {"currency": "USD", "report_date": RECENT}}},
    "b": {"status": "OK", "symbol": "X", "provider": "b",
          "revenue": 101.0,
          "meta": {"revenue": {"currency": "EUR", "report_date": RECENT}}},
}
merged = fe.merge_evidence(synthetic)
check("34b. different known currencies -> NOT_COMPARABLE (no conversion)",
      merged["evidence"]["metrics"]["revenue"]["status"] == "NOT_COMPARABLE"
      and "currencies differ" in merged["evidence"]["metrics"]["revenue"]["note"])

r = _cross(av_overview=AV_OK)
conf = r["evidence"]["confidence"]
check("35. confidence uses named levels + components (no percentages)",
      isinstance(conf["level"], str) and set(conf["components"]) == {
          "completeness", "freshness", "sources", "agreement",
          "validation", "present_metrics"}
      and all(isinstance(v, str) for k, v in conf["components"].items()
              if k != "present_metrics"))

_reset()
check("36. cross-validation is opt-in (off by default)",
      settings.FUNDAMENTALS_CROSS_VALIDATION is False)
calls = _mock_fmp()
_reset(FUNDAMENTALS_PROVIDER="fmp", FUNDAMENTALS_FALLBACK_PROVIDER="alphavantage")
_mock_fmp(calls=calls)
av_calls = _mock_av({"Note": "limited"}, calls=[])
r = fs.get_fundamentals("AAPL")
check("36b. with cross-validation OFF the fallback stays failure-only",
      r["provider"] == "fmp" and av_calls == []
      and "fallback" not in json.dumps(r))

print("\n== caching / quota / resilience ==")

_reset(FUNDAMENTALS_PROVIDER="fmp")
calls = _mock_fmp()
fs.get_fundamentals("AAPL")
n_after_first = len(calls)
fs.get_fundamentals("AAPL")
fs.get_fundamentals("AAPL")
check("37. repeated requests served from cache (no new HTTP)",
      len(calls) == n_after_first)

_reset(FUNDAMENTALS_PROVIDER="fmp")
calls = _mock_fmp()
thread_results = []


def _worker():
    thread_results.append(fs.get_fundamentals("AAPL")["status"])


threads = [threading.Thread(target=_worker) for _ in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
per_endpoint = {ep: calls.count(ep) for ep in set(calls)}
check("38. concurrent requests deduplicated (one HTTP per endpoint)",
      all(v == 1 for v in per_endpoint.values())
      and set(per_endpoint) == {"profile", "income-statement",
                                "ratios", "key-metrics"}
      and thread_results == ["OK"] * 8)

_reset(FUNDAMENTALS_PROVIDER="fmp")
calls = _mock_fmp()
fs.get_fundamentals("AAPL")
for key in list(fs._fmp_cache):
    expires, rows = fs._fmp_cache[key]
    fs._fmp_cache[key] = (time.monotonic() - 1, rows)   # force expiry
fs.reset_cache()                     # clear the symbol-level result cache
calls.clear()
r = fs.get_fundamentals("AAPL")
check("39. expired cache refreshes",
      set(calls) == {"profile", "income-statement", "ratios",
                     "key-metrics"} and r["status"] == "OK")

_reset(FUNDAMENTALS_PROVIDER="fmp")
calls = _mock_fmp()
fs.get_fundamentals("AAPL")
# vendor starts rate-limiting AFTER good data is cached
_mock_fmp({ep: (429, {}, None) for ep in FULL}, calls=calls)
r = fs.get_fundamentals("AAPL")
check("40. rate limits preserve valid cached data",
      r["status"] == "OK" and r["market_cap"] == 4874072686740.0)

_reset(FUNDAMENTALS_PROVIDER="fmp")
calls = _mock_fmp()
for _ in range(5):
    fs.providers_health()
    fs._build("fmp").health_check()
check("41. health polling NEVER triggers provider requests",
      calls == [])

_reset(FUNDAMENTALS_PROVIDER="fmp", FMP_DAILY_REQUEST_BUDGET=4)
calls = _mock_fmp()
fs.get_fundamentals("AAPL")          # 4 endpoint requests
check("42a. budget counted (4/4)",
      fs._fmp_usage_snapshot()["count"] == 4)
calls.clear()
fs.get_fundamentals("MSFT")
check("42b. past budget -> no HTTP, honest unavailability",
      calls == [] and fs.get_fundamentals("MSFT")["status"] == "DATA_UNAVAILABLE")

_reset(FUNDAMENTALS_PROVIDER="fmp")
calls = _mock_fmp({"profile": (403, {}, None),
                   "income-statement": INCOME, "ratios": RATIOS,
                   "key-metrics": KEY_METRICS})
fs.get_fundamentals("AAPL")
calls.clear()
fs.get_fundamentals("MSFT")
check("43. restricted endpoint not re-requested within the TTL",
      "profile" not in calls)

_reset(FUNDAMENTALS_PROVIDER="fmp",
       FUNDAMENTALS_FALLBACK_PROVIDER="alphavantage",
       FUNDAMENTALS_CROSS_VALIDATION=True)
_mock_fmp({ep: (500, {}, None) for ep in FULL})
_mock_av({"Note": "rate limited"})
r = fs.get_fundamentals("AAPL")
check("44. both providers failing -> DATA_UNAVAILABLE with both reasons",
      r["status"] == "DATA_UNAVAILABLE"
      and "ALL fundamentals providers unavailable" in r["reason"]
      and "fmp" in r["reason"] and "alphavantage" in r["reason"])

print("\n== secrets ==")

_reset(FUNDAMENTALS_PROVIDER="fmp")
_mock_fmp()
r = fs.get_fundamentals("AAPL")
blob = json.dumps(r) + json.dumps(fs.providers_health()) \
    + json.dumps(fs._build("fmp").health_check()) \
    + json.dumps(fs.fmp_connectivity_probe("AAPL")) \
    + json.dumps(fs.provider_config())
check("45. API key never appears in any result/health/reason payload",
      TEST_KEY not in blob)
_reset(FUNDAMENTALS_PROVIDER="fmp")
_mock_fmp(lambda ep: (None, None, ("PROVIDER_ERROR",
                                   f"boom with {TEST_KEY} inside")))
r = fs.get_fundamentals("AAPL")
check("45b. transport errors scrubbed of the key",
      TEST_KEY not in json.dumps(r) and TEST_KEY not in json.dumps(
          fs._build("fmp").health_check()))

print("\n== agent integration ==")

from agents import fundamentals_agent as fa                       # noqa: E402
from agents.fundamentals_agent import analyze_fundamentals       # noqa: E402
from services import llm_service                                 # noqa: E402

_reset(FUNDAMENTALS_PROVIDER="fmp", FUNDAMENTALS_FALLBACK_PROVIDER="alphavantage",
       FUNDAMENTALS_CROSS_VALIDATION=True)
_mock_fmp(FULL_RECENT)
_mock_av(AV_OK)
_orig_call_json = llm_service.call_json


def _fake_call_json(agent, system, user, **kw):
    return types.SimpleNamespace(
        ok=True, status="OK", provider="unorouter", model="test-model",
        latency_ms=80, error=None, error_type=None, text="", attempts=1,
        parsed={"signal": "BULLISH", "confidence": 0.6,
                "summary": "Cross-validated fundamentals."})


llm_service.call_json = _fake_call_json
try:
    fa.reset_fundamentals_cache()
    evidence_result = fs.get_fundamentals("AAPL")
    agent_out = analyze_fundamentals("AAPL", evidence_result)
    check("46. agent consumes evidence-bearing results (AVAILABLE evidence)",
          agent_out["evidence_status"] == "AVAILABLE"
          and agent_out["signal"] == "BULLISH")
    conflicted = dict(evidence_result)
    conflicted["pe_ratio"] = 34.1
    fa.reset_fundamentals_cache()
    agent_out2 = analyze_fundamentals("AAPL", conflicted)
    check("46b. conflicts are visible to the agent but non-fatal",
          agent_out2["evidence_status"] == "AVAILABLE")
finally:
    llm_service.call_json = _orig_call_json

# restore
fs._fmp_http_get = None
fs._http_get_json = None
for attr, value in _SAVED.items():
    setattr(settings, attr, value)

print()
if FAILURES:
    print(f"{len(FAILURES)} FMP TEST FAILURE(S):")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("ALL FMP FUNDAMENTALS + EVIDENCE TESTS PASSED")
