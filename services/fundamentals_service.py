"""
services/fundamentals_service.py

FundamentalsProvider abstraction — the ONLY way fundamentals data enters the
bot. yfinance is gone: no Yahoo scraping, no cookies/crumbs, no unofficial
endpoints, and no trading decision depends on any of that.

Architecture:

    FundamentalsProvider (base class, normalized contract)
        ├── primary provider    (FUNDAMENTALS_PROVIDER, default "none")
        └── optional fallback   (FUNDAMENTALS_FALLBACK_PROVIDER)

Normalized response (NEVER fabricated — unavailable fields are null):

    {
      "status": "OK" | "DATA_UNAVAILABLE" | "ERROR",
      "symbol": "AAPL",
      "provider": "none",
      "market_cap": null,
      "pe_ratio": null,
      "eps": null,
      "revenue": null,
      "profit_margin": null,
      "roe": null,
      "debt_to_equity": null,
      "timestamp": 1691000000.0,
      "reason": "..."          # only when status != OK
    }

Built-in providers:
  - "none": the honest default. No fundamentals provider configured —
    returns DATA_UNAVAILABLE / NO_PROVIDER_CONFIGURED. The bot keeps
    trading cycles useful on price data, technicals, news and risk.
  - "alphavantage": Alpha Vantage's official HTTP API (OVERVIEW, plus
    BALANCE_SHEET for the debt/equity field). A DATA provider only — the
    LLM interpretation is routed separately (LLM_FUNDAMENTALS_PROVIDER).

Adding a real provider (no guessing required): subclass FundamentalsProvider,
implement get_fundamentals() using the vendor's OFFICIAL API, and register
it in _PROVIDER_CLASSES. Nothing else in the bot changes.
"""

import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from config import settings

logger = logging.getLogger("fundamentals_service")

_FIELDS = (
    "market_cap", "pe_ratio", "eps", "revenue",
    "profit_margin", "roe", "debt_to_equity",
)


def _normalized(symbol: str, provider: str, status: str,
                reason: str = None, **fields) -> dict:
    """Builds the normalized payload; missing metrics are null, never made up."""
    out = {
        "status": status,
        "symbol": symbol,
        "provider": provider,
        "timestamp": time.time(),
    }
    for field in _FIELDS:
        out[field] = fields.get(field)
    if reason:
        out["reason"] = reason
    return out


class FundamentalsProvider:
    """Base class. Subclasses fetch from an OFFICIAL provider API and return
    the normalized dict via _normalized(); unavailable fields stay null."""

    name = "base"

    def get_fundamentals(self, symbol: str) -> dict:
        raise NotImplementedError

    def health_check(self) -> dict:
        """Lightweight configuration/connectivity description for /api/health."""
        return {"provider": self.name, "status": "READY", "detail": ""}


class NoneProvider(FundamentalsProvider):
    """The default: no fundamentals source configured. Returns an explicit,
    structured DATA_UNAVAILABLE so the fundamentals agent and the cycle can
    degrade honestly instead of scraping something unreliable."""

    name = "none"

    def get_fundamentals(self, symbol: str) -> dict:
        return _normalized(
            symbol, self.name, "DATA_UNAVAILABLE",
            reason="NO_PROVIDER_CONFIGURED",
        )

    def health_check(self) -> dict:
        return {
            "provider": self.name,
            "status": "DATA_UNAVAILABLE",
            "detail": "FUNDAMENTALS_PROVIDER=none — no fundamentals source configured; the bot runs on price/technical/news/risk evidence",
        }


# ---------------------------------------------------------------------------
# Alpha Vantage — fundamentals DATA provider (official HTTP API only).
#
# Architecture (strict separation):
#   Alpha Vantage  -> FACTUAL FINANCIAL DATA (this class)
#   LLM (NVIDIA)   -> INTERPRETATION of that data (agents/fundamentals_agent
#                     via llm_service.call_json("fundamentals", ...))
# The LLM is NOT a data provider; this class is NOT an LLM provider and
# never calls llm_service.
#
# Request design (free tier: 25 requests/day, 5/minute):
#   1. OVERVIEW — carries 6 of the 7 contract fields
#      (MarketCapitalization, PERatio, EPS, RevenueTTM, ProfitMargin,
#      ReturnOnEquityTTM).
#   2. BALANCE_SHEET — ONLY when OVERVIEW yielded usable data, for the one
#      contract field OVERVIEW does not carry (debt_to_equity =
#      shortLongTermDebtTotal / totalShareholderEquity, latest annual
#      report). At most 2 requests per symbol per cache window; a failed
#      BALANCE_SHEET never discards the OVERVIEW data (debt_to_equity stays
#      null). No other endpoints are called — no unnecessary API calls.
# ---------------------------------------------------------------------------

_AV_BASE_URL = "https://www.alphavantage.co/query"
_AV_TIMEOUT_SECONDS = 10.0          # bounded, like LLM_REQUEST_TIMEOUT_SECONDS
_AV_VENDOR_MESSAGE_MAX = 160        # excerpt cap for vendor messages in reasons

# Free-tier request throttle: the vendor asks for <= 1 request/second.
# EVERY Alpha Vantage HTTP request is serialized through one lock with at
# least ALPHAVANTAGE_MIN_REQUEST_INTERVAL_SECONDS between consecutive
# request starts (OVERVIEW -> BALANCE_SHEET included). Bounded by the
# interval + the request timeout; never blocks other services.
_av_http_lock = threading.Lock()
_av_last_request_at = 0.0


def _av_reset_state() -> None:
    """Test hook: clears the last-fetch record and the throttle clock."""
    global _av_last_request_at
    _av_last_fetch["status"] = None
    _av_last_fetch["reason"] = None
    with _av_http_lock:
        _av_last_request_at = 0.0


def _av_min_interval_s() -> float:
    return max(0.0, float(settings.ALPHAVANTAGE_MIN_REQUEST_INTERVAL_SECONDS or 0.0))

# Module-level record of the most recent fetch outcome (for health_check;
# provider instances are rebuilt per resolution, so state lives here).
_av_last_fetch = {"status": None, "reason": None}


def _to_float(value):
    """Alpha Vantage returns numbers as strings; missing values arrive as
    'None', '' or '-'. Unparseable/missing -> None (never 0, never made up)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if value != value else float(value)   # NaN -> None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in ("none", "null", "-", "n/a"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _http_get_json(url: str, timeout: float) -> dict:
    """GET <url> and parse JSON. Standard library only, bounded timeout.
    Raises on timeout / network / HTTP / JSON failures (caller categorizes)."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "ai-trader-paper/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    return parsed


class AlphaVantageProvider(FundamentalsProvider):
    """Fundamentals DATA provider using Alpha Vantage's official /query API.
    Returns the normalized contract; failures are structured
    DATA_UNAVAILABLE/ERROR with a categorized reason — never an exception
    through the trading cycle, and never the API key."""

    name = "alphavantage"

    def _scrub(self, text: str) -> str:
        """Defense in depth: the key must never appear in any reason/log."""
        key = settings.ALPHAVANTAGE_API_KEY
        text = str(text)
        if key:
            text = text.replace(key, "***")
        return text[:400]

    def _throttled_query(self, function: str, symbol: str):
        """One official-endpoint call THROUGH THE SHARED THROTTLE: requests
        are serialized and spaced at least ALPHAVANTAGE_MIN_REQUEST_INTERVAL_
        SECONDS apart (vendor free-tier limit), so OVERVIEW and BALANCE_SHEET
        never fire back-to-back and concurrent consumers cannot burst."""
        global _av_last_request_at
        with _av_http_lock:
            wait = _av_min_interval_s() - (time.monotonic() - _av_last_request_at)
            if wait > 0:
                time.sleep(wait)
            _av_last_request_at = time.monotonic()
            return self._query(function, symbol)

    def _query(self, function: str, symbol: str):
        """One official-endpoint call. Returns (data, None) or
        (None, normalized-DATA_UNAVAILABLE/ERROR-dict)."""
        params = urllib.parse.urlencode(
            {"function": function, "symbol": symbol,
             "apikey": settings.ALPHAVANTAGE_API_KEY})
        try:
            return _http_get_json(f"{_AV_BASE_URL}?{params}",
                                  _AV_TIMEOUT_SECONDS), None
        except (socket.timeout, TimeoutError) as exc:
            return None, ("ERROR", f"TIMEOUT: {function} request exceeded "
                                   f"{_AV_TIMEOUT_SECONDS:.0f}s ({self._scrub(exc)})")
        except urllib.error.HTTPError as exc:
            return None, ("ERROR", f"HTTP_ERROR: {function} returned "
                                   f"HTTP {exc.code}")
        except urllib.error.URLError as exc:
            return None, ("ERROR", f"NETWORK_ERROR: {function} request failed "
                                   f"({self._scrub(exc.reason)})")
        except (ValueError, json.JSONDecodeError) as exc:
            return None, ("ERROR", f"MALFORMED_RESPONSE: {function} returned "
                                   f"invalid JSON ({self._scrub(exc)})")
        except Exception as exc:  # noqa: BLE001 — provider failures are data
            return None, ("ERROR", f"REQUEST_ERROR: {function} "
                                   f"({self._scrub(exc)})")

    @staticmethod
    def _vendor_refusal(data: dict):
        """Detects Alpha Vantage's structured refusal payloads. Returns
        (status, reason) or None when the response is real data."""
        if "Error Message" in data:
            return "ERROR", ("VENDOR_ERROR: " + str(data["Error Message"])
                             [:_AV_VENDOR_MESSAGE_MAX])
        if "Note" in data:   # rate limit / call-frequency notice
            # DISTINCT status: the provider is NOT broken — we exceeded the
            # free-tier call frequency. Bounded backoff follows; the fetch
            # is never immediately retried.
            return "RATE_LIMITED", ("RATE_LIMITED: " + str(data["Note"])
                                    [:_AV_VENDOR_MESSAGE_MAX])
        if "Information" in data:   # invalid/entitled key guidance
            return "DATA_UNAVAILABLE", ("VENDOR_INFORMATION: "
                                        + str(data["Information"])
                                        [:_AV_VENDOR_MESSAGE_MAX])
        return None

    def get_fundamentals(self, symbol: str) -> dict:
        if not settings.ALPHAVANTAGE_API_KEY:
            reason = "ALPHAVANTAGE_API_KEY_NOT_SET"
            _av_last_fetch.update({"status": "DATA_UNAVAILABLE",
                                   "reason": reason})
            return _normalized(symbol, self.name, "DATA_UNAVAILABLE",
                               reason=reason)

        # --- 1. OVERVIEW (primary: carries most contract fields) ----------
        overview, failure = self._throttled_query("OVERVIEW", symbol)
        if failure:
            status, reason = failure
            _av_last_fetch.update({"status": status, "reason": reason})
            logger.warning("alphavantage OVERVIEW failed for %s: %s",
                           symbol, reason)
            return _normalized(symbol, self.name, status, reason=reason)
        refusal = self._vendor_refusal(overview)
        if refusal:
            status, reason = refusal
            _av_last_fetch.update({"status": status, "reason": reason})
            logger.warning("alphavantage refused OVERVIEW for %s: %s",
                           symbol, reason)
            return _normalized(symbol, self.name, status, reason=reason)
        if not overview:
            reason = ("EMPTY_RESPONSE: OVERVIEW returned no data "
                      "(symbol may be an ETF or unknown to the vendor)")
            _av_last_fetch.update({"status": "DATA_UNAVAILABLE",
                                   "reason": reason})
            return _normalized(symbol, self.name, "DATA_UNAVAILABLE",
                               reason=reason)

        fields = {
            "market_cap": _to_float(overview.get("MarketCapitalization")),
            "pe_ratio": _to_float(overview.get("PERatio")),
            "eps": _to_float(overview.get("EPS")),
            "revenue": _to_float(overview.get("RevenueTTM")),
            "profit_margin": _to_float(overview.get("ProfitMargin")),
            "roe": _to_float(overview.get("ReturnOnEquityTTM")),
            # OVERVIEW carries no debt field on the official API; a future
            # vendor addition would be picked up here for free.
            "debt_to_equity": _to_float(overview.get("DebtEquityRatio")
                                        or overview.get("DebtToEquityRatio")),
        }

        # --- 2. BALANCE_SHEET (only the debt/equity field, only on usable
        #        OVERVIEW data; a failure here never discards OVERVIEW) -----
        balance = None
        if fields["debt_to_equity"] is None and any(
                v is not None for v in fields.values()):
            balance, bs_failure = self._throttled_query("BALANCE_SHEET", symbol)
            bs_refusal = self._vendor_refusal(balance) if balance else None
            if bs_failure:
                logger.warning("alphavantage BALANCE_SHEET failed for %s: %s "
                               "(keeping OVERVIEW data; debt_to_equity=null)",
                               symbol, bs_failure[1])
            elif bs_refusal:
                logger.warning("alphavantage BALANCE_SHEET refused for %s: %s "
                               "(keeping OVERVIEW data; debt_to_equity=null)",
                               symbol, bs_refusal[1])
            elif balance and balance.get("annualReports"):
                latest = balance["annualReports"][0] or {}
                debt = _to_float(latest.get("shortLongTermDebtTotal"))
                equity = _to_float(latest.get("totalShareholderEquity"))
                if debt is not None and equity:
                    fields["debt_to_equity"] = round(debt / equity, 4)

        if not any(v is not None for v in fields.values()):
            reason = ("NO_USABLE_METRICS: OVERVIEW returned no usable "
                      "fundamental fields (symbol may not have fundamentals "
                      "coverage)")
            _av_last_fetch.update({"status": "DATA_UNAVAILABLE",
                                   "reason": reason})
            return _normalized(symbol, self.name, "DATA_UNAVAILABLE",
                               reason=reason)

        _av_last_fetch.update({"status": "OK", "reason": None})
        logger.info("alphavantage fundamentals OK for %s (%d/7 fields)",
                    symbol, sum(v is not None for v in fields.values()))
        result = _normalized(symbol, self.name, "OK", **fields)
        # Per-metric provenance (source field, endpoint, reporting date,
        # period, currency where the vendor reports one). AV's OVERVIEW
        # carries no reporting dates or currency — that absence is recorded
        # honestly instead of being invented.
        result["meta"] = self._provenance(fields, balance)
        return result

    @staticmethod
    def _provenance(fields: dict, balance: dict) -> dict:
        """Provenance for each Alpha Vantage metric: original source field,
        endpoint, reporting date / fiscal period / currency where available.
        AV does not report dates or currency for OVERVIEW fields; the
        balance-sheet debt/equity carries fiscalDateEnding of the annual
        report actually used."""
        bs_date = None
        if isinstance(balance, dict) and balance.get("annualReports"):
            bs_date = str((balance["annualReports"][0] or {})
                          .get("fiscalDateEnding") or "") or None
        meta = {}
        overview_fields = {
            "market_cap": "MarketCapitalization",
            "pe_ratio": "PERatio",
            "eps": "EPS",
            "revenue": "RevenueTTM",
            "profit_margin": "ProfitMargin",
            "roe": "ReturnOnEquityTTM",
        }
        for metric, source_field in overview_fields.items():
            if fields.get(metric) is not None:
                meta[metric] = {
                    "source_field": source_field,
                    "endpoint": "OVERVIEW",
                    "report_date": None,
                    "period": "TTM" if source_field.endswith("TTM") else "current",
                    "currency": None,
                    "note": "reporting date/currency not provided by this vendor",
                }
        if fields.get("debt_to_equity") is not None:
            meta["debt_to_equity"] = {
                "source_field": "shortLongTermDebtTotal/totalShareholderEquity",
                "endpoint": "BALANCE_SHEET",
                "report_date": bs_date,
                "period": "FY (annual)",
                "currency": None,
                "note": "derived from the latest annual report; vendor debt "
                        "scope excludes some lease obligations",
            }
        return meta

    def health_check(self) -> dict:
        """Configuration + last-fetch state. No network call (the free-tier
        quota is never spent on health checks)."""
        if not settings.ALPHAVANTAGE_API_KEY:
            return {
                "provider": self.name,
                "status": "NOT_CONFIGURED",
                "detail": ("fundamentals data provider: alphavantage — "
                           "ALPHAVANTAGE_API_KEY is not set"),
            }
        last_status = _av_last_fetch.get("status")
        if last_status == "OK":
            return {
                "provider": self.name,
                "status": "READY",
                "detail": ("fundamentals data provider: alphavantage — "
                           "last fetch OK (LLM interpretation is routed "
                           "separately via LLM_FUNDAMENTALS_PROVIDER)"),
            }
        if last_status == "RATE_LIMITED":
            # The provider is NOT broken: we exceeded the vendor's free-tier
            # call frequency. Distinct reason + bounded backoff, never a
            # permanent-failure claim.
            return {
                "provider": self.name,
                "status": "DATA_UNAVAILABLE",
                "reason": "RATE_LIMITED",
                "detail": ("provider=alphavantage — reason=RATE_LIMITED: "
                           "free-tier call frequency exceeded; bounded "
                           f"backoff {max(1.0, float(settings.ALPHAVANTAGE_RATE_LIMIT_BACKOFF_SECONDS or 120.0)):.0f}s "
                           "before the next attempt (the provider itself is "
                           "not marked broken)"),
            }
        if last_status in ("DATA_UNAVAILABLE", "ERROR"):
            return {
                "provider": self.name,
                "status": last_status,
                "detail": ("fundamentals data provider: alphavantage — "
                           "last fetch failed: "
                           + str(_av_last_fetch.get("reason"))[:200]),
            }
        return {
            "provider": self.name,
            "status": "PENDING",
            "detail": ("fundamentals data provider: alphavantage — key "
                       "configured, no fetch yet (data is fetched during "
                       "cycles; no request is sent at boot to preserve "
                       "quota)"),
        }



# ---------------------------------------------------------------------------
# Financial Modeling Prep (FMP) — fundamentals DATA provider (official
# Stable API: https://financialmodelingprep.com/stable).
#
# Strict separation (same as Alpha Vantage):
#   FMP            -> FACTUAL FINANCIAL DATA (this class)
#   LLM (routed)   -> INTERPRETATION of that data (agents/fundamentals_agent
#                     via llm_service.call_json("fundamentals", ...))
# The LLM is NOT a data provider; this class never calls llm_service.
#
# Endpoint design (free tier: 250 requests/day):
#   get_fundamentals() uses the MINIMAL endpoint set that fills the 7-field
#   contract, in priority order (each response is cached per symbol AND per
#   endpoint for FMP_CACHE_TTL_SECONDS — statements are NOT re-fetched every
#   trading cycle):
#     1. /profile                  -> market_cap (+ currency, company name);
#                                     cached for TTL/6 (marketCap is
#                                     price-dependent, the statements are not)
#     2. /income-statement         -> revenue, eps (+ netIncome for a
#                                     clearly-labeled derived margin)
#     3. /ratios                   -> pe_ratio, profit_margin, debt_to_equity
#     4. /key-metrics              -> roe
#     5. /balance-sheet-statement  -> ONLY as a clearly-labeled derivation
#                                     fallback (debt_to_equity, roe) when
#                                     /ratios or /key-metrics is restricted
#   /cash-flow-statement is never used by the trading path (no contract
#   field needs it); it is probed ONLY by the operator connectivity
#   diagnostic so the endpoint capability registry stays honest.
#
# Endpoint capability registry: every REAL response updates an in-memory
# record (classification, HTTP status, duration, records, latest reporting
# date, provider error code, validation result). Health checks and
# dashboard polling NEVER issue requests — they only READ the registry.
# Access-restricted / authentication-failed endpoints are not re-tried for
# FMP_CACHE_TTL_SECONDS (per endpoint). A restricted endpoint is reported
# as a subscription limitation, never bypassed.
#
# Quota protection:
#   * daily request accounting — once FMP_DAILY_REQUEST_BUDGET requests
#     have been issued (UTC day), further requests are refused WITHOUT
#     hitting the API; valid cached evidence keeps being served;
#   * bounded exponential backoff after HTTP 429 / rate-limit responses
#     (FMP_RATE_LIMIT_BACKOFF_SECONDS * 2^(consecutive-1), capped at 24h) —
#     never an immediate retry, never a permanent lockout;
#   * per-(symbol, endpoint) cache + in-flight deduplication.
# ---------------------------------------------------------------------------

_FMP_ENDPOINTS = (
    "profile", "income-statement", "balance-sheet-statement",
    "cash-flow-statement", "key-metrics", "ratios",
)

# Fixed classification vocabulary for every FMP response (also used by the
# operator connectivity diagnostic):
#   AVAILABLE             valid data returned (structurally validated)
#   EMPTY_DATA            HTTP 200 but an empty array (unknown symbol)
#   AUTHENTICATION_FAILED 401 / invalid-key payload
#   ACCESS_RESTRICTED     403 / subscription-tier payload
#   RATE_LIMITED          429 / limit-reached payload / local daily budget
#   PROVIDER_ERROR        5xx / other vendor error payloads
#   TIMEOUT               request exceeded FMP_REQUEST_TIMEOUT_SECONDS
#   INVALID_RESPONSE      unparseable JSON / wrong shape

_FMP_VENDOR_MESSAGE_MAX = 160
_FMP_MAX_BACKOFF_S = 24 * 3600.0

# Module-level FMP runtime state (provider instances are rebuilt per
# resolution, so state lives here — same pattern as _av_last_fetch).
_fmp_state = {
    "endpoints": {},                 # endpoint -> capability record
    "usage": {"date": None, "count": 0},   # daily accounting (UTC date)
    "backoff": {"until": 0.0, "consecutive": 0, "reason": None},
    "auth_status": "UNKNOWN",        # UNKNOWN | OK | AUTHENTICATION_FAILED | ACCESS_RESTRICTED
    "last_request_mono": None,       # last FMP HTTP request completed
    "last_success_mono": None,       # last HTTP success with usable payload
    "last_data_mono": None,          # last get_fundamentals() == OK
    "last_report_date": None,        # latest reporting date seen
    "last_status": None,             # last get_fundamentals() outcome
    "last_reason": None,
    "cache_hits": 0,
    "cache_misses": 0,
}
_fmp_state_lock = threading.Lock()
_fmp_http_lock = threading.Lock()          # serializes FMP HTTP requests
_fmp_cache = {}          # (symbol, endpoint) -> (expires_at, rows)
_fmp_inflight = {}       # (symbol, endpoint) -> {"event", "rows", "record"}
_fmp_inflight_lock = threading.Lock()
_FMP_INFLIGHT_WAIT_S = 30.0


def _fmp_reset_state() -> None:
    """Test hook: clears all FMP runtime state."""
    with _fmp_state_lock:
        _fmp_state.update({
            "endpoints": {}, "usage": {"date": None, "count": 0},
            "backoff": {"until": 0.0, "consecutive": 0, "reason": None},
            "auth_status": "UNKNOWN", "last_request_mono": None,
            "last_success_mono": None, "last_data_mono": None,
            "last_report_date": None, "last_status": None, "last_reason": None,
            "cache_hits": 0, "cache_misses": 0,
        })
    _fmp_cache.clear()
    with _fmp_inflight_lock:
        for slot in _fmp_inflight.values():
            slot["event"].set()
        _fmp_inflight.clear()


def _fmp_scrub(text) -> str:
    """Defense in depth: the FMP key must never appear in any reason/log."""
    key = settings.FMP_API_KEY
    text = str(text)
    if key:
        text = text.replace(key, "***")
    return text[:400]


def _fmp_timeout_s() -> float:
    return max(1.0, float(settings.FMP_REQUEST_TIMEOUT_SECONDS or 10.0))


def _fmp_endpoint_ttl_s(endpoint: str) -> float:
    """Cache lifetime per endpoint: financial statements / key metrics /
    ratios are stable between filings (full TTL); the profile carries the
    price-dependent marketCap (TTL/6, min 60s)."""
    ttl = max(0.0, float(settings.FMP_CACHE_TTL_SECONDS or 21600.0))
    if endpoint == "profile":
        return max(60.0, ttl / 6.0)
    return ttl


def _fmp_budget() -> int:
    return max(0, int(settings.FMP_DAILY_REQUEST_BUDGET or 0))


def _fmp_spend_request() -> bool:
    """Counts ONE request against today's (UTC) budget BEFORE it is issued.
    Returns False when the budget is exhausted (the caller must NOT hit the
    API; cached evidence is still served)."""
    today = time.strftime("%Y-%m-%d", time.gmtime())
    with _fmp_state_lock:
        usage = _fmp_state["usage"]
        if usage["date"] != today:
            usage.update({"date": today, "count": 0})
        budget = _fmp_budget()
        if budget and usage["count"] >= budget:
            return False
        usage["count"] += 1
        return True


def _fmp_usage_snapshot() -> dict:
    today = time.strftime("%Y-%m-%d", time.gmtime())
    with _fmp_state_lock:
        usage = dict(_fmp_state["usage"])
    if usage.get("date") != today:
        usage = {"date": today, "count": 0}
    return {"date": usage.get("date"), "count": usage.get("count", 0),
            "budget": _fmp_budget()}


def _fmp_classify(http_status, payload):
    """Maps one FMP HTTP outcome to the fixed classification vocabulary.
    HTTP 200 is NOT classified AVAILABLE unless the payload is an array —
    FMP signals several refusals inside 200 bodies."""
    if http_status == 401:
        return "AUTHENTICATION_FAILED", "HTTP 401"
    if http_status == 403:
        return "ACCESS_RESTRICTED", "HTTP 403"
    if http_status == 429:
        return "RATE_LIMITED", "HTTP 429"
    if http_status is not None and http_status >= 500:
        return "PROVIDER_ERROR", f"HTTP {http_status}"
    if isinstance(payload, dict):
        text = str(payload.get("Error Message")
                   or payload.get("message") or payload.get("error") or "")
        low = text.lower()
        if text:
            if "unauthorized" in low or ("invalid" in low and
                                         ("key" in low or "api" in low)):
                return "AUTHENTICATION_FAILED", text[:_FMP_VENDOR_MESSAGE_MAX]
            if ("subscription" in low or "plan" in low or "upgrade" in low
                    or "entitle" in low or "access" in low):
                return "ACCESS_RESTRICTED", text[:_FMP_VENDOR_MESSAGE_MAX]
            if "limit" in low or "rate" in low:
                return "RATE_LIMITED", text[:_FMP_VENDOR_MESSAGE_MAX]
            return "PROVIDER_ERROR", text[:_FMP_VENDOR_MESSAGE_MAX]
        return "INVALID_RESPONSE", "JSON object instead of the expected array"
    if isinstance(payload, list):
        return ("AVAILABLE" if payload else "EMPTY_DATA"), ""
    return "INVALID_RESPONSE", "payload was not JSON"


def _fmp_validate(endpoint: str, rows):
    """Structural validation of one endpoint payload. Returns
    (validated, records, latest_date). Validation is per the endpoint's
    documented shape: profile rows carry symbol + marketCap/price; statement
    / ratio rows carry date + symbol."""
    if not isinstance(rows, list) or not rows:
        return False, (len(rows) if isinstance(rows, list) else 0), None
    dates = [str(r.get("date") or "")[:10] for r in rows
             if isinstance(r, dict) and r.get("date")]
    latest = max(dates) if dates else None
    newest = rows[0] if isinstance(rows[0], dict) else {}
    if endpoint == "profile":
        validated = bool(newest.get("symbol")) and any(
            newest.get(k) is not None for k in ("marketCap", "mktCap", "price"))
    else:
        validated = bool(newest.get("date")) and bool(
            newest.get("symbol") or newest.get("reportedCurrency"))
    return validated, len(rows), latest


def _fmp_record_endpoint(endpoint: str, symbol: str, status: str,
                         http_status, duration_ms, records, latest_date,
                         error_code, validated) -> None:
    """Updates the endpoint capability registry (in memory; read by health
    checks — never written by them). The error text is scrubbed here as
    well — defense in depth so NO pathway can leak the API key."""
    with _fmp_state_lock:
        record = {
            "endpoint": endpoint, "symbol": symbol, "status": status,
            "http_status": http_status,
            "duration_ms": round(float(duration_ms), 1) if duration_ms else None,
            "records": records, "latest_date": latest_date,
            "error_code": _fmp_scrub(error_code) if error_code else error_code,
            "validated": bool(validated),
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if status in ("ACCESS_RESTRICTED", "AUTHENTICATION_FAILED"):
            # the endpoint is not re-requested for one cache TTL (the
            # restriction is a subscription fact, not a transient error)
            record["_restricted_until"] = (time.monotonic()
                                           + _fmp_endpoint_ttl_s(endpoint))
        _fmp_state["endpoints"][endpoint] = record
        if status in ("AVAILABLE", "EMPTY_DATA"):
            # an empty array is an AUTHORIZED, executed request — but it
            # must not erase a hard auth failure observed moments earlier
            # in the same pass (a broken key does not un-break mid-request;
            # a later AVAILABLE fetch still clears the failure).
            sticky = (status == "EMPTY_DATA" and
                      _fmp_state["auth_status"] in ("AUTHENTICATION_FAILED",
                                                    "ACCESS_RESTRICTED"))
            if not sticky:
                _fmp_state["auth_status"] = "OK"
                _fmp_state["last_success_mono"] = time.monotonic()
        elif status == "AUTHENTICATION_FAILED":
            _fmp_state["auth_status"] = "AUTHENTICATION_FAILED"
        elif status == "ACCESS_RESTRICTED":
            _fmp_state["auth_status"] = "ACCESS_RESTRICTED"
        if latest_date and status == "AVAILABLE":
            current = _fmp_state["last_report_date"]
            if not current or str(latest_date) > str(current):
                _fmp_state["last_report_date"] = str(latest_date)


def _fmp_backoff_snapshot() -> dict:
    with _fmp_state_lock:
        backoff = dict(_fmp_state["backoff"])
    remaining = max(0.0, backoff.get("until", 0.0) - time.monotonic())
    return {"active": remaining > 0, "remaining_s": round(remaining, 1),
            "reason": backoff.get("reason") if remaining > 0 else None}


def _fmp_http_get(endpoint: str, symbol: str):
    """ONE bounded FMP HTTP GET through the shared FMP lock. Returns
    (http_status, payload, transport_error) where transport_error is None
    or (classification, detail). Never raises, never logs the key."""
    params = urllib.parse.urlencode(
        {"symbol": symbol, "apikey": settings.FMP_API_KEY})
    url = f"{str(settings.FMP_BASE_URL).rstrip('/')}/{endpoint}?{params}"
    timeout_s = _fmp_timeout_s()
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "ai-trader-paper/1.0"})
        with _fmp_http_lock:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read().decode("utf-8", errors="replace")
                status = response.status
        try:
            payload = json.loads(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            return status, None, ("INVALID_RESPONSE",
                                  f"invalid JSON ({_fmp_scrub(exc)})")
        return status, payload, None
    except (socket.timeout, TimeoutError) as exc:
        return None, None, ("TIMEOUT",
                            f"request exceeded {timeout_s:.0f}s "
                            f"({_fmp_scrub(exc)})")
    except urllib.error.HTTPError as exc:
        # 4xx/5xx: body may still carry the vendor's structured message
        body = {}
        try:
            body = json.loads(exc.read().decode("utf-8", errors="replace"))
            if not isinstance(body, dict):
                body = {}
        except Exception:  # noqa: BLE001 — body is best-effort context
            body = {}
        return exc.code, body or None, None
    except urllib.error.URLError as exc:
        return None, None, ("PROVIDER_ERROR",
                            f"network error ({_fmp_scrub(exc.reason)})")
    except Exception as exc:  # noqa: BLE001 — provider failures are data
        return None, None, ("PROVIDER_ERROR", f"request error "
                                              f"({_fmp_scrub(exc)})")


def _fmp_fetch_endpoint(symbol: str, endpoint: str) -> tuple:
    """Fetches one endpoint for one symbol through ALL the protection
    layers (cache -> registry skip -> budget -> backoff -> single-flight
    -> HTTP). Returns (rows, record) — rows is a list on success, None on
    any failure; record is the capability/failure record (never None)."""
    cache_key = (symbol, endpoint)

    # 1. valid cached response -> no API call
    cached = _fmp_cache.get(cache_key)
    if cached and cached[0] > time.monotonic():
        with _fmp_state_lock:
            _fmp_state["cache_hits"] += 1
        rows = cached[1]
        status = "EMPTY_DATA" if not rows else "AVAILABLE"
        return rows, {"endpoint": endpoint, "status": status,
                      "cached": True, "validated": bool(rows)}
    with _fmp_state_lock:
        _fmp_state["cache_misses"] += 1

    # 2. recently-established endpoint restriction -> do not re-request
    with _fmp_state_lock:
        record = _fmp_state["endpoints"].get(endpoint) or {}
    restricted_until = (record.get("_restricted_until") or 0) \
        if isinstance(record, dict) else 0
    if record.get("status") in ("ACCESS_RESTRICTED", "AUTHENTICATION_FAILED") \
            and time.monotonic() < restricted_until:
        return None, {"endpoint": endpoint, "status": record["status"],
                      "error_code": record.get("error_code"),
                      "detail": "endpoint known restricted (cached); "
                                "not re-requested",
                      "cached": True}

    # 3. daily budget -> refuse WITHOUT hitting the API
    if not _fmp_spend_request():
        usage = _fmp_usage_snapshot()
        return None, {
            "endpoint": endpoint, "status": "RATE_LIMITED",
            "error_code": "LOCAL_DAILY_BUDGET",
            "detail": (f"FMP daily request budget exhausted "
                       f"({usage['count']}/{usage['budget']} on "
                       f"{usage['date']}); fresh retrieval unavailable "
                       f"today — valid cached evidence is preserved"),
        }

    # 4. rate-limit backoff -> refuse WITHOUT hitting the API
    backoff = _fmp_backoff_snapshot()
    if backoff["active"]:
        return None, {
            "endpoint": endpoint, "status": "RATE_LIMITED",
            "error_code": "RATE_LIMIT_BACKOFF",
            "detail": (f"in bounded backoff after a rate-limit response "
                       f"({backoff['remaining_s']:.0f}s remaining; "
                       f"reason: {backoff['reason']})"),
        }

    # 5. single-flight: join an in-flight fetch for this (symbol, endpoint)
    with _fmp_inflight_lock:
        slot = _fmp_inflight.get(cache_key)
        if slot is None:
            slot = {"event": threading.Event(), "rows": None, "record": None}
            _fmp_inflight[cache_key] = slot
            fetcher = True
        else:
            fetcher = False
    if not fetcher:
        if slot["event"].wait(timeout=_FMP_INFLIGHT_WAIT_S) \
                and slot["record"] is not None:
            return slot["rows"], slot["record"]
        # pathological: fetcher vanished — fall through and fetch ourselves

    started = time.monotonic()
    try:
        http_status, payload, transport_error = _fmp_http_get(endpoint, symbol)
        duration_ms = (time.monotonic() - started) * 1000.0

        if transport_error is not None:
            status, detail = transport_error
            if status == "RATE_LIMITED":
                with _fmp_state_lock:
                    bo = _fmp_state["backoff"]
                    bo["consecutive"] += 1
                    base = max(1.0, float(
                        settings.FMP_RATE_LIMIT_BACKOFF_SECONDS or 300.0))
                    wait = min(_FMP_MAX_BACKOFF_S,
                               base * (2 ** (bo["consecutive"] - 1)))
                    bo["until"] = time.monotonic() + wait
                    bo["reason"] = detail
            else:
                with _fmp_state_lock:
                    # any non-rate-limit completion resets the escalation
                    _fmp_state["backoff"]["consecutive"] = 0
            with _fmp_state_lock:
                _fmp_state["last_request_mono"] = time.monotonic()
            _fmp_record_endpoint(endpoint, symbol, status, http_status,
                                 duration_ms, None, None,
                                 detail[:_FMP_VENDOR_MESSAGE_MAX], False)
            rows, record = None, {"endpoint": endpoint, "status": status,
                                  "error_code": _fmp_scrub(
                                      detail)[:_FMP_VENDOR_MESSAGE_MAX],
                                  "http_status": http_status,
                                  "duration_ms": round(duration_ms, 1)}
            if status in ("ACCESS_RESTRICTED", "AUTHENTICATION_FAILED"):
                record["_restricted_until"] = (time.monotonic()
                                               + _fmp_endpoint_ttl_s(endpoint))
        else:
            status, detail = _fmp_classify(http_status, payload)
            with _fmp_state_lock:
                _fmp_state["last_request_mono"] = time.monotonic()
                if status == "RATE_LIMITED":
                    bo = _fmp_state["backoff"]
                    bo["consecutive"] += 1
                    base = max(1.0, float(
                        settings.FMP_RATE_LIMIT_BACKOFF_SECONDS or 300.0))
                    wait = min(_FMP_MAX_BACKOFF_S,
                               base * (2 ** (bo["consecutive"] - 1)))
                    bo["until"] = time.monotonic() + wait
                    bo["reason"] = detail or f"HTTP {http_status}"
                else:
                    _fmp_state["backoff"]["consecutive"] = 0
            validated, records, latest_date = _fmp_validate(endpoint, payload) \
                if isinstance(payload, list) else (False, None, None)
            _fmp_record_endpoint(endpoint, symbol, status, http_status,
                                 duration_ms, records, latest_date,
                                 detail or None, validated)
            rows = payload if (validated and isinstance(payload, list)) else None
            record = {"endpoint": endpoint, "status": status,
                      "http_status": http_status,
                      "duration_ms": round(duration_ms, 1),
                      "records": records, "latest_date": latest_date,
                      "error_code": detail or None,
                      "validated": validated}
            if isinstance(payload, list):
                # cache real arrays (also empty ones — unknown symbols must
                # not burn the quota; shorter TTL for empty results)
                ttl = _fmp_endpoint_ttl_s(endpoint) if payload \
                    else max(60.0, _fmp_endpoint_ttl_s(endpoint) / 12.0)
                _fmp_cache[cache_key] = (time.monotonic() + ttl, payload)
                if record["status"] in ("ACCESS_RESTRICTED",
                                        "AUTHENTICATION_FAILED"):
                    record["_restricted_until"] = time.monotonic() + \
                        _fmp_endpoint_ttl_s(endpoint)
        return rows, record
    finally:
        with _fmp_inflight_lock:
            slot["rows"], slot["record"] = rows, record
            if _fmp_inflight.get(cache_key) is slot:
                _fmp_inflight.pop(cache_key, None)
        slot["event"].set()


class FMPProvider(FundamentalsProvider):
    """Fundamentals DATA provider using FMP's official Stable API. Returns
    the normalized contract; every metric carries provenance (source field,
    endpoint, reporting date, fiscal year, period, currency, retrieval
    time). Failures are structured DATA_UNAVAILABLE/ERROR/RATE_LIMITED
    with a categorized reason — never an exception through the trading
    cycle, and never the API key."""

    name = "fmp"

    @staticmethod
    def _statement_meta(row: dict, endpoint: str) -> dict:
        return {
            "endpoint": endpoint,
            "report_date": str(row.get("date") or "")[:10] or None,
            "fiscal_year": str(row.get("fiscalYear") or "") or None,
            "period": str(row.get("period") or "") or None,
            "currency": str(row.get("reportedCurrency") or "") or None,
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
        }

    def get_fundamentals(self, symbol: str) -> dict:
        if not settings.FMP_API_KEY:
            reason = "FMP_API_KEY_NOT_SET"
            with _fmp_state_lock:
                _fmp_state["last_status"] = "DATA_UNAVAILABLE"
                _fmp_state["last_reason"] = reason
            return _normalized(symbol, self.name, "DATA_UNAVAILABLE",
                               reason=reason)

        fields = {}
        meta = {}
        failures = []

        def _record_failure(endpoint: str, rec: dict) -> None:
            if rec and rec.get("status") not in ("AVAILABLE",):
                failures.append(f"{endpoint}: {rec.get('status')}"
                                + (f" ({rec.get('error_code')})"
                                   if rec.get("error_code") else ""))

        # --- 1. /profile: market_cap + trading currency -------------------
        profile, rec = _fmp_fetch_endpoint(symbol, "profile")
        _record_failure("profile", rec)
        if profile:
            row = profile[0] if isinstance(profile[0], dict) else {}
            mcap = row.get("marketCap")
            if mcap is None:
                mcap = row.get("mktCap")   # legacy field name
            mcap = _to_float(mcap)
            if mcap is not None:
                fields["market_cap"] = mcap
                meta["market_cap"] = {
                    "source_field": "marketCap", "endpoint": "profile",
                    "report_date": None, "fiscal_year": None,
                    "period": "current (price-dependent)",
                    "currency": str(row.get("currency") or "") or None,
                    "note": "market capitalization at retrieval time "
                            "(not enterprise value)",
                    "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime()),
                }

        # --- 2. /income-statement: revenue, eps ---------------------------
        income, rec = _fmp_fetch_endpoint(symbol, "income-statement")
        _record_failure("income-statement", rec)
        net_income = None
        income_row = {}
        if income:
            income_row = income[0] if isinstance(income[0], dict) else {}
            smeta = self._statement_meta(income_row, "income-statement")
            revenue = _to_float(income_row.get("revenue"))
            if revenue is not None:
                fields["revenue"] = revenue
                meta["revenue"] = {**smeta, "source_field": "revenue"}
            eps = _to_float(income_row.get("eps"))
            if eps is not None:
                fields["eps"] = eps
                meta["eps"] = {**smeta, "source_field": "eps",
                               "note": "basic EPS (diluted is epsDiluted; "
                                       "never mixed silently)"}
            net_income = _to_float(income_row.get("netIncome"))

        # --- 3. /ratios: pe_ratio, profit_margin, debt_to_equity ----------
        ratios, rec = _fmp_fetch_endpoint(symbol, "ratios")
        ratios_ok = bool(ratios)
        _record_failure("ratios", rec)
        if ratios:
            row = ratios[0] if isinstance(ratios[0], dict) else {}
            smeta = self._statement_meta(row, "ratios")
            pe = _to_float(row.get("priceEarningsRatio"))
            if pe is not None:
                fields["pe_ratio"] = pe
                meta["pe_ratio"] = {**smeta,
                                    "source_field": "priceEarningsRatio",
                                    "note": "price-dependent at the report date"}
            margin = _to_float(row.get("netProfitMargin"))
            if margin is not None:
                fields["profit_margin"] = margin
                meta["profit_margin"] = {**smeta,
                                         "source_field": "netProfitMargin"}
            dte = _to_float(row.get("debtToEquityRatio"))
            if dte is not None:
                fields["debt_to_equity"] = dte
                meta["debt_to_equity"] = {
                    **smeta, "source_field": "debtToEquityRatio",
                    "note": "vendor definition (totalDebt incl. capital "
                            "leases / equity)"}

        # --- 4. /key-metrics: roe (+ market_cap fallback) ------------------
        keym, rec = _fmp_fetch_endpoint(symbol, "key-metrics")
        keym_ok = bool(keym)
        _record_failure("key-metrics", rec)
        if keym:
            row = keym[0] if isinstance(keym[0], dict) else {}
            smeta = self._statement_meta(row, "key-metrics")
            roe = _to_float(row.get("returnOnEquity"))
            if roe is not None:
                fields["roe"] = roe
                meta["roe"] = {**smeta, "source_field": "returnOnEquity"}
            if "market_cap" not in fields:
                mcap = _to_float(row.get("marketCap"))
                if mcap is not None:
                    fields["market_cap"] = mcap
                    meta["market_cap"] = {
                        **smeta, "source_field": "marketCap",
                        "note": "market cap as of the report date "
                                "(profile was unavailable)"}

        # --- 5. /balance-sheet-statement: derivation fallbacks only -------
        need_balance = ((fields.get("debt_to_equity") is None and not ratios_ok)
                        or (fields.get("roe") is None and not keym_ok))
        if need_balance:
            balance, rec = _fmp_fetch_endpoint(symbol,
                                               "balance-sheet-statement")
            _record_failure("balance-sheet-statement", rec)
            if balance:
                row = balance[0] if isinstance(balance[0], dict) else {}
                smeta = self._statement_meta(row, "balance-sheet-statement")
                debt = _to_float(row.get("totalDebt"))
                equity = _to_float(row.get("totalStockholdersEquity"))
                if (fields.get("debt_to_equity") is None and debt is not None
                        and equity):
                    fields["debt_to_equity"] = round(debt / equity, 4)
                    meta["debt_to_equity"] = {
                        **smeta,
                        "source_field": "totalDebt/totalStockholdersEquity",
                        "derived": True,
                        "note": "DERIVED (ratios endpoint unavailable): "
                                "total debt / shareholders equity from the "
                                "latest statement"}
                if (fields.get("roe") is None and net_income is not None
                        and equity):
                    fields["roe"] = round(net_income / equity, 4)
                    meta["roe"] = {
                        **smeta,
                        "source_field": "netIncome/totalStockholdersEquity",
                        "derived": True,
                        "note": "DERIVED (key-metrics endpoint unavailable): "
                                "net income / shareholders equity — periods "
                                "must match the statement used"}

        # derived profit margin (only when /ratios gave none)
        if (fields.get("profit_margin") is None and net_income is not None
                and fields.get("revenue")):
            fields["profit_margin"] = round(net_income / fields["revenue"], 6)
            meta["profit_margin"] = {
                **self._statement_meta(income_row, "income-statement"),
                "source_field": "netIncome/revenue", "derived": True,
                "note": "DERIVED (ratios endpoint gave no margin): net "
                        "income / revenue from the same statement"}

        if not any(v is not None for v in fields.values()):
            reason = ("NO_USABLE_METRICS: no FMP endpoint returned usable "
                      "fundamental fields"
                      + (f" — endpoint failures: {'; '.join(failures[:3])}"
                         if failures else
                         " (symbol may not have fundamentals coverage)"))
            with _fmp_state_lock:
                _fmp_state["last_status"] = "DATA_UNAVAILABLE"
                _fmp_state["last_reason"] = reason
            logger.warning("fmp fundamentals unavailable for %s: %s",
                           symbol, reason)
            return _normalized(symbol, self.name, "DATA_UNAVAILABLE",
                               reason=reason)

        with _fmp_state_lock:
            _fmp_state["last_status"] = "OK"
            _fmp_state["last_reason"] = None
            _fmp_state["last_data_mono"] = time.monotonic()
            report_dates = [m.get("report_date") for m in meta.values()
                            if isinstance(m, dict) and m.get("report_date")]
            if report_dates:
                _fmp_state["last_report_date"] = max(report_dates)
        logger.info("fmp fundamentals OK for %s (%d/7 fields)",
                    symbol, sum(v is not None for v in fields.values()))
        result = _normalized(symbol, self.name, "OK", **fields)
        result["meta"] = meta
        return result

    def health_check(self) -> dict:
        """Configuration + runtime state for /api/health. NO network call —
        the daily budget is never spent on health checks, and the dashboard
        polling never triggers provider requests."""
        if not settings.FMP_API_KEY:
            status = ("CONFIGURATION_ERROR"
                      if settings.FUNDAMENTALS_PROVIDER == self.name
                      else "NOT_CONFIGURED")
            return {
                "provider": self.name, "status": status,
                "detail": ("fundamentals data provider: fmp — "
                           "FMP_API_KEY is not set"
                           + (" (selected via FUNDAMENTALS_PROVIDER=fmp)"
                              if status == "CONFIGURATION_ERROR" else "")),
            }
        with _fmp_state_lock:
            state = {k: (dict(v) if isinstance(v, dict) else v)
                     for k, v in _fmp_state.items() if k != "endpoints"}
            endpoints = {k: {kk: vv for kk, vv in v.items()
                             if not kk.startswith("_")}
                         for k, v in _fmp_state["endpoints"].items()}
        backoff = _fmp_backoff_snapshot()
        usage = _fmp_usage_snapshot()
        budget_exhausted = (usage["budget"] > 0
                            and usage["count"] >= usage["budget"])
        last_status = state.get("last_status")

        if backoff["active"] or budget_exhausted:
            reason = ("daily budget exhausted" if budget_exhausted
                      else f"rate-limit backoff ({backoff['reason']})")
            return {
                "provider": self.name, "status": "RATE_LIMITED",
                "detail": (f"fundamentals data provider: fmp — {reason}; "
                           f"valid cached evidence is preserved and served"),
                "auth_status": state.get("auth_status"),
                "usage": usage, "backoff": backoff, "endpoints": endpoints,
                "cache": {"hits": state.get("cache_hits", 0),
                          "misses": state.get("cache_misses", 0),
                          "entries": len(_fmp_cache)},
                "last_request_at": _mono_to_iso(state.get("last_request_mono")),
                "last_data_at": _mono_to_iso(state.get("last_data_mono")),
                "last_report_date": state.get("last_report_date"),
            }
        if last_status == "OK":
            return {
                "provider": self.name, "status": "READY",
                "detail": ("fundamentals data provider: fmp — last fetch OK "
                           "(LLM interpretation is routed separately via "
                           "LLM_FUNDAMENTALS_PROVIDER)"),
                "auth_status": state.get("auth_status"),
                "usage": usage, "backoff": backoff, "endpoints": endpoints,
                "cache": {"hits": state.get("cache_hits", 0),
                          "misses": state.get("cache_misses", 0),
                          "entries": len(_fmp_cache)},
                "last_request_at": _mono_to_iso(state.get("last_request_mono")),
                "last_data_at": _mono_to_iso(state.get("last_data_mono")),
                "last_report_date": state.get("last_report_date"),
            }
        if state.get("auth_status") == "AUTHENTICATION_FAILED":
            return {"provider": self.name, "status": "ERROR",
                    "detail": "fundamentals data provider: fmp — "
                              "authentication failed (check FMP_API_KEY)",
                    "auth_status": "AUTHENTICATION_FAILED",
                    "usage": usage, "backoff": backoff, "endpoints": endpoints,
                    "cache": {"hits": state.get("cache_hits", 0),
                              "misses": state.get("cache_misses", 0),
                              "entries": len(_fmp_cache)},
                    "last_request_at": _mono_to_iso(
                        state.get("last_request_mono")),
                    "last_data_at": _mono_to_iso(
                        state.get("last_data_mono")),
                    "last_report_date": state.get("last_report_date")}
        if state.get("auth_status") == "ACCESS_RESTRICTED":
            return {"provider": self.name, "status": "ACCESS_RESTRICTED",
                    "detail": ("fundamentals data provider: fmp — requested "
                               "endpoints are restricted on this "
                               "subscription tier"),
                    "auth_status": "ACCESS_RESTRICTED",
                    "usage": usage, "backoff": backoff, "endpoints": endpoints,
                    "cache": {"hits": state.get("cache_hits", 0),
                              "misses": state.get("cache_misses", 0),
                              "entries": len(_fmp_cache)},
                    "last_request_at": _mono_to_iso(
                        state.get("last_request_mono")),
                    "last_data_at": _mono_to_iso(
                        state.get("last_data_mono")),
                    "last_report_date": state.get("last_report_date")}
        if last_status in ("DATA_UNAVAILABLE", "ERROR"):
            return {"provider": self.name, "status": last_status,
                    "detail": ("fundamentals data provider: fmp — last "
                               "fetch failed: "
                               + str(state.get("last_reason"))[:200]),
                    "auth_status": state.get("auth_status"),
                    "usage": usage, "backoff": backoff, "endpoints": endpoints,
                    "cache": {"hits": state.get("cache_hits", 0),
                              "misses": state.get("cache_misses", 0),
                              "entries": len(_fmp_cache)},
                    "last_request_at": _mono_to_iso(
                        state.get("last_request_mono")),
                    "last_data_at": _mono_to_iso(
                        state.get("last_data_mono")),
                    "last_report_date": state.get("last_report_date")}
        return {
            "provider": self.name, "status": "PENDING",
            "detail": ("fundamentals data provider: fmp — key configured, "
                       "no fetch yet (data is fetched during cycles; no "
                       "request is sent at boot to preserve the daily "
                       "budget; READY is only shown after a successful "
                       "fetch)"),
            "auth_status": state.get("auth_status"),
            "usage": usage, "backoff": backoff, "endpoints": endpoints,
            "cache": {"hits": state.get("cache_hits", 0),
                      "misses": state.get("cache_misses", 0),
                      "entries": len(_fmp_cache)},
            "last_request_at": _mono_to_iso(state.get("last_request_mono")),
            "last_data_at": _mono_to_iso(state.get("last_data_mono")),
            "last_report_date": state.get("last_report_date"),
        }


def fmp_connectivity_probe(symbol: str = "AAPL") -> dict:
    """ON-DEMAND FMP connectivity diagnostic (operator-initiated; NEVER run
    at startup or by health checks). Probes each documented Stable API
    endpoint once, through the same budget/backoff/cache gates as the
    trading path, and records the outcome in the endpoint capability
    registry. Endpoints probed within the last 10 minutes reuse their
    registry record (no repeated requests). Returns the per-endpoint
    classification table — an endpoint is only "accessible" when its
    payload passed STRUCTURAL validation, not merely HTTP 200."""
    symbol = str(symbol or "AAPL").strip().upper() or "AAPL"
    if not settings.FMP_API_KEY:
        return {
            "symbol": symbol, "configured": False,
            "error": "FMP_API_KEY is not set — nothing to probe",
            "endpoints": [],
        }
    usage = _fmp_usage_snapshot()
    backoff = _fmp_backoff_snapshot()
    endpoints_out = []
    for endpoint in _FMP_ENDPOINTS:
        rows, record = _fmp_fetch_endpoint(symbol, endpoint)
        entry = {
            "endpoint": endpoint,
            "http_status": record.get("http_status"),
            "request_duration_ms": record.get("duration_ms"),
            "classification": record.get("status"),
            "accessible": record.get("status") == "AVAILABLE",
            "records_returned": record.get("records"),
            "latest_reporting_date": record.get("latest_date"),
            "provider_error": record.get("error_code"),
            "validated": bool(record.get("validated")),
            "cached": bool(record.get("cached")),
        }
        if record.get("status") == "ACCESS_RESTRICTED":
            entry["note"] = ("restricted on this subscription tier — "
                             "reported as a limitation, never bypassed")
        elif record.get("status") == "RATE_LIMITED":
            entry["note"] = "rate limited (budget/backoff preserved cached data)"
        endpoints_out.append(entry)
    available = [e["endpoint"] for e in endpoints_out if e["accessible"]]
    restricted = [e["endpoint"] for e in endpoints_out
                  if e["classification"] == "ACCESS_RESTRICTED"]
    return {
        "symbol": symbol,
        "configured": True,
        "base_url": settings.FMP_BASE_URL,
        "usage": usage,
        "backoff": backoff,
        "endpoints": endpoints_out,
        "summary": {
            "available": available,
            "restricted": restricted,
            "available_count": len(available),
            "probed_count": len(endpoints_out),
        },
    }


def _mono_to_iso(monotonic_value) -> str:
    """Best-effort wall-clock rendering of a monotonic stamp recorded by
    this process (None -> None)."""
    if monotonic_value is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


_PROVIDER_CLASSES = {
    "none": NoneProvider,
    "alphavantage": AlphaVantageProvider,
    "fmp": FMPProvider,
}


def available_providers() -> list:
    """Registered provider names (for validation and /api/config)."""
    return sorted(_PROVIDER_CLASSES)


def _build(name: str) -> FundamentalsProvider:
    cls = _PROVIDER_CLASSES.get(name)
    return cls() if cls else None


# ---------------------------------------------------------------------------
# Service-level resolution: primary -> optional fallback, with a short
# per-symbol result cache so one cycle never asks twice.
# ---------------------------------------------------------------------------

_result_cache: dict = {}  # symbol -> (expires_at_monotonic, result)
# Concurrent-consumer coalescing: one in-flight fetch per symbol. Everyone
# else asking for the same symbol waits (bounded) and shares the result —
# a cycle thread and an operator inspection can never double-fetch.
_fetch_inflight: dict = {}          # symbol -> {"event", "result"}
_fetch_inflight_lock = threading.Lock()
_INFLIGHT_WAIT_S = 30.0             # bounded wait (fetch itself is <= ~11s)


def reset_cache() -> None:
    """Test hook: clears the result cache."""
    _result_cache.clear()
    with _fetch_inflight_lock:
        _fetch_inflight.clear()


def _result_ttl_s(result: dict) -> float:
    """Cache lifetime for one result. Successful/normal results use the
    configured freshness TTL; a vendor RATE_LIMITED result is held for the
    BOUNDED BACKOFF window instead (never an immediate retry, never a
    permanent lockout). Other failures keep the normal TTL."""
    if isinstance(result, dict) and result.get("status") == "RATE_LIMITED":
        return max(1.0, float(settings.ALPHAVANTAGE_RATE_LIMIT_BACKOFF_SECONDS or 120.0))
    return max(0.0, float(settings.FUNDAMENTALS_RESULT_CACHE_TTL_SECONDS))


# Last multi-provider evidence outcome (for health reporting; NO network).
_last_evidence = {"at": None, "symbol": None, "result": None}


def _fetch_provider_safely(name: str, symbol: str) -> dict:
    """Runs one provider's get_fundamentals with exception containment."""
    provider = _build(name)
    if provider is None:
        return _normalized(
            symbol, name, "DATA_UNAVAILABLE",
            reason=f"UNKNOWN_PROVIDER ({name}); registered: "
                   f"{', '.join(available_providers())}")
    try:
        return provider.get_fundamentals(symbol) or {}
    except Exception as exc:  # noqa: BLE001 — provider failures are data
        logger.error(f"fundamentals provider '{name}' raised for {symbol}: {exc}")
        return _normalized(symbol, name, "ERROR", reason=str(exc))


def _fetch_cross_validated(symbol: str, primary_name: str,
                           fallback_name: str) -> dict:
    """FUNDAMENTALS_CROSS_VALIDATION=true: BOTH providers are fetched
    INDEPENDENTLY (a failure of one never blocks the other) and their
    metrics are cross-validated by services/fundamentals_evidence before
    the agent sees the evidence. Values are never averaged; conflicts stay
    visible per metric."""
    from services import fundamentals_evidence  # deferred: no import cycle

    results = {
        primary_name: _fetch_provider_safely(primary_name, symbol),
        fallback_name: _fetch_provider_safely(fallback_name, symbol),
    }
    merged = fundamentals_evidence.merge_evidence(results)
    _last_evidence.update({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime()),
                           "symbol": symbol, "result": merged})
    return merged


def _fetch_once(symbol: str) -> dict:
    """Primary provider first; on DATA_UNAVAILABLE/ERROR, the configured
    fallback (if any). With FUNDAMENTALS_CROSS_VALIDATION=true and a real
    fallback configured, BOTH are fetched and cross-validated. Never
    raises."""
    primary_name = settings.FUNDAMENTALS_PROVIDER
    fallback_name = settings.FUNDAMENTALS_FALLBACK_PROVIDER

    if (settings.FUNDAMENTALS_CROSS_VALIDATION and fallback_name
            and fallback_name in _PROVIDER_CLASSES
            and fallback_name != "none"
            and primary_name in _PROVIDER_CLASSES
            and primary_name != "none"):
        return _fetch_cross_validated(symbol, primary_name, fallback_name)

    primary = _build(primary_name)
    if primary is None:
        return _normalized(
            symbol, primary_name, "DATA_UNAVAILABLE",
            reason=f"UNKNOWN_PROVIDER ({primary_name}); registered: {', '.join(available_providers())}",
        )

    try:
        result = primary.get_fundamentals(symbol) or {}
    except Exception as exc:  # noqa: BLE001 — provider failures are data
        logger.error(f"fundamentals provider '{primary_name}' raised for {symbol}: {exc}")
        result = _normalized(symbol, primary_name, "ERROR", reason=str(exc))

    if result.get("status") == "OK" or not fallback_name:
        return result

    fallback = _build(fallback_name)
    if fallback is None:
        logger.warning(f"FUNDAMENTALS_FALLBACK_PROVIDER '{fallback_name}' is not registered.")
        return result

    logger.info(
        f"fundamentals: primary '{primary_name}' returned {result.get('status')} "
        f"for {symbol}; trying fallback '{fallback_name}'."
    )
    try:
        fb_result = fallback.get_fundamentals(symbol) or {}
    except Exception as exc:  # noqa: BLE001
        logger.error(f"fundamentals fallback '{fallback_name}' raised for {symbol}: {exc}")
        fb_result = _normalized(symbol, fallback_name, "ERROR", reason=str(exc))
    return fb_result if fb_result.get("status") == "OK" else result


def get_fundamentals(symbol: str) -> dict:
    """Normalized fundamentals for `symbol`. Never raises; failures carry
    status DATA_UNAVAILABLE/ERROR/RATE_LIMITED with a reason.

    Free-tier protection (all three layers):
      1. per-symbol result cache — a valid cached result is returned WITHOUT
         any API call (rate-limited results are held for the bounded backoff
         window, not the normal TTL);
      2. per-symbol in-flight coalescing — simultaneous requests for the
         same symbol share ONE fetch;
      3. the provider-level throttle spaces every HTTP request.
    """
    cached = _result_cache.get(symbol)
    if cached and cached[0] > time.monotonic():
        return cached[1]

    # --- single-flight: join an in-flight fetch for this symbol ----------
    with _fetch_inflight_lock:
        slot = _fetch_inflight.get(symbol)
        if slot is None:
            slot = {"event": threading.Event(), "result": None}
            _fetch_inflight[symbol] = slot
            fetcher = True
        else:
            fetcher = False
    if not fetcher:
        if slot["event"].wait(timeout=_INFLIGHT_WAIT_S) and slot["result"] is not None:
            return slot["result"]
        # pathological: the fetcher vanished without setting the event —
        # fall through and fetch ourselves (still throttled + cached)

    result = None
    try:
        result = _fetch_once(symbol)
    finally:
        # always publish + wake waiters, even on a pathological raise
        with _fetch_inflight_lock:
            slot["result"] = result
            if _fetch_inflight.get(symbol) is slot:
                _fetch_inflight.pop(symbol, None)
        slot["event"].set()

    ttl = _result_ttl_s(result)
    if ttl > 0:
        _result_cache[symbol] = (time.monotonic() + ttl, result)
    return result


def provider_config() -> dict:
    """Configured fundamentals providers (for /api/config; no secrets).
    Clearly separates the DATA provider from the LLM that INTERPRETS it:
    an LLM is never a fundamentals data source."""
    # Deferred import: llm_service must not be a hard dependency of the
    # data layer at import time.
    try:
        from services import llm_service
        route = llm_service.route_info("fundamentals")
        llm_route = {"provider": route.get("provider"),
                     "model": route.get("model"),
                     "role": "interpretation (LLM; never a data source)"}
    except Exception:  # noqa: BLE001 — config view must never crash
        llm_route = {"provider": None, "model": None,
                     "role": "interpretation (LLM; never a data source)"}
    return {
        "primary": settings.FUNDAMENTALS_PROVIDER,
        "fallback": settings.FUNDAMENTALS_FALLBACK_PROVIDER or None,
        "available": available_providers(),
        "llm_route": llm_route,
    }


def health_check() -> dict:
    """Fundamentals section of the startup/health report."""
    primary = _build(settings.FUNDAMENTALS_PROVIDER)
    if primary is None:
        return {
            "provider": settings.FUNDAMENTALS_PROVIDER,
            "fallback": settings.FUNDAMENTALS_FALLBACK_PROVIDER or None,
            "status": "DATA_UNAVAILABLE",
            "detail": f"unknown provider '{settings.FUNDAMENTALS_PROVIDER}'",
        }
    health = primary.health_check()
    if settings.FUNDAMENTALS_FALLBACK_PROVIDER:
        fb = _build(settings.FUNDAMENTALS_FALLBACK_PROVIDER)
        health["fallback"] = settings.FUNDAMENTALS_FALLBACK_PROVIDER if fb else \
            f"unknown ({settings.FUNDAMENTALS_FALLBACK_PROVIDER})"
    return health


def providers_health() -> dict:
    """Per-provider fundamentals health for /api/health (FMP + Alpha
    Vantage rows + the multi-provider evidence summary). Pure in-memory
    state — health checks and dashboard polling NEVER trigger provider
    requests."""
    providers = {}
    for name in ("fmp", "alphavantage"):
        provider = _build(name)
        if provider is None:
            continue
        key_configured = bool(
            getattr(settings, "FMP_API_KEY", None) if name == "fmp"
            else getattr(settings, "ALPHAVANTAGE_API_KEY", None))
        selected = settings.FUNDAMENTALS_PROVIDER == name
        if not (key_configured or selected
                or settings.FUNDAMENTALS_FALLBACK_PROVIDER == name):
            continue   # not configured at all — not shown as failing
        providers[name] = provider.health_check()

    evidence = {
        "cross_validation_enabled": bool(settings.FUNDAMENTALS_CROSS_VALIDATION),
        "primary": settings.FUNDAMENTALS_PROVIDER,
        "fallback": settings.FUNDAMENTALS_FALLBACK_PROVIDER or None,
        "last_result": None,
    }
    if _last_evidence.get("result") is not None:
        result = _last_evidence["result"]
        ev = result.get("evidence") or {}
        confidence = ev.get("confidence") or {}
        evidence["last_result"] = {
            "at": _last_evidence.get("at"),
            "symbol": _last_evidence.get("symbol"),
            "status": result.get("status"),
            "available_providers": ev.get("available_providers"),
            "failed_providers": list((ev.get("failed_providers") or {}).keys()),
            "conflicted_metrics": ev.get("conflicted_metrics"),
            "missing_metrics": ev.get("missing_metrics"),
            "confidence": confidence.get("level"),
            "components": confidence.get("components"),
        }
    return {"providers": providers, "evidence": evidence}


def reset_evidence_state() -> None:
    """Test hook: clears the last-evidence record."""
    _last_evidence.update({"at": None, "symbol": None, "result": None})
