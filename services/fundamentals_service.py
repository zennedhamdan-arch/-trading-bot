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
        return _normalized(symbol, self.name, "OK", **fields)

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


_PROVIDER_CLASSES = {
    "none": NoneProvider,
    "alphavantage": AlphaVantageProvider,
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


def _fetch_once(symbol: str) -> dict:
    """Primary provider first; on DATA_UNAVAILABLE/ERROR, the configured
    fallback (if any). Never raises."""
    primary_name = settings.FUNDAMENTALS_PROVIDER
    fallback_name = settings.FUNDAMENTALS_FALLBACK_PROVIDER

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
