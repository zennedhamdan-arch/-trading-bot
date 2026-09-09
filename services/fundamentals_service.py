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

Adding a real provider (no guessing required): subclass FundamentalsProvider,
implement get_fundamentals() using the vendor's OFFICIAL API, and register
it in _PROVIDER_CLASSES. Nothing else in the bot changes.
"""

import logging
import time

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


_PROVIDER_CLASSES = {
    "none": NoneProvider,
    # Register real providers here, e.g.:
    # "vendorname": VendorNameProvider,
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


def reset_cache() -> None:
    """Test hook: clears the result cache."""
    _result_cache.clear()


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
    """Normalized fundamentals for `symbol` (cached briefly per symbol so a
    single cycle asks at most once). Never raises; failures carry status
    DATA_UNAVAILABLE/ERROR with a reason."""
    cached = _result_cache.get(symbol)
    ttl = max(0.0, float(settings.FUNDAMENTALS_RESULT_CACHE_TTL_SECONDS))
    now = time.monotonic()
    if cached and cached[0] > now:
        return cached[1]

    result = _fetch_once(symbol)
    if ttl > 0:
        _result_cache[symbol] = (now + ttl, result)
    return result


def provider_config() -> dict:
    """Configured fundamentals providers (for /api/config; no secrets)."""
    return {
        "primary": settings.FUNDAMENTALS_PROVIDER,
        "fallback": settings.FUNDAMENTALS_FALLBACK_PROVIDER or None,
        "available": available_providers(),
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
