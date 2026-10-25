"""
operator_tools/fundamentals.py — read-only fundamentals-provider
diagnostics. The FMP connectivity probe is OPERATOR-INITIATED only (never
startup, never health polling) and goes through the existing
fundamentals_service abstraction with all budget/backoff/cache gates.
"""

from services import fundamentals_service
from services.operator.operator_tools._registry import tool
from services.operator.schemas import sanitize


@tool("fmp_connectivity",
      "diagnostics",
      "On-demand Financial Modeling Prep (FMP) connectivity check: probes "
      "each documented Stable API endpoint for a symbol (/profile, "
      "/income-statement, /balance-sheet-statement, /cash-flow-statement, "
      "/key-metrics, /ratios) and reports HTTP status, duration, "
      "classification (AVAILABLE / EMPTY_DATA / AUTHENTICATION_FAILED / "
      "ACCESS_RESTRICTED / RATE_LIMITED / PROVIDER_ERROR / TIMEOUT / "
      "INVALID_RESPONSE), record counts, latest reporting date and whether "
      "the payload passed structural validation. Costs up to 6 FMP "
      "requests against the daily budget (cached endpoints are not "
      "re-requested). Read-only; never trades.",
      args={"symbol": {"type": "string",
                       "description": "ticker to probe (default AAPL)",
                       "required": False}},
      label="Probing FMP endpoints")
def _fmp_connectivity(args):
    symbol = str(args.get("symbol") or "AAPL").strip().upper() or "AAPL"
    return sanitize(fundamentals_service.fmp_connectivity_probe(symbol))


@tool("fundamentals_provider_health",
      "diagnostics",
      "Fundamentals provider health WITHOUT any network requests: FMP and "
      "Alpha Vantage configuration/authentication status, endpoint "
      "capability registry, cache and daily-budget usage, plus the last "
      "multi-provider evidence outcome (agreement/conflicts, confidence).",
      args={},
      label="Reading fundamentals provider health")
def _fundamentals_provider_health(args):
    return sanitize(fundamentals_service.providers_health())
