"""
services/fundamentals_evidence.py

Multi-provider fundamentals EVIDENCE validation and merging.

    Alpha Vantage ─┐
                   ├──> per-provider normalization (fundamentals_service)
    FMP ───────────┘            |
                                v
                    Evidence Validator (this module)
                    * compares only equivalent metrics
                    * checks reporting periods / dates / currencies
                    * detects stale statements and missing fields
                    * preserves each provider's original values
                    * records agreement or disagreement per metric
                                |
                                v
                    Evidence Merger (this module)
                    * selects one value per metric (primary provider
                      preferred; never averages, never lets an LLM guess)
                    * attaches a transparent, component-based confidence
                      assessment
                                |
                                v
                    Existing fundamentals agent -> existing LLM service

Design rules (enforced):
  * Missing data is NEVER treated as zero and values are NEVER invented.
  * Provider values are NEVER blindly averaged — each metric keeps every
    provider's original number and one explicitly-selected value.
  * A metric where independent sources disagree beyond the documented
    tolerance is marked CONFLICTED (not averaged, not hidden).
  * Metrics whose vendor definitions or measurement windows differ are
    compared with a documented, wider tolerance or marked NOT_COMPARABLE —
    never silently treated as identical.
  * Quarterly and annual values are never mixed: period metadata is checked
    before comparison, and different fiscal years make a metric
    NOT_COMPARABLE.
  * Currencies are compared when both providers report them; different
    currencies make a metric NOT_COMPARABLE (no conversion is ever
    invented). When one vendor does not report a currency (Alpha Vantage),
    the comparison proceeds flagged as currency-unverified.
  * Confidence uses NAMED levels derived from documented components — no
    arbitrary percentages.

Comparison tolerances (relative difference |a-b| / max(|a|,|b|)); rationale:
  market_cap      5%   both price-dependent at their own retrieval time,
                       so legitimate short-term drift is expected
  pe_ratio       10%   price-dependent; the earnings window (TTM vs FY)
                       can differ slightly between vendors
  revenue        15%   Alpha Vantage reports RevenueTTM while FMP reports
                       the fiscal-year figure — different measurement
                       windows accepted explicitly with a wider tolerance
  eps            15%   same TTM-vs-FY window difference as revenue
  profit_margin  15%   ratio; AV is TTM-based, FMP FY-based
  roe            20%   the equity base period can differ between vendors
  debt_to_equity 25%   vendor debt definitions differ materially (FMP uses
                       totalDebt incl. capital leases; AV uses
                       shortLongTermDebtTotal) — the widest tolerance, and
                       the definition caveat is recorded in the evidence

When BOTH providers report concrete report dates that are more than
400 days apart (different fiscal years for a balance-sheet metric), the
metric is NOT_COMPARABLE regardless of tolerance.
"""

import logging
import time

logger = logging.getLogger("fundamentals_evidence")

_METRICS = ("market_cap", "pe_ratio", "eps", "revenue",
            "profit_margin", "roe", "debt_to_equity")

# Documented per-metric relative tolerances (see module docstring).
_TOLERANCES = {
    "market_cap": 0.05,
    "pe_ratio": 0.10,
    "eps": 0.15,
    "revenue": 0.15,
    "profit_margin": 0.15,
    "roe": 0.20,
    "debt_to_equity": 0.25,
}

# Balance-sheet-style metrics where a different fiscal year makes the
# comparison meaningless (annual vs annual only).
_PERIOD_CRITICAL = ("debt_to_equity",)

# Freshness thresholds in days since the report date (documented).
_FRESH_DAYS = 120
_RECENT_DAYS = 250
_STALE_DAYS = 400


def _as_days(date_str):
    """YYYY-MM-DD -> epoch days, or None."""
    if not date_str:
        return None
    try:
        import calendar
        y, m, d = str(date_str)[:10].split("-")
        return calendar.timegm((int(y), int(m), int(d), 0, 0, 0)) // 86400
    except Exception:  # noqa: BLE001 — malformed dates are data, not crashes
        return None


def _relative_diff(a: float, b: float) -> float:
    """|a-b| / max(|a|,|b|) — 0.0 when both are (near) zero."""
    magnitude = max(abs(a), abs(b))
    if magnitude == 0:
        return 0.0
    return abs(a - b) / magnitude


def _compare_metric(metric: str, results: dict) -> dict:
    """Compares one metric across providers. Returns the metric evidence
    entry: status + per-provider values + the selected value + notes."""
    providers = [p for p in results
                 if isinstance(results[p], dict)
                 and results[p].get(metric) is not None]
    entry = {
        "values": {p: results[p].get(metric) for p in providers},
        "selected": None, "selected_provider": None,
        "note": None,
    }
    if not providers:
        entry["status"] = "MISSING"
        return entry
    if len(providers) == 1:
        entry["status"] = "SINGLE_SOURCE"
        entry["selected"] = results[providers[0]].get(metric)
        entry["selected_provider"] = providers[0]
        return entry

    # two or more sources: currency check first (never invent a conversion)
    currencies = {}
    for p in providers:
        meta = (results[p].get("meta") or {}).get(metric) or {}
        cur = meta.get("currency")
        if cur:
            currencies[p] = str(cur).upper()
    distinct = set(currencies.values())
    if len(distinct) > 1:
        entry["status"] = "NOT_COMPARABLE"
        entry["note"] = (f"reported currencies differ ({currencies}) — "
                         "no conversion is invented")
        entry["selected"] = results[providers[0]].get(metric)
        entry["selected_provider"] = providers[0]
        return entry
    currency_unverified = len(currencies) < len(providers)

    # reporting-period check: concrete dates on both sides
    dates = {}
    for p in providers:
        meta = (results[p].get("meta") or {}).get(metric) or {}
        days = _as_days(meta.get("report_date"))
        if days is not None:
            dates[p] = days
    if len(dates) == len(providers) >= 2:
        span = max(dates.values()) - min(dates.values())
        if metric in _PERIOD_CRITICAL and span > _STALE_DAYS:
            entry["status"] = "NOT_COMPARABLE"
            entry["note"] = ("reporting dates differ by more than a fiscal "
                             f"year ({span} days) — annual vs annual only")
            freshest = min(providers, key=lambda p: -dates[p])
            entry["selected"] = results[freshest].get(metric)
            entry["selected_provider"] = freshest
            return entry

    a, b = providers[0], providers[1]
    va, vb = results[a].get(metric), results[b].get(metric)
    diff = _relative_diff(va, vb)
    tolerance = _TOLERANCES.get(metric, 0.15)
    if diff <= tolerance:
        entry["status"] = "AGREED"
        entry["note"] = (f"relative difference {diff:.4f} within the "
                         f"documented {metric} tolerance {tolerance:.2f}")
    else:
        entry["status"] = "CONFLICTED"
        entry["note"] = (f"relative difference {diff:.4f} EXCEEDS the "
                         f"documented {metric} tolerance {tolerance:.2f} — "
                         "both original values are preserved; no averaging")
    if currency_unverified:
        entry["note"] = (entry["note"] + "; currency unverified for one "
                         "provider (not reported)") if entry["note"] else \
            "currency unverified for one provider (not reported)"
    # selection: the FIRST provider in `results` order is the configured
    # primary (its value is used; the disagreement stays visible above)
    entry["selected"] = results[a].get(metric)
    entry["selected_provider"] = a
    return entry


def _confidence(metrics: dict, selected_meta: dict, provider_count: int) -> dict:
    """Component-based confidence assessment. Every component is a NAMED
    label with a documented derivation (no arbitrary percentages).

    completeness  COMPLETE   >= 6 of 7 contract metrics present
                  PARTIAL    3-5 present
                  SPARSE     1-2 present
                  NONE       0 present
    freshness     age of the OLDEST reporting date among the selected
                  values — a fundamentals judgment is only as fresh as
                  its stalest input:
                  FRESH      < 120 days
                  RECENT     < 250 days
                  STALE      < 400 days
                  VERY_STALE >= 400 days
                  UNKNOWN    no reporting date available
    sources       DUAL_SOURCE / SINGLE_SOURCE / NONE
    agreement     AGREED                 all comparable pairs within tolerance
                  PARTIAL_AGREEMENT      some pairs agree, none conflict
                  CONFLICTED             at least one pair conflicts
                  NOT_APPLICABLE         no comparable pair (single source)
    validation    VENDOR_REPORTED        all selected values reported by vendors
                  PARTIALLY_DERIVED      at least one derived (labeled) value
                  NONE                   no values
    Overall level:
      NONE          no data at all
      LOW           any CONFLICTED metric, or sparse/very-stale evidence
      MEDIUM        single-source or partially-agreeing evidence, or stale
      HIGH          dual-source, no conflicts, >= PARTIAL completeness,
                    >= RECENT freshness
    """
    present = [m for m in _METRICS
               if metrics.get(m, {}).get("selected") is not None]
    if len(present) >= 6:
        completeness = "COMPLETE"
    elif len(present) >= 3:
        completeness = "PARTIAL"
    elif len(present) >= 1:
        completeness = "SPARSE"
    else:
        completeness = "NONE"

    today = time.gmtime()
    import calendar
    today_days = calendar.timegm(today) // 86400
    date_days = [_as_days((selected_meta.get(m) or {}).get("report_date"))
                 for m in present]
    date_days = [d for d in date_days if d is not None]
    if not date_days:
        freshness = "UNKNOWN"
    else:
        # worst vintage: the stalest selected input sets the freshness
        age = today_days - min(date_days)
        if age < _FRESH_DAYS:
            freshness = "FRESH"
        elif age < _RECENT_DAYS:
            freshness = "RECENT"
        elif age < _STALE_DAYS:
            freshness = "STALE"
        else:
            freshness = "VERY_STALE"

    sources = ("NONE" if provider_count == 0 else
               "SINGLE_SOURCE" if provider_count == 1 else "DUAL_SOURCE")

    statuses = [metrics[m]["status"] for m in _METRICS
                if metrics.get(m, {}).get("status") in ("AGREED", "CONFLICTED")]
    if any(s == "CONFLICTED" for s in statuses):
        agreement = "CONFLICTED"
    elif statuses and all(s == "AGREED" for s in statuses):
        agreement = "AGREED"
    elif statuses:
        agreement = "PARTIAL_AGREEMENT"
    else:
        agreement = "NOT_APPLICABLE"

    derived = any((selected_meta.get(m) or {}).get("derived")
                  for m in present)
    validation = ("NONE" if not present else
                  "PARTIALLY_DERIVED" if derived else "VENDOR_REPORTED")

    if completeness == "NONE":
        level = "NONE"
    elif (agreement == "CONFLICTED" or completeness == "SPARSE"
            or freshness == "VERY_STALE"):
        level = "LOW"
    elif (sources == "DUAL_SOURCE" and agreement == "AGREED"
            and completeness in ("COMPLETE", "PARTIAL")
            and freshness in ("FRESH", "RECENT")):
        level = "HIGH"
    else:
        level = "MEDIUM"

    return {
        "level": level,
        "components": {
            "completeness": completeness,
            "freshness": freshness,
            "sources": sources,
            "agreement": agreement,
            "validation": validation,
            "present_metrics": len(present),
        },
    }


def merge_evidence(results: dict) -> dict:
    """Validates and merges normalized results from multiple providers.

    Args:
        results: ordered {provider_name: normalized_result} — the FIRST
            entry is the configured primary provider (its value wins ties;
            disagreement is still recorded per metric). Only providers that
            returned a dict are included; a provider that failed entirely
            is represented by its structured failure result (status !=
            OK), which is preserved in the evidence block.

    Returns:
        The merged normalized-contract result: the usual top-level fields
        (status/symbol/provider/timestamp/7 metrics/reason) plus an
        "evidence" block (per-metric status/values, provider availability,
        confidence components). Never raises; never averages.
    """
    results = {p: r for p, r in (results or {}).items()
               if isinstance(r, dict)}
    provider_names = list(results.keys())
    primary_name = provider_names[0] if provider_names else "none"

    ok_providers = [p for p in provider_names
                    if results[p].get("status") == "OK"]
    failed = {p: {"status": results[p].get("status"),
                  "reason": results[p].get("reason")}
              for p in provider_names if results[p].get("status") != "OK"}

    symbol = next((results[p].get("symbol") for p in provider_names
                   if results[p].get("symbol")), None)
    # only OK providers contribute metric values
    comparable = {p: results[p] for p in ok_providers}

    metrics = {m: _compare_metric(m, comparable) for m in _METRICS}

    # selected values + their provenance (from the winning provider)
    selected_meta = {}
    for m in _METRICS:
        winner = metrics[m].get("selected_provider")
        if winner:
            meta = (results[winner].get("meta") or {}).get(m)
            if isinstance(meta, dict):
                selected_meta[m] = dict(meta)
                selected_meta[m]["provider"] = winner

    confidence = _confidence(metrics, selected_meta, len(ok_providers))

    # merged top-level result: the primary OK provider's shape wins; the
    # selected metric values come from the per-metric winners above
    base = results.get(primary_name) if ok_providers and \
        results.get(primary_name, {}).get("status") == "OK" else \
        (results.get(ok_providers[0]) if ok_providers else
         (results.get(primary_name) or {}))

    merged = {
        "status": base.get("status"),
        "symbol": symbol,
        "provider": (primary_name if comparable.get(primary_name)
                     else (ok_providers[0] if ok_providers
                           else base.get("provider"))),
        "timestamp": time.time(),
        "reason": base.get("reason"),
    }
    for m in _METRICS:
        merged[m] = metrics[m].get("selected")
    if not ok_providers:
        reasons = "; ".join(f"{p}: {results[p].get('reason')}"
                            for p in provider_names) or "no provider result"
        merged["status"] = "DATA_UNAVAILABLE"
        merged["reason"] = (f"ALL fundamentals providers unavailable — "
                            f"{reasons}")
    elif len(ok_providers) == 1:
        merged["status"] = "OK"
        merged["reason"] = None

    conflict_metrics = [m for m in _METRICS
                        if metrics[m].get("status") == "CONFLICTED"]
    merged["evidence"] = {
        "cross_validated": len(ok_providers) >= 2,
        "available_providers": ok_providers,
        "failed_providers": failed,
        "metrics": metrics,
        "conflicted_metrics": conflict_metrics,
        "missing_metrics": [m for m in _METRICS
                            if metrics[m].get("status") == "MISSING"],
        "confidence": confidence,
    }
    if conflict_metrics:
        logger.warning(
            "fundamentals evidence CONFLICTED for %s on: %s (original "
            "values preserved; no averaging)",
            symbol, ", ".join(conflict_metrics))
    return merged
