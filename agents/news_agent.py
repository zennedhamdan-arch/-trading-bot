"""
agents/news_agent.py

News Agent V2 — a thin adapter over the persistent News Intelligence
System (services/news_intelligence.py). The EXPENSIVE work (fetching,
normalizing, deduplicating, relevance-filtering and LLM analysis) happens
in the independent News Worker on its own schedule; this agent only READS
the cached intelligence and shapes it into the report the pipeline expects.

Consequences:
  * a trading cycle NEVER waits on UnoRouter, Groq or the Alpaca News API
  * the same article is NEVER analyzed twice (persistent fingerprints)
  * fresh intelligence is used as-is; stale intelligence is used but marked
    is_stale=true; when no intelligence exists the deterministic fallback
    applies (reduced confidence, source "deterministic_fallback")
  * an LLM is never called from this agent's cycle path

Report semantics (unchanged from V1): NEUTRAL sentiment only ever means a
genuine directionally-neutral read. A missing/failed news source yields
UNAVAILABLE with null sentiment and null confidence — never a fake NEUTRAL.
"""

import logging

from config import settings
from services import llm_service, news_intelligence

logger = logging.getLogger("news_agent")

# Retained for backwards compatibility with tests/direct callers (the
# analysis prompt now lives in news_intelligence._ANALYSIS_INSTRUCTIONS).
SYSTEM_INSTRUCTIONS = news_intelligence._ANALYSIS_INSTRUCTIONS


def _extract_json(text: str) -> dict:
    """Tolerant JSON extraction — kept for local use/testing."""
    return llm_service.extract_json(text)


def reset_news_analysis_cache() -> None:
    """Test/introspection hook (the intelligence cache lives in SQLite)."""
    news_intelligence.init_db()


def analyze_news(symbol: str, headlines: list = None) -> dict:
    """Builds the news report for one symbol from the persistent News
    Intelligence cache. `headlines` is accepted for signature compatibility
    with the V1 pipeline; the intelligence record is the source of truth.

    Returns a dict:
        {
          "agent": "news",
          "symbol": symbol,
          "evidence_status": "AVAILABLE"|"UNAVAILABLE",
          "sentiment": "BULLISH"/"BEARISH"/"NEUTRAL" | None,
          "confidence": float | None,
          "summary": str,
          "key_headline": str,
          "importance": "HIGH"|"MEDIUM"|"LOW"|None,
          "impact_horizon": "SHORT_TERM"|"LONG_TERM"|None,
          "event_risk": bool, "event_type": str|None,
          "event_risk_level": str|None,
          "headline_count": int, "new_headline_count": int,
          "is_stale": bool, "cache_age_minutes": float|None,
          "error": str | null,
          "provider": str, "model": str, "llm_status": str,
          "latency_ms": float | null, "cached": bool
        }
    """
    intel = news_intelligence.get_cached_news_intelligence(symbol)

    route = llm_service.route_info("news")
    source = str(intel.get("source") or "")
    provider = source if source else route["provider"]
    model = intel.get("model")

    base_result = {
        "agent": "news",
        "symbol": symbol,
        "evidence_status": None,
        "sentiment": None,
        "confidence": None,
        "summary": "",
        "key_headline": intel.get("key_headline") or "",
        "importance": intel.get("importance"),
        "impact_horizon": intel.get("impact_horizon"),
        "event_risk": bool(intel.get("event_risk")),
        "event_type": intel.get("event_type"),
        "event_risk_level": intel.get("event_risk_level"),
        "headline_count": int(intel.get("headline_count") or 0),
        "new_headline_count": int(intel.get("new_headline_count") or 0),
        "is_stale": bool(intel.get("is_stale")),
        "cache_age_minutes": intel.get("cache_age_minutes"),
        "error": None,
        "provider": provider,
        "model": model,
        "llm_status": "OK",
        "latency_ms": 0.0,
        "cached": True,
    }

    has_intelligence = (
        intel.get("last_updated") is not None
        and (intel.get("sentiment") is not None or int(intel.get("headline_count") or 0) > 0)
    )

    if not has_intelligence:
        # Nothing cached yet (worker has not run / no news ever fetched).
        base_result["llm_status"] = "SKIPPED_NO_DATA"
        base_result["evidence_status"] = "UNAVAILABLE"
        base_result["confidence"] = None
        base_result["sentiment"] = None
        base_result["summary"] = (
            "No news intelligence cached yet — no news verdict "
            "(the news worker fills this cache on its own schedule)."
        )
        base_result["provider"] = route["provider"]
        return base_result

    # Fresh or stale cached intelligence: used as-is (stale is marked).
    sentiment = intel.get("sentiment")
    base_result["sentiment"] = sentiment if sentiment in ("BULLISH", "BEARISH", "NEUTRAL") else None
    base_result["confidence"] = intel.get("confidence")
    base_result["summary"] = intel.get("summary") or ""
    base_result["evidence_status"] = "AVAILABLE"

    if source and "deterministic_fallback" in source:
        # Honest labeling: a crude keyword read (in whole or in part), never
        # presented as advanced AI reasoning.
        base_result["llm_status"] = "DETERMINISTIC_FALLBACK"
    elif intel.get("is_stale"):
        base_result["llm_status"] = "STALE_CACHE"
    else:
        base_result["llm_status"] = "OK"

    if base_result["is_stale"]:
        base_result["summary"] = (
            (base_result["summary"] + " " if base_result["summary"] else "")
            + f"[News intelligence {intel.get('cache_age_minutes')} minutes old — "
              f"marked stale, used with reduced confidence.]"
        ).strip()

    return base_result
