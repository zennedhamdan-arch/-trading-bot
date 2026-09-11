"""
services/news_worker.py

The independent News Worker (Part 11). News processing NEVER runs inside a
trading cycle — this worker runs on its own schedule (NEWS_REFRESH_MINUTES)
and maintains the persistent News Intelligence cache:

    1. gets the tracked symbols
    2. fetches recent news (Alpaca News API)
    3. normalizes articles (stable fingerprints)
    4. checks duplicates (persistent — survives restarts)
    5. scores relevance (deterministic)
    6. analyzes only NEW, IMPORTANT articles (HIGH/MEDIUM only)
    7. updates the persistent per-symbol News Intelligence

The trading cycle only ever calls
news_intelligence.get_cached_news_intelligence(symbol) — a cache read that
cannot block on UnoRouter, Groq or the News API.

Failure policy: any per-symbol failure is recorded in the worker stats and
logged; the worker itself never raises into the scheduler and never blocks
trading cycles (it runs as its own scheduled job).
"""

import logging
import threading
from datetime import datetime, timezone

from config import settings
from services import news_intelligence

logger = logging.getLogger("news_worker")

# Worker stats (process-lifetime; last refresh time is persisted).
_stats = {
    "last_refresh": None,
    "last_refresh_duration_s": None,
    "symbols_processed": [],
    "articles_fetched": 0,
    "duplicates_ignored": 0,
    "articles_analyzed": 0,
    "provider_failures": 0,
    "refreshes": 0,
    "last_errors": [],
}

_run_lock = threading.Lock()


def stats() -> dict:
    """Worker stats for /api/news/status (no secrets). Includes the cached
    per-symbol intelligence for the trade universe — pure cache reads, this
    never triggers LLM analysis."""
    out = dict(_stats)
    out["last_errors"] = list(_stats["last_errors"][-10:])
    out["enabled"] = bool(settings.NEWS_ENABLED)
    out["refresh_minutes"] = float(settings.NEWS_REFRESH_MINUTES)
    out.update(news_intelligence.cache_counts())
    out["intelligence"] = {
        symbol: news_intelligence.get_cached_news_intelligence(symbol)
        for symbol in settings.TRADE_UNIVERSE
    }
    return out


def reset_stats() -> None:
    """Test hook."""
    _stats.update({
        "last_refresh": None,
        "last_refresh_duration_s": None,
        "symbols_processed": [],
        "articles_fetched": 0,
        "duplicates_ignored": 0,
        "articles_analyzed": 0,
        "provider_failures": 0,
        "refreshes": 0,
        "last_errors": [],
    })


def refresh_all(symbols=None) -> dict:
    """One full worker pass over the trade universe. Never raises."""
    if not settings.NEWS_ENABLED:
        logger.info("News worker disabled (NEWS_ENABLED=false).")
        return {"skipped": True, "reason": "NEWS_ENABLED=false"}

    # Never overlap runs (e.g. a slow pass still finishing when the next
    # interval fires).
    if not _run_lock.acquire(blocking=False):
        logger.warning("News worker: previous refresh still running — skipping.")
        return {"skipped": True, "reason": "previous refresh still running"}

    started = datetime.now(timezone.utc)
    symbols = [s.upper() for s in (symbols if symbols is not None else settings.TRADE_UNIVERSE)]
    try:
        per_symbol = []
        fetched = dupes = analyzed = failures = 0
        errors = []
        for symbol in symbols:
            try:
                result = news_intelligence.refresh_symbol(symbol)
            except Exception as exc:  # noqa: BLE001 — one symbol never kills the pass
                result = {"symbol": symbol, "error": str(exc),
                          "articles_fetched": 0, "duplicates_ignored": 0,
                          "articles_analyzed": 0, "provider_failures": 1}
            per_symbol.append(result)
            fetched += result.get("articles_fetched", 0)
            dupes += result.get("duplicates_ignored", 0)
            analyzed += result.get("articles_analyzed", 0)
            failures += result.get("provider_failures", 0)
            if result.get("error"):
                errors.append(f"{symbol}: {result['error']}")

        finished = datetime.now(timezone.utc)
        _stats.update({
            "last_refresh": finished.isoformat(),
            "last_refresh_duration_s": round((finished - started).total_seconds(), 1),
            "symbols_processed": symbols,
            "articles_fetched": fetched,
            "duplicates_ignored": dupes,
            "articles_analyzed": analyzed,
            "provider_failures": failures,
            "refreshes": _stats["refreshes"] + 1,
            "last_errors": errors,
        })
        logger.info(
            f"News worker pass complete: {len(symbols)} symbols, {fetched} articles "
            f"fetched, {dupes} duplicates ignored, {analyzed} new articles "
            f"analyzed, {failures} provider failure(s)."
        )
        return {"skipped": False, "symbols": per_symbol}
    finally:
        _run_lock.release()


async def scheduled_refresh() -> None:
    """APScheduler entry point (async)."""
    refresh_all()
