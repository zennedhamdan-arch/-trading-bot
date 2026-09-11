"""
services/news_intelligence.py

News Intelligence V2 — a persistent, deduplicating news analysis layer.

MOST IMPORTANT RULE: NEVER ANALYZE THE SAME ARTICLE TWICE.

Pipeline (executed by the independent news worker, NEVER inside a trading
cycle):

    ALPACA NEWS
      -> ARTICLE NORMALIZATION      (article_id, headline, summary, source,
                                     url, published_at, symbols)
      -> DEDUPLICATION              (persistent fingerprints; survives
                                     restarts, cycles and scheduler runs)
      -> RELEVANCE FILTER           (deterministic HIGH/MEDIUM/LOW/IRRELEVANT)
      -> NEW IMPORTANT ARTICLE?  no -> USE EXISTING CACHE
                              yes -> ANALYZE (LLM; deterministic fallback when
                                     every provider fails)
                                 -> STORE RESULT
                                 -> UPDATE NEWS INTELLIGENCE (per symbol)

The trading cycle only calls get_cached_news_intelligence(symbol) — a pure
cache read that can never block on UnoRouter, Groq or the Alpaca News API.

Persistence: the SAME SQLite database file as memory_service
(settings.MEMORY_DB_PATH, default data/memory.db) — no duplicate storage
engine, new tables only. Intelligence and article fingerprints therefore
survive application restarts.

Deterministic fallback (Part 13): when all LLM providers fail, articles are
analyzed with simple positive/negative keyword matching (confidence 0.30,
importance LOW, source "deterministic_fallback"). It is a crude fallback and
says so — it never pretends to be advanced AI reasoning.
"""

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from config import settings

logger = logging.getLogger("news_intelligence")

DB_PATH = Path(settings.MEMORY_DB_PATH)

RELEVANCE_LEVELS = ("HIGH", "MEDIUM", "LOW", "IRRELEVANT")
_RELEVANCE_SCORE = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "IRRELEVANT": 0}

# ---------------------------------------------------------------------------
# Deterministic relevance vocabulary (Part 9 / Part 14)
# ---------------------------------------------------------------------------

# Company-name hints for the default universe (symbol token matching is the
# primary signal; company names catch headlines that never use the ticker).
_COMPANY_NAMES = {
    "AAPL": ("apple", "iphone", "ipad", "tim cook", "app store"),
    "MSFT": ("microsoft", "azure", "copilot", "satya nadella", "windows"),
    "NVDA": ("nvidia", "geforce", "jensen huang", "cuda", "data center"),
    "TSLA": ("tesla", "elon musk", "model 3", "model y", "cybertruck"),
    "SPY": ("s&p 500", "s&p500", "stock market", "wall street", "spdr"),
}

# Material, stock-moving events -> HIGH relevance when combined with a
# symbol/company mention; also the event-risk vocabulary (Part 14).
_MATERIAL_KEYWORDS = (
    "earnings", "revenue", "guidance", "outlook", "profit", "loss",
    "upgrade", "downgrade", "price target",
    "sec", "subpoena", "investigation", "lawsuit", "litigation", "recall",
    "regulat", "fine", "probe",
    "ceo", "cfo", "resign", "steps down", "appoints",
    "acquisition", "acquire", "merger", "takeover", "buyout", "stake",
    "bankruptcy", "restructuring", "layoffs", "job cuts",
    "launch", "unveil", "announc", "release",
    "fda", "approval",
    "dividend", "buyback", "split",
)

# Event-risk detection: keyword group -> (event_type, risk_level).
_EVENT_RISK_PATTERNS = (
    (("earnings", "quarterly results", "q1", "q2", "q3", "q4"), "EARNINGS", "HIGH"),
    (("sec", "subpoena", "investigation", "lawsuit", "litigation", "regulat",
      "probe", "fine"), "REGULATORY", "HIGH"),
    (("recall", "halt", "bankruptcy", "fraud"), "ANNOUNCEMENT", "HIGH"),
    (("ceo", "cfo", "resign", "steps down", "appoints"), "CEO_CHANGE", "MEDIUM"),
    (("merger", "acquisition", "acquire", "takeover", "buyout"), "ANNOUNCEMENT", "MEDIUM"),
    (("fed", "federal reserve", "interest rate", "inflation", "cpi",
      "jobs report", "gdp", "recession", "tariff"), "MACRO", "MEDIUM"),
)

# Macro events move the whole market — HIGH relevance for the index (SPY),
# MEDIUM for single names.
_MACRO_KEYWORDS = (
    "fed", "federal reserve", "interest rate", "rate cut", "rate hike",
    "inflation", "cpi", "jobs report", "payroll", "gdp", "recession",
    "tariff", "treasury", "earnings season",
)

# Deterministic fallback sentiment vocabulary (Part 13) — crude by design.
_POSITIVE_KEYWORDS = (
    "beat", "growth", "record", "upgrade", "approval", "profit",
    "revenue increase",
)
_NEGATIVE_KEYWORDS = (
    "miss", "downgrade", "lawsuit", "investigation", "recall", "decline",
    "loss", "warning",
)

# ---------------------------------------------------------------------------
# SQLite (same DB file as memory_service; new tables only)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS news_articles (
    article_id TEXT PRIMARY KEY,       -- Alpaca article id, or SHA256 fingerprint
    symbol TEXT NOT NULL,
    headline TEXT NOT NULL,
    normalized_headline TEXT NOT NULL, -- lowercase, trimmed, single-spaced
    summary TEXT,
    source TEXT,
    url TEXT,
    author TEXT,
    published_at TEXT,
    fetched_at TEXT NOT NULL,
    relevance TEXT,                    -- HIGH | MEDIUM | LOW | IRRELEVANT
    relevance_reasons TEXT,            -- JSON list of matched signals
    analyzed INTEGER NOT NULL DEFAULT 0,
    analysis TEXT                      -- JSON per-article verdict
);
CREATE INDEX IF NOT EXISTS idx_news_articles_symbol
    ON news_articles(symbol, published_at);

CREATE TABLE IF NOT EXISTS news_intelligence (
    symbol TEXT PRIMARY KEY,
    sentiment TEXT,
    confidence REAL,
    importance TEXT,
    impact_horizon TEXT,
    event_risk INTEGER NOT NULL DEFAULT 0,
    event_type TEXT,
    event_risk_level TEXT,
    headline_count INTEGER,
    new_headline_count INTEGER,
    key_headline TEXT,
    summary TEXT,
    source TEXT,                       -- provider id or "deterministic_fallback"
    model TEXT,
    last_updated TEXT,
    is_stale INTEGER NOT NULL DEFAULT 0,
    error TEXT
);

CREATE TABLE IF NOT EXISTS news_worker_state (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

_init_lock = threading.Lock()
_initialized = False


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Creates the news tables in the shared memory DB (idempotent)."""
    global _initialized
    with _init_lock:
        if _initialized:
            return
        with _conn() as conn:
            conn.executescript(_SCHEMA)
        _initialized = True
        logger.info(f"News intelligence tables ready in {DB_PATH}")


# ---------------------------------------------------------------------------
# Article normalization (Part 7)
# ---------------------------------------------------------------------------


def normalize_headline(text: str) -> str:
    """lowercase, trim, collapse repeated whitespace."""
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def fingerprint(headline: str, source: str, published_at) -> str:
    """SHA256(normalized_headline + source + published_at) — the stable
    article identity used when the provider supplies no article id."""
    basis = f"{normalize_headline(headline)}|{str(source or '').strip().lower()}|{str(published_at or '')}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def normalize_article(symbol: str, raw: dict) -> dict:
    """Normalizes one raw article (as returned by
    market_data_service.get_news_articles) into the stored shape. Uses the
    provider's article id when present; otherwise the content fingerprint.
    """
    headline = str(raw.get("headline") or "").strip()
    source = str(raw.get("source") or "").strip()
    published_at = raw.get("published_at")
    published_at = str(published_at) if published_at is not None else None
    article_id = str(raw.get("id") or "").strip()
    if not article_id:
        article_id = fingerprint(headline, source, published_at)
    return {
        "article_id": article_id,
        "symbol": str(symbol).upper(),
        "headline": headline,
        "normalized_headline": normalize_headline(headline),
        "summary": str(raw.get("summary") or "").strip(),
        "source": source,
        "url": str(raw.get("url") or "").strip() or None,
        "author": str(raw.get("author") or "").strip() or None,
        "published_at": published_at,
        "relevance": None,
        "relevance_reasons": [],
    }


# ---------------------------------------------------------------------------
# Relevance filter (Part 9) + event risk (Part 14) — fully deterministic
# ---------------------------------------------------------------------------


def score_relevance(symbol: str, headline: str, summary: str = ""):
    """(relevance, reasons) — deterministic, no LLM involved.

    HIGH    direct symbol/company mention + a material event, or a major
            macro event for the index symbol
    MEDIUM  symbol/company mention alone, or a macro event for a single name
    LOW     only generic finance vocabulary
    IRRELEVANT  nothing matched
    """
    symbol = str(symbol).upper()
    text = normalize_headline(headline + " . " + (summary or ""))
    reasons = []

    mentions_symbol = False
    if re.search(rf"\b{re.escape(symbol.lower())}\b", text):
        mentions_symbol = True
        reasons.append(f"symbol mention: {symbol}")
    for name in _COMPANY_NAMES.get(symbol, ()):
        if name in text:
            mentions_symbol = True
            reasons.append(f"company mention: {name}")
            break

    material = [kw for kw in _MATERIAL_KEYWORDS if kw in text]
    macro = [kw for kw in _MACRO_KEYWORDS if kw in text]

    if mentions_symbol and material:
        return "HIGH", reasons + [f"material event: {material[0]}"]
    if symbol == "SPY" and macro:
        return "HIGH", reasons + [f"macro event: {macro[0]}"]
    if mentions_symbol:
        return "MEDIUM", reasons or ["symbol/company mention"]
    if macro:
        return "MEDIUM", [f"macro event: {macro[0]}"]
    if any(w in text for w in ("stock", "shares", "market", "trading", "investor")):
        return "LOW", ["generic finance vocabulary only"]
    return "IRRELEVANT", []


def detect_event_risk(headline: str, summary: str = "") -> dict:
    """Deterministic event-risk detection (Part 14)."""
    text = normalize_headline(headline + " . " + (summary or ""))
    for keywords, event_type, risk_level in _EVENT_RISK_PATTERNS:
        for kw in keywords:
            if kw in text:
                return {
                    "event_risk": True,
                    "event_type": event_type,
                    "risk_level": risk_level,
                    "matched": kw,
                }
    return {"event_risk": False, "event_type": None, "risk_level": None}


def relevance_threshold_score() -> int:
    configured = str(settings.NEWS_RELEVANCE_THRESHOLD or "MEDIUM").upper()
    return _RELEVANCE_SCORE.get(configured, _RELEVANCE_SCORE["MEDIUM"])


def article_reaches_llm(relevance: str) -> bool:
    """Only articles at/above the configured threshold are ever analyzed."""
    return _RELEVANCE_SCORE.get(relevance, 0) >= relevance_threshold_score()


# ---------------------------------------------------------------------------
# Persistence: dedup + storage (Part 8)
# ---------------------------------------------------------------------------


def upsert_articles(symbol: str, articles: list) -> dict:
    """Stores normalized articles; returns {"new": [...], "duplicates": n}.

    Deduplication is persistent (SQLite): an article whose article_id was
    ever seen before is counted as a duplicate and is NEVER re-analyzed —
    this survives trading cycles, scheduler runs and application restarts.
    """
    init_db()
    new_articles = []
    duplicates = 0
    now = _now_iso()
    with _conn() as conn:
        for art in articles:
            row = conn.execute(
                "SELECT article_id FROM news_articles WHERE article_id = ?",
                (art["article_id"],),
            ).fetchone()
            if row is not None:
                duplicates += 1
                continue
            relevance, reasons = score_relevance(symbol, art["headline"], art.get("summary", ""))
            art = dict(art)
            art["relevance"] = relevance
            art["relevance_reasons"] = reasons
            conn.execute(
                """INSERT INTO news_articles
                   (article_id, symbol, headline, normalized_headline, summary,
                    source, url, author, published_at, fetched_at,
                    relevance, relevance_reasons, analyzed, analysis)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
                (
                    art["article_id"], art["symbol"], art["headline"],
                    art["normalized_headline"], art["summary"], art["source"],
                    art["url"], art["author"], art["published_at"], now,
                    relevance, json.dumps(reasons),
                ),
            )
            new_articles.append(art)
    return {"new": new_articles, "duplicates": duplicates}


def get_recent_articles(symbol: str, limit: int = None) -> list:
    """Known recent articles for a symbol (API + worker use; no LLM)."""
    init_db()
    limit = limit or settings.NEWS_MAX_ARTICLES_PER_SYMBOL
    with _conn() as conn:
        rows = conn.execute(
            """SELECT article_id, symbol, headline, summary, source, url, author,
                      published_at, relevance, relevance_reasons, analyzed,
                      analysis, fetched_at
               FROM news_articles WHERE symbol = ?
               ORDER BY COALESCE(published_at, fetched_at) DESC LIMIT ?""",
            (str(symbol).upper(), int(limit)),
        ).fetchall()
    out = []
    for r in rows:
        art = dict(r)
        try:
            art["relevance_reasons"] = json.loads(art.get("relevance_reasons") or "[]")
        except (ValueError, TypeError):
            art["relevance_reasons"] = []
        try:
            art["analysis"] = json.loads(art["analysis"]) if art.get("analysis") else None
        except (ValueError, TypeError):
            art["analysis"] = None
        out.append(art)
    return out


def _store_article_analysis(article_id: str, analysis: dict) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE news_articles SET analyzed = 1, analysis = ? WHERE article_id = ?",
            (json.dumps(analysis), article_id),
        )


# ---------------------------------------------------------------------------
# Analysis: LLM with deterministic fallback (Parts 6, 13)
# ---------------------------------------------------------------------------

_ANALYSIS_INSTRUCTIONS = """You are a financial news analyst. You will be given a stock ticker and a
numbered list of NEW articles about it. For each article, judge its sentiment
for the stock, your confidence, its importance, and the impact horizon.
Then give an overall summary of what the news flow means for the stock.

Respond ONLY with a single valid JSON object, no markdown fences, no preamble, in this exact shape:
{
  "articles": [
    {"index": <the article number>, "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
     "confidence": <float 0.0 to 1.0>, "importance": "HIGH" | "MEDIUM" | "LOW",
     "impact_horizon": "SHORT_TERM" | "LONG_TERM"}
  ],
  "overall_sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
  "overall_confidence": <float 0.0 to 1.0>,
  "summary": "<two sentence synthesis of the news flow>"
}
"""


def deterministic_article_analysis(article: dict) -> dict:
    """The crude keyword fallback (Part 13) — a fallback, not AI reasoning."""
    text = normalize_headline(article.get("headline", "") + " . " + str(article.get("summary") or ""))
    pos = sum(1 for kw in _POSITIVE_KEYWORDS if kw in text)
    neg = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in text)
    if pos > neg:
        sentiment = "BULLISH"
    elif neg > pos:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"
    event = detect_event_risk(article.get("headline", ""), article.get("summary", ""))
    return {
        "sentiment": sentiment,
        "confidence": 0.30,
        "importance": "LOW",
        "impact_horizon": "SHORT_TERM",
        "source": "deterministic_fallback",
        "event": event,
    }


def analyze_new_articles(symbol: str, articles: list) -> dict:
    """Analyzes NEW, relevance-filtered articles.

    One batched LLM request per symbol (never one per article). If every
    provider in the chain fails, each article gets the deterministic keyword
    fallback (source="deterministic_fallback") — analysis still completes.

    Returns {"analyzed": n, "llm_ok": bool, "source": str, "model": str,
             "error": str | None, "summary": str, "overall_sentiment": ...,
             "overall_confidence": ...}.
    """
    if not articles:
        return {"analyzed": 0, "llm_ok": True, "source": None, "model": None,
                "error": None, "summary": "", "overall_sentiment": None,
                "overall_confidence": None}

    listing = []
    for i, art in enumerate(articles):
        summary_txt = f" — {art.get('summary')}" if art.get("summary") else ""
        listing.append(f"{i}. {art.get('headline')}{summary_txt} (source: {art.get('source') or 'unknown'})")
    prompt = f"Ticker: {symbol}\nNew articles:\n" + "\n".join(listing)

    from services import llm_service
    result = llm_service.call_json(
        "news", system=_ANALYSIS_INSTRUCTIONS, user=prompt,
        temperature=0.2, max_tokens=900, symbol=symbol,
    )

    per_article = {}
    llm_summary = ""
    overall_sentiment = None
    overall_confidence = None
    llm_ok = False

    if result.ok and isinstance(result.parsed, dict):
        llm_ok = True
        for item in (result.parsed.get("articles") or []):
            try:
                idx = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            sentiment = str(item.get("sentiment", "NEUTRAL")).upper()
            if sentiment not in ("BULLISH", "BEARISH", "NEUTRAL"):
                sentiment = "NEUTRAL"
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            importance = str(item.get("importance", "LOW")).upper()
            if importance not in ("HIGH", "MEDIUM", "LOW"):
                importance = "LOW"
            horizon = str(item.get("impact_horizon", "SHORT_TERM")).upper()
            if horizon not in ("SHORT_TERM", "LONG_TERM"):
                horizon = "SHORT_TERM"
            per_article[idx] = {
                "sentiment": sentiment, "confidence": confidence,
                "importance": importance, "impact_horizon": horizon,
                "source": result.provider, "model": result.model,
            }
        llm_summary = str(result.parsed.get("summary", ""))
        overall_sentiment = str(result.parsed.get("overall_sentiment", "")).upper() or None
        try:
            overall_confidence = min(1.0, max(0.0, float(result.parsed.get("overall_confidence", 0.0))))
        except (TypeError, ValueError):
            overall_confidence = None
    else:
        logger.warning(
            f"News analysis LLM chain failed for {symbol} ({result.status}): "
            f"{result.error} — using deterministic keyword fallback."
        )

    stored = 0
    for i, art in enumerate(articles):
        event = detect_event_risk(art.get("headline", ""), art.get("summary", ""))
        if llm_ok and i in per_article:
            analysis = {**per_article[i], "event": event}
        else:
            analysis = deterministic_article_analysis(art)
        analysis["event"] = analysis.get("event") or event
        _store_article_analysis(art["article_id"], analysis)
        stored += 1

    return {
        "analyzed": stored,
        "llm_ok": llm_ok,
        "source": result.provider if llm_ok else "deterministic_fallback",
        "model": result.model if llm_ok else None,
        "error": None if llm_ok else (result.error or "LLM chain failed"),
        "summary": llm_summary,
        "overall_sentiment": overall_sentiment,
        "overall_confidence": overall_confidence,
    }


# ---------------------------------------------------------------------------
# Per-symbol intelligence aggregation + cache (Part 10)
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def update_intelligence(symbol: str, analysis_outcome: dict, stats: dict) -> dict:
    """Aggregates the analyzed articles for a symbol into the persistent
    intelligence record (the Part 10 shape)."""
    init_db()
    symbol = str(symbol).upper()
    articles = get_recent_articles(symbol)
    analyzed = [a for a in articles if a.get("analysis")]
    total = len(articles)

    event = {"event_risk": False, "event_type": None, "risk_level": None}
    best_event_level = -1
    level_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    bullish = bearish = 0
    confidences = []
    key_article = None
    key_score = -1.0
    llm_any = False
    sources = set()

    for a in analyzed:
        an = a["analysis"]
        sources.add(an.get("source") or "unknown")
        if an.get("source") not in (None, "deterministic_fallback"):
            llm_any = True
        if an.get("sentiment") == "BULLISH":
            bullish += 1
        elif an.get("sentiment") == "BEARISH":
            bearish += 1
        confidences.append(float(an.get("confidence") or 0.0))
        score = float(an.get("confidence") or 0.0) + 0.5 * (
            level_rank.get(str(an.get("importance")).upper(), 0)
        )
        if score > key_score:
            key_score = score
            key_article = a
        ev = an.get("event") or {}
        if ev.get("event_risk"):
            rank = level_rank.get(str(ev.get("risk_level")).upper(), 1)
            if rank > best_event_level:
                best_event_level = rank
                event = {
                    "event_risk": True,
                    "event_type": ev.get("event_type"),
                    "risk_level": ev.get("risk_level"),
                }

    # The LLM produced a valid overall verdict (possibly with partial
    # per-article coverage): the top-line verdict IS the LLM's; articles the
    # LLM did not cover remain labeled deterministic at the article level.
    overall_from_llm = bool(
        analysis_outcome.get("llm_ok")
        and analysis_outcome.get("overall_sentiment") in ("BULLISH", "BEARISH", "NEUTRAL")
    )
    if overall_from_llm:
        sentiment = analysis_outcome["overall_sentiment"]
    elif bullish > bearish:
        sentiment = "BULLISH"
    elif bearish > bullish:
        sentiment = "BEARISH"
    elif analyzed:
        sentiment = "NEUTRAL"
    else:
        sentiment = None

    confidence = None
    if analyzed and confidences:
        # Transparent statistic: the mean of the per-article confidences
        # (no invented formula beyond that).
        confidence = round(sum(confidences) / len(confidences), 3)
    if analysis_outcome.get("overall_confidence") is not None and llm_any:
        confidence = analysis_outcome["overall_confidence"]

    importance = None
    if analyzed:
        importance = "HIGH" if any(
            str(a["analysis"].get("importance")).upper() == "HIGH" for a in analyzed
        ) else "MEDIUM" if analyzed else None

    if not analyzed:
        source = None
    elif analysis_outcome.get("llm_ok"):
        source = analysis_outcome.get("source") or "llm"
    elif llm_any:
        # Mixed history: earlier articles were LLM-analyzed, the newest
        # refresh fell back to deterministic keyword analysis.
        llm_sources = [s for s in sources if s not in (None, "deterministic_fallback", "unknown")]
        source = f"{llm_sources[0]}+deterministic_fallback" if llm_sources else "deterministic_fallback"
    else:
        source = "deterministic_fallback"

    summary = analysis_outcome.get("summary") or (
        f"Keyword fallback over {len(analyzed)} articles — crude sentiment read, "
        f"not advanced AI reasoning." if analyzed and not analysis_outcome.get("llm_ok") else ""
    )

    record = {
        "symbol": symbol,
        "sentiment": sentiment,
        "confidence": confidence,
        "importance": importance,
        "impact_horizon": (key_article["analysis"].get("impact_horizon") if key_article else None),
        "event_risk": bool(event["event_risk"]),
        "event_type": event.get("event_type"),
        "event_risk_level": event.get("risk_level"),
        "headline_count": total,
        "new_headline_count": int(stats.get("new_headline_count", 0)),
        "key_headline": (key_article.get("headline") if key_article else None),
        "summary": summary,
        "source": source,
        "model": analysis_outcome.get("model") if analysis_outcome.get("llm_ok") else None,
        "last_updated": _now_iso(),
        "is_stale": 0,
        "error": analysis_outcome.get("error"),
    }

    with _conn() as conn:
        conn.execute(
            """INSERT INTO news_intelligence
               (symbol, sentiment, confidence, importance, impact_horizon,
                event_risk, event_type, event_risk_level, headline_count,
                new_headline_count, key_headline, summary, source, model,
                last_updated, is_stale, error)
               VALUES (:symbol, :sentiment, :confidence, :importance,
                       :impact_horizon, :event_risk, :event_type,
                       :event_risk_level, :headline_count, :new_headline_count,
                       :key_headline, :summary, :source, :model, :last_updated,
                       :is_stale, :error)
               ON CONFLICT(symbol) DO UPDATE SET
                 sentiment=excluded.sentiment, confidence=excluded.confidence,
                 importance=excluded.importance,
                 impact_horizon=excluded.impact_horizon,
                 event_risk=excluded.event_risk, event_type=excluded.event_type,
                 event_risk_level=excluded.event_risk_level,
                 headline_count=excluded.headline_count,
                 new_headline_count=excluded.new_headline_count,
                 key_headline=excluded.key_headline, summary=excluded.summary,
                 source=excluded.source, model=excluded.model,
                 last_updated=excluded.last_updated, is_stale=excluded.is_stale,
                 error=excluded.error""",
            record,
        )
    return record


def get_cached_news_intelligence(symbol: str) -> dict:
    """The ONLY call a trading cycle makes: read the persistent per-symbol
    intelligence (Part 10 shape). Never triggers LLM analysis, never blocks.

    Fresh intelligence is returned as stored; stale intelligence is returned
    marked is_stale=true; when nothing exists yet the deterministic
    no-intelligence fallback is returned (reduced confidence, source
    "deterministic_fallback") so the cycle always continues safely.
    """
    init_db()
    symbol = str(symbol).upper()
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM news_intelligence WHERE symbol = ?", (symbol,)
        ).fetchone()

    if row is None:
        return {
            "symbol": symbol,
            "sentiment": "NEUTRAL",
            "confidence": 0.30,
            "importance": "LOW",
            "impact_horizon": None,
            "event_risk": False,
            "event_type": None,
            "event_risk_level": None,
            "headline_count": 0,
            "new_headline_count": 0,
            "key_headline": None,
            "summary": "No news intelligence cached yet — trading continues "
                       "with reduced confidence (deterministic fallback).",
            "source": "deterministic_fallback",
            "model": None,
            "cache_age_minutes": None,
            "is_stale": True,
            "last_updated": None,
            "error": None,
        }

    record = dict(row)
    record["event_risk"] = bool(record.get("event_risk"))
    updated = _parse_iso(record.get("last_updated"))
    age_minutes = None
    if updated is not None:
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age_minutes = round(
            (datetime.now(timezone.utc) - updated).total_seconds() / 60.0, 1
        )
    record["cache_age_minutes"] = age_minutes
    max_age = float(settings.NEWS_CACHE_MAX_AGE_MINUTES or 60)
    stale = record.get("is_stale") or (age_minutes is not None and age_minutes > max_age)
    record["is_stale"] = bool(stale)
    return record


def cache_counts() -> dict:
    """Cache statistics for /api/news/status."""
    init_db()
    with _conn() as conn:
        symbols = conn.execute(
            "SELECT COUNT(*) FROM news_intelligence"
        ).fetchone()[0]
        articles = conn.execute(
            "SELECT COUNT(*) FROM news_articles"
        ).fetchone()[0]
        analyzed = conn.execute(
            "SELECT COUNT(*) FROM news_articles WHERE analyzed = 1"
        ).fetchone()[0]
    return {"symbols_with_intelligence": symbols,
            "articles_known": articles, "articles_analyzed": analyzed}


# ---------------------------------------------------------------------------
# One-symbol refresh pipeline (used by the news worker)
# ---------------------------------------------------------------------------


def refresh_symbol(symbol: str, fetch_articles=None) -> dict:
    """The Part 6 pipeline for one symbol. Returns per-symbol stats.

    `fetch_articles` is injectable for tests (defaults to
    market_data_service.get_news_articles).
    """
    symbol = str(symbol).upper()
    stats = {
        "symbol": symbol,
        "articles_fetched": 0,
        "duplicates_ignored": 0,
        "new_important": 0,
        "articles_analyzed": 0,
        "provider_failures": 0,
        "error": None,
    }
    try:
        if fetch_articles is None:
            from services import market_data_service
            raw_articles = market_data_service.get_news_articles(symbol)
        else:
            raw_articles = fetch_articles(symbol)
        if isinstance(raw_articles, dict):
            error = raw_articles.get("error")
            raw_articles = raw_articles.get("articles", [])
            if error:
                stats["error"] = str(error)
    except Exception as exc:  # noqa: BLE001 — the worker never crashes the app
        stats["error"] = str(exc)
        raw_articles = []

    stats["articles_fetched"] = len(raw_articles)

    normalized = [normalize_article(symbol, raw) for raw in raw_articles]
    outcome = upsert_articles(symbol, normalized)
    stats["duplicates_ignored"] = outcome["duplicates"]

    new_important = [
        a for a in outcome["new"]
        if article_reaches_llm(a.get("relevance") or "IRRELEVANT")
    ]
    stats["new_important"] = len(new_important)

    if new_important:
        analysis = analyze_new_articles(symbol, new_important)
        stats["articles_analyzed"] = analysis["analyzed"]
        if not analysis.get("llm_ok"):
            stats["provider_failures"] += 1
        # LOW/IRRELEVANT new articles are stored but never analyzed — no LLM.
        update_intelligence(symbol, analysis, {
            "new_headline_count": len(outcome["new"]),
        })
    elif outcome["new"]:
        # Only unimportant new articles: refresh counts/summary without
        # touching any LLM.
        update_intelligence(symbol, {"llm_ok": True}, {
            "new_headline_count": len(outcome["new"]),
        })
    else:
        stats["new_headline_count"] = 0

    return stats
