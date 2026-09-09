"""
services/realtime_service.py

Real-time market data via the official Alpaca SDK streams
(StockDataStream + NewsDataStream): trades, quotes, minute bars and news
for the trade universe, exposed through /api/realtime for the dashboard.

ARCHITECTURE — dictated by the installed alpaca-py 0.44.0 SDK:

    DataStream.run() is a SYNCHRONOUS method that internally runs its
    private forever-coroutine through asyncio.run(): it creates and OWNS
    its own event loop. It must never be awaited or called from inside
    the FastAPI event loop — asyncio.run() inside a running loop raises
    RuntimeError, the SDK's run() finally-clause then calls stop(),
    which dereferences a None loop ("'NoneType' object has no attribute
    'is_running'") and the forever coroutine is never awaited.
    That is exactly the production failure this service now avoids.

    Therefore each stream runs in its own dedicated background thread
    and the SDK keeps ownership of its loop:

        FastAPI event loop              stream threads (SDK-owned loops)
        ------------------              -------------------------------
        start_async() ─────> supervisor thread ──> StockDataStream.run()
             │                                     NewsDataStream.run()
        _drain_loop()  <──── queue.Queue (thread-safe, bounded,
             │                drop-oldest — never blocks a handler)
        _state / _status   (only mutated on the FastAPI loop)

Rules honored here:
  - A market tick NEVER triggers an LLM call (this module imports no LLM
    code); the AI cycle stays on its scheduled interval.
  - Handlers only enqueue normalized events — they never block and never
    run slow operations on the SDK's loop.
  - Exactly one stream instance per kind at any time: a dead stream is
    stopped via its public stop() and its thread joined BEFORE a
    replacement is created; reconnects use bounded exponential backoff
    with jitter (2s, 4s, 8s, 16s, ... capped), never a fixed loop.
  - Honesty: "connected" is only reported (and logged) once the stream
    is genuinely running — observed via the SDK's own running flag (the
    SDK exposes no public connection-state accessor; read defensively)
    or the first real market tick. Before that the status is CONNECTING.
  - A connected-but-silent stream is detected by the supervisor when the
    market is OPEN and no tick arrived for REALTIME_STALE_TICK_SECONDS;
    it is then recycled. While the market is closed IEX legitimately
    sends nothing, so no reconnect storm occurs.
  - Optional: without Alpaca keys (or REALTIME_ENABLED=false) the
    service stays down and the bot is unaffected.
"""

import asyncio
import logging
import queue
import random
import threading
import time
from collections import deque

from config import settings
from services import alpaca_service

logger = logging.getLogger("realtime_service")

# ---------------------------------------------------------------------------
# Normalized state — mutated ONLY by the drain task on the FastAPI loop
# (readers on the same loop therefore never see torn writes).
# ---------------------------------------------------------------------------

_state: dict = {}            # symbol -> {"last_trade": ..., "last_quote": ..., "minute_bar": ..., "updated_at": ...}
_recent_news = deque(maxlen=50)

_status = {
    "enabled": False,
    "connected_at": None,    # when the stock stream was first observed genuinely running (this epoch)
    "last_tick_at": None,    # last trade/quote/bar event
    "last_event_at": None,   # last event of any kind (incl. news)
    "last_error": None,
    "reconnect_attempt": 0,  # current backoff attempt for the stock stream
    "reconnects": 0,         # total supervisor-initiated restarts (both streams)
    "symbols": [],
    "feed": None,
}

# Thread-safe, bounded bridge from the SDK stream loops to the FastAPI loop.
_EVENT_QUEUE_MAX = 5000
_events: "queue.Queue" = queue.Queue(maxsize=_EVENT_QUEUE_MAX)


def _get(obj, attr):
    """Reads an attribute from a stream payload (model or dict)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)


def _iso(ts) -> str:
    return ts.isoformat() if hasattr(ts, "isoformat") else (str(ts) if ts else None)


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Handlers — async coroutines invoked by the SDK on ITS loop. They only
# enqueue; they never block, never call LLMs, never touch shared state.
# ---------------------------------------------------------------------------


def _enqueue(event: dict) -> None:
    """Non-blocking handoff; if the queue is full the oldest event is
    dropped so the newest market data always flows through."""
    try:
        _events.put_nowait(event)
    except queue.Full:
        try:
            _events.get_nowait()
            _events.put_nowait(event)
        except (queue.Empty, queue.Full):
            pass


async def _on_trade(trade) -> None:
    symbol = _get(trade, "symbol")
    if not symbol:
        return
    _enqueue({
        "type": "trade",
        "symbol": symbol,
        "payload": {
            "price": _get(trade, "price"),
            "size": _get(trade, "size"),
            "timestamp": _iso(_get(trade, "timestamp")),
        },
    })


async def _on_quote(quote) -> None:
    symbol = _get(quote, "symbol")
    if not symbol:
        return
    _enqueue({
        "type": "quote",
        "symbol": symbol,
        "payload": {
            "bid_price": _get(quote, "bid_price"),
            "ask_price": _get(quote, "ask_price"),
            "bid_size": _get(quote, "bid_size"),
            "ask_size": _get(quote, "ask_size"),
            "timestamp": _iso(_get(quote, "timestamp")),
        },
    })


async def _on_bar(bar) -> None:
    symbol = _get(bar, "symbol")
    if not symbol:
        return
    _enqueue({
        "type": "bar",
        "symbol": symbol,
        "payload": {
            "open": _get(bar, "open"),
            "high": _get(bar, "high"),
            "low": _get(bar, "low"),
            "close": _get(bar, "close"),
            "volume": _get(bar, "volume"),
            "timestamp": _iso(_get(bar, "timestamp")),
        },
    })


async def _on_news(news) -> None:
    _enqueue({
        "type": "news",
        "payload": {
            "headline": _get(news, "headline"),
            "summary": _get(news, "summary"),
            "source": _get(news, "source"),
            "symbols": _get(news, "symbols") or [],
            "created_at": _iso(_get(news, "created_at")),
        },
    })


def _apply_event(event: dict) -> None:
    """Applies one normalized event to the shared state. Runs on the
    FastAPI loop (via _drain_loop) — the single writer."""
    etype = event.get("type")
    now = _now()
    if etype == "news":
        _recent_news.appendleft(event.get("payload") or {})
        _status["last_event_at"] = now
        return
    symbol = event.get("symbol")
    if not symbol:
        return
    slot = _state.setdefault(symbol, {})
    if etype == "trade":
        slot["last_trade"] = event.get("payload")
    elif etype == "quote":
        slot["last_quote"] = event.get("payload")
    elif etype == "bar":
        slot["minute_bar"] = event.get("payload")
    else:
        return
    slot["updated_at"] = now
    _status["last_tick_at"] = now
    _status["last_event_at"] = now
    # First real market data of this epoch: the stream is genuinely alive.
    worker = _workers.get("stock")
    if worker is not None:
        worker.ticks_seen = True


async def _drain_loop() -> None:
    """FastAPI-loop side of the bridge: drains the queue into the shared
    state. Never blocks on the queue (get_nowait) and never touches the
    network or any LLM."""
    while True:
        try:
            processed = 0
            while processed < 2000:
                try:
                    event = _events.get_nowait()
                except queue.Empty:
                    break
                _apply_event(event)
                processed += 1
            await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            # Drain whatever is left so no tick is lost on shutdown.
            while True:
                try:
                    _apply_event(_events.get_nowait())
                except queue.Empty:
                    break
            raise


# ---------------------------------------------------------------------------
# Stream workers — one dedicated thread per stream; the SDK owns its loop
# ---------------------------------------------------------------------------


class _StreamWorker:
    """One SDK stream + the thread that runs it via the public run()."""

    def __init__(self, kind: str, stream):
        self.kind = kind
        self.stream = stream
        self.thread = None
        self.exit_reason = None
        self.started_at = _now()
        self.ticks_seen = False

    @property
    def alive(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def genuinely_running(self) -> bool:
        """True once the stream is genuinely connected+authed+subscribed.

        The installed SDK exposes no public connection-state accessor or
        on-connect callback; ``_running`` is the SDK's own flag, set True
        only after connect+auth+subscribe succeed and False on close.
        We read it defensively (None if a future SDK removes it) and fall
        back to tick-based evidence, so an SDK upgrade degrades
        gracefully instead of breaking. This is a read-only probe of a
        stable attribute — the service never calls the SDK's private
        forever-loop implementation.
        """
        running = getattr(self.stream, "_running", None)
        if isinstance(running, bool):
            return running
        return self.ticks_seen


def _safe_stream_stop(stream) -> None:
    """Stops a stream via its public stop(). stop() is designed to be
    called from another thread (it schedules stop_ws onto the stream's
    own loop); any failure (e.g. the loop already gone) is swallowed —
    the worker thread exiting is what actually ends the stream."""
    if stream is None:
        return
    try:
        stream.stop()
    except Exception:  # noqa: BLE001 — best-effort stop, never fatal
        pass


def _worker_main(worker: _StreamWorker) -> None:
    try:
        # Blocks for the lifetime of the connection; the SDK reconnects
        # internally with its own bounded backoff. Returns on stop() or
        # a fatal error (e.g. "insufficient subscription").
        worker.stream.run()
        worker.exit_reason = "run() returned"
    except BaseException as exc:  # noqa: BLE001 — recorded, supervisor restarts
        worker.exit_reason = f"{type(exc).__name__}: {exc}"
    finally:
        _safe_stream_stop(worker.stream)


def _build_stock_stream():
    """Constructs the stock stream on the CONFIGURED feed. Tests may
    monkeypatch this factory."""
    from alpaca.data.live import StockDataStream

    feed = alpaca_service._resolve_feed()
    symbols = [s for s in settings.TRADE_UNIVERSE]
    kwargs = {
        "api_key": settings.ALPACA_API_KEY,
        "secret_key": settings.ALPACA_SECRET_KEY,
        "feed": feed,
    }
    if settings.REALTIME_DATA_TIMEOUT_SECONDS > 0:
        # Opt-in transport-level silent-socket detection.
        kwargs["data_timeout"] = float(settings.REALTIME_DATA_TIMEOUT_SECONDS)
    stock = StockDataStream(**kwargs)
    stock.subscribe_trades(_on_trade, *symbols)
    stock.subscribe_quotes(_on_quote, *symbols)
    stock.subscribe_bars(_on_bar, *symbols)
    return stock, feed, symbols


def _build_news_stream():
    """Constructs the news stream (no staleness timeout: news is
    legitimately sporadic). Tests may monkeypatch this factory."""
    from alpaca.data.live import NewsDataStream

    news = NewsDataStream(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_SECRET_KEY,
    )
    news.subscribe_news(_on_news, *settings.TRADE_UNIVERSE)
    return news


_BUILDERS = {"stock": _build_stock_stream, "news": _build_news_stream}

# ---------------------------------------------------------------------------
# Supervisor — owns (re)starts, bounded exponential backoff with jitter,
# honest connection observation, market-aware staleness recycling.
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_workers: dict = {}          # kind -> _StreamWorker (at most one per kind)
_supervisor: threading.Thread = None
_stop_event = threading.Event()
_stopped = True              # True until start_async() and after shutdown()
_drain_task = None
_live_announced = False      # "connected" logged for the current epoch?
_market_open_cache = {"ts": 0.0, "is_open": None}


def _backoff_delay(attempt: int) -> float:
    """Bounded exponential backoff with jitter: attempt 1 ~2s, 2 ~4s,
    3 ~8s, 4 ~16s ... capped at REALTIME_RECONNECT_MAX_SECONDS. ±25%
    jitter avoids thundering reconnects."""
    base = max(0.5, float(settings.REALTIME_RECONNECT_SECONDS))
    cap = max(base, float(settings.REALTIME_RECONNECT_MAX_SECONDS))
    raw = min(base * (2 ** (max(1, attempt) - 1)), cap)
    return raw * random.uniform(0.75, 1.25)


def _record_error(message: str) -> None:
    _status["last_error"] = message
    logger.warning(f"Real-time stream: {message}")


def _market_open() -> bool:
    """Is the market currently open? Cached for 60s. On a clock failure
    the answer is unknown (False) — staleness recycling then stays off
    rather than risking a reconnect storm on API errors."""
    now = _now()
    cached = _market_open_cache
    if cached["ts"] and now - cached["ts"] < 60.0:
        return bool(cached["is_open"])
    try:
        clock = alpaca_service.get_clock()
        is_open = bool(clock.get("is_open")) if not clock.get("error") else None
    except Exception:  # noqa: BLE001
        is_open = None
    cached.update({"ts": now, "is_open": is_open})
    return bool(is_open)


def _start_worker(kind: str) -> None:
    """Creates and starts exactly one worker for `kind`. A previous
    worker for the same kind must already have exited (enforced by the
    callers) — never two concurrent streams of the same kind."""
    build = _BUILDERS[kind]
    if kind == "stock":
        stream, feed, symbols = build()
        _status["symbols"] = symbols
        _status["feed"] = getattr(feed, "value", str(feed))
    else:
        stream = build()
    worker = _StreamWorker(kind, stream)
    worker.thread = threading.Thread(
        target=_worker_main, args=(worker,),
        name=f"alpaca-{kind}-stream", daemon=True,
    )
    with _lock:
        _workers[kind] = worker
    worker.thread.start()


def _recycle_worker(kind: str, reason: str) -> None:
    """Stops + closes the old stream, waits for its thread to exit, and
    only then creates the replacement. If the old thread refuses to die,
    NO duplicate is created — the existing stream is left running and
    the condition is reported honestly."""
    with _lock:
        worker = _workers.get(kind)
    if worker is None:
        return
    if worker.alive:
        _safe_stream_stop(worker.stream)
        if worker.thread:
            worker.thread.join(timeout=15.0)
    if worker.alive:
        _record_error(
            f"{kind} stream thread did not exit after stop() ({reason}); "
            f"keeping the existing instance — no duplicate stream created"
        )
        return
    _status["reconnects"] += 1
    logger.info(f"Real-time {kind} stream recycled ({reason}).")
    _start_worker(kind)


def _announce_live_once(worker: _StreamWorker) -> None:
    """Logs 'connected' exactly once per epoch, only when genuinely
    running — never when the thread has merely been spawned."""
    global _live_announced
    if _live_announced or not worker.genuinely_running():
        return
    _live_announced = True
    _status["connected_at"] = _now()
    logger.info(
        f"Real-time layer connected (feed={_status.get('feed')}, "
        f"symbols={_status.get('symbols')}). Ticks never trigger LLM calls."
    )


def _check_staleness(worker: _StreamWorker) -> None:
    """Recycles the stock stream when it is live but silent while the
    market is OPEN (connected-but-mute socket). Silence with the market
    closed is expected and never triggers a reconnect."""
    stale_s = float(settings.REALTIME_STALE_TICK_SECONDS)
    if stale_s <= 0 or not worker.genuinely_running():
        return
    if not _market_open():
        return
    last_tick = _status.get("last_tick_at")
    reference = last_tick or worker.started_at
    silent_for = _now() - reference
    if silent_for > stale_s:
        _recycle_worker(
            "stock",
            f"no market tick for {silent_for:.0f}s while the market is open",
        )


def _supervisor_main() -> None:
    """Background thread: watches the stream workers, restarts dead ones
    with bounded exponential backoff + jitter, observes honest connection
    state, and recycles silent-but-connected streams (market open only)."""
    global _live_announced
    attempts = {"stock": 0, "news": 0}
    while not _stop_event.is_set():
        with _lock:
            workers = dict(_workers)
        for kind, worker in workers.items():
            if worker.alive:
                continue
            # Worker thread died (stream error/return): restart it, but
            # only after a backoff — and never while another instance of
            # the same kind could still be alive (recycle guarantees this).
            attempts[kind] += 1
            if kind == "stock":
                _status["reconnect_attempt"] = attempts[kind]
            delay = _backoff_delay(attempts[kind])
            _record_error(
                f"{kind} stream stopped ({worker.exit_reason or 'unknown reason'}); "
                f"restarting in {delay:.1f}s (attempt #{attempts[kind]})"
            )
            if _stop_event.wait(delay):
                return
            if _stop_event.is_set():
                return
            with _lock:
                still_dead = not (_workers.get(kind) and _workers[kind].alive)
            if still_dead:
                _status["reconnects"] += 1
                _start_worker(kind)

        stock = workers.get("stock")
        if stock is not None and stock.alive:
            _announce_live_once(stock)
            if stock.genuinely_running():
                attempts["stock"] = 0
                _status["reconnect_attempt"] = 0
            _check_staleness(stock)

        _stop_event.wait(1.0)
    logger.info("Real-time supervisor stopped.")


# ---------------------------------------------------------------------------
# Lifecycle (called from the FastAPI lifespan)
# ---------------------------------------------------------------------------


async def start_async() -> None:
    """Starts the real-time layer: the supervisor thread (which starts
    both stream workers on their own dedicated threads) plus the drain
    task on the FastAPI event loop. Idempotent; honest about being
    disabled/unconfigured. Must be called from a running event loop."""
    global _supervisor, _stopped, _drain_task, _live_announced
    if not settings.REALTIME_ENABLED:
        _status["enabled"] = False
        _status["last_error"] = "disabled via REALTIME_ENABLED"
        logger.info("Real-time layer disabled (REALTIME_ENABLED=false).")
        return
    if not settings.ALPACA_API_KEY or not settings.ALPACA_SECRET_KEY:
        _status["enabled"] = False
        _status["last_error"] = "Alpaca API keys not configured"
        logger.info("Real-time layer idle: Alpaca API keys not configured.")
        return
    with _lock:
        if _supervisor is not None and _supervisor.is_alive():
            return  # already running — never a second supervisor/streams
        _stop_event.clear()
        _stopped = False
        _live_announced = False
        _status["enabled"] = True
        _status["connected_at"] = None
        _status["last_tick_at"] = None
        _status["reconnect_attempt"] = 0
        _start_worker("stock")
        _start_worker("news")
        _supervisor = threading.Thread(
            target=_supervisor_main, name="realtime-supervisor", daemon=True
        )
        _supervisor.start()
        _drain_task = asyncio.get_running_loop().create_task(_drain_loop())
    # Deliberately NOT "connected": the streams are starting; connection
    # is announced only once a stream is genuinely running.
    logger.info(
        f"Real-time layer starting (feed={_status.get('feed')}, "
        f"symbols={_status.get('symbols')}); status will report CONNECTED "
        f"only once the stream is genuinely running."
    )


def _stop_workers() -> None:
    """Blocking stop of every worker (runs on a worker thread via
    asyncio.to_thread — stream.stop() may block up to a few seconds)."""
    with _lock:
        workers = list(_workers.values())
    for worker in workers:
        if worker.alive:
            _safe_stream_stop(worker.stream)
        if worker.thread:
            worker.thread.join(timeout=15.0)
    with _lock:
        for worker in workers:
            if worker.alive:
                logger.warning(
                    f"Real-time {worker.kind} stream thread did not exit on shutdown."
                )
        _workers.clear()


async def shutdown() -> None:
    """Clean shutdown: signals stop to both streams, joins their threads
    (no orphan loops), cancels the drain task. Safe to call repeatedly."""
    global _stopped, _drain_task, _live_announced
    _stop_event.set()
    if _supervisor is not None and _supervisor.is_alive():
        _supervisor.join(timeout=20.0)
    # stream.stop() blocks (SDK waits up to 5s) — run it off the event loop.
    await asyncio.to_thread(_stop_workers)
    task, _drain_task = _drain_task, None
    if task is not None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    _stopped = True
    _live_announced = False
    logger.info("Real-time layer stopped (shutdown).")


# ---------------------------------------------------------------------------
# Read API (dashboard + health)
# ---------------------------------------------------------------------------


def _connection_status() -> str:
    if not settings.REALTIME_ENABLED:
        return "DISABLED"
    if not settings.ALPACA_API_KEY or not settings.ALPACA_SECRET_KEY:
        return "NO_KEYS"
    if _stopped:
        return "DISCONNECTED"
    with _lock:
        stock = _workers.get("stock")
    if stock is None:
        return "DISCONNECTED"
    if not stock.alive:
        return "ERROR"
    if not stock.genuinely_running():
        return "CONNECTING"
    # Live stream: report STALE when it has been silent while the market
    # is open (the supervisor recycles it); CONNECTED otherwise (a quiet
    # closed market is normal and stays CONNECTED).
    stale_s = float(settings.REALTIME_STALE_TICK_SECONDS)
    if stale_s > 0 and _status.get("last_tick_at") and _market_open():
        if _now() - _status["last_tick_at"] > stale_s:
            return "STALE"
    return "CONNECTED"


def get_state() -> dict:
    """Normalized real-time snapshot for /api/realtime: connection
    status, timestamps, per-worker detail, per-symbol data and news."""
    status = _connection_status()
    with _lock:
        workers = dict(_workers)
    worker_detail = {}
    for kind, worker in workers.items():
        worker_detail[kind] = {
            "alive": worker.alive,
            "genuinely_running": worker.genuinely_running(),
            "started_at": worker.started_at,
            "exit_reason": worker.exit_reason,
        }
    last_tick = _status.get("last_tick_at")
    seconds_since_tick = round(_now() - last_tick, 1) if last_tick else None
    ticks_stale = (
        status == "STALE"
        or (last_tick is not None and _market_open()
            and float(settings.REALTIME_STALE_TICK_SECONDS) > 0
            and _now() - last_tick > float(settings.REALTIME_STALE_TICK_SECONDS))
    )
    return {
        "status": status,               # legacy key (same value)
        "connection_status": status,
        "enabled": bool(_status.get("enabled")),
        "feed": _status.get("feed"),
        "symbols": _status.get("symbols") or list(settings.TRADE_UNIVERSE),
        "connected_at": _status.get("connected_at"),
        "last_tick_at": last_tick,
        "last_event_at": _status.get("last_event_at"),
        "last_error": _status.get("last_error"),
        "reconnect_attempt": _status.get("reconnect_attempt", 0),
        "reconnects": _status.get("reconnects", 0),
        "seconds_since_last_tick": seconds_since_tick,
        "ticks_stale": bool(ticks_stale),
        "workers": worker_detail,
        "data": dict(_state),
        "news": list(_recent_news),
    }


def reset_state() -> None:
    """Test hook: clears the normalized state. Does NOT stop running
    threads — tests that started the layer must call shutdown() first."""
    _state.clear()
    _recent_news.clear()
    _status.update({
        "enabled": False, "connected_at": None, "last_tick_at": None,
        "last_event_at": None, "last_error": None, "reconnect_attempt": 0,
        "reconnects": 0, "symbols": [], "feed": None,
    })
    with _events.mutex:
        _events.queue.clear()
