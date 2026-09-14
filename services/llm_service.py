"""
services/llm_service.py

Centralized LLM router (the "LLMService" of the V2 architecture):

    LLM REQUEST
      |
      v
    CHECK CACHE  --(hit)--> RETURN CACHE
      | miss
      v
    UNOROUTER PRIMARY MODEL
      | fail
      v
    UNOROUTER FALLBACK MODEL 1
      | fail
      v
    UNOROUTER FALLBACK MODEL 2          (<= LLM_MAX_MODEL_ATTEMPTS models)
      | fail
      v
    GROQ                               (final provider fallback)
      | fail
      v
    CACHED RESULT (if one exists, even stale)
      | none
      v
    DETERMINISTIC FALLBACK             (agent-level, e.g. news keywords)

Providers (OpenAI-compatible unless noted):
    unorouter (PRIMARY), groq, nvidia, gemini, openrouter

Hard latency policy (no long retries, ever):
    * every HTTP call has a LLM_REQUEST_TIMEOUT_SECONDS timeout
    * SDK-level retries are LLM_MAX_RETRIES (default 0)
    * a failed attempt records the failure, updates the circuit breaker and
      IMMEDIATELY moves to the next model/provider — the trading cycle never
      waits 30-60s for a provider to recover.

Circuit breakers per provider:
    * classic consecutive-failure breaker: CLOSED -> OPEN (after
      LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD consecutive failures, for
      LLM_CIRCUIT_BREAKER_BACKOFF_SECONDS) -> HALF_OPEN probe -> CLOSED/OPEN.
      While OPEN no requests are sent to that provider at all.
    * plus the specific circuits: 429 quota (provider paused), 401/403 auth
      (provider paused), 404 model (that provider+model pair paused).

Standardized response: every result can be rendered via LLMResult.standard():
    {"success": true, "provider": "...", "model": "...", "content": "...",
     "cached": false, "fallback_used": false, "latency_ms": 0, "error": null}

Agents never implement provider-specific HTTP calls. They call

    llm_service.complete(task="debate", messages=[...])      # messages-style
    llm_service.call(agent="debate", system=..., user=...)   # system/user-style
    llm_service.call_json(agent="news", system=..., user=...)  # + JSON parsing

Statuses (every result carries provider/model/status/latency/error_type):
    OK, NOT_CONFIGURED, MODEL_NOT_FOUND, PROVIDER_QUOTA_EXCEEDED,
    RATE_LIMITED, PROVIDER_ERROR, AUTH_ERROR, NETWORK_ERROR,
    INVALID_RESPONSE, CIRCUIT_OPEN

Model selection is EXPLICIT and owned by this application — UnoRouter (or any
gateway) never chooses models for us. Configured ids are verified against each
provider's LIVE /models catalog at startup; an id that no longer exists is
skipped at runtime with LLM_MODEL_UNAVAILABLE logged and the next configured
model is tried. No unlimited model guessing: at most LLM_MAX_MODEL_ATTEMPTS
models per request.
"""

import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeoutError
from datetime import datetime, timezone

from config import settings
from services import gemini_service

logger = logging.getLogger("llm_service")

# Generic tolerant JSON extraction (single implementation; lives in the
# gemini transport module for backward compatibility with direct callers).
extract_json = gemini_service._extract_json

PROVIDERS = ("unorouter", "groq", "nvidia", "gemini", "openrouter")

# ---------------------------------------------------------------------------
# Provider classes
# ---------------------------------------------------------------------------


class BaseLLMProvider:
    """One LLM provider behind the router. Subclasses own transport details."""

    name = "base"
    key_attr = ""                    # settings attribute holding the API key
    enabled_attr = ""                # settings attribute gating the provider
    daily_limit_attr = ""            # settings attribute for the local 24h budget
    quota_backoff_attr = ""          # settings attribute for the 429 backoff

    def get_client(self):
        raise NotImplementedError

    def send(self, model, system, user, temperature, max_tokens) -> str:
        raise NotImplementedError

    def list_models(self) -> set:
        raise NotImplementedError


class GroqProvider(BaseLLMProvider):
    name = "groq"
    key_attr = "GROQ_API_KEY"
    enabled_attr = "GROQ_ENABLED"
    daily_limit_attr = "GROQ_DAILY_REQUEST_LIMIT"
    quota_backoff_attr = "GROQ_QUOTA_BACKOFF_MINUTES"

    def get_client(self):
        client = _clients.get(self.name)
        if client is None:
            from groq import Groq
            client = Groq(
                api_key=_provider_key(self.name),
                timeout=max(1.0, float(settings.LLM_REQUEST_TIMEOUT_SECONDS)),
                max_retries=max(0, int(settings.LLM_MAX_RETRIES)),
            )
            _clients[self.name] = client
        return client

    def send(self, model, system, user, temperature, max_tokens):
        completion = self.get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content or ""

    def list_models(self):
        return _extract_model_ids(self.get_client().models.list())


class _OpenAICompatProvider(BaseLLMProvider):
    """OpenAI-compatible endpoint (UnoRouter, OpenRouter, NVIDIA NIM)."""

    def get_client(self):
        client = _clients.get(self.name)
        if client is None:
            from openai import OpenAI
            client = OpenAI(
                api_key=_provider_key(self.name),
                base_url=self.base_url(),
                timeout=max(1.0, float(settings.LLM_REQUEST_TIMEOUT_SECONDS)),
                max_retries=max(0, int(settings.LLM_MAX_RETRIES)),
            )
            _clients[self.name] = client
        return client

    def base_url(self) -> str:
        raise NotImplementedError

    def send(self, model, system, user, temperature, max_tokens):
        completion = self.get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content or ""

    def list_models(self):
        return _extract_model_ids(self.get_client().models.list())


class UnoRouterProvider(_OpenAICompatProvider):
    """UnoRouter — OpenAI-compatible gateway (api.unorouter.com/v1).

    The APPLICATION owns model selection: primary + fallback model ids come
    from config and are tried in order. UnoRouter never chooses for us.
    """
    name = "unorouter"
    key_attr = "UNOROUTER_API_KEY"
    enabled_attr = "UNOROUTER_ENABLED"
    daily_limit_attr = "UNOROUTER_DAILY_REQUEST_LIMIT"
    quota_backoff_attr = "UNOROUTER_QUOTA_BACKOFF_MINUTES"

    def base_url(self):
        return settings.UNOROUTER_BASE_URL


class OpenRouterProvider(_OpenAICompatProvider):
    name = "openrouter"
    key_attr = "OPENROUTER_API_KEY"
    daily_limit_attr = "OPENROUTER_DAILY_REQUEST_LIMIT"
    quota_backoff_attr = "OPENROUTER_QUOTA_BACKOFF_MINUTES"

    def base_url(self):
        return settings.OPENROUTER_BASE_URL


class NVIDIAProvider(_OpenAICompatProvider):
    name = "nvidia"
    key_attr = "NVIDIA_API_KEY"
    daily_limit_attr = "NVIDIA_DAILY_REQUEST_LIMIT"
    quota_backoff_attr = "NVIDIA_QUOTA_BACKOFF_MINUTES"

    def base_url(self):
        return settings.NVIDIA_BASE_URL


class GeminiProvider(BaseLLMProvider):
    name = "gemini"
    key_attr = "GEMINI_API_KEY"
    daily_limit_attr = "GEMINI_DAILY_REQUEST_LIMIT"
    quota_backoff_attr = "GEMINI_QUOTA_BACKOFF_MINUTES"

    def get_client(self):
        return gemini_service._get_client()

    def send(self, model, system, user, temperature, max_tokens):
        # The Interactions API transport ignores chat-style sampling params;
        # temperature is embedded in the agent prompts where it matters.
        return gemini_service.generate_raw(system, user, model=model)

    def list_models(self):
        return _extract_model_ids(self.get_client().models.list())


_PROVIDER_INSTANCES = {
    "unorouter": UnoRouterProvider(),
    "groq": GroqProvider(),
    "nvidia": NVIDIAProvider(),
    "gemini": GeminiProvider(),
    "openrouter": OpenRouterProvider(),
}

# ---------------------------------------------------------------------------
# Routing tables
# ---------------------------------------------------------------------------

# task -> settings attribute for its provider
_TASK_PROVIDERS = {
    "technical": "LLM_TECH_PROVIDER",
    "debate": "LLM_DEBATE_PROVIDER",
    "risk": "LLM_RISK_PROVIDER",
    "cio": "LLM_CIO_PROVIDER",
    "news": "LLM_NEWS_PROVIDER",
    "fundamentals": "LLM_FUNDAMENTALS_PROVIDER",
}

# (provider, task) -> settings attribute for the model id
_TASK_MODELS = {
    ("groq", "technical"): "GROQ_TECH_MODEL",
    ("groq", "debate"): "GROQ_DEBATE_MODEL",
    ("groq", "risk"): "GROQ_RISK_MODEL",
    ("groq", "cio"): "GROQ_CIO_MODEL",
    ("gemini", "news"): "GEMINI_MODEL",
    ("gemini", "fundamentals"): "GEMINI_MODEL",
    ("openrouter", "technical"): "OPENROUTER_MODEL",
    ("openrouter", "debate"): "OPENROUTER_MODEL",
    ("openrouter", "risk"): "OPENROUTER_MODEL",
    ("openrouter", "cio"): "OPENROUTER_MODEL",
    ("openrouter", "news"): "OPENROUTER_MODEL",
    ("openrouter", "fundamentals"): "OPENROUTER_MODEL",
    ("nvidia", "technical"): "NVIDIA_MODEL",
    ("nvidia", "debate"): "NVIDIA_MODEL",
    ("nvidia", "risk"): "NVIDIA_MODEL",
    ("nvidia", "cio"): "NVIDIA_MODEL",
    ("nvidia", "news"): "NVIDIA_MODEL",
    ("nvidia", "fundamentals"): "NVIDIA_MODEL",
    # UnoRouter tasks all use the application's explicit model chain
    # (primary + UNOROUTER_FALLBACK_MODELS), resolved in _candidate_chain().
    ("unorouter", "technical"): "UNOROUTER_PRIMARY_MODEL",
    ("unorouter", "debate"): "UNOROUTER_PRIMARY_MODEL",
    ("unorouter", "risk"): "UNOROUTER_PRIMARY_MODEL",
    ("unorouter", "cio"): "UNOROUTER_PRIMARY_MODEL",
    ("unorouter", "news"): "UNOROUTER_PRIMARY_MODEL",
    ("unorouter", "fundamentals"): "UNOROUTER_PRIMARY_MODEL",
}


def _task_model(provider: str, task: str) -> str:
    attr = _TASK_MODELS.get((provider, task))
    return str(getattr(settings, attr, "") or "") if attr else ""


def route_info(agent: str) -> dict:
    """Resolves a task's (provider, model) plus the global fallback route
    from settings at call time. The provider is returned exactly as
    configured — an unknown provider name surfaces as a PROVIDER_ERROR in
    call() rather than being silently normalized."""
    provider_attr = _TASK_PROVIDERS[agent]
    provider = str(getattr(settings, provider_attr, "") or "").strip().lower()
    return {
        "provider": provider,
        "model": _task_model(provider, agent) if provider in _PROVIDER_INSTANCES else "",
        "fallback_provider": settings.LLM_FALLBACK_PROVIDER,
        "fallback_model": settings.LLM_FALLBACK_MODEL,
    }


def llm_routes() -> dict:
    """All task routes (for /api/config and startup validation)."""
    return {agent: route_info(agent) for agent in sorted(_TASK_PROVIDERS)}


def _provider_key(provider: str) -> str:
    return str(getattr(settings, _PROVIDER_INSTANCES[provider].key_attr, "") or "")


def _provider_enabled(provider: str) -> bool:
    inst = _PROVIDER_INSTANCES[provider]
    if not inst.enabled_attr:
        return True
    return bool(getattr(settings, inst.enabled_attr, True))


def _daily_limit(provider: str) -> int:
    return int(getattr(settings, _PROVIDER_INSTANCES[provider].daily_limit_attr, 0) or 0)


def _quota_backoff_seconds(provider: str) -> float:
    minutes = float(getattr(settings, _PROVIDER_INSTANCES[provider].quota_backoff_attr, 30) or 30)
    return max(30.0, minutes * 60.0)


def _groq_chain_model(task: str) -> str:
    """The Groq model used when the chain falls back to Groq for a task
    routed to UnoRouter: the task's Groq model when configured, else the
    generic Groq fallback model."""
    return _task_model("groq", task) or str(settings.GROQ_FALLBACK_MODEL or "")


# ---------------------------------------------------------------------------
# State (per process)
# ---------------------------------------------------------------------------

_clients = {}  # provider -> cached SDK client (tests may inject)
_verified_models = {}  # provider -> set of model ids seen in the live catalog
_last_validation = None  # report dict returned by validate_models()

# Model-catalog layer: /v1/models is fetched AT MOST once per provider per
# LLM_CATALOG_CACHE_TTL_MINUTES window (5-10 min) — one shared catalog for
# health checks, validations and chain resolution. A hanging provider is
# bounded by LLM_CATALOG_TIMEOUT_SECONDS in a worker thread, so a slow
# Gemini (or any provider) can never block startup or a health-check cycle.
_catalog_cache = {}  # provider -> {"ids": set|None, "fetched_at": float,
#                                "error": str|None, "timed_out": bool}
_catalog_lock = threading.Lock()
# TWO disjoint pools so nested work can never starve itself:
#   _catalog_pool  — runs RAW list_models() fetches (never submits anything)
#   _validation_pool — runs whole get_model_catalog() wrappers in parallel
# A single shared pool deadlocks: N wrappers occupy all workers and each
# wrapper's inner fetch stays queued behind them until its timeout fires
# (the exact "validation times out" symptom this fix removes). Abandoned
# hung fetches self-terminate via the SDK's own HTTP timeout.
_catalog_pool_ref = None
_validation_pool_ref = None
_model_warned = {}  # (provider, model_id) -> monotonic deadline: an
#                    LLM_MODEL_UNAVAILABLE warning is logged AT MOST once per
#                    model per catalog/health-check period (never 6x).


def _catalog_pool():
    global _catalog_pool_ref
    if _catalog_pool_ref is None:
        _catalog_pool_ref = ThreadPoolExecutor(
            max_workers=max(2, len(PROVIDERS)), thread_name_prefix="llm-catalog")
    return _catalog_pool_ref


def _validation_pool():
    global _validation_pool_ref
    if _validation_pool_ref is None:
        _validation_pool_ref = ThreadPoolExecutor(
            max_workers=max(2, len(PROVIDERS)), thread_name_prefix="llm-validate")
    return _validation_pool_ref


def _catalog_ttl_s() -> float:
    return max(60.0, float(settings.LLM_CATALOG_CACHE_TTL_MINUTES or 8) * 60.0)


def _normalize_model_id(model_id) -> str:
    """Exact-match normalization: trim whitespace and strip a 'models/'
    prefix (Gemini convention). Case is NOT mangled — ids match exactly."""
    return str(model_id or "").strip().removeprefix("models/")


def get_model_catalog(provider: str, force: bool = False) -> dict:
    """The provider's LIVE model catalog, TTL-cached.

    Returns {"ids": set|None, "fetched_at": float, "error": str|None,
             "timed_out": bool, "cached": bool, "age_s": float|None}.
    ids is None when the catalog could not be fetched (error/timeout/not
    configured) — callers must treat that as 'unverified', never as empty.
    Exactly ONE /v1/models request per provider per TTL window, shared by
    everything; force=True refetches (used when the TTL has expired)."""
    if provider not in _PROVIDER_INSTANCES:
        return {"ids": None, "fetched_at": 0.0, "error": f"unknown provider '{provider}'",
                "timed_out": False, "cached": False, "age_s": None}
    if not force:
        with _catalog_lock:
            entry = _catalog_cache.get(provider)
            if entry and entry.get("fetched_at") \
                    and (time.monotonic() - entry["fetched_at"]) < _catalog_ttl_s():
                out = dict(entry)
                out["cached"] = True
                out["age_s"] = round(time.monotonic() - entry["fetched_at"], 1)
                return out

    fetched = {"ids": None, "fetched_at": time.monotonic(),
               "error": None, "timed_out": False}
    inst = _PROVIDER_INSTANCES[provider]
    if not _provider_enabled(provider) or not _provider_key(provider):
        fetched["error"] = (f"disabled via {inst.enabled_attr}=false"
                            if not _provider_enabled(provider)
                            else f"{inst.key_attr} not configured")
    else:
        timeout_s = max(1.0, float(settings.LLM_CATALOG_TIMEOUT_SECONDS or 8))
        try:
            future = _catalog_pool().submit(inst.list_models)
            try:
                ids = future.result(timeout=timeout_s)
                fetched["ids"] = {_normalize_model_id(i) for i in (ids or set())}
                _verified_models[provider] = set(fetched["ids"])
            except _FutureTimeoutError:
                fetched["timed_out"] = True
                fetched["error"] = (f"model catalog fetch timed out after "
                                    f"{timeout_s:.0f}s")
                # Retrieve/swallow the late result so the abandoned worker
                # thread never logs an unretrieved-exception warning.
                future.add_done_callback(lambda f: f.exception() if f.done() else None)
        except Exception as exc:  # noqa: BLE001 — catalog fetch must never raise
            fetched["error"] = f"could not list models: {exc}"

    # A fresh catalog starts a new warning period for this provider: a model
    # that is STILL missing warns once for the new period (not zero times,
    # not six times).
    if fetched["ids"] is not None:
        for key in [k for k in _model_warned if k[0] == provider]:
            _model_warned.pop(key, None)
    with _catalog_lock:
        _catalog_cache[provider] = fetched
    out = dict(fetched)
    out["cached"] = False
    out["age_s"] = 0.0
    return out


def _warn_model_unavailable(provider: str, model: str, detail: str) -> bool:
    """Logs LLM_MODEL_UNAVAILABLE for (provider, model) AT MOST once per
    catalog/health-check period. Repeated 404s, six routed tasks or several
    validations within one period never produce duplicate warnings."""
    key = (provider, _normalize_model_id(model))
    now = time.monotonic()
    if now < _model_warned.get(key, 0.0):
        return False
    _model_warned[key] = now + _catalog_ttl_s()
    logger.warning("LLM_MODEL_UNAVAILABLE: %s: model '%s' %s", provider, model, detail)
    return True

_quota_backoff_until = {p: 0.0 for p in PROVIDERS}      # 429 circuit
_auth_backoff_until = {p: 0.0 for p in PROVIDERS}       # 401/403 circuit
_model_unavailable_until = {}                           # (provider, model) -> deadline (404 circuit)
_request_times = {p: deque() for p in PROVIDERS}        # rolling 24h send stamps
_last_failure = {p: None for p in PROVIDERS}            # {"status", "at"} for DEGRADED display

# Classic consecutive-failure circuit breaker (CLOSED -> OPEN -> HALF_OPEN).
_breaker = {
    p: {"failures": 0, "open_until": 0.0,
        "last_success": None, "last_failure": None}
    for p in PROVIDERS
}

# Router-level response cache: identical (agent, prompt) within the TTL is
# served without a network call; on total chain failure the last cached
# response is the "cached result" stop before the agent deterministic
# fallback. Bounded (LRU eviction), never stores secrets beyond prompt text
# the agents themselves already hold.
_response_cache: "OrderedDict[tuple, dict]" = OrderedDict()

_cycle_usage = {}

_ROLLING_WINDOW_S = 24 * 3600.0
_BACKOFF_CAP_S = 24 * 3600.0
_DEGRADED_WINDOW_S = 120.0


def reset_cycle_usage() -> None:
    """Zeroes the per-cycle usage counters. main.run_trading_cycle calls
    this at the start of every cycle."""
    _cycle_usage.clear()
    for provider in PROVIDERS:
        _cycle_usage[provider] = {
            "requests": 0,
            "ok": 0,
            "errors": {},
            "by_agent": {},
            "by_symbol": {},
        }


def reset_all_state() -> None:
    """Test hook: clears clients, circuits, breakers, caches, rolling
    budgets, usage and validation cache. Does NOT touch settings."""
    _clients.clear()
    _verified_models.clear()
    with _catalog_lock:
        _catalog_cache.clear()
    _model_warned.clear()
    global _last_validation
    _last_validation = None
    for p in PROVIDERS:
        _quota_backoff_until[p] = 0.0
        _auth_backoff_until[p] = 0.0
        _request_times[p].clear()
        _last_failure[p] = None
        _breaker[p] = {"failures": 0, "open_until": 0.0,
                       "last_success": None, "last_failure": None}
    _model_unavailable_until.clear()
    _response_cache.clear()
    reset_cycle_usage()


def cycle_usage() -> dict:
    """Snapshot of the current cycle's LLM usage (per provider, per agent,
    per symbol)."""
    out = {}
    for provider, u in _cycle_usage.items():
        if not (u["requests"] or u["ok"] or u["errors"]):
            continue  # provider untouched this cycle
        out[provider] = {
            "requests": u["requests"],
            "ok": u["ok"],
            "errors": dict(u["errors"]),
            "by_agent": {a: dict(v) for a, v in u["by_agent"].items()},
            "by_symbol": {s: dict(v) for s, v in u["by_symbol"].items()},
        }
    return out


# ---------------------------------------------------------------------------
# Classic circuit breaker (CLOSED / OPEN / HALF_OPEN)
# ---------------------------------------------------------------------------

def _breaker_state(provider: str) -> str:
    entry = _breaker[provider]
    if entry["failures"] >= _breaker_threshold() and entry["open_until"] > time.monotonic():
        return "OPEN"
    if entry["failures"] >= _breaker_threshold():
        return "HALF_OPEN"  # backoff expired: the next request is a probe
    return "CLOSED"


def _breaker_threshold() -> int:
    return max(1, int(settings.LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD or 3))


def _breaker_backoff_s() -> float:
    return max(1.0, float(settings.LLM_CIRCUIT_BREAKER_BACKOFF_SECONDS or 600.0))


def _breaker_check(provider: str):
    """(allowed, reason). While OPEN no request is sent to the provider."""
    if not settings.LLM_CIRCUIT_BREAKER_ENABLED:
        return True, None
    state = _breaker_state(provider)
    if state == "OPEN":
        remaining = int(round(_breaker[provider]["open_until"] - time.monotonic()))
        return False, (
            f"circuit OPEN after {_breaker[provider]['failures']} consecutive "
            f"failures — no request sent for another {remaining}s"
        )
    return True, None


def _breaker_success(provider: str) -> None:
    _breaker[provider]["failures"] = 0
    _breaker[provider]["open_until"] = 0.0
    _breaker[provider]["last_success"] = _now_iso()


def _breaker_failure(provider: str, status: str) -> None:
    if not settings.LLM_CIRCUIT_BREAKER_ENABLED:
        _breaker[provider]["last_failure"] = _now_iso()
        return
    entry = _breaker[provider]
    entry["failures"] += 1
    entry["last_failure"] = _now_iso()
    if entry["failures"] >= _breaker_threshold():
        was_open = _breaker_state(provider) in ("OPEN", "HALF_OPEN")
        entry["open_until"] = time.monotonic() + _breaker_backoff_s()
        if not was_open:
            logger.warning(
                f"{provider}: circuit breaker OPEN after {entry['failures']} "
                f"consecutive failures (last status {status}); no requests for "
                f"{_breaker_backoff_s():.0f}s."
            )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def provider_states() -> dict:
    """Circuit-breaker state per provider (for /api/health, /api/config,
    /api/providers/health). No secrets.

    state: READY | NOT_CONFIGURED | QUOTA_EXHAUSTED | MODEL_UNAVAILABLE |
           AUTH_ERROR | NETWORK_ERROR | DEGRADED
    circuit: CLOSED | OPEN | HALF_OPEN   (classic consecutive-failure breaker)
    """
    now = time.monotonic()
    states = {}
    for provider in PROVIDERS:
        inst = _PROVIDER_INSTANCES[provider]
        limit = _daily_limit(provider)
        stamps = _request_times[provider]
        while stamps and stamps[0] <= now - _ROLLING_WINDOW_S:
            stamps.popleft()

        breaker = _breaker[provider]
        detail = ""
        if not _provider_enabled(provider):
            state = "NOT_CONFIGURED"
            detail = f"disabled via {inst.enabled_attr}=false"
        elif not _provider_key(provider):
            state = "NOT_CONFIGURED"
            detail = f"{inst.key_attr} not set"
        elif _auth_backoff_until[provider] > now:
            state = "AUTH_ERROR"
            detail = f"auth circuit open for another {int(round(_auth_backoff_until[provider] - now))}s"
        elif _quota_backoff_until[provider] > now:
            state = "QUOTA_EXHAUSTED"
            detail = f"quota circuit open for another {int(round(_quota_backoff_until[provider] - now))}s"
        elif limit > 0 and len(stamps) >= limit:
            state = "QUOTA_EXHAUSTED"
            detail = f"local rolling-24h budget reached ({len(stamps)}/{limit})"
        else:
            dead_models = [
                m for (p, m), until in _model_unavailable_until.items()
                if p == provider and until > now
            ]
            if dead_models:
                state = "MODEL_UNAVAILABLE"
                detail = f"model circuit open: {', '.join(sorted(dead_models))}"
            else:
                last = _last_failure[provider]
                if last and (now - last["at"]) < _DEGRADED_WINDOW_S and last["status"] in (
                    "NETWORK_ERROR", "PROVIDER_ERROR", "RATE_LIMITED",
                ):
                    state = last["status"] if last["status"] == "NETWORK_ERROR" else "DEGRADED"
                    detail = f"last request failed with {last['status']}"
                else:
                    state = "READY"
                    detail = f"{len(stamps)} requests in last 24h"

        states[provider] = {
            "state": state,
            "detail": detail,
            "enabled": _provider_enabled(provider) and bool(_provider_key(provider)),
            "circuit": _breaker_state(provider),
            "failure_count": breaker["failures"],
            "open_until": (
                datetime.fromtimestamp(
                    time.time() + (breaker["open_until"] - now), tz=timezone.utc
                ).isoformat()
                if breaker["open_until"] > now else None
            ),
            "last_success": breaker["last_success"],
            "last_failure": breaker["last_failure"],
            "requests_last_24h": len(stamps),
            "daily_request_limit": limit,
            "quota_cooldown_remaining_s": round(max(0.0, _quota_backoff_until[provider] - now), 1),
            "auth_cooldown_remaining_s": round(max(0.0, _auth_backoff_until[provider] - now), 1),
        }
    return states


def quota_state() -> dict:
    """Quota posture per provider (kept for backward compatibility; the
    richer view is provider_states())."""
    states = provider_states()
    out = {}
    for provider, s in states.items():
        out[provider] = {
            "daily_request_limit": s["daily_request_limit"],
            "requests_last_24h": s["requests_last_24h"],
            "backoff_active": s["state"] == "QUOTA_EXHAUSTED",
            "backoff_seconds_remaining": s["quota_cooldown_remaining_s"],
        }
    return out


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

_RETRY_DELAY_PATTERNS = (
    re.compile(r"[Rr]etry\s+(?:in|after)\s+([\d.]+)\s*s(?:ec(?:ond)?s?)?\b"),
    re.compile(r'"retryDelay"\s*:\s*"?(\d+)\s*s', re.IGNORECASE),
)

_QUOTA_METRIC_PATTERNS = (
    "free_tier", "freetier", "daily", "per day", "per_day", "exhaust",
)


def _parse_retry_delay(message: str):
    """Extracts a retry-after delay (seconds) from a provider error message."""
    for pattern in _RETRY_DELAY_PATTERNS:
        m = pattern.search(message)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _looks_like_quota_exhaustion(message: str, retry_after) -> bool:
    """Distinguishes quota exhaustion (stop; do not retry) from a transient
    per-minute rate limit (safe to retry after the given delay)."""
    low = message.lower()
    if any(marker in low for marker in _QUOTA_METRIC_PATTERNS):
        return True
    if retry_after is None:
        return True  # no retry hint -> assume hard exhaustion, do not hammer
    return retry_after > _max_rate_limit_wait()


def _max_rate_limit_wait() -> float:
    return max(1.0, float(settings.LLM_RATE_LIMIT_MAX_WAIT_SECONDS or 30))


def _max_retries() -> int:
    return max(0, int(settings.LLM_RATE_LIMIT_MAX_RETRIES or 0))


def _is_network_error(exc: Exception) -> bool:
    """Transport-level failures (connection/timeout), by base type or name.
    Never keyword-matches arbitrary messages."""
    try:
        import httpx
        if isinstance(exc, httpx.TransportError):
            return True
    except ImportError:
        pass
    for cls in exc.__class__.__mro__:
        name = cls.__name__
        if name in ("APIConnectionError", "ConnectError", "ConnectTimeout",
                    "ReadTimeout", "WriteTimeout", "PoolTimeout", "TimeoutError"):
            return True
    return False


def _classify_exception(exc: Exception):
    """Maps a provider exception to (status, retry_after).

    Uses the exception's HTTP status code when the SDK exposes one — never
    message keyword matching on generic exceptions.
    """
    code = getattr(exc, "status_code", None)
    if not isinstance(code, int):
        code = getattr(exc, "code", None)
    if not isinstance(code, int):
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
    message = str(exc)
    retry_after = _parse_retry_delay(message)

    if isinstance(code, int) and code == 429:
        if _looks_like_quota_exhaustion(message, retry_after):
            return "PROVIDER_QUOTA_EXCEEDED", retry_after
        return "RATE_LIMITED", retry_after
    if isinstance(code, int) and code == 404:
        return "MODEL_NOT_FOUND", None
    if isinstance(code, int) and code in (401, 403):
        return "AUTH_ERROR", None
    if _is_network_error(exc):
        return "NETWORK_ERROR", None
    return "PROVIDER_ERROR", None


# ---------------------------------------------------------------------------
# Specific circuits (quota / auth / dead model)
# ---------------------------------------------------------------------------

def _quota_blocked(provider: str):
    """(blocked, reason). Checked BEFORE any request is sent."""
    now = time.monotonic()
    until = _quota_backoff_until[provider]
    if until > now:
        remaining = int(round(until - now))
        return True, (
            f"provider quota exhausted (server 429); circuit open for another "
            f"{remaining}s — no request sent"
        )
    limit = _daily_limit(provider)
    if limit > 0:
        stamps = _request_times[provider]
        while stamps and stamps[0] <= now - _ROLLING_WINDOW_S:
            stamps.popleft()
        if len(stamps) >= limit:
            return True, (
                f"local rolling-24h request budget for {provider} reached "
                f"({len(stamps)}/{limit}) — no request sent"
            )
    return False, None


def _mark_quota_backoff(provider: str, retry_after) -> None:
    """Records a server-side quota exhaustion; stops further requests to
    this provider for the window. Never retried past this point."""
    if retry_after is not None and 0 < retry_after <= _BACKOFF_CAP_S:
        window = retry_after
        why = f"server retry delay {int(retry_after)}s"
    else:
        window = _quota_backoff_seconds(provider)
        why = f"configured {_PROVIDER_INSTANCES[provider].quota_backoff_attr}"
    _quota_backoff_until[provider] = max(
        _quota_backoff_until[provider], time.monotonic() + window
    )
    logger.warning(
        f"{provider}: quota exhausted (429). Circuit open for {int(window)}s "
        f"({why}). Subsequent calls return PROVIDER_QUOTA_EXCEEDED."
    )


def _mark_model_unavailable(provider: str, model: str) -> None:
    """404 circuit: stop requesting this (provider, model) pair. The warning
    is deduplicated per catalog/health-check period — repeated 404s on the
    same model produce ONE warning, not one per request/task."""
    minutes = max(0.5, float(settings.MODEL_UNAVAILABLE_COOLDOWN_MINUTES or 30))
    _model_unavailable_until[(provider, model)] = time.monotonic() + minutes * 60.0
    _warn_model_unavailable(
        provider, model,
        f"not found (404) or absent from the live catalog. Circuit open for "
        f"{minutes:.0f} min; requests to it short-circuit and the next "
        f"configured model is tried."
    )


def _mark_auth_failure(provider: str) -> None:
    """401/403 circuit: stop authenticating repeatedly."""
    minutes = max(0.5, float(settings.AUTH_COOLDOWN_MINUTES or 60))
    _auth_backoff_until[provider] = max(
        _auth_backoff_until[provider], time.monotonic() + minutes * 60.0
    )
    logger.warning(
        f"{provider}: authentication failed (401/403). Circuit open for "
        f"{minutes:.0f} min; no further requests until it expires."
    )


def _count_request(provider: str) -> None:
    now = time.monotonic()
    stamps = _request_times[provider]
    stamps.append(now)
    while stamps and stamps[0] <= now - _ROLLING_WINDOW_S:
        stamps.popleft()


# ---------------------------------------------------------------------------
# Result type + usage recording
# ---------------------------------------------------------------------------

# Standardized error codes for LLMResult.standard() (Part 5 contract).
_ERROR_CODES = {
    "NOT_CONFIGURED": "PROVIDER_NOT_CONFIGURED",
    "MODEL_NOT_FOUND": "LLM_MODEL_UNAVAILABLE",
    "PROVIDER_QUOTA_EXCEEDED": "QUOTA_EXCEEDED",
    "RATE_LIMITED": "RATE_LIMITED",
    "PROVIDER_ERROR": "PROVIDER_ERROR",
    "AUTH_ERROR": "AUTH_ERROR",
    "NETWORK_ERROR": "NETWORK_ERROR",
    "INVALID_RESPONSE": "INVALID_RESPONSE",
    "CIRCUIT_OPEN": "PROVIDER_UNAVAILABLE",
}


class LLMResult:
    """Outcome of one task request (standardized via standard())."""

    def __init__(self, agent, provider, model, status, text="", parsed=None,
                 latency_ms=None, error=None, error_type=None, attempts=0,
                 fallback_used=False, cached=False, cache_age_s=None,
                 cache_replay=False):
        self.agent = agent
        self.provider = provider
        self.model = model
        self.status = status
        self.text = text
        self.parsed = parsed
        self.latency_ms = latency_ms
        self.error = error
        self.error_type = error_type or (None if status == "OK" else status)
        self.attempts = attempts
        self.fallback_used = fallback_used
        self.cached = cached
        self.cache_age_s = cache_age_s
        # cache_replay: served from cache AFTER a total chain failure
        # (the "cached result" stop before the deterministic fallback).
        self.cache_replay = cache_replay

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def standard(self) -> dict:
        """The standardized response shape every provider funnel returns."""
        if self.ok:
            return {
                "success": True,
                "provider": self.provider,
                "model": self.model,
                "content": self.text,
                "cached": self.cached,
                "fallback_used": self.fallback_used,
                "latency_ms": self.latency_ms,
                "error": None,
            }
        return {
            "success": False,
            "provider": None,
            "model": None,
            "content": None,
            "cached": False,
            "fallback_used": self.fallback_used,
            "latency_ms": self.latency_ms,
            "error": _ERROR_CODES.get(self.status, "PROVIDER_UNAVAILABLE"),
        }

    def as_dict(self) -> dict:
        return {
            "agent": self.agent,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "error_type": self.error_type,
            "attempts": self.attempts,
            "fallback_used": self.fallback_used,
            "cached": self.cached,
            "cache_replay": self.cache_replay,
        }


def _record(result: LLMResult, symbol=None) -> LLMResult:
    """Records a finished attempt in the per-cycle usage counters."""
    provider = result.provider
    usage = _cycle_usage.get(provider)
    if usage is None:
        return result  # reset_cycle_usage not called (ad-hoc call); fine

    def _bump(bucket):
        bucket["requests"] += 1
        if result.ok:
            bucket["ok"] += 1
        else:
            bucket["errors"][result.status] = bucket["errors"].get(result.status, 0) + 1

    _bump(usage)
    _bump(usage["by_agent"].setdefault(result.agent, {"requests": 0, "ok": 0, "errors": {}}))
    if symbol:
        _bump(usage["by_symbol"].setdefault(symbol, {"requests": 0, "ok": 0, "errors": {}}))
    return result


def _reclassify(result: LLMResult, symbol=None) -> LLMResult:
    """Adjusts already-recorded counters when an OK response turns out
    unparseable (call_json): move it from ok to errors."""
    provider = result.provider
    usage = _cycle_usage.get(provider)
    if usage is None:
        return result

    def _move(bucket):
        if bucket["ok"] > 0:
            bucket["ok"] -= 1
        bucket["errors"][result.status] = bucket["errors"].get(result.status, 0) + 1

    _move(usage)
    agent_u = usage["by_agent"].get(result.agent)
    if agent_u:
        _move(agent_u)
    if symbol:
        sym_u = usage["by_symbol"].get(symbol)
        if sym_u:
            _move(sym_u)
    return result


# ---------------------------------------------------------------------------
# Transport (one attempt = NO retries unless a short explicit 429 delay;
# failures immediately hand over to the next model/provider)
# ---------------------------------------------------------------------------


def _attempt(provider: str, model: str, system: str, user: str,
             temperature: float, max_tokens: int):
    inst = _PROVIDER_INSTANCES[provider]
    last = ("PROVIDER_ERROR", "request failed", None, None)
    for attempt in range(_max_retries() + 1):
        _count_request(provider)
        try:
            text = inst.send(model, system, user, temperature, max_tokens)
            if not str(text).strip():
                return "INVALID_RESPONSE", "model returned an empty response", text, None
            _last_failure[provider] = None
            return "OK", None, str(text), None
        except Exception as exc:  # noqa: BLE001 — classified below
            status, retry_after = _classify_exception(exc)
            last = (status, str(exc) or exc.__class__.__name__, None, retry_after)
            _last_failure[provider] = {"status": status, "at": time.monotonic()}
            if status == "PROVIDER_QUOTA_EXCEEDED":
                _mark_quota_backoff(provider, retry_after)
                return last  # never retry an exhausted quota
            if status == "MODEL_NOT_FOUND":
                _mark_model_unavailable(provider, model)
                return last  # never retry a dead model
            if status == "AUTH_ERROR":
                _mark_auth_failure(provider)
                return last  # never retry a failing credential
            if status != "RATE_LIMITED":
                return last
            can_retry = (
                attempt < _max_retries()
                and retry_after is not None
                and 0 < retry_after <= _max_rate_limit_wait()
            )
            if not can_retry:
                return last
            wait_s = min(retry_after, _max_rate_limit_wait())
            logger.info(
                f"{provider}: transient rate limit (429), retrying in {wait_s:.1f}s "
                f"(attempt {attempt + 1}/{_max_retries()})"
            )
            time.sleep(wait_s)
    return last


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------


def _cache_key(agent: str, system: str, user: str) -> str:
    payload = f"{agent}\x00{system}\x00{user}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(key: str, fresh_only: bool = True):
    entry = _response_cache.get(key)
    if entry is None:
        return None
    if fresh_only and entry["expires"] <= time.monotonic():
        return None  # stale entries only serve total-failure replay
    _response_cache.move_to_end(key)
    return entry


def _cache_put(key: str, text: str, parsed, provider: str, model: str) -> None:
    ttl = max(0.0, float(settings.LLM_RESPONSE_CACHE_TTL_MINUTES) * 60.0)
    if ttl <= 0:
        return
    _response_cache[key] = {
        "expires": time.monotonic() + ttl,
        "stored": time.monotonic(),
        "text": text,
        "parsed": parsed,
        "provider": provider,
        "model": model,
    }
    _response_cache.move_to_end(key)
    max_entries = max(1, int(settings.LLM_RESPONSE_CACHE_MAX_ENTRIES or 500))
    while len(_response_cache) > max_entries:
        _response_cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Candidate chain resolution (the application's OWN model fallback logic)
# ---------------------------------------------------------------------------


def _candidate_chain(agent: str):
    """Resolves the ordered (provider, model) candidates for a task.

    - task routed to UnoRouter: [uno primary, uno fallback 1, uno fallback 2,
      ...] capped at LLM_MAX_MODEL_ATTEMPTS models, then Groq (the final
      provider fallback) when Groq is configured.
    - task routed to Groq: [groq model] (+ the optional verified global
      fallback route, unchanged legacy behavior).
    - any other provider: [provider model] (+ optional global fallback).
    """
    route = route_info(agent)
    provider, model = route["provider"], route["model"]
    fb_provider, fb_model = route["fallback_provider"], route["fallback_model"]

    if provider == "unorouter":
        # Catalog-verified chain: configured primary + fallback candidates,
        # filtered to ids that EXIST in the live /v1/models catalog (TTL
        # cached — no extra catalog calls). A model absent from the catalog
        # is never attempted (no 404 round-trip); the cap still applies.
        effective = unorouter_effective_chain()
        chain = [("unorouter", m) for m in effective["chain"]]
        # Final provider fallback: Groq, when it is usable at all.
        if _provider_enabled("groq") and _provider_key("groq"):
            groq_model = _groq_chain_model(agent)
            if groq_model:
                chain.append(("groq", groq_model))
        return chain, route

    # Legacy/single-route providers (+ optional verified global fallback).
    chain = [(provider, model)] if provider in _PROVIDER_INSTANCES else []
    fallback_usable = (
        bool(fb_provider) and bool(fb_model)
        and fb_provider in _PROVIDER_INSTANCES
        and fb_provider != provider
        and fb_model in _verified_models.get(fb_provider, set())
    )
    if fallback_usable:
        chain.append((fb_provider, fb_model))
    return chain, route


def _candidate_precheck(provider: str, model: str):
    """(status, error) when a candidate must be skipped WITHOUT a network
    call (circuits), else (None, None)."""
    now = time.monotonic()
    if not _provider_enabled(provider):
        return "NOT_CONFIGURED", f"{_PROVIDER_INSTANCES[provider].enabled_attr}=false"
    if not _provider_key(provider):
        key_name = _PROVIDER_INSTANCES[provider].key_attr
        return "NOT_CONFIGURED", f"{key_name} not configured."
    allowed, breaker_reason = _breaker_check(provider)
    if not allowed:
        return "CIRCUIT_OPEN", f"CIRCUIT_OPEN: {breaker_reason}."
    blocked, reason = _quota_blocked(provider)
    if blocked:
        return "PROVIDER_QUOTA_EXCEEDED", f"PROVIDER_QUOTA_EXCEEDED: {reason}."
    if _auth_backoff_until[provider] > now:
        remaining = int(round(_auth_backoff_until[provider] - now))
        return "AUTH_ERROR", (f"AUTH_ERROR: authentication previously failed on {provider}; "
                              f"circuit open ({remaining}s remaining).")
    if _model_unavailable_until.get((provider, model), 0.0) > now:
        remaining = int(round(_model_unavailable_until[(provider, model)] - now))
        return "MODEL_NOT_FOUND", (f"LLM_MODEL_UNAVAILABLE: model '{model}' previously "
                                   f"returned 404 on {provider}; circuit open "
                                   f"({remaining}s remaining).")
    # Catalog-verified skip: a model id that is not in the (TTL-cached) live
    # catalog is never sent — no 404 round-trip is needed. Only applied when
    # a catalog was actually fetched; an unavailable catalog never blocks.
    catalog = get_model_catalog(provider)
    live_ids = catalog.get("ids")
    if live_ids is not None and _normalize_model_id(model) not in live_ids:
        _warn_model_unavailable(
            provider, model,
            f"is not in the live /v1/models catalog ({len(live_ids)} live "
            f"models, catalog {catalog.get('age_s')}s old); skipped without "
            f"a request, the next configured model is used.")
        return "MODEL_NOT_FOUND", (
            f"LLM_MODEL_UNAVAILABLE: model '{model}' is not in the live "
            f"catalog ({len(live_ids)} live models).")
    return None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def call(agent: str, system: str, user: str, temperature: float = 0.2,
         max_tokens: int = 500, symbol: str = None,
         expect_json: bool = False) -> LLMResult:
    """Runs one task request through the unified chain:

    cache -> provider chain (primary model -> fallback models -> Groq)
    -> cached replay on total failure. Never raises for provider-side
    problems; never blocks on long retries (10s timeout, 0 SDK retries).
    """
    t0 = time.monotonic()
    key = _cache_key(agent, system, user)

    # 1. CACHE HIT (fresh) -> return without any network call.
    entry = _cache_get(key, fresh_only=True)
    if entry is not None:
        return LLMResult(
            agent, entry["provider"], entry["model"], "OK",
            text=entry["text"], parsed=entry["parsed"],
            latency_ms=0.0, cached=True,
            cache_age_s=round(time.monotonic() - entry["stored"], 1),
        )

    chain, route = _candidate_chain(agent)

    # Configuration problem with no usable candidate at all.
    if not chain or route["provider"] not in _PROVIDER_INSTANCES:
        return _record(LLMResult(
            agent, route["provider"], route["model"], "PROVIDER_ERROR",
            error=f"PROVIDER_ERROR: unknown LLM provider '{route['provider']}' "
                  f"(check LLM_*_PROVIDER settings).",
        ), symbol)
    if not route["model"] and route["provider"] != "unorouter":
        model_env = _TASK_MODELS.get((route["provider"], agent), "?")
        return _record(LLMResult(
            agent, route["provider"], "", "NOT_CONFIGURED",
            error=f"No model configured for {agent} on {route['provider']} (set {model_env}).",
        ), symbol)

    # 2. Walk the chain: each failure records, updates circuits, and
    #    IMMEDIATELY tries the next candidate. No sleeping between models.
    last_result = None
    for idx, (cand_provider, cand_model) in enumerate(chain):
        if not cand_model:
            continue
        pre_status, pre_error = _candidate_precheck(cand_provider, cand_model)
        if pre_status is None:
            status, attempt_error, text, _ = _attempt(
                cand_provider, cand_model, system, user, temperature, max_tokens
            )
            parsed = None
            if status == "OK" and expect_json:
                # Malformed JSON is treated as a failed ATTEMPT: the next
                # model in the chain gets a chance instead of failing the
                # whole call.
                try:
                    parsed = extract_json(text)
                except Exception as exc:  # noqa: BLE001 — parse errors are data
                    status = "INVALID_RESPONSE"
                    attempt_error = f"unparseable JSON response: {exc}"
            if status == "OK":
                _breaker_success(cand_provider)
                if expect_json and parsed is None:
                    parsed = _safe_parse(text)
                _cache_put(key, text, parsed, cand_provider, cand_model)
                return _record(LLMResult(
                    agent, cand_provider, cand_model, "OK", text=text,
                    parsed=parsed,
                    latency_ms=round((time.monotonic() - t0) * 1000, 1),
                    attempts=idx + 1, fallback_used=idx > 0,
                ), symbol)
            error = f"{status}: {attempt_error}"
        else:
            # Circuit/config skip (no network call): the specific circuit
            # (quota/auth/model/breaker) already owns its window — a skip is
            # NOT counted as a new failure, so the breaker can close.
            status, error = pre_status, pre_error

        # Only an ACTUAL failed attempt updates the failure breaker.
        if pre_status is None:
            _breaker_failure(cand_provider, status)
        last_result = LLMResult(
            agent, cand_provider, cand_model, status,
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
            error=error, attempts=idx + 1, fallback_used=idx > 0,
        )
        if idx == 0 and len(chain) > 1:
            log = logger.info if status == "NOT_CONFIGURED" else logger.warning
            log(
                f"{agent}: {cand_provider}/{cand_model} unavailable ({status}); "
                f"trying next fallback in the chain."
            )
        # Quota exhaustion ends the chain for THAT provider only; other
        # providers (e.g. Groq after UnoRouter) still get their turn.

    # 3. Total chain failure -> cached replay (even a stale entry) if one
    #    exists; the agent's deterministic fallback is the next stop after.
    if last_result is None:
        last_result = LLMResult(
            agent, route["provider"], route["model"], "PROVIDER_ERROR",
            error="PROVIDER_ERROR: no usable provider/model candidate in the chain.",
        )
    replay = _response_cache.get(key)
    if replay is not None:
        _record(last_result, symbol)  # the failure is still accounted
        return LLMResult(
            agent, replay["provider"], replay["model"], "OK",
            text=replay["text"], parsed=replay["parsed"],
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
            cached=True, fallback_used=True,
            cache_age_s=round(time.monotonic() - replay["stored"], 1),
            cache_replay=True,
        )

    return _record(last_result, symbol)


def _safe_parse(text: str):
    try:
        return extract_json(text)
    except Exception:  # noqa: BLE001 — best effort only
        return None


def complete(task: str, messages: list = None, system: str = None,
             user: str = None, temperature: float = 0.2, max_tokens: int = 500,
             symbol: str = None) -> LLMResult:
    """Messages-style entry point: complete(task="debate", messages=[...]).

    Accepts an OpenAI-style messages list (system/user roles) or explicit
    system/user strings. Equivalent to call(); returns the same LLMResult.
    """
    if messages:
        sys_parts, user_parts = [], []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
            if role == "system":
                sys_parts.append(content)
            else:
                user_parts.append(content)
        system = "\n\n".join(p for p in sys_parts if p) or system
        user = "\n\n".join(p for p in user_parts if p) or user
    return call(task, system=system or "", user=user or "",
                temperature=temperature, max_tokens=max_tokens, symbol=symbol)


def call_json(agent: str, system: str, user: str, temperature: float = 0.2,
              max_tokens: int = 500, symbol: str = None) -> LLMResult:
    """call() + tolerant JSON parsing. A malformed response is retried down
    the model chain inside call(); if every model returns unparseable output
    the result comes back with status INVALID_RESPONSE (never fabricated
    content)."""
    result = call(agent, system, user, temperature=temperature,
                  max_tokens=max_tokens, symbol=symbol, expect_json=True)
    if not result.ok:
        return result
    if result.parsed is None:
        try:
            result.parsed = extract_json(result.text)
        except Exception as exc:  # noqa: BLE001 — parse errors are data, not crashes
            result.status = "INVALID_RESPONSE"
            result.error = f"INVALID_RESPONSE: {exc}"
            result.error_type = "INVALID_RESPONSE"
            _reclassify(result, symbol)
    return result


# ---------------------------------------------------------------------------
# Startup validation — verify configured models against the live catalogs
# ---------------------------------------------------------------------------


def _extract_model_ids(response) -> set:
    """Pulls model id strings out of a provider's /v1/models response.
    Handles OpenAI-style objects (.data[] of items with .id), raw dicts
    ({"data": [{"id": ...}]}) and plain lists — always the ACTUAL returned
    ids from data[].id, never assumptions from public documentation."""
    ids = set()
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    items = data if data is not None else response
    if items is None:
        return ids
    if isinstance(items, dict):
        items = items.get("data") or items.get("models") or []
    try:
        for item in items:
            if isinstance(item, dict):
                model_id = item.get("id") or item.get("name")
            else:
                model_id = getattr(item, "id", None) or getattr(item, "name", None)
            if model_id:
                ids.add(_normalize_model_id(model_id))
    except TypeError:
        pass
    return ids


def _model_attempt_cap() -> int:
    return max(1, int(settings.LLM_MAX_MODEL_ATTEMPTS or 3))


def unorouter_model_chain() -> list:
    """The CONFIGURED UnoRouter model chain: primary + ALL ordered fallback
    candidates (uncapped — this is what the operator asked for, reported in
    diagnostics). The attempt cap is applied to the EFFECTIVE (catalog-
    verified) chain so a dead candidate never wastes an attempt slot."""
    models = [str(settings.UNOROUTER_PRIMARY_MODEL or "")]
    models += [m for m in (settings.UNOROUTER_FALLBACK_MODELS or []) if m]
    return [_normalize_model_id(m) for m in models if m]


def unorouter_effective_chain(catalog_ids=None) -> dict:
    """The catalog-VERIFIED UnoRouter chain.

    Every configured id (primary first, then the ordered fallback
    candidates) is kept ONLY if that exact id is present in the live
    /v1/models catalog. Ids are never invented or substituted: a missing
    candidate is reported and skipped, and the next candidate that DOES
    exist is selected. When the catalog itself is unavailable (fetch
    failed/timed out) the configured order is kept — verification is
    impossible, and the runtime 404 circuits still protect every request —
    with catalog_available=False reporting that honestly.

    Returns {"chain": [...], "configured": [...], "missing": [...],
             "selected_fallback": str|None, "catalog_available": bool}."""
    configured = unorouter_model_chain()
    if catalog_ids is None:
        catalog_ids = get_model_catalog("unorouter").get("ids")
    if catalog_ids is None:
        return {"chain": list(configured)[:_model_attempt_cap()],
                "configured": configured, "missing": [],
                "selected_fallback": (
                    configured[1] if len(configured) > 1 else None),
                "catalog_available": False}
    live = {_normalize_model_id(m) for m in catalog_ids}
    chain, missing = [], []
    for model in configured:
        (chain if model in live else missing).append(model)
    # The attempt cap applies to LIVE models only (never unlimited guessing;
    # a dead candidate never consumes an attempt slot).
    chain = chain[:_model_attempt_cap()]
    return {"chain": chain, "configured": configured, "missing": missing,
            "selected_fallback": (chain[1] if len(chain) > 1 else None),
            "catalog_available": True}


def _configured_models_for(provider: str) -> list:
    """The model ids this deployment actually uses on a provider (ordered,
    unique, normalized). unorouter: primary + fallback candidates. groq:
    task-specific models for groq-routed tasks plus the chain-final
    GROQ_FALLBACK_MODEL whenever Groq backs up UnoRouter-routed tasks.
    gemini/nvidia/openrouter: their model when a task is routed there."""
    out = []

    def add(model):
        model = _normalize_model_id(model)
        if model and model not in out:
            out.append(model)

    routed = [str(getattr(settings, attr, "") or "").lower()
              for attr in _TASK_PROVIDERS.values()]
    if provider == "unorouter":
        for m in unorouter_model_chain():
            add(m)
    elif provider == "groq":
        for task, attr in sorted(_TASK_PROVIDERS.items()):
            if str(getattr(settings, attr, "") or "").lower() == "groq":
                add(_task_model("groq", task))
        if "unorouter" in routed and _provider_enabled("groq") and _provider_key("groq"):
            add(_groq_chain_model("technical") or str(settings.GROQ_FALLBACK_MODEL or ""))
    elif provider == "gemini":
        if "gemini" in routed:
            add(settings.GEMINI_MODEL)
    elif provider == "nvidia":
        if "nvidia" in routed:
            add(settings.NVIDIA_MODEL)
    elif provider == "openrouter":
        if "openrouter" in routed:
            add(settings.OPENROUTER_MODEL)
    return out


def active_chain_providers() -> list:
    """Providers the ACTIVE request chain depends on: every provider a task
    is routed to, plus Groq when any task is UnoRouter-routed (Groq is the
    chain-final provider fallback). Used by health to decide which LLM
    providers are REQUIRED."""
    routed = {str(getattr(settings, attr, "") or "").lower()
              for attr in _TASK_PROVIDERS.values()}
    routed.discard("")
    if "unorouter" in routed:
        routed.add("groq")
    return sorted(p for p in routed if p in _PROVIDER_INSTANCES)


def validate_models(refresh_if_stale: bool = False) -> dict:
    """Verifies every configured model id against each provider's LIVE
    /v1/models catalog.

    Catalog policy (one fetch per provider per health-check period):
      * /v1/models is fetched AT MOST once per provider per
        LLM_CATALOG_CACHE_TTL_MINUTES (5-10 min) — health checks, startup
        validation, chain resolution and /api/providers/health all share the
        same TTL-cached catalog.
      * Each fetch is bounded by LLM_CATALOG_TIMEOUT_SECONDS in a worker
        thread (a hanging provider, e.g. Gemini, is marked DEGRADED and the
        check continues — startup never blocks).
      * ids are compared with EXACT normalization (trim + 'models/' prefix);
        nothing is assumed from public documentation.

    Per-provider status:
      READY            catalog fetched, all configured models matched
      NO_USABLE_MODEL  catalog fetched but ZERO configured models matched
      DEGRADED         catalog fetch timed out (provider kept, unverified)
      ERROR            catalog fetch failed
      NOT_CONFIGURED   no key / disabled — catalog not fetched

    Returns: {"providers": {name: {"ok", "status", "models_found",
                                   "configured", "matched", "missing",
                                   "checked", "selected_fallback",
                                   "catalog_age_s", "catalog_cached",
                                   "timed_out", "error"}},
              "routes": {task: {provider, model, fallback_provider,
                                fallback_model}}}
    """
    global _last_validation

    report = {"providers": {}, "routes": {}}

    # --- fetch every enabled+keyed provider's catalog ONCE (concurrently,
    #     each bounded by LLM_CATALOG_TIMEOUT_SECONDS) ----------------------
    to_fetch = [p for p in PROVIDERS
                if _provider_enabled(p) and _provider_key(p)]
    force_providers = set()
    if refresh_if_stale:
        now = time.monotonic()
        with _catalog_lock:
            for p in to_fetch:
                entry = _catalog_cache.get(p)
                if not entry or not entry.get("fetched_at") \
                        or (now - entry["fetched_at"]) >= _catalog_ttl_s():
                    force_providers.add(p)
    futures = {}
    for p in to_fetch:
        futures[p] = _validation_pool().submit(
            get_model_catalog, p, p in force_providers)
    catalogs = {}
    for p in to_fetch:
        try:
            catalogs[p] = futures[p].result(
                timeout=max(2.0, float(settings.LLM_CATALOG_TIMEOUT_SECONDS)) + 2.0)
        except Exception:  # noqa: BLE001 — one provider never fails validation
            catalogs[p] = get_model_catalog(p)

    for provider in PROVIDERS:
        inst = _PROVIDER_INSTANCES[provider]
        configured = _configured_models_for(provider)
        entry = {
            "ok": False, "status": "NOT_CONFIGURED", "models_found": 0,
            "configured": configured, "matched": [], "missing": [],
            "missing_models": [], "checked": {}, "selected_fallback": None,
            "catalog_age_s": None, "catalog_cached": False,
            "timed_out": False, "error": None,
        }
        report["providers"][provider] = entry

        if not _provider_enabled(provider):
            entry["error"] = f"disabled via {inst.enabled_attr}=false — catalog not fetched"
            continue
        if not _provider_key(provider):
            entry["error"] = f"{inst.key_attr} not configured — catalog not fetched"
            continue

        catalog = catalogs.get(provider) or get_model_catalog(provider)
        entry["catalog_age_s"] = catalog.get("age_s")
        entry["catalog_cached"] = bool(catalog.get("cached"))
        entry["timed_out"] = bool(catalog.get("timed_out"))
        live_ids = catalog.get("ids")

        if live_ids is None:
            entry["error"] = catalog.get("error") or "catalog unavailable"
            if catalog.get("timed_out"):
                entry["status"] = "DEGRADED"
                logger.warning(
                    f"LLM CATALOG {provider}: {entry['error']} — provider "
                    f"marked DEGRADED, validation continues without it.")
            else:
                entry["status"] = "ERROR"
                logger.warning(f"LLM CATALOG {provider}: {entry['error']}")
            continue

        entry["ok"] = True
        entry["models_found"] = len(live_ids)
        matched = [m for m in configured if m in live_ids]
        missing = [m for m in configured if m not in live_ids]
        entry["matched"] = matched
        entry["missing_models"] = missing

        if provider == "unorouter":
            effective = unorouter_effective_chain(live_ids)
            entry["selected_fallback"] = effective["selected_fallback"]
            entry["effective_chain"] = effective["chain"]
            entry["catalog_available"] = True

        if configured and not matched:
            # Endpoint works but NOTHING configured is usable — never READY.
            entry["status"] = "NO_USABLE_MODEL"
        else:
            entry["status"] = "READY"

    # --- per-task route checks (back-compat label mapping) ------------------
    def _mark(provider, model, label):
        entry = report["providers"][provider]
        if not entry["ok"]:
            return
        if label in entry["checked"]:
            return  # dedup: 6 tasks sharing one label -> ONE row, ONE warning
        present = _normalize_model_id(model) in (live_catalog_ids.get(provider) or set())
        entry["checked"][label] = present
        if not present:
            entry["missing"].append(f"{label}: {_normalize_model_id(model)}")

    live_catalog_ids = {p: (report["providers"][p].get("ok") and
                            get_model_catalog(p).get("ids")) or set()
                        for p in PROVIDERS}

    for agent in sorted(_TASK_PROVIDERS):
        route = route_info(agent)
        if route["provider"] == "unorouter":
            for i, model in enumerate(unorouter_model_chain()):
                _mark("unorouter", model, "primary" if i == 0 else f"fallback[{i - 1}]")
        elif route["provider"] in report["providers"]:
            _mark(route["provider"], route["model"], f"{agent}.model")
        if route["fallback_provider"] in report["providers"] and route["fallback_model"]:
            _mark(route["fallback_provider"], route["fallback_model"], f"{agent}.fallback")
        report["routes"][agent] = dict(route)

    # --- ONE warning per missing model per period + safe diagnostics --------
    for provider in PROVIDERS:
        entry = report["providers"][provider]
        configured = entry.get("configured") or []
        # warnings: the provider's OWN missing configured models — exactly
        # ONE warning per model per catalog period (never 6 identical lines)
        if entry["ok"] and entry.get("missing_models"):
            for model in entry["missing_models"]:
                _warn_model_unavailable(
                    provider, model,
                    "is configured but not in the live /v1/models catalog "
                    f"({entry['models_found']} live models); it is skipped "
                    f"and the next valid configured model is used.")
        # safe diagnostics line (NEVER any key/header/secret)
        if entry["status"] != "NOT_CONFIGURED":
            def shown(ids):
                ids = list(ids or [])
                return (",".join(ids[:8]) + ("…" if len(ids) > 8 else "")) or "none"
            logger.info(
                "LLM CATALOG %s: status=%s live_models=%s configured=%s matched=%s missing=%s%s",
                provider, entry["status"], entry["models_found"],
                shown(configured), shown(entry.get("matched") or []),
                shown(entry.get("missing_models") or []),
                (f" selected_fallback={entry['selected_fallback']}"
                 if provider == "unorouter" and entry.get("selected_fallback") else ""),
            )

    _last_validation = report
    return report


def last_validation() -> dict:
    return _last_validation
