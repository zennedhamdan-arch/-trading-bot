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
import json
import logging
import re
import sqlite3
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeoutError
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
    # Trading Partner / System Operator: a READ-ONLY observer task. Routed
    # like any other task (OPERATOR_LLM_PROVIDER/OPERATOR_LLM_MODEL) but
    # deliberately EXCLUDED from active_chain_providers() — the trading
    # system's health must never depend on the operator's provider.
    "operator": "OPERATOR_LLM_PROVIDER",
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
    # Operator: one optional model override honored across providers.
    ("groq", "operator"): "OPERATOR_LLM_MODEL",
    ("gemini", "operator"): "OPERATOR_LLM_MODEL",
    ("nvidia", "operator"): "OPERATOR_LLM_MODEL",
    ("openrouter", "operator"): "OPERATOR_LLM_MODEL",
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
    model = str(getattr(settings, attr, "") or "") if attr else ""
    if not model and task == "operator":
        # OPERATOR_LLM_MODEL is an optional override; without it the
        # operator rides the provider's own configured default (never an
        # invented id — these are all configured settings attributes).
        model = str({
            "groq": getattr(settings, "GROQ_FALLBACK_MODEL", ""),
            "gemini": getattr(settings, "GEMINI_MODEL", ""),
            "nvidia": getattr(settings, "NVIDIA_MODEL", ""),
            "openrouter": getattr(settings, "OPENROUTER_MODEL", ""),
        }.get(provider, "") or "")
    return model


# Alternates tried (in order) when a routed provider is VERIFIED to have
# zero usable configured models. Groq is the chain-final fallback by design;
# NVIDIA is the secondary verified alternative. An alternate is only used
# when it is enabled, keyed, and its task model is verified in its LIVE
# catalog — ids are never invented.
_ROUTE_ALTERNATES = ("groq", "nvidia")


def _provider_has_usable_models(provider: str, task: str) -> "bool | None":
    """True/False when the live catalog VERIFIES the provider's configured
    task models; None when the catalog is not verified right now (a missing
    catalog is 'unverified', never 'empty' — no switch decision on it)."""
    live = _catalog_verified_models(provider)
    if live is None:
        return None
    live = {_normalize_model_id(m) for m in live}
    if provider == "unorouter":
        # the application's explicit model chain (primary + fallbacks)
        usable = [m for m in unorouter_model_chain()
                  if _normalize_model_id(m) in live]
        return bool(usable)
    model = _task_model(provider, task)
    return bool(model) and _normalize_model_id(model) in live


def effective_route(agent: str) -> dict:
    """The route a task ACTUALLY uses, with the switch RECORDED.

    A provider whose live catalog is VERIFIED to contain NONE of its
    configured models (NO_USABLE_MODEL — a permanent configuration problem,
    not a transient failure) is routed around: the first verified alternate
    (groq, then nvidia) serves the task. The switch is never silent — it is
    visible in route_info / health / startup diagnostics with its reason,
    and it self-heals when the catalog verification changes.

    Transient failures (catalog unreachable, open breakers, quota) NEVER
    trigger the switch: the provider stays routed and its failure is
    reported honestly as a degradation.
    """
    provider_attr = _TASK_PROVIDERS[agent]
    configured_provider = str(getattr(settings, provider_attr, "") or "").strip().lower()
    configured_model = (_task_model(configured_provider, agent)
                        if configured_provider in _PROVIDER_INSTANCES else "")
    route = {
        "provider": configured_provider,
        "model": configured_model,
        "configured_provider": configured_provider,
        "configured_model": configured_model,
        "switched": False,
        "switch_reason": None,
        "fallback_provider": settings.LLM_FALLBACK_PROVIDER,
        "fallback_model": settings.LLM_FALLBACK_MODEL,
    }
    if configured_provider not in _PROVIDER_INSTANCES:
        return route
    usable = _provider_has_usable_models(configured_provider, agent)
    if usable is not False:
        return route
    # Verified zero usable models -> try the alternates, verified only.
    for alt in _ROUTE_ALTERNATES:
        if alt == configured_provider:
            continue
        if not (_provider_enabled(alt) and _provider_key(alt)):
            continue
        alt_live = _catalog_verified_models(alt)
        if alt_live is None:
            continue   # unverified alternate is never selected
        alt_live = {_normalize_model_id(m) for m in alt_live}
        # Groq serves ANY task through its chain-final fallback model when
        # no task-specific Groq model is configured (news/fundamentals have
        # none) — the same resolution _candidate_chain uses.
        alt_model = (_groq_chain_model(agent) if alt == "groq"
                     else _task_model(alt, agent))
        if not alt_model or _normalize_model_id(alt_model) not in alt_live:
            continue
        route.update({
            "provider": alt,
            "model": alt_model,
            "switched": True,
            "switch_reason": (
                f"{configured_provider} has zero usable configured models in "
                f"its live catalog (NO_USABLE_MODEL); tasks served by verified "
                f"{alt} {alt_model}"),
        })
        return route
    # No verified alternate either: keep the configured route; health will
    # report NO_USABLE_MODEL on the active chain (nothing hidden).
    route["switch_reason"] = (
        f"{configured_provider} has zero usable configured models and no "
        f"verified alternate is available")
    return route


def route_info(agent: str) -> dict:
    """Resolves a task's EFFECTIVE (provider, model) plus the configured
    route and the global fallback, from settings + verified catalogs at
    call time (catalog reads are cache peeks — never a network call). The
    provider is returned exactly as configured unless a VERIFIED
    NO_USABLE_MODEL state switched the task to a verified alternate (the
    switch is recorded in `switched`/`switch_reason`). An unknown provider
    name surfaces as a PROVIDER_ERROR in call() rather than being silently
    normalized."""
    return effective_route(agent)


def llm_routes() -> dict:
    """All task routes (for /api/config and startup validation)."""
    return {agent: route_info(agent) for agent in sorted(_TASK_PROVIDERS)}


def provider_enabled(provider: str) -> bool:
    """Public read-only view: is the provider's ENABLE SWITCH on? (Distinct
    from having a key configured — health distinguishes DISABLED from
    NOT_CONFIGURED.)"""
    return bool(_provider_enabled(provider))


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

# Inference evidence (independent of model DISCOVERY): the last successful
# completion per provider, with the exact model that answered. A catalog
# match is not proof of inference; a served completion is.
_last_success_info = {}  # provider -> {"model", "at", "monotonic"}

# Obsolete-model registry: a configured id observed ABSENT from a freshly
# fetched live catalog. Discovery facts OUTLIVE a catalog outage — otherwise
# an obsolete id returns to the routing chain the moment /v1/models times out
# (which is exactly how an obsolete id keeps getting re-sent). An id is
# removed again only when a fresh catalog actually lists it.
_obsolete_models = {}   # provider -> {normalized model id -> {"since", "live_models"}}

# Model-catalog layer: /v1/models is fetched AT MOST once per provider per
# LLM_CATALOG_CACHE_TTL_MINUTES window (5-10 min) — one shared catalog for
# health checks, validations and chain resolution. A hanging provider is
# bounded by LLM_CATALOG_TIMEOUT_SECONDS in a worker thread, so a slow
# Gemini (or any provider) can never block startup or a health-check cycle.
_catalog_cache = {}  # provider -> {"ids": set|None, "fetched_at": float,
#                                "error": str|None, "timed_out": bool}
_catalog_lock = threading.Lock()
# Single-flight catalog fetches: concurrent health checks / validations /
# operator inspections at a cache-expiry moment join ONE in-flight fetch
# instead of stampeding the provider with duplicate /v1/models requests.
_catalog_inflight = {}   # provider -> {"event": Event, "result": dict|None}


def _catalog_failure_ttl_s() -> float:
    """Bounded negative cache: a FAILED catalog fetch (timeout/network) is
    retried after this period — short enough that a transient outage does
    not blind model verification for a full TTL window, long enough that a
    down provider is never hammered."""
    return max(5.0, float(settings.LLM_CATALOG_FAILURE_TTL_SECONDS or 60.0))


def _catalog_entry_ttl_s(entry: dict) -> float:
    """Freshness window for one cached catalog entry: full TTL on success
    and on not-configured/disabled states (a config fact, not a failure —
    no churn), short bounded window on actual fetch failures."""
    if not entry or entry.get("ids") is None:
        if entry and entry.get("no_fetch"):
            return _catalog_ttl_s()
        return _catalog_failure_ttl_s()
    return _catalog_ttl_s()


def _catalog_fresh(entry: dict) -> bool:
    return bool(entry and entry.get("fetched_at")
                and (time.monotonic() - entry["fetched_at"]) < _catalog_entry_ttl_s(entry))


# --- catalog vs inference status separation -----------------------------------
#
# A model-catalog (/v1/models) failure is a DISCOVERY problem, not proof
# that inference is unavailable. The two concepts are reported separately:
#
#   catalog_status:   VERIFIED            live catalog fetched
#                     UNAVAILABLE         fetch failed temporarily (timeout /
#                                         network) — retry after the bounded
#                                         failure TTL; inference NOT tested
#                     UNAVAILABLE_AUTH    fetch failed with a credentials
#                                         error — a real provider failure
#                     NOT_CONFIGURED      no key / disabled (no fetch)
#
#   inference_status: the LIVE circuit state for actual completions:
#                     READY / QUOTA_EXHAUSTED / AUTH_ERROR / MODEL_UNAVAILABLE /
#                     NETWORK_ERROR / DEGRADED / NOT_CONFIGURED
#                     UNKNOWN             no recent inference attempt — the
#                                         catalog says nothing about inference
#
# Policy (unchanged): a temporarily unavailable catalog does NOT disable
# configured models — the chain still uses them and the runtime 404/429
# circuits protect every request. Legitimate validation (a fetched catalog
# with zero matching models -> NO_USABLE_MODEL) is never bypassed.

_CATALOG_AUTH_RE = re.compile(
    r"\b40[13]\b|unauthorized|forbidden|invalid[ _-]?(api[ _-]?key|key|token|credential)"
    r"|authentication|permission denied", re.IGNORECASE)
_CATALOG_TEMPORARY_RE = re.compile(
    r"timed?[ _-]?out|timeout|connection|network|unreachable|temporarily"
    r"|read operation|reset by peer|broken pipe|ssl|eof|rate limit", re.IGNORECASE)


def _classify_catalog_failure(error_text: str) -> str:
    """'AUTH' for credentials-class failures (real provider failure),
    'TEMPORARY' for timeout/network-class failures (discovery degraded)."""
    text = str(error_text or "")
    if _CATALOG_AUTH_RE.search(text):
        return "AUTH"
    return "TEMPORARY"   # unknown failures default to temporary/honest-DEGRADED


def _inference_status(provider: str, states: dict) -> str:
    """Honest inference-endpoint status from the live circuit, independent
    of model discovery. An OPEN circuit (quota/auth/model/network) is itself
    evidence of real inference attempts; READY is only claimed when a
    request has actually been served — otherwise UNKNOWN (the catalog says
    nothing about inference)."""
    state = (states or {}).get(provider) or {}
    circuit = state.get("state")
    if circuit == "NOT_CONFIGURED":
        return "NOT_CONFIGURED"
    if circuit not in ("READY", None):
        return circuit   # open circuits are evidence of attempts
    with _state_lock:
        attempted = bool(_request_times.get(provider)) or \
            _last_failure.get(provider) is not None
    if not attempted:
        return "UNKNOWN"
    return circuit or "UNKNOWN"
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


def peek_model_catalog(provider: str) -> dict:
    """The provider's FRESH cached catalog WITHOUT ever fetching: ids is a
    set when a fresh (TTL-valid) catalog entry exists, else None. Used by
    route-level decisions (effective_route / active_chain_providers) so
    resolving a route never triggers a network call — the switch engages
    only once a validation/health check has actually verified the catalog.
    ids=None here means 'unverified', NEVER 'empty'."""
    with _catalog_lock:
        entry = _catalog_cache.get(provider)
        if _catalog_fresh(entry):
            out = dict(entry)
            out["cached"] = True
            out["age_s"] = round(time.monotonic() - entry["fetched_at"], 1)
            return out
    return {"ids": None, "fetched_at": 0.0, "error": None,
            "timed_out": False, "cached": False, "age_s": None}


def _catalog_verified_models(provider: str) -> "set | None":
    """Live-catalog ids for a provider when a FRESH VERIFIED catalog exists
    (ids is a set). None = catalog not verified right now (never fetches)."""
    if not (_provider_enabled(provider) and _provider_key(provider)):
        return None
    ids = peek_model_catalog(provider).get("ids")
    return ids if isinstance(ids, (set, list)) else None


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
            if _catalog_fresh(entry):
                out = dict(entry)
                out["cached"] = True
                out["age_s"] = round(time.monotonic() - entry["fetched_at"], 1)
                return out

    # --- single-flight: join any in-flight fetch for this provider -------
    with _catalog_lock:
        slot = _catalog_inflight.get(provider)
        if slot is None:
            slot = {"event": threading.Event(), "result": None}
            _catalog_inflight[provider] = slot
            fetcher = True
        else:
            fetcher = False
    if not fetcher:
        # bounded wait: the fetching thread is itself bounded by the catalog
        # timeout, so this can never hang indefinitely
        if slot["event"].wait(timeout=max(2.0, float(settings.LLM_CATALOG_TIMEOUT_SECONDS)) + 5.0) \
                and slot["result"] is not None:
            out = dict(slot["result"])
            out["cached"] = True
            out["shared"] = True
            out["age_s"] = round(time.monotonic() - out["fetched_at"], 1)
            return out
        # pathological: fetcher vanished — fall through and fetch ourselves

    fetched = {"ids": None, "fetched_at": time.monotonic(),
               "error": None, "timed_out": False}
    try:
        inst = _PROVIDER_INSTANCES[provider]
        if not _provider_enabled(provider) or not _provider_key(provider):
            fetched["no_fetch"] = True   # config state: cache for the full TTL
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
                    _record_catalog_obsolescence(provider, fetched["ids"])
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
    finally:
        with _catalog_lock:
            _catalog_cache[provider] = fetched
            if _catalog_inflight.get(provider) is slot:
                _catalog_inflight.pop(provider, None)
        slot["result"] = fetched
        slot["event"].set()
    out = dict(fetched)
    out["cached"] = False
    out["age_s"] = 0.0
    return out


def _record_catalog_obsolescence(provider: str, live_ids) -> dict:
    """Updates the obsolete-model registry from a FRESHLY fetched catalog.

    A configured id that the live catalog does not list is remembered as
    obsolete for this process; an id that appears again is restored at once.
    This is what keeps an obsolete id out of routing while a later catalog
    fetch is unavailable (ids=None would otherwise mean "unverified", and
    the configured order — including the dead id — would be used again).
    Returns the provider's registry entry (safe, no secrets)."""
    live = {_normalize_model_id(m) for m in (live_ids or set())}
    registry = _obsolete_models.setdefault(provider, {})
    for model in _configured_models_for(provider):
        norm = _normalize_model_id(model)
        if not norm:
            continue
        if norm in live:
            if registry.pop(norm, None):
                logger.info(
                    "LLM_MODEL_RESTORED: %s: '%s' is listed in the live "
                    "catalog again — routing may use it.", provider, norm)
        else:
            entry = registry.get(norm) or {"since": _now_iso()}
            entry["live_models"] = len(live)
            registry[norm] = entry
    return {m: dict(v) for m, v in registry.items()}


def obsolete_models(provider: str = None) -> dict:
    """Configured model ids known to be absent from a live catalog (per
    provider, or all). Safe diagnostics only — ids and timestamps."""
    if provider is not None:
        return {m: dict(v) for m, v in (_obsolete_models.get(provider) or {}).items()}
    return {p: {m: dict(v) for m, v in models.items()}
            for p, models in _obsolete_models.items() if models}


def _is_known_obsolete(provider: str, model: str) -> bool:
    return _normalize_model_id(model) in (_obsolete_models.get(provider) or {})


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

# Quota management:
_model_last_send = {}     # (provider, model) -> monotonic ts of the last send
_state_lock = threading.Lock()
_cycle_send_count = 0     # NETWORK SENDS this trading cycle (budget counter;
#                          reset by reset_cycle_usage, enforced in call())
_request_log_seq = 0      # occasional prune trigger for the persisted log

# Persisted per-request usage log (the SAME SQLite file as memory_service —
# no second storage engine). One row per network send / cache hit / replay:
# powers /api/llm/usage (requests_last_hour, requests_today, 429_count,
# cache_hits/misses, calls_by_agent, calls_by_model). Metadata only —
# NEVER prompts, responses, API keys or headers.
_REQUEST_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_request_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_epoch REAL NOT NULL,
    ts_iso TEXT NOT NULL,
    date TEXT NOT NULL,
    kind TEXT NOT NULL,          -- send | cache_hit | cache_replay | cache_miss
    provider TEXT,
    model TEXT,
    agent TEXT,
    symbol TEXT,
    reason TEXT,
    ok INTEGER NOT NULL DEFAULT 0,
    status TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_req_log_ts ON llm_request_log(ts_epoch);
CREATE INDEX IF NOT EXISTS idx_llm_req_log_date ON llm_request_log(date);
"""


_request_log_ready = False


def _request_log_conn():
    db_path = Path(str(settings.MEMORY_DB_PATH))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    return conn


def _ensure_request_log(conn) -> None:
    """Creates the llm_request_log table when first needed (idempotent)."""
    global _request_log_ready
    if _request_log_ready:
        return
    conn.executescript(_REQUEST_LOG_SCHEMA)
    _request_log_ready = True


def _log_request(kind: str, provider=None, model=None, agent=None,
                 symbol=None, reason=None, ok=False, status=None,
                 latency_ms=None) -> None:
    """Counts + logs + persists ONE LLM request event.

    Structured log line: provider, model, caller/agent, symbol, reason —
    never any API key, header, prompt or response content. Persistence is
    best-effort: accounting must never break a request."""
    try:
        logger.info(
            "LLM_REQUEST kind=%s provider=%s model=%s agent=%s symbol=%s "
            "reason=%s ok=%s status=%s%s",
            kind, provider or "-", model or "-", agent or "-", symbol or "-",
            reason or "-", ok, status or "-",
            f" latency_ms={latency_ms:.0f}" if latency_ms is not None else "",
        )
    except Exception:  # noqa: BLE001
        pass
    global _request_log_seq
    _request_log_seq += 1
    try:
        now = datetime.now(timezone.utc)
        with _request_log_conn() as conn:
            _ensure_request_log(conn)
            if _request_log_seq % 64 == 1:
                keep_days = max(1, int(settings.LLM_REQUEST_LOG_KEEP_DAYS or 2))
                cutoff = (now - timedelta(days=keep_days)).strftime("%Y-%m-%d")
                conn.execute("DELETE FROM llm_request_log WHERE date < ?", (cutoff,))
            conn.execute(
                "INSERT INTO llm_request_log (ts_epoch, ts_iso, date, kind, "
                "provider, model, agent, symbol, reason, ok, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now.timestamp(), now.isoformat(), now.strftime("%Y-%m-%d"),
                 kind, provider, model, agent, symbol, reason,
                 1 if ok else 0, status),
            )
    except Exception:  # noqa: BLE001 — persistence is best-effort only
        pass


def _model_min_interval_s(provider: str) -> float:
    """Conservative minimum interval between sends to the SAME model.
    Free UnoRouter models default to 60s; other providers 0 (their own
    rate limits and circuits apply)."""
    if provider == "unorouter":
        return max(0.0, float(settings.LLM_MODEL_MIN_INTERVAL_SECONDS_UNOROUTER or 0))
    return max(0.0, float(settings.LLM_MODEL_MIN_INTERVAL_SECONDS or 0))


def _model_rate_limited(provider: str, model: str) -> float:
    """Seconds the caller must wait before this model may be used again
    (0.0 = allowed now). Enforced per (provider, model)."""
    interval = _model_min_interval_s(provider)
    if interval <= 0:
        return 0.0
    with _state_lock:
        last = _model_last_send.get((provider, model))
    if last is None:
        return 0.0
    remaining = interval - (time.monotonic() - last)
    return max(0.0, remaining)


def _mark_model_sent(provider: str, model: str) -> None:
    with _state_lock:
        _model_last_send[(provider, model)] = time.monotonic()


def _cycle_send_budget() -> int:
    return max(1, int(settings.LLM_CYCLE_MAX_REQUESTS or 12))


def _cycle_sends_remaining() -> int:
    return max(0, _cycle_send_budget() - _cycle_send_count)

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
    """Zeroes the per-cycle usage counters AND the per-cycle network-send
    budget. main.run_trading_cycle calls this at the start of every cycle
    (the news worker runs OUTSIDE cycles and never calls this — its volume
    is bounded by design: at most one batched request per symbol per
    refresh, only for NEW important articles)."""
    global _cycle_send_count
    _cycle_send_count = 0
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
    _inference_probes.clear()
    with _catalog_lock:
        _catalog_cache.clear()
        for slot in _catalog_inflight.values():
            slot["event"].set()
        _catalog_inflight.clear()
    _model_warned.clear()
    _last_success_info.clear()
    _obsolete_models.clear()
    global _last_validation, _cycle_send_count
    _last_validation = None
    with _state_lock:
        _model_last_send.clear()
    _cycle_send_count = 0
    global _request_log_seq, _request_log_ready
    _request_log_seq = 0
    _request_log_ready = False
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


def usage_status() -> dict:
    """The LLM usage/quota status payload (/api/llm/usage):

    requests_this_cycle, requests_last_hour, requests_today, 429_count,
    cache_hits, cache_misses, current_circuit_state (per provider) +
    cooldown_remaining, calls_by_agent, calls_by_model. Served from the
    in-process cycle counters and the persisted llm_request_log (same
    SQLite file as memory_service). No secrets, no prompts, no responses."""
    now = datetime.now(timezone.utc)
    hour_ago = now.timestamp() - 3600.0
    today = now.strftime("%Y-%m-%d")
    stats = {
        "requests_this_cycle": 0, "network_sends_this_cycle": _cycle_send_count,
        "cycle_send_budget": _cycle_send_budget(),
        "requests_last_hour": 0, "requests_today": 0, "429_count": 0,
        "cache_hits": 0, "cache_misses": 0,
        "calls_by_agent": {}, "calls_by_model": {},
    }
    # this-cycle accounting (in-process; includes cache hits and skips)
    for usage in _cycle_usage.values():
        stats["requests_this_cycle"] += usage.get("requests", 0)
    try:
        with _request_log_conn() as conn:
            conn.row_factory = sqlite3.Row
            _ensure_request_log(conn)
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM llm_request_log "
                "WHERE kind='send' AND ts_epoch >= ?", (hour_ago,)).fetchone()
            stats["requests_last_hour"] = row["n"]
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM llm_request_log "
                "WHERE kind='send' AND date = ?", (today,)).fetchone()
            stats["requests_today"] = row["n"]
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM llm_request_log "
                "WHERE kind='send' AND status='PROVIDER_QUOTA_EXCEEDED' "
                "AND date = ?", (today,)).fetchone()
            stats["429_count"] = row["n"]
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM llm_request_log "
                "WHERE kind='cache_hit' AND date = ?", (today,)).fetchone()
            stats["cache_hits"] = row["n"]
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM llm_request_log "
                "WHERE kind='cache_miss' AND date = ?", (today,)).fetchone()
            stats["cache_misses"] = row["n"]
            for row in conn.execute(
                    "SELECT agent, COUNT(*) AS n FROM llm_request_log "
                    "WHERE kind='send' AND date = ? GROUP BY agent", (today,)):
                stats["calls_by_agent"][row["agent"] or "unknown"] = row["n"]
            for row in conn.execute(
                    "SELECT provider, model, COUNT(*) AS n FROM llm_request_log "
                    "WHERE kind='send' AND date = ? GROUP BY provider, model",
                    (today,)):
                key = f"{row['provider']}/{row['model']}"
                stats["calls_by_model"][key] = row["n"]
    except Exception as exc:  # noqa: BLE001 — stats must never fail the API
        stats["log_error"] = str(exc)
    hits = stats["cache_hits"]
    lookups = hits + stats["cache_misses"]
    stats["cache_hit_rate"] = round(hits / lookups, 3) if lookups else None

    circuits = {}
    cooldown_remaining = 0.0
    for provider, state in provider_states().items():
        provider_cooldown = max(
            state.get("quota_cooldown_remaining_s") or 0.0,
            state.get("auth_cooldown_remaining_s") or 0.0,
        )
        cooldown_remaining = max(cooldown_remaining, provider_cooldown)
        circuits[provider] = {
            "state": state.get("state"),
            "circuit": state.get("circuit"),
            "cooldown_remaining_s": provider_cooldown,
            "model_cooldowns_s": {
                model.split("/", 1)[1]: round(max(0.0, until - time.monotonic()), 0)
                for (p, model), until in _model_unavailable_until.items()
                if p == provider and until > time.monotonic()
            },
        }
    return {
        "checked_at": now.isoformat(),
        **stats,
        "current_circuit_state": circuits,
        "cooldown_remaining": round(cooldown_remaining, 1),
        "active_chain_providers": active_chain_providers(),
    }


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

# --- what a failure is allowed to open ---------------------------------------
# MODEL-scoped: the model id is wrong/obsolete for this account. Such a
# failure must NEVER open (or re-open, e.g. from a HALF_OPEN probe) the
# PROVIDER circuit — other models on the same provider are still healthy, and
# an obsolete candidate must not be able to poison them.
_MODEL_SCOPED_FAILURES = frozenset({"MODEL_NOT_FOUND"})
# LOCAL policy stops: not the provider's fault at all.
_NON_PROVIDER_FAILURES = frozenset({"CYCLE_BUDGET_EXCEEDED", "RATE_LIMITED_LOCAL"})


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


def _breaker_model_failure(provider: str, status: str) -> None:
    """Records a MODEL-scoped failure (dead/obsolete id) without touching the
    provider's consecutive-failure count, so it can neither open nor re-open
    the provider circuit. The per-model 404 circuit already owns the window."""
    _breaker[provider]["last_failure"] = _now_iso()
    logger.debug(
        "%s: model-scoped failure (%s) — provider circuit unchanged; the "
        "model circuit owns the cooldown.", provider, status)


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
            # Safe last-failure diagnostics: status + category (TIMEOUT /
            # CONNECTION / TLS / HTTP_5XX / ...) + original exception type.
            # Never a message, header, URL or key.
            "last_error": dict(_last_failure[provider])
            if _last_failure[provider] else None,
            "last_success_model": (_last_success_info.get(provider) or {}).get("model"),
            "obsolete_models": sorted(_obsolete_models.get(provider) or {}),
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


def _http_status_code(exc: Exception):
    """The HTTP status carried by an SDK exception, when one exists."""
    code = getattr(exc, "status_code", None)
    if not isinstance(code, int):
        code = getattr(exc, "code", None)
    if not isinstance(code, int):
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def _exception_type_names(exc: Exception) -> list:
    """ORIGINAL exception class names, root cause first.

    An SDK wrapper (e.g. openai.APIConnectionError) hides the real cause
    (httpx.ConnectError, ssl.SSLCertVerificationError, ...); walking
    __cause__/__context__ preserves that category. Type NAMES only — no
    message, header, URL or key is ever extracted here."""
    names, seen, node = [], set(), exc
    while node is not None and id(node) not in seen and len(names) < 8:
        seen.add(id(node))
        name = node.__class__.__name__
        if name not in names:
            names.append(name)
        node = node.__cause__ or node.__context__
    return list(reversed(names))


# Transport category hints (lower-case class-name fragments). TLS and timeout
# are checked before the generic connection family so a ConnectTimeout stays a
# TIMEOUT and a handshake failure stays a TLS failure.
_TLS_NAME_HINTS = ("ssl", "tls", "certificate")
_TIMEOUT_NAME_HINTS = ("timeout",)
_CONNECTION_NAME_HINTS = ("connection", "connecterror", "protocolerror",
                          "networkerror", "serverdisconnected", "remotedisconnected")


def _exception_details(exc: Exception) -> dict:
    """SAFE, type-based classification of a provider exception.

    Distinguishes connection failures, timeouts, TLS failures, HTTP 4xx/5xx,
    rate limits (429 transient) from quota exhaustion (429 hard), missing
    models (404) and auth failures (401/403). Preserves the ORIGINAL
    exception class name(s) so NVIDIA-style NETWORK_ERROR failures are not
    collapsed into one opaque category. Contains no message, URL, header or
    credential."""
    code = _http_status_code(exc)
    names = _exception_type_names(exc)
    message = str(exc)
    retry_after = _parse_retry_delay(message)
    category = "PROVIDER_ERROR"
    if code is not None:
        if code == 404:
            category = "MODEL_NOT_FOUND"
        elif code in (401, 403):
            category = "AUTH"
        elif code == 429:
            category = ("QUOTA_EXHAUSTED"
                        if _looks_like_quota_exhaustion(message, retry_after)
                        else "RATE_LIMIT")
        elif 400 <= code < 500:
            category = "HTTP_4XX"
        elif 500 <= code < 600:
            category = "HTTP_5XX"
    if category == "PROVIDER_ERROR":
        lowered = [n.lower() for n in names]
        if any(h in n for h in _TLS_NAME_HINTS for n in lowered):
            category = "TLS"
        elif any(h in n for h in _TIMEOUT_NAME_HINTS for n in lowered):
            category = "TIMEOUT"
        elif any(h in n for h in _CONNECTION_NAME_HINTS for n in lowered):
            category = "CONNECTION"
        else:
            try:
                import httpx
                if isinstance(exc, httpx.TransportError):
                    category = "CONNECTION"
            except ImportError:
                pass
    return {"category": category, "http_status": code,
            "exception_type": names[0] if names else exc.__class__.__name__,
            "exception_chain": names[:6], "retry_after": retry_after}


def _status_from_details(details: dict):
    """Maps classified details to the router's (status, retry_after)."""
    code = details.get("http_status")
    category = details.get("category")
    retry_after = details.get("retry_after")
    if code == 429:
        return (("PROVIDER_QUOTA_EXCEEDED" if category == "QUOTA_EXHAUSTED"
                 else "RATE_LIMITED"), retry_after)
    if code == 404:
        return "MODEL_NOT_FOUND", None
    if code in (401, 403):
        return "AUTH_ERROR", None
    if category in ("CONNECTION", "TIMEOUT", "TLS"):
        return "NETWORK_ERROR", None
    return "PROVIDER_ERROR", None


def _classify_exception(exc: Exception):
    """Maps a provider exception to (status, retry_after).

    Uses the exception's HTTP status code when the SDK exposes one — never
    message keyword matching on generic exceptions — and the original
    exception type chain for transport failures.
    """
    return _status_from_details(_exception_details(exc))


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
    "RATE_LIMITED_LOCAL": "RATE_LIMITED_LOCAL",
    "CYCLE_BUDGET_EXCEEDED": "CYCLE_BUDGET_EXCEEDED",
}


class LLMResult:
    """Outcome of one task request (standardized via standard())."""

    def __init__(self, agent, provider, model, status, text="", parsed=None,
                 latency_ms=None, error=None, error_type=None, attempts=0,
                 fallback_used=False, cached=False, cache_age_s=None,
                 cache_replay=False, error_category=None, http_status=None,
                 exception_type=None):
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
        # Safe failure attribution: what KIND of failure it was and which
        # HTTP status / original exception type carried it (never a message
        # with credentials, headers or a signed URL).
        self.error_category = error_category
        self.http_status = http_status
        self.exception_type = exception_type
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
            "error_category": self.error_category,
            "http_status": self.http_status,
            "exception_type": self.exception_type,
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
             temperature: float, max_tokens: int, details: dict = None,
             count_cycle: bool = False):
    """One logical attempt (up to LLM_RATE_LIMIT_MAX_RETRIES + 1 sends).

    `details` (optional dict) is filled with SAFE outcome diagnostics: the
    category (CONNECTION / TIMEOUT / TLS / HTTP_4XX / HTTP_5XX / RATE_LIMIT /
    QUOTA_EXHAUSTED / MODEL_NOT_FOUND / AUTH / EMPTY_RESPONSE /
    MALFORMED_RESPONSE / PROVIDER_ERROR / CYCLE_BUDGET_EXCEEDED), the HTTP
    status when one exists, and the ORIGINAL exception class name.

    `count_cycle=True` charges EVERY network send (including retries) to the
    per-cycle budget, so bounded retries can never exceed
    LLM_CYCLE_MAX_REQUESTS. `count_cycle=False` (probes) spends no cycle
    budget.
    """
    global _cycle_send_count
    inst = _PROVIDER_INSTANCES[provider]
    last = ("PROVIDER_ERROR", "request failed", None, None)
    for attempt in range(_max_retries() + 1):
        if count_cycle and _cycle_sends_remaining() <= 0:
            if details is not None:
                details.update({"category": "CYCLE_BUDGET_EXCEEDED",
                                "http_status": None, "exception_type": None})
            return ("CYCLE_BUDGET_EXCEEDED",
                    (f"CYCLE_BUDGET_EXCEEDED: this cycle already sent "
                     f"{_cycle_send_count} LLM requests (cap "
                     f"{_cycle_send_budget()}); failing fast to protect the "
                     f"quota."), None, None)
        if count_cycle:
            _cycle_send_count += 1
        _count_request(provider)
        try:
            text = inst.send(model, system, user, temperature, max_tokens)
            if not str(text).strip():
                _last_failure[provider] = {
                    "status": "INVALID_RESPONSE", "at": time.monotonic(),
                    "model": model, "category": "EMPTY_RESPONSE"}
                if details is not None:
                    details.update({"category": "EMPTY_RESPONSE",
                                    "http_status": None,
                                    "exception_type": None})
                return "INVALID_RESPONSE", "model returned an empty response", text, None
            _last_failure[provider] = None
            _last_success_info[provider] = {
                "model": model, "at": _now_iso(), "monotonic": time.monotonic()}
            if details is not None:
                details.update({"category": "OK", "http_status": None,
                                "exception_type": None})
            return "OK", None, str(text), None
        except Exception as exc:  # noqa: BLE001 — classified below
            classified = _exception_details(exc)
            status, retry_after = _status_from_details(classified)
            last = (status, str(exc) or exc.__class__.__name__, None, retry_after)
            _last_failure[provider] = {
                "status": status, "at": time.monotonic(), "model": model,
                "category": classified["category"],
                "http_status": classified["http_status"],
                "exception_type": classified["exception_type"],
            }
            if details is not None:
                details.update(classified)
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
                and (not count_cycle or _cycle_sends_remaining() > 0)
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
    # Chain-final VERIFIED alternate: ONE provider the live catalog confirms
    # can serve this task (Groq). This is what lets a routed provider whose
    # inference endpoint is unreachable (e.g. NVIDIA -> NETWORK_ERROR) still
    # complete the task. At most one extra provider — never unlimited
    # provider hopping, and never an unverified model id.
    if (chain and provider not in ("unorouter", "groq")
            and all(p != "groq" for p, _m in chain)):
        groq_model = _groq_chain_model(agent)
        groq_live = _catalog_verified_models("groq")
        if (_provider_enabled("groq") and _provider_key("groq") and groq_model
                and groq_live is not None
                and _normalize_model_id(groq_model) in
                {_normalize_model_id(m) for m in groq_live}):
            chain.append(("groq", groq_model))
    return chain[:_model_attempt_cap()], route


def _candidate_precheck(provider: str, model: str):
    """(status, error) when a candidate must be skipped WITHOUT a network
    call (circuits), else (None, None)."""
    now = time.monotonic()
    if not _provider_enabled(provider):
        return "NOT_CONFIGURED", f"{_PROVIDER_INSTANCES[provider].enabled_attr}=false"
    if not _provider_key(provider):
        key_name = _PROVIDER_INSTANCES[provider].key_attr
        return "NOT_CONFIGURED", f"{key_name} not configured."
    # Recorded-obsolete id: excluded from routing even while the live catalog
    # is unavailable. No request is sent, and the provider breaker is not
    # touched (the model is the problem, not the provider).
    if _is_known_obsolete(provider, model):
        _warn_model_unavailable(
            provider, model,
            "was absent from a fetched live catalog and is recorded as "
            "obsolete for this process; routing excludes it even while the "
            "catalog is unavailable (no request sent).")
        return "MODEL_NOT_FOUND", (
            f"LLM_MODEL_UNAVAILABLE: model '{model}' is recorded as obsolete "
            f"on {provider} (absent from a fetched live catalog); skipped "
            f"without a request.")
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
         expect_json: bool = False, reason: str = None) -> LLMResult:
    """Runs one task request through the unified chain:

    budget -> cache -> provider chain (primary model -> fallback models ->
    Groq) -> cached replay on total failure. Never raises for provider-side
    problems; never blocks on long retries (10s timeout, 0 SDK retries).

    Quota management (why this cycle will never hammer an endpoint):
      * a per-model minimum interval (60s for free UnoRouter models) — a
        request that would arrive sooner is NOT sent; the chain instantly
        fails over to the next model (never waits),
      * a global per-cycle network-send budget (LLM_CYCLE_MAX_REQUESTS),
      * open circuits (quota/auth/model/breaker) are skipped with no call,
      * every event (send / cache_hit / cache_replay / cache_miss) is
        logged with provider, model, agent, symbol and reason, and
        persisted for /api/llm/usage.
    """
    global _cycle_send_count
    t0 = time.monotonic()
    key = _cache_key(agent, system, user)

    # 1. CACHE HIT (fresh) -> return without any network call.
    entry = _cache_get(key, fresh_only=True)
    if entry is not None:
        _log_request("cache_hit", provider=entry["provider"],
                     model=entry["model"], agent=agent, symbol=symbol,
                     reason=reason, ok=True, status="OK",
                     latency_ms=0.0)
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

    # Fresh cache had nothing: this invocation is a cache miss.
    _log_request("cache_miss", provider=route["provider"],
                 model=route["model"], agent=agent, symbol=symbol,
                 reason=reason)

    # 2. Walk the chain: each failure records, updates circuits, and
    #    IMMEDIATELY tries the next candidate. No sleeping between models.
    #    Checks are cheapest-first: cycle budget -> circuits -> per-model
    #    rate limit -> send.
    last_result = None
    for idx, (cand_provider, cand_model) in enumerate(chain):
        if not cand_model:
            continue
        attempt_details = {}
        # 2a. Global per-cycle network-send budget: beyond the cap the call
        #     fails fast (never a send); cached replay / deterministic
        #     fallback take over. Protects the quota by construction.
        #     The operator task is EXEMPT: it is an observer running
        #     OUTSIDE trading cycles (operator_service enforces its own
        #     per-request send cap) and must never starve a trading cycle.
        if agent != "operator" and _cycle_sends_remaining() <= 0:
            last_result = LLMResult(
                agent, cand_provider, cand_model, "CYCLE_BUDGET_EXCEEDED",
                latency_ms=round((time.monotonic() - t0) * 1000, 1),
                error=(f"CYCLE_BUDGET_EXCEEDED: this cycle already sent "
                       f"{_cycle_send_count} LLM requests (cap "
                       f"{_cycle_send_budget()}); failing fast to protect "
                       f"the quota."),
                attempts=idx + 1, fallback_used=idx > 0,
            )
            break
        pre_status, pre_error = _candidate_precheck(cand_provider, cand_model)
        # 2b. Per-model minimum interval (quota management): a request to a
        #     model used too recently is NOT sent — the chain instantly
        #     moves on (trading never waits). The "operator" task is
        #     EXEMPT from this check: it is an interactive, user-initiated
        #     chat bounded by OPERATOR_MAX_LLM_SENDS (never a scheduled
        #     cycle), and a 60s spacing between its own tool-loop rounds
        #     would break every multi-round conversation. Its sends are
        #     still MARKED below (shared quota accounting) and the 429
        #     circuits + daily limits still protect the endpoint.
        if pre_status is None and agent != "operator":
            wait_s = _model_rate_limited(cand_provider, cand_model)
            if wait_s > 0:
                pre_status, pre_error = "RATE_LIMITED_LOCAL", (
                    f"RATE_LIMITED_LOCAL: model '{cand_model}' was used "
                    f"{_model_min_interval_s(cand_provider) - wait_s:.0f}s ago; "
                    f"minimum interval is {wait_s:.0f}s more — trying the next "
                    f"model instead of sending (quota protection).")
        if pre_status is None:
            # One NETWORK SEND: count it against the per-model interval
            # (always — the quota is shared) and the trading-cycle budget
            # (trading tasks only) BEFORE the request leaves the process.
            _mark_model_sent(cand_provider, cand_model)
            attempt_t0 = time.monotonic()
            # count_cycle: every send — including a bounded retry — is charged
            # to the per-cycle budget inside _attempt (retries can never
            # exceed LLM_CYCLE_MAX_REQUESTS).
            status, attempt_error, text, _ = _attempt(
                cand_provider, cand_model, system, user, temperature, max_tokens,
                details=attempt_details, count_cycle=(agent != "operator")
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
                    attempt_details = {"category": "MALFORMED_RESPONSE",
                                       "http_status": None,
                                       "exception_type": exc.__class__.__name__}
            _log_request(
                "send", provider=cand_provider, model=cand_model, agent=agent,
                symbol=symbol, reason=reason, ok=(status == "OK"),
                status=status, latency_ms=(time.monotonic() - attempt_t0) * 1000,
            )
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

        # Only an ACTUAL failed attempt updates the failure breaker — and a
        # MODEL-scoped failure (dead/obsolete id) never opens or re-opens the
        # PROVIDER circuit, so it cannot poison the models that still work.
        if pre_status is None and status not in _NON_PROVIDER_FAILURES:
            if status in _MODEL_SCOPED_FAILURES:
                _breaker_model_failure(cand_provider, status)
            else:
                _breaker_failure(cand_provider, status)
        last_result = LLMResult(
            agent, cand_provider, cand_model, status,
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
            error=error, attempts=idx + 1, fallback_used=idx > 0,
            error_category=(attempt_details.get("category")
                            if pre_status is None else status),
            http_status=attempt_details.get("http_status"),
            exception_type=attempt_details.get("exception_type"),
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
        _log_request("cache_replay", provider=replay["provider"],
                     model=replay["model"], agent=agent, symbol=symbol,
                     reason=reason, ok=True, status="OK",
                     latency_ms=(time.monotonic() - t0) * 1000)
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
              max_tokens: int = 500, symbol: str = None,
              reason: str = None) -> LLMResult:
    """call() + tolerant JSON parsing. A malformed response is retried down
    the model chain inside call(); if every model returns unparseable output
    the result comes back with status INVALID_RESPONSE (never fabricated
    content)."""
    result = call(agent, system, user, temperature=temperature,
                  max_tokens=max_tokens, symbol=symbol, expect_json=True,
                  reason=reason)
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
    failed/timed out) the configured order is kept MINUS ids a previous
    catalog proved obsolete — verifying the rest is impossible, and the
    runtime 404 circuits still protect every request — with
    catalog_available=False reporting that honestly.

    A candidate recorded as OBSOLETE (absent from a catalog that WAS fetched)
    stays out of the chain even when the current catalog is unavailable — an
    obsolete id must not re-enter routing just because /v1/models timed out.

    Returns {"chain": [...], "configured": [...], "missing": [...],
             "excluded_obsolete": [...], "selected_fallback": str|None,
             "catalog_available": bool, "routing_basis": str}."""
    configured = unorouter_model_chain()
    obsolete = [m for m in configured if _is_known_obsolete("unorouter", m)]
    routable = [m for m in configured if m not in obsolete]
    if catalog_ids is None:
        catalog_ids = get_model_catalog("unorouter").get("ids")
    if catalog_ids is None:
        # Verification impossible: keep the configured ORDER, minus ids that
        # a previous fetch already proved absent. Runtime 404 circuits still
        # protect every remaining request.
        return {"chain": routable[:_model_attempt_cap()],
                "configured": configured, "missing": [],
                "excluded_obsolete": obsolete,
                "selected_fallback": (
                    routable[1] if len(routable) > 1 else None),
                "catalog_available": False,
                "routing_basis": "configured-minus-known-obsolete"}
    live = {_normalize_model_id(m) for m in catalog_ids}
    # A fetched catalog is the source of truth: remember configured ids it
    # does not list, and restore any id it lists again.
    _record_catalog_obsolescence("unorouter", live)
    obsolete = [m for m in configured if _is_known_obsolete("unorouter", m)]
    chain, missing = [], []
    for model in configured:
        (chain if model in live else missing).append(model)
        if model not in live and model not in obsolete:
            obsolete.append(model)
    # The attempt cap applies to LIVE models only (never unlimited guessing;
    # a dead candidate never consumes an attempt slot).
    chain = chain[:_model_attempt_cap()]
    return {"chain": chain, "configured": configured, "missing": missing,
            "excluded_obsolete": obsolete,
            "selected_fallback": (chain[1] if len(chain) > 1 else None),
            "catalog_available": True,
            "routing_basis": "catalog-verified"}


def _model_env_source(provider: str, model: str) -> str:
    """'env:VAR_NAME' when the configured model id came from an environment
    override, 'default' when it is the repository default. Never returns a
    value — only the variable NAME (no secrets)."""
    import os as _os
    model = _normalize_model_id(model)
    if provider == "unorouter":
        chain = unorouter_model_chain()
        if model in chain:
            idx = chain.index(model)
            var = ("UNOROUTER_PRIMARY_MODEL" if idx == 0
                   else "UNOROUTER_FALLBACK_MODELS")
            return f"env:{var}" if _os.getenv(var) is not None else "default"
        return "default"
    for (p, _task), attr in _TASK_MODELS.items():
        if p != provider:
            continue
        if _normalize_model_id(str(getattr(settings, attr, "") or "")) == model \
                or (p == "groq" and model == _normalize_model_id(
                    str(settings.GROQ_FALLBACK_MODEL or ""))):
            return f"env:{attr}" if _os.getenv(attr) is not None else "default"
    return "default"


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
        if str(settings.OPERATOR_LLM_PROVIDER).lower() == "gemini":
            add(_task_model("gemini", "operator"))
    elif provider == "nvidia":
        if "nvidia" in routed:
            add(settings.NVIDIA_MODEL)
        if str(settings.OPERATOR_LLM_PROVIDER).lower() == "nvidia":
            add(_task_model("nvidia", "operator"))
    elif provider == "openrouter":
        if "openrouter" in routed:
            add(settings.OPENROUTER_MODEL)
        if str(settings.OPERATOR_LLM_PROVIDER).lower() == "openrouter":
            add(_task_model("openrouter", "operator"))
    return out


def active_chain_providers() -> list:
    """Providers the ACTIVE request chain depends on, computed from the
    EFFECTIVE routes: every provider a task actually runs on (after a
    verified NO_USABLE_MODEL switch), plus Groq when any task still runs on
    UnoRouter (Groq is the chain-final provider fallback). Used by health
    to decide which LLM providers are REQUIRED.

    A provider that was routed around by a VERIFIED NO_USABLE_MODEL switch
    is NOT required: its failure must not degrade the overall system while
    every active task route has a verified working provider and fallback."""
    routed = set()
    for agent, attr in _TASK_PROVIDERS.items():
        if agent == "operator":
            continue   # observer layer: never a required provider
        try:
            routed.add(str(effective_route(agent)["provider"] or "").lower())
        except Exception:  # noqa: BLE001 — health must never crash
            routed.add(str(getattr(settings, attr, "") or "").lower())
    routed.discard("")
    if "unorouter" in routed:
        # Groq is the chain-final provider fallback for every UnoRouter-
        # routed task — including when UnoRouter is disabled or dead (the
        # request chain serves those tasks on Groq).
        routed.add("groq")
    # A provider that is disabled or has no key cannot serve anything — it
    # is not REQUIRED (the request chain already skips it at runtime; e.g.
    # a disabled UnoRouter leaves every task on Groq, so Groq alone is
    # required and the system is HEALTHY).
    routed = {p for p in routed
              if p in _PROVIDER_INSTANCES
              and _provider_enabled(p) and _provider_key(p)}
    return sorted(routed)


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

    Per-provider status (back-compat label) plus the SEPARATED views:
      status / catalog_status / inference_status
      READY              catalog VERIFIED, all configured models matched
      NO_USABLE_MODEL    catalog VERIFIED but ZERO configured models matched
      DEGRADED           catalog UNAVAILABLE temporarily (timeout/network) —
                         discovery degraded only; inference reported
                         separately and configured models stay usable
      ERROR              catalog UNAVAILABLE_AUTH (credentials-class failure)
      NOT_CONFIGURED     no key / disabled — catalog not fetched
      catalog_status:    VERIFIED | UNAVAILABLE | UNAVAILABLE_AUTH | NOT_CONFIGURED
      inference_status:  live circuit state, or UNKNOWN when never exercised

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
    # Catalog fetch policy: get_model_catalog(force=False) already serves a
    # fresh entry from cache and refetches a STALE one (success TTL, or the
    # short bounded failure TTL for failed fetches). No per-thread force
    # flag is computed: a thread that decided "stale" while the cache was
    # empty must NOT blindly refetch when another thread's fetch has since
    # refreshed the entry (that was a duplicate-fetch race under concurrent
    # health checks). refresh_if_stale therefore changes nothing here — it
    # is kept for API compatibility; single-flight + TTLs do the work.
    futures = {}
    for p in to_fetch:
        futures[p] = _validation_pool().submit(get_model_catalog, p, False)
    catalogs = {}
    for p in to_fetch:
        try:
            catalogs[p] = futures[p].result(
                timeout=max(2.0, float(settings.LLM_CATALOG_TIMEOUT_SECONDS)) + 2.0)
        except Exception:  # noqa: BLE001 — one provider never fails validation
            catalogs[p] = get_model_catalog(p)

    states = provider_states()   # live inference circuits (no network)

    def _inference_view(provider: str) -> str:
        return _probe_view(provider, _inference_status(provider, states))

    for provider in PROVIDERS:
        inst = _PROVIDER_INSTANCES[provider]
        configured = _configured_models_for(provider)
        entry = {
            "ok": False, "status": "NOT_CONFIGURED", "models_found": 0,
            "configured": configured, "matched": [], "missing": [],
            "missing_models": [], "checked": {}, "selected_fallback": None,
            "catalog_matched": [], "selected": [], "routing_excluded": [],
            "routing_basis": None, "inference_tested": None,
            "inference_verified": False,
            "catalog_age_s": None, "catalog_cached": False,
            "timed_out": False, "error": None,
            "catalog_status": "NOT_CONFIGURED", "inference_status": "NOT_CONFIGURED",
        }
        report["providers"][provider] = entry

        if not _provider_enabled(provider):
            # DISABLED is distinct from NOT_CONFIGURED: the deployment turned
            # the provider off on purpose — it is not a missing credential.
            entry["status"] = "DISABLED"
            entry["error"] = f"disabled via {inst.enabled_attr}=false — catalog not fetched"
            entry["inference_status"] = _inference_view(provider)
            continue
        if not _provider_key(provider):
            entry["status"] = "NOT_CONFIGURED"
            entry["error"] = f"{inst.key_attr} not configured — catalog not fetched"
            entry["inference_status"] = _inference_view(provider)
            continue

        catalog = catalogs.get(provider) or get_model_catalog(provider)
        entry["catalog_age_s"] = catalog.get("age_s")
        entry["catalog_cached"] = bool(catalog.get("cached"))
        entry["timed_out"] = bool(catalog.get("timed_out"))
        live_ids = catalog.get("ids")

        if live_ids is None:
            # Catalog discovery failed — that is NOT proof inference is
            # down. Temporary failures (timeout/network) degrade DISCOVERY
            # only; only credentials-class failures are a real provider
            # ERROR. Inference status is reported independently.
            entry["error"] = catalog.get("error") or "catalog unavailable"
            failure_class = ("TEMPORARY" if catalog.get("timed_out")
                             else _classify_catalog_failure(entry["error"]))
            entry["inference_status"] = _inference_view(provider)
            # log the discovery failure ONCE per catalog period (a cached
            # failure was already logged when it was fetched)
            if not catalog.get("cached"):
                if failure_class == "AUTH":
                    logger.warning(
                        f"LLM CATALOG {provider}: {entry['error']} — "
                        f"credentials-class failure; provider marked ERROR.")
                else:
                    logger.warning(
                        f"LLM CATALOG {provider}: {entry['error']} — catalog "
                        f"temporarily unavailable (discovery degraded); "
                        f"inference status: {entry['inference_status']} — "
                        f"configured models remain usable per validation "
                        f"policy.")
            if failure_class == "AUTH":
                entry["catalog_status"] = "UNAVAILABLE_AUTH"
                entry["status"] = "ERROR"
            else:
                # CATALOG_UNAVAILABLE is its own status (discovery degraded
                # ONLY — never proof that inference is down; that is reported
                # separately in inference_status).
                entry["catalog_status"] = "UNAVAILABLE"
                entry["status"] = "CATALOG_UNAVAILABLE"
            # What routing would use RIGHT NOW, without a catalog: the
            # configured chain minus ids a previous catalog proved obsolete.
            entry["routing_excluded"] = sorted(_obsolete_models.get(provider) or {})
            if provider == "unorouter":
                eff = unorouter_effective_chain()
                entry["selected"] = eff["chain"]
                entry["routing_basis"] = eff["routing_basis"]
            continue

        entry["ok"] = True
        entry["models_found"] = len(live_ids)
        entry["catalog_status"] = "VERIFIED"
        entry["inference_status"] = _inference_view(provider)
        matched = [m for m in configured if m in live_ids]
        missing = [m for m in configured if m not in live_ids]
        entry["matched"] = matched
        entry["missing_models"] = missing

        entry["catalog_matched"] = list(matched)
        entry["routing_excluded"] = sorted(_obsolete_models.get(provider) or {})

        if provider == "unorouter":
            effective = unorouter_effective_chain(live_ids)
            entry["selected_fallback"] = effective["selected_fallback"]
            entry["effective_chain"] = effective["chain"]
            entry["catalog_available"] = True
            entry["selected"] = effective["chain"]
            entry["routing_excluded"] = effective["excluded_obsolete"]
            entry["routing_basis"] = effective["routing_basis"]

        if configured and not matched:
            # Endpoint works but NOTHING configured is usable — never READY.
            entry["status"] = "NO_USABLE_MODEL"
        elif missing:
            # Catalog verified, some models usable, but configured ids have
            # DRIFTED (obsolete ids in the deployment configuration). The
            # provider still serves its matched models: DEGRADED, with the
            # obsolete ids named — never silently substituted.
            entry["status"] = "DEGRADED"
        else:
            entry["status"] = "READY"

    # --- inference evidence + selected models (independent of catalogs) ----
    for provider in PROVIDERS:
        entry = report["providers"][provider]
        probe = _inference_probes.get(provider)
        entry["inference_probe"] = dict(probe) if probe else None
        last_ok = _last_success_info.get(provider)
        if probe:
            entry["inference_tested"] = {
                "source": "probe", "model": probe.get("model"),
                "ok": bool(probe.get("ok")), "status": probe.get("status"),
                "error_category": probe.get("error_category"),
                "at": probe.get("at")}
        elif last_ok:
            entry["inference_tested"] = {
                "source": "inference", "model": last_ok.get("model"),
                "ok": True, "status": "OK", "error_category": None,
                "at": last_ok.get("at")}
        entry["inference_verified"] = bool(
            entry["inference_tested"] and entry["inference_tested"]["ok"])
        if not entry.get("selected"):
            # Models this provider actually serves on the EFFECTIVE routes.
            selected = []
            for agent in _TASK_PROVIDERS:
                if agent == "operator":
                    continue
                try:
                    route = effective_route(agent)
                except Exception:  # noqa: BLE001 - reporting must not crash
                    continue
                model = _normalize_model_id(_task_model(provider, agent))
                if route["provider"] == provider and model and model not in selected:
                    selected.append(model)
            entry["selected"] = selected

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
        if route["configured_provider"] == "unorouter":
            for i, model in enumerate(unorouter_model_chain()):
                _mark("unorouter", model, "primary" if i == 0 else f"fallback[{i - 1}]")
        elif route["configured_provider"] in report["providers"]:
            _mark(route["configured_provider"], route["configured_model"],
                  f"{agent}.model")
        if route["provider"] in report["providers"] and route["provider"] != route["configured_provider"]:
            _mark(route["provider"], route["model"], f"{agent}.effective")
        if route["fallback_provider"] in report["providers"] and route["fallback_model"]:
            _mark(route["fallback_provider"], route["fallback_model"], f"{agent}.fallback")
        chain, _r = _candidate_chain(agent)
        route["chain"] = [f"{p}:{m}" for p, m in chain]
        route["operational"] = bool(chain)
        report["routes"][agent] = dict(route)

    # --- ONE warning per missing model per period + safe diagnostics --------
    for provider in PROVIDERS:
        entry = report["providers"][provider]
        configured = entry.get("configured") or []
        # warnings: the provider's OWN missing configured models — exactly
        # ONE warning per model per catalog period (never 6 identical lines)
        if entry["ok"]:
            entry["model_sources"] = {
                m: _model_env_source(provider, m)
                for m in (entry.get("matched") or []) + (entry.get("missing_models") or [])
            }
        if entry["ok"] and entry.get("missing_models"):
            for model in entry["missing_models"]:
                source = _model_env_source(provider, model)
                _warn_model_unavailable(
                    provider, model,
                    "is configured but not in the live /v1/models catalog "
                    f"({entry['models_found']} live models); it is skipped "
                    f"and the next valid configured model is used "
                    f"[source: {source}]")
                if source.startswith("env:"):
                    # Configuration DRIFT: the deployment environment
                    # overrides the repository default with an id the
                    # provider no longer serves. Never silently replaced —
                    # the operator must fix the environment variable.
                    logger.warning(
                        "LLM_CONFIG_DRIFT: %s (from %s) is not served by %s "
                        "— update the deployment environment variable to a "
                        "catalog-verified model id (never auto-substituted).",
                        model, source, provider)
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


# --- bounded, non-trading inference health probe ---------------------------
# One tiny request per provider, explicitly labeled (reason="inference_probe")
# and recorded separately. NEVER run automatically unless
# LLM_INFERENCE_PROBE_ENABLED=true — the default keeps zero extra provider
# requests. A probe result NEVER changes routing by itself; it only makes
# the inference dimension VERIFIED / INFERENCE_FAILED instead of UNKNOWN.
_inference_probes: dict = {}
_PROBE_FRESH_S = 600.0


def inference_probe(provider: str, task: str = "cio") -> dict:
    """Sends ONE bounded, non-trading inference request to the provider's
    task model: verifies that a catalog-listed model can actually serve a
    completion (a catalog match is not proof of inference). Respects the
    per-request timeout and the no-retry policy; records the outcome for
    health/diagnostics. Never raises, never exposes secrets."""
    result = {"provider": provider, "ok": False, "model": None,
              "status": None, "latency_ms": None, "error": None,
              "at": datetime.now(timezone.utc).isoformat(),
              # monotonic stamp for freshness (wall-clock epochs and
              # time.monotonic() are different scales — never mixed)
              "at_monotonic": time.monotonic()}
    try:
        if provider not in _PROVIDER_INSTANCES:
            result["error"] = f"unknown provider '{provider}'"
            result["status"] = "NOT_CONFIGURED"
            return result
        if not (_provider_enabled(provider) and _provider_key(provider)):
            result["status"] = "NOT_CONFIGURED"
            result["error"] = "provider disabled or key not configured"
            return result
        model = _task_model(provider, task)
        if provider == "unorouter":
            usable = _provider_has_usable_models(provider, task)
            if usable:
                model = unorouter_effective_chain()["chain"][0]
        if not model:
            result["status"] = "NO_USABLE_MODEL"
            result["error"] = "no task model configured for this provider"
            return result
        result["model"] = model
        t0 = time.monotonic()
        details = {}
        status, error, text, _retry = _attempt(
            provider, model,
            system="You are a health probe. Reply with exactly: OK",
            user="health probe", temperature=0.0, max_tokens=8,
            details=details)
        result["status"] = status
        result["error_category"] = details.get("category")
        result["http_status"] = details.get("http_status")
        result["exception_type"] = details.get("exception_type")
        result["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
        result["ok"] = status == "OK"
        result["error"] = error
        _log_request("send", provider=provider, model=model, agent="probe",
                     symbol=None, reason="inference_probe", ok=result["ok"],
                     status=status, latency_ms=result["latency_ms"])
    except Exception as exc:  # noqa: BLE001 — a probe never crashes anything
        result["ok"] = False
        result["error"] = str(exc)[:200]
        result["status"] = result["status"] or "PROVIDER_ERROR"
    _inference_probes[provider] = result
    logger.info("LLM INFERENCE PROBE %s: %s (model=%s, %sms)%s",
                provider, "OK" if result["ok"] else result["status"],
                result.get("model"), result.get("latency_ms"),
                f" — {result['error']}" if result.get("error") else "")
    return result


def probe_active_chain() -> dict:
    """Runs the bounded inference probe once per ACTIVE-CHAIN provider
    (effective routes only — optional providers are not probed). Explicitly
    opt-in via LLM_INFERENCE_PROBE_ENABLED; never called by a trading
    cycle."""
    out = {}
    if not getattr(settings, "LLM_INFERENCE_PROBE_ENABLED", False):
        return out
    for provider in active_chain_providers():
        representative = next(
            (a for a in ("cio", "risk", "technical", "news", "fundamentals",
                         "debate")
             if effective_route(a)["provider"] == provider), "cio")
        out[provider] = inference_probe(provider, representative)
    return out


def inference_probe_results() -> dict:
    """Recorded probe outcomes (no secrets), for /api/providers/health."""
    return {p: dict(r) for p, r in _inference_probes.items()}


def _probe_view(provider: str, inference_status: str) -> str:
    """Merges a fresh probe outcome into the inference status: a successful
    probe turns UNKNOWN into VERIFIED; a failed probe is INFERENCE_FAILED.
    Live circuits always take precedence over an older probe."""
    probe = _inference_probes.get(provider)
    if not probe:
        return inference_status
    try:
        age = time.monotonic() - float(probe["at_monotonic"])
    except Exception:  # noqa: BLE001
        return inference_status
    if age > _PROBE_FRESH_S or age < 0:
        return inference_status
    if probe.get("ok"):
        # A successful probe upgrades READY/UNKNOWN to VERIFIED. A circuit
        # that already recorded a specific failure (quota/auth/model) keeps
        # its more specific verdict even against one good probe.
        if inference_status in ("READY", "UNKNOWN", None):
            return "VERIFIED"
        return inference_status
    # A FAILED probe is direct evidence of inference failure: READY/UNKNOWN
    # (never exercised) and the generic DEGRADED window both yield to it.
    # Specific circuit states (quota/auth/model) stay — they name the cause.
    if inference_status in ("READY", "UNKNOWN", None, "DEGRADED"):
        return "INFERENCE_FAILED"
    return inference_status


# --- startup configuration diagnostics --------------------------------------
def startup_diagnostics(log: bool = True) -> dict:
    """One structured, secret-free snapshot of the EFFECTIVE LLM
    configuration, logged at startup (and available via
    /api/providers/health): per task — configured provider/model with its
    source (default or env override), live-catalog match, effective route
    after any verified NO_USABLE_MODEL switch, chain, fallback, and
    inference verification state."""
    routes = {}
    for agent in sorted(_TASK_PROVIDERS):
        route = route_info(agent)
        provider, model = route["provider"], route["model"]
        live = _catalog_verified_models(route["configured_provider"])
        configured_match = (
            None if live is None
            else _normalize_model_id(route["configured_model"]) in
            {_normalize_model_id(m) for m in live})
        routes[agent] = {
            "configured_provider": route["configured_provider"],
            "configured_model": route["configured_model"],
            "configured_model_source": _model_env_source(
                route["configured_provider"], route["configured_model"]),
            "configured_model_in_catalog": configured_match,
            "provider": provider,
            "model": model,
            "switched": route["switched"],
            "switch_reason": route["switch_reason"],
            "chain": None,
            "fallback": (f"{route['fallback_provider']}:{route['fallback_model']}"
                         if route["fallback_provider"] and route["fallback_model"]
                         else None),
        }
        try:
            chain, _r = _candidate_chain(agent)
            routes[agent]["chain"] = [f"{p}:{m}" for p, m in chain]
        except Exception:  # noqa: BLE001
            pass
    states = provider_states()
    providers = {}
    for provider in PROVIDERS:
        providers[provider] = {
            "enabled": _provider_enabled(provider),
            "key_configured": bool(_provider_key(provider)),
            "circuit": states.get(provider, {}).get("state"),
            "inference_status": _probe_view(
                provider, _inference_status(provider, states)),
        }
        if provider == "unorouter":
            providers[provider]["configured_chain"] = unorouter_model_chain()
            eff = unorouter_effective_chain()
            providers[provider]["effective_chain"] = eff["chain"]
            providers[provider]["catalog_available"] = eff["catalog_available"]
    diag = {
        "active_chain_providers": active_chain_providers(),
        "routes": routes,
        "providers": providers,
        "inference_probes": inference_probe_results(),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    # Log the human-readable table ONCE at startup (INFO; no keys/secrets).
    # API callers pass log=False — dashboard polling never spams the log.
    if not log:
        return diag
    logger.info("LLM ROUTING DIAGNOSTICS — active chain: %s",
                ", ".join(diag["active_chain_providers"]) or "none")
    for agent, r in routes.items():
        logger.info(
            "LLM ROUTE %-12s configured=%s:%s [%s, catalog=%s] -> effective=%s:%s%s",
            agent, r["configured_provider"], r["configured_model"] or "-",
            r["configured_model_source"],
            ("VERIFIED" if r["configured_model_in_catalog"]
             else "MISSING" if r["configured_model_in_catalog"] is False
             else "UNVERIFIED"),
            r["provider"], r["model"] or "-",
            (" [SWITCHED: " + r["switch_reason"] + "]"
             if r["switched"] else ""))
    for provider, p in providers.items():
        if provider == "unorouter" and p.get("configured_chain"):
            logger.info(
                "LLM PROVIDER unorouter: configured_chain=%s "
                "effective_chain=%s catalog_available=%s",
                ",".join(p["configured_chain"]),
                ",".join(p.get("effective_chain") or []) or "EMPTY",
                p.get("catalog_available"))
    return diag if log else diag


def last_validation() -> dict:
    return _last_validation
