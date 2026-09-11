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
import time
from collections import OrderedDict, deque
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
    """404 circuit: stop requesting this (provider, model) pair."""
    minutes = max(0.5, float(settings.MODEL_UNAVAILABLE_COOLDOWN_MINUTES or 30))
    _model_unavailable_until[(provider, model)] = time.monotonic() + minutes * 60.0
    logger.warning(
        f"LLM_MODEL_UNAVAILABLE: {provider}: model '{model}' not found (404) "
        f"or absent from the live catalog. Circuit open for {minutes:.0f} min; "
        f"requests to it short-circuit and the next configured model is tried."
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
        models = [str(settings.UNOROUTER_PRIMARY_MODEL or "")]
        models += [m for m in (settings.UNOROUTER_FALLBACK_MODELS or []) if m]
        # Cap MODEL attempts (never unlimited model guessing).
        cap = max(1, int(settings.LLM_MAX_MODEL_ATTEMPTS or 3))
        models = models[:cap]
        chain = [("unorouter", m) for m in models if m]
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
    """Pulls model id strings out of a provider's models.list() response."""
    ids = set()
    data = getattr(response, "data", None)
    items = data if data is not None else response
    if items is None:
        return ids
    try:
        for item in items:
            model_id = getattr(item, "id", None) or getattr(item, "name", None)
            if model_id:
                ids.add(str(model_id).removeprefix("models/"))
    except TypeError:
        pass
    return ids


def unorouter_model_chain() -> list:
    """The application's explicit UnoRouter model chain (primary +
    fallbacks, capped at LLM_MAX_MODEL_ATTEMPTS)."""
    models = [str(settings.UNOROUTER_PRIMARY_MODEL or "")]
    models += [m for m in (settings.UNOROUTER_FALLBACK_MODELS or []) if m]
    cap = max(1, int(settings.LLM_MAX_MODEL_ATTEMPTS or 3))
    return [m for m in models if m][:cap]


def validate_models() -> dict:
    """Verifies every configured model id against each provider's LIVE
    /models catalog. Providers without an API key are skipped (reported as
    such). A model id missing from the catalog is reported as
    LLM_MODEL_UNAVAILABLE and short-circuited at request time — the next
    configured model is used automatically.

    Returns: {"providers": {name: {"ok", "models_found", "checked",
                                   "missing", "error"}},
              "routes": {task: {provider, model, fallback_provider,
                                fallback_model}}}
    """
    global _last_validation
    report = {"providers": {}, "routes": {}}

    for provider in PROVIDERS:
        inst = _PROVIDER_INSTANCES[provider]
        entry = {"ok": False, "models_found": 0, "checked": {}, "missing": [], "error": None}
        report["providers"][provider] = entry
        if not _provider_enabled(provider):
            entry["error"] = f"disabled via {inst.enabled_attr}=false — catalog not fetched"
            continue
        if not _provider_key(provider):
            entry["error"] = f"{inst.key_attr} not configured — catalog not fetched"
            continue
        try:
            ids = inst.list_models()
            entry["models_found"] = len(ids)
            entry["ok"] = True
            _verified_models[provider] = ids
        except Exception as exc:  # noqa: BLE001 — validation must never crash startup
            entry["error"] = f"could not list models: {exc}"
            logger.warning(f"{provider}: model validation failed: {exc}")
            continue

    def _check(provider, model, label):
        if not model:
            return
        entry = report["providers"][provider]
        if not entry["ok"]:
            return
        present = model in _verified_models.get(provider, set())
        entry["checked"][label] = present
        if not present:
            entry["missing"].append(f"{label}: {model}")

    for agent in sorted(_TASK_PROVIDERS):
        route = route_info(agent)
        if route["provider"] == "unorouter":
            for i, model in enumerate(unorouter_model_chain()):
                _check("unorouter", model, "primary" if i == 0 else f"fallback[{i - 1}]")
        elif route["provider"] in report["providers"]:
            _check(route["provider"], route["model"], f"{agent}.model")
        if route["fallback_provider"] in report["providers"] and route["fallback_model"]:
            _check(route["fallback_provider"], route["fallback_model"], f"{agent}.fallback")
        report["routes"][agent] = dict(route)

    # Report dead UnoRouter model ids loudly (the runtime skips them).
    uno_entry = report["providers"].get("unorouter", {})
    chain = unorouter_model_chain()
    for label, present in (uno_entry.get("checked") or {}).items():
        if not present:
            try:
                idx = 0 if label == "primary" else int(label.split("[")[1].rstrip("]")) + 1
            except (IndexError, ValueError):
                idx = -1
            missing_model = chain[idx] if 0 <= idx < len(chain) else "?"
            logger.warning(
                "LLM_MODEL_UNAVAILABLE: unorouter %s model '%s' is not in the "
                "live catalog; it is skipped and the next configured model is used.",
                label, missing_model,
            )

    _last_validation = report
    return report


def last_validation() -> dict:
    return _last_validation
