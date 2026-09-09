"""
services/llm_service.py

Centralized LLM router (the "LLMRouter" of the architecture):

    LLMRouter
      ├── GroqProvider
      ├── NVIDIAProvider
      ├── GeminiProvider
      └── OpenRouterProvider

Agents never implement provider-specific HTTP calls. They call

    llm_service.complete(task="debate", messages=[...])      # messages-style
    llm_service.call(agent="debate", system=..., user=...)   # system/user-style
    llm_service.call_json(agent="news", system=..., user=...)  # + JSON parsing

Responsibilities:
  - Routing: task -> (provider, model), resolved from settings at CALL time
    (LLM_<TASK>_PROVIDER plus per-provider model envs). No model ids are
    hardcoded in agent source files.
  - Optional global fallback route (LLM_FALLBACK_PROVIDER/LLM_FALLBACK_MODEL),
    used only when explicitly configured AND verified in the fallback
    provider's live catalog — never for quota errors, never silently.
  - Statuses (every result carries provider/model/status/latency/error_type):
        OK, NOT_CONFIGURED, MODEL_NOT_FOUND, PROVIDER_QUOTA_EXCEEDED,
        RATE_LIMITED, PROVIDER_ERROR, AUTH_ERROR, NETWORK_ERROR,
        INVALID_RESPONSE
  - Circuit breakers per provider:
        * 429 quota-exhausted  -> provider paused for the server retry delay
          (or the configured backoff); NEVER retried; later calls short-
          circuit with PROVIDER_QUOTA_EXCEEDED. No per-agent/per-symbol
          retry storms, no duplicate requests.
        * 429 transient        -> at most LLM_RATE_LIMIT_MAX_RETRIES retries,
          only when the server gave a short explicit retry delay.
        * 404 MODEL_NOT_FOUND  -> the (provider, model) pair is short-circuited
          for MODEL_UNAVAILABLE_COOLDOWN_MINUTES; the same request is never
          re-sent blindly.
        * 401/403 auth failure -> provider paused for AUTH_COOLDOWN_MINUTES;
          no repeated auth attempts.
        * local rolling-24h request budget per provider (e.g. Gemini free
          tier = 20/day) short-circuits BEFORE requests are sent.
  - Usage accounting: per-cycle counters per provider/agent/symbol.
  - Startup validation: validate_models() checks every configured model id
    against each provider's LIVE /models catalog.
  - provider_states(): READY / DEGRADED / QUOTA_EXHAUSTED / MODEL_UNAVAILABLE /
    AUTH_ERROR / NETWORK_ERROR / NOT_CONFIGURED for dashboards and health.
"""

import logging
import re
import time
from collections import deque

from config import settings
from services import gemini_service

logger = logging.getLogger("llm_service")

# Generic tolerant JSON extraction (single implementation; lives in the
# gemini transport module for backward compatibility with direct callers).
extract_json = gemini_service._extract_json

PROVIDERS = ("groq", "nvidia", "gemini", "openrouter")

# ---------------------------------------------------------------------------
# Provider classes
# ---------------------------------------------------------------------------


class BaseLLMProvider:
    """One LLM provider behind the router. Subclasses own transport details."""

    name = "base"
    key_attr = ""                    # settings attribute holding the API key
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
    daily_limit_attr = "GROQ_DAILY_REQUEST_LIMIT"
    quota_backoff_attr = "GROQ_QUOTA_BACKOFF_MINUTES"

    def get_client(self):
        client = _clients.get(self.name)
        if client is None:
            from groq import Groq
            client = Groq(api_key=_provider_key(self.name))
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
    """OpenAI-compatible endpoint (OpenRouter, NVIDIA NIM)."""

    def get_client(self):
        client = _clients.get(self.name)
        if client is None:
            from openai import OpenAI
            client = OpenAI(
                api_key=_provider_key(self.name),
                base_url=self.base_url(),
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


def _daily_limit(provider: str) -> int:
    return int(getattr(settings, _PROVIDER_INSTANCES[provider].daily_limit_attr, 0) or 0)


def _quota_backoff_seconds(provider: str) -> float:
    minutes = float(getattr(settings, _PROVIDER_INSTANCES[provider].quota_backoff_attr, 30) or 30)
    return max(30.0, minutes * 60.0)


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
    """Test hook: clears clients, circuits, rolling budgets, usage and
    validation cache. Does NOT touch settings."""
    _clients.clear()
    _verified_models.clear()
    global _last_validation
    _last_validation = None
    for p in PROVIDERS:
        _quota_backoff_until[p] = 0.0
        _auth_backoff_until[p] = 0.0
        _request_times[p].clear()
        _last_failure[p] = None
    _model_unavailable_until.clear()
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


def provider_states() -> dict:
    """Circuit-breaker state per provider (for /api/health, /api/config).

    READY | NOT_CONFIGURED | QUOTA_EXHAUSTED | MODEL_UNAVAILABLE |
    AUTH_ERROR | NETWORK_ERROR | DEGRADED
    """
    now = time.monotonic()
    states = {}
    for provider in PROVIDERS:
        inst = _PROVIDER_INSTANCES[provider]
        limit = _daily_limit(provider)
        stamps = _request_times[provider]
        while stamps and stamps[0] <= now - _ROLLING_WINDOW_S:
            stamps.popleft()

        detail = ""
        if not _provider_key(provider):
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
# Circuit breakers
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
        f"{provider}: model '{model}' not found (404). Circuit open for "
        f"{minutes:.0f} min; requests to it short-circuit with MODEL_NOT_FOUND."
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


class LLMResult:
    """Outcome of one task->provider request."""

    def __init__(self, agent, provider, model, status, text="", parsed=None,
                 latency_ms=None, error=None, error_type=None, attempts=0,
                 fallback_used=False):
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

    @property
    def ok(self) -> bool:
        return self.status == "OK"

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
# Transport (one attempt = bounded retries for transient rate limits only)
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
# Public API
# ---------------------------------------------------------------------------


def _resolve_route(agent: str):
    """Returns (provider, model, fallback_provider, fallback_model)."""
    route = route_info(agent)
    provider, model = route["provider"], route["model"]
    fb_provider = route["fallback_provider"]
    fb_model = route["fallback_model"]
    fallback_usable = (
        bool(fb_provider) and bool(fb_model)
        and fb_provider in _PROVIDER_INSTANCES
        and fb_provider != provider
        and fb_model in _verified_models.get(fb_provider, set())
    )
    return provider, model, (fb_provider if fallback_usable else None), (fb_model if fallback_usable else None)


def call(agent: str, system: str, user: str, temperature: float = 0.2,
         max_tokens: int = 500, symbol: str = None) -> LLMResult:
    """Runs one task request against its configured provider/model.
    Never raises for provider-side problems."""
    provider, model, fb_provider, fb_model = _resolve_route(agent)

    # --- configuration checks (no fallback: these are config problems the
    #     operator must see, not transient failures to route around) --------
    if provider not in _PROVIDER_INSTANCES:
        return _record(LLMResult(
            agent, provider, model, "PROVIDER_ERROR",
            error=f"PROVIDER_ERROR: unknown LLM provider '{provider}' "
                  f"(check LLM_*_PROVIDER settings).",
        ), symbol)

    if not _provider_key(provider):
        key_name = _PROVIDER_INSTANCES[provider].key_attr
        return _record(LLMResult(
            agent, provider, model, "NOT_CONFIGURED",
            error=f"{key_name} not configured.",
        ), symbol)

    if not model:
        model_env = _TASK_MODELS.get((provider, agent), "?")
        return _record(LLMResult(
            agent, provider, "", "NOT_CONFIGURED",
            error=f"No model configured for {agent} on {provider} (set {model_env}).",
        ), symbol)

    # --- quota circuit: reported honestly, NEVER routed around -------------
    blocked, reason = _quota_blocked(provider)
    if blocked:
        return _record(LLMResult(
            agent, provider, model, "PROVIDER_QUOTA_EXCEEDED",
            error=f"PROVIDER_QUOTA_EXCEEDED: {reason}.",
        ), symbol)

    # --- attempt plan: primary, then the verified fallback (if any). A
    #     circuit-blocked candidate (auth / dead model) produces a synthetic
    #     result WITHOUT a network call and falls through to the fallback.
    t0 = time.monotonic()
    routes = [(provider, model)]
    if fb_provider and fb_model:
        routes.append((fb_provider, fb_model))

    last_result = None
    for idx, (cand_provider, cand_model) in enumerate(routes):
        now = time.monotonic()
        if _auth_backoff_until[cand_provider] > now:
            status = "AUTH_ERROR"
            error = (f"AUTH_ERROR: authentication previously failed on {cand_provider}; "
                     f"circuit open ({int(round(_auth_backoff_until[cand_provider] - now))}s remaining).")
        elif _model_unavailable_until.get((cand_provider, cand_model), 0.0) > now:
            status = "MODEL_NOT_FOUND"
            error = (f"MODEL_NOT_FOUND: model '{cand_model}' previously returned 404 on "
                     f"{cand_provider}; circuit open "
                     f"({int(round(_model_unavailable_until[(cand_provider, cand_model)] - now))}s remaining).")
        else:
            status, attempt_error, text, _ = _attempt(
                cand_provider, cand_model, system, user, temperature, max_tokens
            )
            if status == "OK":
                return _record(LLMResult(
                    agent, cand_provider, cand_model, "OK", text=text,
                    latency_ms=round((time.monotonic() - t0) * 1000, 1),
                    attempts=idx + 1, fallback_used=idx > 0,
                ), symbol)
            error = f"{status}: {attempt_error}"

        last_result = LLMResult(
            agent, cand_provider, cand_model, status,
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
            error=error, attempts=idx + 1, fallback_used=idx > 0,
        )
        if status == "PROVIDER_QUOTA_EXCEEDED":
            break  # quota is reported honestly, never routed around
        if idx == 0 and len(routes) > 1:
            logger.warning(
                f"{agent}: {provider}/{model} failed ({status}); trying verified "
                f"fallback {fb_provider}/{fb_model}."
            )
            continue
        break

    return _record(last_result, symbol)


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
    """call() + tolerant JSON parsing. Parse failures come back with status
    INVALID_RESPONSE (never fabricated content)."""
    result = call(agent, system, user, temperature=temperature,
                  max_tokens=max_tokens, symbol=symbol)
    if not result.ok:
        return result
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


def validate_models() -> dict:
    """Verifies every configured model id against each provider's LIVE
    /models catalog. Providers without an API key are skipped (reported as
    such). Results gate fallback eligibility: an unverified fallback model
    is never used.

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
        if route["provider"] in report["providers"]:
            _check(route["provider"], route["model"], f"{agent}.model")
        if route["fallback_provider"] in report["providers"] and route["fallback_model"]:
            _check(route["fallback_provider"], route["fallback_model"], f"{agent}.fallback")
        report["routes"][agent] = dict(route)

    _last_validation = report
    return report


def last_validation() -> dict:
    return _last_validation
