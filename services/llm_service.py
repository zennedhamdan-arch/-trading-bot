"""
services/llm_service.py

Common LLM provider layer for every agent in the bot.

Responsibilities (single place, NOT scattered across agent files):
  - Routing: each agent maps to (provider, model, optional fallback model),
    resolved from settings at CALL time so environment variables always win
    and no model id is hardcoded in agent source files.
  - One entry point per agent: call() / call_json() -> LLMResult carrying
    provider, model, status, latency_ms, error and error_type.
  - Status model (reported per agent, surfaced in cycle records):
        OK                       request succeeded
        NOT_CONFIGURED           provider API key missing
        MODEL_NOT_FOUND          provider says the model does not exist or
                                 the account has no access to it
        PROVIDER_QUOTA_EXCEEDED  the account's request quota is exhausted
                                 (local rolling-24h budget reached, or the
                                 provider returned a quota-exhausted 429)
        RATE_LIMITED             transient rate limit; bounded retries
                                 (short delay only) did not succeed
        PROVIDER_ERROR           any other provider/transport failure
        INVALID_RESPONSE         model answered but not parseable JSON
  - Quota handling (per provider, no blind retries):
        * a local rolling-24h request budget (e.g. Gemini free tier is 20
          requests/day) short-circuits requests with PROVIDER_QUOTA_EXCEEDED
          BEFORE they are sent;
        * a server 429 is classified: quota-exhaustion indicators -> NO
          retry, a backoff window is set and later calls short-circuit;
          a short "retry in Ns" delay -> at most LLM_RATE_LIMIT_MAX_RETRIES
          retries after waiting at most LLM_RATE_LIMIT_MAX_WAIT_SECONDS;
  - Optional fallback models: used ONLY when explicitly configured via env
    AND verified against the provider's live model list during
    validate_models(); never used for quota errors (same provider = same
    quota) and never as a silent provider switch.
  - Usage accounting: per-cycle counters (reset_cycle_usage/cycle_usage)
    broken down per provider, per agent and per symbol — this is how the
    number of LLM calls per cycle is reported honestly.
  - Startup validation: validate_models() lists each provider's live model
    catalog and verifies every configured model id (primary + fallback).

Design notes:
  - Groq and OpenRouter go through their OpenAI-compatible chat-completions
    SDK clients (cached, one per provider). Gemini goes through
    services.gemini_service (the Interactions API transport).
  - All request/response accounting is in-memory per process; the rolling
    budget window uses time.monotonic().
"""

import logging
import re
import time
from collections import deque

from config import settings
from services import gemini_service

logger = logging.getLogger("llm_service")

# Generic tolerant JSON extraction (single implementation, lives in the
# gemini transport module for backward compatibility with direct callers).
extract_json = gemini_service._extract_json

PROVIDERS = ("groq", "openrouter", "gemini")

# Agent pipeline keys -> settings attribute names (resolved at call time).
_ROUTES = {
    "technical": ("groq", "GROQ_TECH_MODEL", "GROQ_TECH_FALLBACK_MODEL"),
    "debate": ("groq", "GROQ_DEBATE_MODEL", "GROQ_DEBATE_FALLBACK_MODEL"),
    "cio": ("groq", "GROQ_CIO_MODEL", "GROQ_CIO_FALLBACK_MODEL"),
    "risk": ("openrouter", "OPENROUTER_RISK_MODEL", "OPENROUTER_RISK_FALLBACK_MODEL"),
    "news": ("gemini", "GEMINI_MODEL", "GEMINI_FALLBACK_MODEL"),
    "fundamentals": ("gemini", "GEMINI_MODEL", "GEMINI_FALLBACK_MODEL"),
}

_PROVIDER_KEYS = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

_DAILY_LIMIT_SETTINGS = {
    "groq": "GROQ_DAILY_REQUEST_LIMIT",
    "openrouter": "OPENROUTER_DAILY_REQUEST_LIMIT",
    "gemini": "GEMINI_DAILY_REQUEST_LIMIT",
}

_QUOTA_BACKOFF_SETTINGS = {
    "groq": "GROQ_QUOTA_BACKOFF_MINUTES",
    "openrouter": "OPENROUTER_QUOTA_BACKOFF_MINUTES",
    "gemini": "GEMINI_QUOTA_BACKOFF_MINUTES",
}

_ROLLING_WINDOW_S = 24 * 3600.0
_BACKOFF_CAP_S = 24 * 3600.0  # never back off longer than a day

# ---------------------------------------------------------------------------
# State (per process)
# ---------------------------------------------------------------------------

_clients = {}  # provider -> cached SDK client (tests may inject)
_verified_models = {}  # provider -> set of model ids seen in the live catalog
_last_validation = None  # report dict returned by validate_models()

_quota_backoff_until = {p: 0.0 for p in PROVIDERS}  # monotonic deadlines
_request_times = {p: deque() for p in PROVIDERS}  # rolling 24h send stamps

_cycle_usage = {}  # per-cycle counters, see reset_cycle_usage()


def reset_cycle_usage() -> None:
    """Zeroes the per-cycle usage counters. main.run_trading_cycle calls
    this at the start of every cycle so llm_usage reflects exactly one
    cycle."""
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
    """Test hook: clears clients, quota windows, rolling budgets, usage and
    validation cache. Does NOT touch settings."""
    _clients.clear()
    _verified_models.clear()
    global _last_validation
    _last_validation = None
    for p in PROVIDERS:
        _quota_backoff_until[p] = 0.0
        _request_times[p].clear()
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


def quota_state() -> dict:
    """Current quota posture per provider (for /api/config; no secrets)."""
    now = time.monotonic()
    state = {}
    for provider in PROVIDERS:
        limit = int(getattr(settings, _DAILY_LIMIT_SETTINGS[provider], 0) or 0)
        stamps = _request_times[provider]
        while stamps and stamps[0] <= now - _ROLLING_WINDOW_S:
            stamps.popleft()
        backoff = _quota_backoff_until[provider]
        state[provider] = {
            "daily_request_limit": limit,
            "requests_last_24h": len(stamps),
            "backoff_active": backoff > now,
            "backoff_seconds_remaining": round(max(0.0, backoff - now), 1) if backoff > now else 0,
        }
    return state


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_info(agent: str) -> dict:
    """Resolves an agent's (provider, model, fallback) from settings at call
    time. Agents attach this to their reports so every result says exactly
    which provider and model produced it."""
    provider, model_attr, fallback_attr = _ROUTES[agent]
    return {
        "provider": provider,
        "model": str(getattr(settings, model_attr, "") or ""),
        "fallback": str(getattr(settings, fallback_attr, "") or ""),
    }


def llm_routes() -> dict:
    """All agent routes (for /api/config and startup validation)."""
    return {agent: route_info(agent) for agent in sorted(_ROUTES)}


def _provider_key(provider: str) -> str:
    return str(getattr(settings, _PROVIDER_KEYS[provider], "") or "")


def _daily_limit(provider: str) -> int:
    return int(getattr(settings, _DAILY_LIMIT_SETTINGS[provider], 0) or 0)


def _quota_backoff_seconds(provider: str) -> float:
    minutes = float(getattr(settings, _QUOTA_BACKOFF_SETTINGS[provider], 30) or 30)
    return max(30.0, minutes * 60.0)


# ---------------------------------------------------------------------------
# Clients (cached; tests may pre-inject via _clients[provider] = fake)
# ---------------------------------------------------------------------------

def _get_groq_client():
    client = _clients.get("groq")
    if client is None:
        from groq import Groq
        client = Groq(api_key=_provider_key("groq"))
        _clients["groq"] = client
    return client


def _get_openrouter_client():
    client = _clients.get("openrouter")
    if client is None:
        from openai import OpenAI
        client = OpenAI(
            api_key=_provider_key("openrouter"),
            base_url=settings.OPENROUTER_BASE_URL,
        )
        _clients["openrouter"] = client
    return client


def _get_client(provider: str):
    if provider == "groq":
        return _get_groq_client()
    if provider == "openrouter":
        return _get_openrouter_client()
    return None  # gemini transport manages its own client


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
    """Extracts a retry-after delay (seconds) from a provider error message,
    e.g. 'Please retry in 25.46s' or '"retryDelay": "3600s"'."""
    for pattern in _RETRY_DELAY_PATTERNS:
        m = pattern.search(message)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _looks_like_quota_exhaustion(message: str, retry_after) -> bool:
    """Distinguishes 'quota exhausted' (stop; do not retry) from a transient
    per-minute rate limit (safe to retry after the given delay).

    Daily/free-tier quota errors mention daily quotas / free-tier request
    metrics, or carry no (or a huge) retry delay. Per-minute limits carry a
    short explicit retry delay.
    """
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


def _classify_exception(exc: Exception):
    """Maps a provider exception to (status, retry_after_s).

    Uses the exception's HTTP status code when the SDK exposes one — never
    message keyword matching on generic exceptions, so ordinary programming
    errors are never mistaken for quota events.
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
        return "PROVIDER_ERROR", None
    return "PROVIDER_ERROR", None


# ---------------------------------------------------------------------------
# Quota gates
# ---------------------------------------------------------------------------

def _quota_blocked(provider: str):
    """(blocked, reason). Checked BEFORE any request is sent."""
    now = time.monotonic()
    until = _quota_backoff_until[provider]
    if until > now:
        remaining = int(round(until - now))
        return True, (
            f"provider quota exhausted (server 429); backing off for another "
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
                f"({len(stamps)}/{limit}, {_DAILY_LIMIT_SETTINGS[provider]}) — "
                f"no request sent"
            )
    return False, None


def _mark_quota_backoff(provider: str, retry_after) -> None:
    """Records a server-side quota exhaustion and stops further requests to
    this provider for the window. Never retried past this point."""
    if retry_after is not None and 0 < retry_after <= _BACKOFF_CAP_S:
        window = retry_after
        why = f"server retry delay {int(retry_after)}s"
    else:
        window = _quota_backoff_seconds(provider)
        why = f"configured {_QUOTA_BACKOFF_SETTINGS[provider]}"
    _quota_backoff_until[provider] = max(
        _quota_backoff_until[provider], time.monotonic() + window
    )
    logger.warning(
        f"{provider}: quota exhausted (429). No further requests for "
        f"{int(window)}s ({why}). Subsequent calls return PROVIDER_QUOTA_EXCEEDED."
    )


def _count_request(provider: str) -> None:
    """Counts one outgoing request against the rolling 24h budget."""
    now = time.monotonic()
    stamps = _request_times[provider]
    stamps.append(now)
    while stamps and stamps[0] <= now - _ROLLING_WINDOW_S:
        stamps.popleft()


# ---------------------------------------------------------------------------
# Result type + usage recording
# ---------------------------------------------------------------------------

class LLMResult:
    """Outcome of one agent->provider request. Carries everything an agent
    report needs: provider, model, status, latency, error and error type."""

    def __init__(self, agent, provider, model, status, text="", parsed=None,
                 latency_ms=None, error=None, error_type=None, attempts=0):
        self.agent = agent
        self.provider = provider
        self.model = model
        self.status = status  # OK / NOT_CONFIGURED / MODEL_NOT_FOUND / ...
        self.text = text
        self.parsed = parsed  # dict when call_json succeeded
        self.latency_ms = latency_ms
        self.error = error  # human-readable message including the status
        self.error_type = error_type or (None if status == "OK" else status)
        self.attempts = attempts

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
        }


def _record(result: LLMResult, symbol=None) -> LLMResult:
    """Records a finished attempt in the per-cycle usage counters."""
    provider = result.provider
    usage = _cycle_usage.get(provider)
    if usage is None:
        return result  # reset_cycle_usage not called (e.g. ad-hoc call); fine

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
    """Adjusts already-recorded counters when a response that arrived OK
    turns out unparseable (call_json): move it from ok to errors."""
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
# Transport (single request, no retries)
# ---------------------------------------------------------------------------

def _send_once(provider: str, model: str, system: str, user: str,
               temperature: float, max_tokens: int) -> str:
    """Sends exactly one chat-style request. Raises whatever the SDK raises;
    classification happens in the caller."""
    if provider == "gemini":
        return gemini_service.generate_raw(system, user, model=model)
    client = _get_client(provider)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content or ""


def _attempt(provider: str, model: str, system: str, user: str,
             temperature: float, max_tokens: int):
    """One attempt = up to (1 + LLM_RATE_LIMIT_MAX_RETRIES) sends, where a
    retry happens ONLY for a transient rate-limit 429 with a short,
    explicit retry delay. Quota exhaustion is never retried."""
    last = ("PROVIDER_ERROR", "request failed", None, None)
    for attempt in range(_max_retries() + 1):
        _count_request(provider)
        try:
            text = _send_once(provider, model, system, user, temperature, max_tokens)
            if not str(text).strip():
                return "INVALID_RESPONSE", "model returned an empty response", text, None
            return "OK", None, str(text), None
        except Exception as exc:  # noqa: BLE001 — classified below
            status, retry_after = _classify_exception(exc)
            last = (status, str(exc) or exc.__class__.__name__, None, retry_after)
            if status == "PROVIDER_QUOTA_EXCEEDED":
                _mark_quota_backoff(provider, retry_after)
                return last  # never retry an exhausted quota
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

def call(agent: str, system: str, user: str, temperature: float = 0.2,
         max_tokens: int = 500, symbol: str = None) -> LLMResult:
    """Runs one agent request against its configured provider/model.

    Args:
        agent: pipeline key — technical | debate | cio | risk | news | fundamentals
        system: system prompt (role + output contract)
        user: user input carrying the data to analyze
        temperature / max_tokens: passthrough to the chat-style providers
        symbol: optional, for per-symbol usage accounting

    Returns LLMResult (never raises for provider-side problems).
    """
    route = route_info(agent)
    provider, model = route["provider"], route["model"]

    if not _provider_key(provider):
        key_name = _PROVIDER_KEYS[provider]
        return _record(LLMResult(
            agent, provider, model, "NOT_CONFIGURED",
            error=f"NOT_CONFIGURED: {key_name} not configured.",
        ), symbol)

    blocked, reason = _quota_blocked(provider)
    if blocked:
        return _record(LLMResult(
            agent, provider, model, "PROVIDER_QUOTA_EXCEEDED",
            error=f"PROVIDER_QUOTA_EXCEEDED: {reason}.",
        ), symbol)

    # Fallback eligibility: explicitly configured AND verified present in the
    # provider's live model catalog during validate_models(). Never on quota
    # errors (same provider shares the quota), never silently.
    fallback = route["fallback"]
    fallback_usable = bool(fallback) and fallback != model and fallback in _verified_models.get(provider, set())

    t0 = time.monotonic()
    attempts = 0
    models_to_try = [model] + ([fallback] if fallback_usable else [])
    last_result = None

    for idx, candidate in enumerate(models_to_try):
        attempts += 1
        status, error, text, _ = _attempt(
            provider, candidate, system, user, temperature, max_tokens
        )
        if status == "OK":
            return _record(LLMResult(
                agent, provider, candidate, "OK", text=text,
                latency_ms=round((time.monotonic() - t0) * 1000, 1),
                attempts=attempts,
            ), symbol)
        last_result = LLMResult(
            agent, provider, candidate, status,
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
            error=f"{status}: {error}", attempts=attempts,
        )
        if status == "PROVIDER_QUOTA_EXCEEDED":
            break  # a fallback on the same provider shares the quota
        if idx == 0 and len(models_to_try) > 1:
            logger.warning(
                f"{agent}: model '{model}' failed ({status}); trying verified "
                f"fallback '{fallback}' on {provider}."
            )
            continue
        break

    if last_result is None:  # defensive: no model to try
        last_result = LLMResult(
            agent, provider, model, "PROVIDER_ERROR",
            error="PROVIDER_ERROR: no model configured for this agent.",
        )
    return _record(last_result, symbol)


def call_json(agent: str, system: str, user: str, temperature: float = 0.2,
              max_tokens: int = 500, symbol: str = None) -> LLMResult:
    """call() + tolerant JSON parsing of the response. Parse failures come
    back with status INVALID_RESPONSE (never fabricated content)."""
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
        _reclassify(result, symbol)  # move the already-counted ok -> error
    return result


# ---------------------------------------------------------------------------
# Startup validation — verify configured models against the live catalogs
# ---------------------------------------------------------------------------

def _extract_model_ids(provider: str, response) -> set:
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


def _list_models(provider: str) -> set:
    """Fetches the provider's live model catalog (requires an API key)."""
    if provider == "groq":
        return _extract_model_ids(provider, _get_groq_client().models.list())
    if provider == "openrouter":
        return _extract_model_ids(provider, _get_openrouter_client().models.list())
    if provider == "gemini":
        return _extract_model_ids(provider, gemini_service._get_client().models.list())
    return set()


def validate_models() -> dict:
    """Verifies every configured model id against each provider's LIVE
    /models catalog. Providers without an API key are skipped (reported as
    such). Results are cached in _verified_models, which gates fallback
    eligibility: an unverified fallback is never used.

    Returns a report dict (also stored as the module's last validation):
        {"providers": {name: {"ok", "models_found", "checked",
                              "missing": [...], "error"}},
         "routes": {agent: {provider, model, fallback, ...}}}
    """
    global _last_validation
    report = {"providers": {}, "routes": {}}

    for provider in PROVIDERS:
        entry = {"ok": False, "models_found": 0, "checked": {}, "missing": [], "error": None}
        report["providers"][provider] = entry
        if not _provider_key(provider):
            entry["error"] = f"{_PROVIDER_KEYS[provider]} not configured — catalog not fetched"
            continue
        try:
            ids = _list_models(provider)
            entry["models_found"] = len(ids)
            entry["ok"] = True
            _verified_models[provider] = ids
        except Exception as exc:  # noqa: BLE001 — validation must never crash startup
            entry["error"] = f"could not list models: {exc}"
            logger.warning(f"{provider}: model validation failed: {exc}")
            continue

    for agent in sorted(_ROUTES):
        route = route_info(agent)
        provider = route["provider"]
        entry = report["providers"][provider]
        checked = entry["checked"]
        if entry["ok"]:
            for role, model in (("model", route["model"]), ("fallback", route["fallback"])):
                if not model:
                    continue
                present = model in _verified_models.get(provider, set())
                checked[f"{agent}.{role}"] = present
                if not present:
                    entry["missing"].append(f"{agent}.{role}: {model}")
        report["routes"][agent] = dict(route)

    _last_validation = report
    return report


def last_validation() -> dict:
    """The most recent validate_models() report (or None)."""
    return _last_validation
