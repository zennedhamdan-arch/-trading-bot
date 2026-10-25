"""
operator_tools/llm.py — LLM provider/model/circuit/usage inspection tools.

The operator can distinguish: provider unavailable, authentication failure,
quota exhaustion, timeout, model unavailable, model-catalog mismatch,
circuit breaker open, malformed response, structured-output failure, and
application configuration error. Never exposes keys (sanitize() redacts).
"""

import json

from services import llm_service
from services.operator.operator_tools._registry import tool
from services.operator.schemas import sanitize

# failure status -> human classification (the taxonomy the operator reports)
FAILURE_TAXONOMY = {
    "PROVIDER_QUOTA_EXCEEDED": "quota exhaustion (429): the provider's quota is used up; the circuit is open for the cooldown window",
    "AUTH_ERROR": "authentication failure (401/403): the API key was rejected or lacks permission",
    "MODEL_NOT_FOUND": "model unavailable (404): the model id does not exist on this provider",
    "NETWORK_ERROR": "network failure: the provider endpoint could not be reached",
    "RATE_LIMITED": "transient rate limit: the provider asked to slow down (bounded retry policy applies)",
    "RATE_LIMITED_LOCAL": "local quota protection: the per-model minimum interval stopped this send before it left the process",
    "INVALID_RESPONSE": "malformed/empty response: the model answered but the content was empty or unparseable JSON",
    "CIRCUIT_OPEN": "circuit breaker open: calls are short-circuited until the cooldown expires",
    "CYCLE_BUDGET_EXCEEDED": "per-cycle LLM budget reached: further sends were blocked to protect the quota",
    "NOT_CONFIGURED": "application configuration error: the provider/task is not configured (key or model missing)",
    "PROVIDER_ERROR": "provider/application error: the request failed for another reason",
}

PROVIDERS = ("unorouter", "groq", "nvidia", "gemini", "openrouter")


@tool("get_llm_provider_status", "llm",
      "Circuit-breaker status of every LLM provider: state (READY/"
      "NOT_CONFIGURED/QUOTA_EXHAUSTED/AUTH_ERROR/MODEL_UNAVAILABLE/"
      "NETWORK_ERROR/DEGRADED), breaker, failure counts, last success/"
      "failure, cooldowns.", label="Checking LLM providers")
def _get_llm_provider_status(args):
    states = llm_service.provider_states()
    return {"providers": {p: {
        "state": s.get("state"), "circuit": s.get("circuit"),
        "enabled": s.get("enabled"), "detail": s.get("detail"),
        "failure_count": s.get("failure_count"),
        "last_success": s.get("last_success"),
        "last_failure": s.get("last_failure"),
        "cooldown_remaining_s": max(s.get("quota_cooldown_remaining_s") or 0.0,
                                    s.get("auth_cooldown_remaining_s") or 0.0),
    } for p, s in states.items()},
        "failure_taxonomy": FAILURE_TAXONOMY}


@tool("get_llm_provider_health", "llm",
      "Full LLM provider health: per-provider model-catalog validation "
      "(live model count, configured/matched/missing model ids, selected "
      "fallback, timeouts) + circuit states.",
      label="Checking provider health")
def _get_llm_provider_health(args):
    validation = llm_service.validate_models(refresh_if_stale=True)
    states = llm_service.provider_states()
    providers = {}
    for provider, entry in (validation.get("providers") or {}).items():
        providers[provider] = {
            "status": entry.get("status"),
            "circuit_state": (states.get(provider) or {}).get("state"),
            "live_models": entry.get("models_found"),
            "configured_models": entry.get("configured"),
            "matched_models": entry.get("matched"),
            "missing_models": entry.get("missing_models"),
            "selected_fallback": entry.get("selected_fallback"),
            "catalog_age_s": entry.get("catalog_age_s"),
            "catalog_cached": entry.get("catalog_cached"),
            "timed_out": entry.get("timed_out"),
            "error": entry.get("error"),
        }
    return {"providers": providers, "routes": validation.get("routes")}


@tool("get_llm_model_catalog", "llm",
      "The LIVE model catalog of one provider (from /v1/models, TTL-cached "
      "8 min): model count, cached age, and the id list (capped).",
      args={"provider": {"type": "string", "description": "provider name",
                         "required": True}},
      label="Checking model catalog")
def _get_llm_model_catalog(args):
    provider = str(args.get("provider") or "").strip().lower()
    if provider not in PROVIDERS:
        return {"error": f"unknown provider '{provider}'", "known_providers": list(PROVIDERS)}
    catalog = llm_service.get_model_catalog(provider)
    ids = sorted(catalog.get("ids") or [])
    return {"provider": provider, "live_models": len(ids),
            "catalog_age_s": catalog.get("age_s"),
            "cached": catalog.get("cached"), "timed_out": catalog.get("timed_out"),
            "error": catalog.get("error"),
            "model_ids_sample": ids[:200]}


@tool("get_configured_models", "llm",
      "The model ids this deployment CONFIGURES on one provider, and which "
      "of them matched the live catalog (validation status).",
      args={"provider": {"type": "string", "description": "provider name",
                         "required": True}},
      label="Checking configured models")
def _get_configured_models(args):
    provider = str(args.get("provider") or "").strip().lower()
    validation = llm_service.last_validation() or llm_service.validate_models()
    entry = (validation.get("providers") or {}).get(provider)
    if entry is None:
        return {"error": f"unknown provider '{provider}'",
                "known_providers": list((validation.get("providers") or {}).keys())}
    return {"provider": provider, "status": entry.get("status"),
            "configured_models": entry.get("configured"),
            "matched_models": entry.get("matched"),
            "missing_models": entry.get("missing_models"),
            "live_models": entry.get("models_found"),
            "selected_fallback": entry.get("selected_fallback")}


@tool("get_llm_failures", "llm",
      "Recent LLM failures from the persisted request log (today): grouped "
      "by status and provider, with the latest failing requests and the "
      "failure taxonomy.", label="Checking LLM failures")
def _get_llm_failures(args):
    from datetime import datetime, timezone
    from services.operator.memory import _conn
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    by_status, by_provider, latest = {}, {}, []
    try:
        with _conn() as conn:
            for row in conn.execute(
                    "SELECT status, provider, model, agent, symbol, reason, "
                    "ts_iso, latency_ms FROM llm_request_log "
                    "WHERE kind='send' AND ok=0 AND date=? "
                    "ORDER BY id DESC LIMIT 200", (today,)):
                by_status[row["status"]] = by_status.get(row["status"], 0) + 1
                by_provider[row["provider"]] = by_provider.get(row["provider"], 0) + 1
                if len(latest) < 25:
                    latest.append(dict(row))
    except Exception as exc:  # noqa: BLE001 — table may not exist yet
        return {"error": f"request log unavailable: {exc}"}
    return {"today": today, "failures_by_status": by_status,
            "failures_by_provider": by_provider, "latest_failures": latest,
            "failure_taxonomy": FAILURE_TAXONOMY}


@tool("get_llm_usage", "llm",
      "LLM usage + quota status: requests this cycle/last hour/today, 429 "
      "count, cache hits/misses, calls by agent and by model, circuit "
      "states with cooldowns.", label="Checking LLM usage")
def _get_llm_usage(args):
    return llm_service.usage_status()


@tool("get_circuit_breaker_states", "llm",
      "Per-provider circuit breaker states (CLOSED/OPEN/HALF_OPEN) with "
      "cooldown-remaining and per-model circuits.", label="Checking circuits")
def _get_circuit_breaker_states(args):
    import time
    states = llm_service.provider_states()
    now = time.monotonic()
    out = {}
    for provider, s in states.items():
        out[provider] = {
            "breaker": s.get("circuit"),
            "state": s.get("state"),
            "failure_count": s.get("failure_count"),
            "open_until": s.get("open_until"),
            "quota_cooldown_remaining_s": s.get("quota_cooldown_remaining_s"),
            "auth_cooldown_remaining_s": s.get("auth_cooldown_remaining_s"),
            "model_circuits_open_s": {
                f"{p}/{m}": round(until - now, 0)
                for (p, m), until in llm_service._model_unavailable_until.items()
                if p == provider and until > now},
        }
    return out


@tool("get_provider_fallback_chain", "llm",
      "The application-owned model fallback chain: UnoRouter configured "
      "candidates vs the catalog-VERIFIED effective chain (with selected "
      "fallback) and the per-task routing table.",
      label="Checking fallback chain")
def _get_provider_fallback_chain(args):
    return {
        "unorouter_configured_chain": llm_service.unorouter_model_chain(),
        "unorouter_effective_chain": llm_service.unorouter_effective_chain(),
        "task_routes": llm_service.llm_routes(),
        "active_chain_providers": llm_service.active_chain_providers(),
        "note": "the APPLICATION owns model selection: cache -> uno primary -> "
                "uno fallbacks (catalog-verified) -> Groq -> cached replay -> "
                "deterministic fallback",
    }


@tool("get_recent_llm_requests", "llm",
      "The most recent persisted LLM request events (sends, cache hits, "
      "replays, misses) with provider/model/agent/symbol/reason/status. "
      "Metadata only — no prompts, responses or keys.",
      args={"limit": {"type": "integer", "description": "max rows (default 25)"}},
      label="Checking recent LLM requests")
def _get_recent_llm_requests(args):
    from services.operator.memory import _conn
    limit = min(int(args.get("limit") or 25), 100)
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT ts_iso, kind, provider, model, agent, symbol, reason, "
                "ok, status FROM llm_request_log ORDER BY id DESC LIMIT ?",
                (limit,)).fetchall()
        return {"requests": [dict(r) for r in rows], "count": len(rows)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"request log unavailable: {exc}"}
