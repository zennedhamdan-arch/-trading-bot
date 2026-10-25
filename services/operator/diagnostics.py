"""
services/operator/diagnostics.py

The diagnostic engine — correlation, not log repetition.

Given a focus area (or auto-detected from the question), it gathers live
evidence from: health, startup rows, recent errors/warnings, agent statuses
from the latest cycle, LLM provider/circuit/catalog state, configuration
warnings, the dependency graph, and persisted failure history.

Then it classifies findings:

  DIRECT CAUSE     the component's own failure/config error (evidence attached)
  DOWNSTREAM EFFECT components that fail BECAUSE an upstream input is missing
  SECONDARY WARNING warnings that are consequences, not causes

Rules of honesty:
  * a finding is only labeled ROOT CAUSE when the evidence chain supports it
    (component's own error + upstream healthy);
  * when the chain cannot be established the engine says
    "Insufficient evidence" and states exactly what is missing;
  * independent problems are kept SEPARATE.

Specialized detectors:
  * fundamentals provider: configured vs registered providers, including the
    LLM-model-id-used-as-data-provider confusion (an LLM is a REASONING
    provider, never a financial DATA source);
  * UnoRouter model-catalog mismatch: endpoint reachable, catalog fetched,
    N live models, configured ids, matched/missing — and whether the failure
    happened BEFORE any inference request (catalog/validation stage) or
    DURING inference;
  * LLM failure taxonomy (quota/auth/timeout/model/breaker/malformed/config).
"""

from services import fundamentals_service, health_service, llm_service
from services.operator.operator_tools.agents import DEPENDENCY_GRAPH
from services.operator.operator_tools.llm import FAILURE_TAXONOMY
from services.operator.schemas import sanitize

_LLM_MODEL_ID_HINTS = ("/", ":free", "deepseek", "llama", "gpt-", "glm",
                       "qwen", "gemini", "claude", "kimi", "mistral")


def _downstream_of(component: str) -> list:
    seen, stack = set(), [component]
    while stack:
        current = stack.pop()
        for dep in DEPENDENCY_GRAPH.get(current, {}).get("downstream", []):
            if dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return sorted(seen)


def _cycle_state():
    import main
    record = main.cycle_history[0] if main.cycle_history else None
    return record


# ---------------------------------------------------------------------------
# Detectors — each returns a list of findings (possibly empty)
# ---------------------------------------------------------------------------

def _detect_fundamentals_issue():
    """The known architectural distinction: DATA provider vs LLM reasoning."""
    findings = []
    cfg = fundamentals_service.provider_config()
    configured = str(cfg.get("provider") or "none")
    registered = fundamentals_service.available_providers()
    if configured in ("", "none"):
        findings.append({
            "component": "fundamentals",
            "classification": "CONFIGURATION (by design)",
            "severity": "info",
            "summary": "No fundamentals DATA provider is configured "
                       "(FUNDAMENTALS_PROVIDER=none).",
            "evidence": [f"configured provider = '{configured}'",
                         f"registered providers = {registered}",
                         "fundamentals agent reports DATA_UNAVAILABLE / "
                         "NO_PROVIDER_CONFIGURED (explicit, not an error)"],
            "impact": "The fundamentals agent supplies no verdict; the CIO "
                      "decides on technical + news + risk evidence "
                      "(evidence quality may be DEGRADED, never fabricated).",
            "recommendation": "To enable fundamentals: register a real "
                              "financial-DATA provider (subclass "
                              "FundamentalsProvider, vendor's official API) "
                              "and keep the LLM (e.g. NVIDIA) as the "
                              "REASONING layer only. Never point "
                              "FUNDAMENTALS_PROVIDER at an LLM model id.",
            "downstream": _downstream_of("fundamentals"),
        })
        return findings
    if configured not in registered:
        looks_like_llm = any(h in configured.lower() for h in _LLM_MODEL_ID_HINTS)
        finding = {
            "component": "fundamentals",
            "classification": "DIRECT CAUSE (configuration)",
            "severity": "error",
            "summary": (f"Configured fundamentals provider '{configured}' is "
                        f"not registered."),
            "evidence": [f"configured provider = '{configured}'",
                         f"registered providers = {registered}",
                         "provider lookup failed -> fundamentals data "
                         "unavailable"],
            "impact": "Fundamentals cannot produce a verdict; downstream "
                      "intelligence (evidence quality, CIO context) degrades.",
            "downstream": _downstream_of("fundamentals"),
            "recommendation": "Set FUNDAMENTALS_PROVIDER to a registered "
                              "provider name (or 'none') and implement a real "
                              "financial-data provider.",
        }
        if looks_like_llm:
            finding["architectural_note"] = (
                f"'{configured}' looks like an LLM MODEL id. An LLM (NVIDIA, "
                "Groq, ...) is a REASONING provider; a fundamentals DATA "
                "provider must supply actual financial metrics "
                "(revenue, EPS, ...). The correct architecture is: "
                "financial data source -> normalized fundamentals -> LLM "
                "reasoning -> structured assessment. The LLM cannot invent "
                "financial data, and the operator will not either.")
            finding["llm_reasoning_route"] = sanitize(
                llm_service.route_info("fundamentals"))
        findings.append(finding)
    return findings


def _detect_unorouter_catalog_issue():
    """Catalog mismatch: /v1/models works but configured ids are absent."""
    findings = []
    validation = llm_service.last_validation() or {}
    entry = (validation.get("providers") or {}).get("unorouter") or {}
    if not entry:
        return findings
    missing = entry.get("missing_models") or []
    states = llm_service.provider_states().get("unorouter") or {}
    if entry.get("status") == "NO_USABLE_MODEL":
        findings.append({
            "component": "unorouter",
            "classification": "DIRECT CAUSE (model catalog mismatch)",
            "severity": "error",
            "summary": "UnoRouter endpoint is reachable but NONE of the "
                       "configured model ids exist in the live /v1/models "
                       "catalog.",
            "evidence": [
                f"catalog fetch OK: {entry.get('models_found')} live models "
                f"(catalog age {entry.get('catalog_age_s')}s)",
                f"configured ids = {entry.get('configured')}",
                "matched ids = [] (zero matches)",
                "failure occurred at the VALIDATION stage — BEFORE any "
                "inference request was sent",
                f"provider circuit state = {states.get('state')}"],
            "impact": "UnoRouter cannot serve any request as configured; the "
                      "chain fails over to the next provider (Groq) or cached "
                      "results.",
            "recommendation": "Verify the model ids against the provider's "
                              "live /v1/models response (the app already "
                              "compares exact ids and never substitutes). "
                              "Update UNOROUTER_PRIMARY_MODEL / "
                              "UNOROUTER_FALLBACK_MODELS with ids that are "
                              "actually present.",
            "downstream": _downstream_of("llm"),
        })
    elif missing and entry.get("status") == "READY":
        effective = llm_service.unorouter_effective_chain()
        findings.append({
            "component": "unorouter",
            "classification": "SECONDARY WARNING (model catalog mismatch)",
            "severity": "warning",
            "summary": f"Configured UnoRouter model id(s) missing from the "
                       f"live catalog: {missing}. The provider still has "
                       f"usable models.",
            "evidence": [
                f"catalog fetch OK: {entry.get('models_found')} live models",
                f"configured = {entry.get('configured')}",
                f"matched = {entry.get('matched')}",
                f"effective chain in use = {effective.get('chain')}",
                f"selected fallback = {effective.get('selected_fallback')}",
                "missing ids are skipped WITHOUT an inference request "
                "(no quota consumed, one warning per catalog period)"],
            "impact": "Reduced model redundancy only; requests route to the "
                      "catalog-verified chain automatically.",
            "recommendation": "Optionally replace the missing ids with live "
                              "ones from /v1/models. Do NOT blindly change "
                              "model ids — check the catalog first.",
        })
    if entry.get("timed_out"):
        findings.append({
            "component": "unorouter",
            "classification": "DIRECT CAUSE (timeout)",
            "severity": "warning",
            "summary": "The UnoRouter model-catalog fetch timed out.",
            "evidence": [f"error = {entry.get('error')}",
                         "bounded by LLM_CATALOG_TIMEOUT_SECONDS; validation "
                         "continued without it"],
            "impact": "Model ids could not be verified this period; the "
                      "runtime 404 circuits still protect requests.",
            "recommendation": "Check network latency to api.unorouter.com; "
                              "the catalog retries next TTL window.",
        })
    return findings


def _detect_llm_provider_issues():
    findings = []
    states = llm_service.provider_states()
    validation = (llm_service.last_validation() or {}).get("providers", {})
    for provider in llm_service.active_chain_providers():
        state = states.get(provider) or {}
        circuit = state.get("state")
        if circuit in ("READY", "NOT_CONFIGURED"):
            continue
        taxonomy = FAILURE_TAXONOMY.get(circuit, circuit)
        ventry = validation.get(provider) or {}
        findings.append({
            "component": provider,
            "classification": "DIRECT CAUSE (LLM provider)",
            "severity": "error" if circuit in ("AUTH_ERROR", "QUOTA_EXHAUSTED") else "warning",
            "summary": f"LLM provider '{provider}' is {circuit}.",
            "evidence": [f"circuit state = {circuit}",
                         f"detail = {state.get('detail')}",
                         f"classification = {taxonomy}",
                         f"last failure = {state.get('last_failure')}",
                         f"validated models matched = {ventry.get('matched')}"],
            "impact": "LLM tasks fail over along the chain (other UnoRouter "
                      "models -> Groq -> cached replay -> deterministic "
                      "fallback / HOLD). Trading continues.",
            "recommendation": ("Wait for the cooldown to expire (one controlled "
                               "retry happens automatically)" if circuit == "QUOTA_EXHAUSTED"
                              else "Check the provider credentials/endpoint."),
            "downstream": _downstream_of("llm"),
        })
    return findings


# startup-report row component -> dependency-graph key (for downstream
# impact). Rows not in the map have no graph entry (e.g. REAL-TIME STREAM).
_ROW_TO_GRAPH = {
    "alpaca_account": "alpaca",
    "alpaca_market_data": "market_data",
    "indicators": "market_data",
    "fundamentals": "fundamentals",
    "news_intelligence": "news_worker",
    "real-time_stream": None,
    "unorouter": "llm", "groq": "llm", "nvidia": "llm",
    "gemini": "llm", "openrouter": "llm",
}

# Honest "not a failure" statuses: PENDING = worker still warming up,
# NOT_CONFIGURED/NO_KEYS = deliberate configuration state, DATA_UNAVAILABLE
# = fundamentals' explicit no-provider state (covered by its own detector).
_NOT_FAILURES = {"READY", "PENDING", "NOT_CONFIGURED", "NO_KEYS", "OFF",
                 "DATA_UNAVAILABLE", "DISABLED"}


def _detect_health_degradations():
    """Any failing startup row that isn't already covered above."""
    findings = []
    startup = health_service.get_startup_report()
    covered = {"fundamentals", "unorouter", "groq", "nvidia", "gemini", "openrouter"}
    for row in (startup.get("rows") or []):
        raw_name = str(row.get("component") or "")
        component = raw_name.lower().replace(" ", "_")
        status = row.get("status")
        if component in covered and status != "NO_USABLE_MODEL":
            continue  # circuit/catalog detectors own provider rows
        if component == "fundamentals":
            continue  # dedicated fundamentals detector (incl. DATA_UNAVAILABLE)
        if status in _NOT_FAILURES:
            continue
        graph_key = _ROW_TO_GRAPH.get(component)
        findings.append({
            "component": component,
            "classification": "DIRECT CAUSE (component status)",
            "severity": "error" if status in ("ERROR", "NO_USABLE_MODEL") else "warning",
            "summary": f"{raw_name} is {status}.",
            "evidence": [f"status = {status}",
                         f"detail = {row.get('detail')}"],
            "impact": ("This is a critical component (trading cycle depends "
                       "on it)." if component in ("alpaca_account",
                                                  "alpaca_market_data",
                                                  "indicators")
                      else None),
            "recommendation": None,
            "downstream": _downstream_of(graph_key) if graph_key else [],
        })
    return findings


def _detect_cycle_degradations():
    """Downstream effects visible in the latest cycle (agent stages)."""
    findings = []
    record = _cycle_state()
    if not record:
        return findings
    if record.get("status") == "PARTIAL_ERROR" or record.get("llm_failures"):
        affected = {}
        for symbol, stages in (record.get("agent_status") or {}).items():
            for agent, status in (stages or {}).items():
                if status in ("ERROR", "UNAVAILABLE"):
                    affected.setdefault(agent, []).append(symbol)
        if affected:
            findings.append({
                "component": "trading_cycle",
                "classification": "DOWNSTREAM EFFECT",
                "severity": "warning",
                "summary": f"Latest cycle ({record.get('id')}, triggered by "
                           f"'{record.get('triggered_by')}') degraded: "
                           f"{record.get('status_label') or record.get('status')}.",
                "evidence": [f"affected agent stages: "
                             + ", ".join(f"{a}[{len(s)} symbols]" for a, s in sorted(affected.items())),
                             f"cycle errors: {len(record.get('errors') or [])}",
                             f"LLM failures: {record.get('llm_failures')}"],
                "impact": "Decisions for affected symbols used cached or "
                          "deterministic fallback evidence (never fabricated).",
                "recommendation": "Correlate the affected agents with the "
                                  "direct-cause findings above.",
                "cycle_id": record.get("id"),
            })
    return findings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_diagnosis(focus: str = None) -> dict:
    """Correlated diagnosis. focus: optional area
    (fundamentals|llm|unorouter|health|cycle|all)."""
    detectors = {
        "fundamentals": _detect_fundamentals_issue,
        "llm": _detect_llm_provider_issues,
        "unorouter": _detect_unorouter_catalog_issue,
        "health": _detect_health_degradations,
        "cycle": _detect_cycle_degradations,
    }
    if focus and focus in detectors:
        detector_names = [focus]
    else:
        detector_names = list(detectors)

    findings = []
    for name in detector_names:
        try:
            findings.extend(detectors[name]())
        except Exception as exc:  # noqa: BLE001 — one detector never kills diagnosis
            findings.append({"component": name,
                             "classification": "DIAGNOSTIC ERROR",
                             "severity": "warning",
                             "summary": f"detector '{name}' failed: {exc}"})

    overall = health_service.overall_status(live=True)
    # Split into direct causes vs downstream/secondary, keeping independent
    # problems separate.
    direct = [f for f in findings if str(f.get("classification", "")).startswith("DIRECT")]
    downstream = [f for f in findings if f.get("classification") == "DOWNSTREAM EFFECT"]
    secondary = [f for f in findings if str(f.get("classification", "")).startswith("SECONDARY")]
    other = [f for f in findings if f not in direct + downstream + secondary]

    # Enrich: attach dependency context for the primary direct cause
    root_cause = None
    if direct:
        primary = direct[0]
        # Only call it ROOT CAUSE when upstream inputs are healthy or the
        # finding itself is a configuration/credential fact.
        upstream_healthy = True
        for dep in DEPENDENCY_GRAPH.get(primary.get("component"), {}).get("upstream", []):
            if dep in {f.get("component") for f in direct}:
                upstream_healthy = False
                break
        if upstream_healthy:
            root_cause = primary
        else:
            root_cause = None

    return sanitize({
        "overall_status": overall.get("status"),
        "overall_reasons": overall.get("reasons"),
        "focus": focus or "all",
        "root_cause": root_cause,
        "direct_causes": direct,
        "downstream_effects": downstream,
        "secondary_warnings": secondary,
        "other_findings": other,
        "uncertainty": ("Insufficient evidence to name a single root cause — "
                        "multiple independent problems or incomplete data; see "
                        "direct_causes." if len(direct) > 1 or (direct and not root_cause)
                        else None),
        "when_uncertain": "say 'Insufficient evidence' rather than guessing",
    })
