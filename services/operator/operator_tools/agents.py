"""
operator_tools/agents.py — agent/decision/cycle tools + the dependency graph.

The graph mirrors main.py's actual pipeline (the repository is the source of
truth): market data feeds technical; the news worker feeds the cached news
verdict; fundamentals data feeds the fundamentals agent; technical+news+
fundamentals (+optional debate) feed the CIO through the evidence gate, with
the deterministic risk engine gating before/alongside; CIO+risk feed
execution; execution feeds memory.
"""

from services.operator.operator_tools._registry import tool
from services.operator.schemas import sanitize

AGENTS = ("technical", "news", "fundamentals", "debate", "risk", "cio",
          "execution", "memory", "market_data")

# component -> {"upstream": [...], "downstream": [...], "role": str}
DEPENDENCY_GRAPH = {
    "alpaca": {"upstream": [], "downstream": ["market_data", "realtime"],
               "role": "broker + market data source (paper account)"},
    "market_data": {"upstream": ["alpaca"], "downstream": ["technical"],
                    "role": "normalized bars/snapshot bundles"},
    "technical": {"upstream": ["market_data"], "downstream": ["evidence", "cio"],
                  "role": "deterministic indicator engine + setup verdict"},
    "news_worker": {"upstream": ["alpaca"], "downstream": ["news"],
                    "role": "independent news fetch/dedup/analysis worker"},
    "news": {"upstream": ["news_worker"], "downstream": ["evidence", "cio"],
             "role": "cached news intelligence reader"},
    "fundamentals_data": {"upstream": [], "downstream": ["fundamentals"],
                          "role": "financial DATA provider (e.g. a vendor API)"},
    "fundamentals": {"upstream": ["fundamentals_data"], "downstream": ["evidence", "cio"],
                     "role": "fundamentals interpretation (needs real data; the "
                             "LLM reasoning provider is NOT a data provider)"},
    "debate": {"upstream": ["technical", "news", "fundamentals"],
               "downstream": ["cio"], "role": "optional bull/bear debate (off by default)"},
    "risk_engine": {"upstream": [], "downstream": ["execution"],
                    "role": "deterministic hard risk gate (LLM cannot override)"},
    "risk": {"upstream": ["technical", "news", "risk_engine"],
             "downstream": ["execution", "cio"],
             "role": "risk agent (deterministic gate first, LLM reasoning supplementary)"},
    "evidence": {"upstream": ["technical", "news", "fundamentals", "debate"],
                 "downstream": ["cio"], "role": "evidence quality gate"},
    "cio": {"upstream": ["technical", "news", "fundamentals", "debate", "risk", "evidence"],
            "downstream": ["execution"], "role": "final decision agent (HOLD fail-safe)"},
    "execution": {"upstream": ["cio", "risk", "risk_engine", "evidence"],
                  "downstream": ["memory"],
                  "role": "validated PAPER order submission"},
    "memory": {"upstream": ["execution"], "downstream": ["cio"],
               "role": "decision history + agent accuracy feedback"},
    "llm": {"upstream": [], "downstream": ["cio", "risk", "news_worker", "fundamentals"],
            "role": "LLM provider chain (unorouter -> groq -> cache)"},
}


def _cycles(limit):
    import main
    return list(main.cycle_history)[:max(1, min(int(limit), 30))]


def component_status(name: str):
    """Unified status for a service/component name (used by
    get_service_status and the diagnostic engine)."""
    from services import health_service, llm_service, fundamentals_service
    from services.operator.operator_tools.system import _latest_cycle
    from config import settings

    startup = health_service.get_startup_report()
    rows = {r["component"].lower().replace(" ", "_").replace("-", "_"): r
            for r in (startup.get("rows") or [])}
    key = str(name or "").strip().lower()

    llm_providers = {"unorouter", "groq", "nvidia", "gemini", "openrouter"}
    if key in llm_providers:
        states = llm_service.provider_states().get(key, {})
        validation = (llm_service.last_validation() or {}).get("providers", {}).get(key, {})
        deps = DEPENDENCY_GRAPH.get("llm", {}).get("downstream", [])
        return {
            "service": key,
            "status": states.get("state", "UNKNOWN"),
            "circuit": states.get("circuit"),
            "detail": states.get("detail"),
            "configured_models": validation.get("configured"),
            "matched_models": validation.get("matched"),
            "missing_models": validation.get("missing_models"),
            "live_catalog_models": validation.get("models_found"),
            "last_success": states.get("last_success"),
            "last_failure": states.get("last_failure"),
            "dependencies": [],
            "dependents": deps,
        }

    row_keys = {
        "alpaca_account": "alpaca_account", "market_data": "alpaca_market_data",
        "indicators": "indicators", "realtime": "real_time_stream",
        "fundamentals": "fundamentals", "news_intelligence": "news_intelligence",
    }
    if key == "news_intelligence" or key == "news":
        row = rows.get("news_intelligence") or rows.get("news intelligence")
    elif key in row_keys:
        row = rows.get(row_keys[key])
    else:
        row = rows.get(key)
    if row is None and key == "memory":
        from services import memory_service
        return {"service": "memory", "status": "READY",
                "detail": f"SQLite at {settings.MEMORY_DB_PATH}",
                "dependencies": ["execution"], "dependents": ["cio"]}
    if row is None:
        return None

    graph_key = {"alpaca_account": "alpaca", "market_data": "market_data",
                 "indicators": "market_data", "realtime": "alpaca"}.get(key, key)
    graph = DEPENDENCY_GRAPH.get(graph_key, {})
    out = {"service": key, "status": row.get("status"),
           "detail": row.get("detail"),
           "dependencies": graph.get("upstream", []),
           "dependents": graph.get("downstream", [])}
    if key == "fundamentals":
        cfg = fundamentals_service.provider_config()
        out["configured_provider"] = cfg.get("provider")
        out["registered_providers"] = fundamentals_service.available_providers()
        out["llm_reasoning_route"] = llm_service.route_info("fundamentals")
        out["architecture_note"] = (
            "A fundamentals DATA provider supplies financial metrics; the LLM "
            "(e.g. NVIDIA) only INTERPRETS that data. An LLM model id is not "
            "a financial-data provider.")
    return out


@tool("get_agent_status", "agents",
      "Per-symbol stage status of one agent from the latest trading cycle "
      "(OK/ERROR/UNAVAILABLE/SKIPPED per symbol + counts).",
      args={"agent": {"type": "string", "description": "one of: " + ", ".join(AGENTS),
                      "required": True}},
      label="Checking agent status")
def _get_agent_status(args):
    agent = str(args.get("agent") or "").strip().lower()
    if agent not in AGENTS:
        return {"error": f"unknown agent '{agent}'", "known_agents": list(AGENTS)}
    cycles = _cycles(1)
    if not cycles:
        return {"agent": agent, "status": "NO_CYCLE_YET",
                "detail": "no trading cycle has run in this process yet"}
    record = cycles[0]
    per_symbol = {}
    for symbol, stages in (record.get("agent_status") or {}).items():
        per_symbol[symbol] = stages.get(agent)
    counts = {}
    for v in per_symbol.values():
        counts[v] = counts.get(v, 0) + 1
    errors = [{"symbol": s, "message": e.get("message")}
              for e in (record.get("errors") or []) if e.get("agent") == agent][:20]
    return {"agent": agent, "cycle_id": record.get("id"),
            "cycle_status": record.get("status"),
            "per_symbol": per_symbol, "status_counts": counts,
            "recent_errors": errors,
            "agent_results": (record.get("agent_results") or {}).get(agent)}


@tool("get_all_agent_statuses", "agents",
      "Every agent's stage status from the latest trading cycle.",
      label="Checking all agents")
def _get_all_agent_statuses(args):
    cycles = _cycles(1)
    if not cycles:
        return {"status": "NO_CYCLE_YET"}
    record = cycles[0]
    results = record.get("agent_results") or {}
    return {"cycle_id": record.get("id"), "cycle_status": record.get("status"),
            "agent_results": results,
            "agent_status_matrix": record.get("agent_status")}


@tool("get_latest_agent_decisions", "agents",
      "The CIO decisions from the latest trading cycle (decision, confidence, "
      "notional, reasoning, evidence quality, blocked reason).",
      label="Inspecting latest decisions")
def _get_latest_agent_decisions(args):
    cycles = _cycles(1)
    if not cycles:
        return {"status": "NO_CYCLE_YET"}
    record = cycles[0]
    return {"cycle_id": record.get("id"), "triggered_by": record.get("triggered_by"),
            "decisions": record.get("decisions") or [],
            "orders": record.get("orders") or []}


@tool("get_agent_decision_history", "agents",
      "Decision history for one agent across recent cycles (from in-memory "
      "cycle records).",
      args={"agent": {"type": "string", "description": "agent name", "required": True},
            "limit": {"type": "integer", "description": "max cycles (default 10)"}},
      label="Checking decision history")
def _get_agent_decision_history(args):
    agent = str(args.get("agent") or "").strip().lower()
    limit = min(int(args.get("limit") or 10), 30)
    history = []
    for record in _cycles(limit):
        if agent == "cio":
            entries = record.get("decisions") or []
        else:
            entries = [e for e in (record.get("errors") or [])
                       if e.get("agent") == agent][:10]
            if not entries:
                stages = {s: (st or {}).get(agent)
                          for s, st in (record.get("agent_status") or {}).items()}
                entries = [{"cycle_id": record.get("id"), "per_symbol": stages}]
        history.append({"cycle_id": record.get("id"),
                        "status": record.get("status"),
                        "at": record.get("started_at"), "entries": entries})
    return {"agent": agent, "cycles": history, "count": len(history)}


@tool("get_latest_trading_cycle", "agents",
      "The FULL latest trading cycle record: per-agent stage matrix, every "
      "decision with reasoning and evidence quality, orders, structured "
      "errors, LLM usage and degradation label.",
      label="Inspecting latest trading cycle")
def _get_latest_trading_cycle(args):
    cycles = _cycles(1)
    if not cycles:
        return {"status": "NO_CYCLE_YET"}
    return sanitize(cycles[0])


@tool("get_trading_cycle_history", "agents",
      "Summaries of the most recent trading cycles (status, decisions, "
      "orders, errors, duration).",
      args={"limit": {"type": "integer", "description": "max cycles (default 10)"}},
      label="Checking cycle history")
def _get_trading_cycle_history(args):
    limit = min(int(args.get("limit") or 10), 30)
    out = []
    for record in _cycles(limit):
        out.append({
            "cycle_id": record.get("id"), "at": record.get("started_at"),
            "triggered_by": record.get("triggered_by"),
            "status": record.get("status"), "status_label": record.get("status_label"),
            "symbols": record.get("symbols_processed"),
            "decisions": len(record.get("decisions") or []),
            "orders": len(record.get("orders") or []),
            "errors": len(record.get("errors") or []),
            "llm_failures": record.get("llm_failures"),
            "duration_s": record.get("duration_s"),
        })
    return {"cycles": out, "count": len(out)}


@tool("get_agent_dependencies", "agents",
      "The agent dependency graph (upstream/downstream) — either for one "
      "component or the whole graph.",
      args={"component": {"type": "string", "description": "optional component name"}},
      label="Tracing dependencies")
def _get_agent_dependencies(args):
    name = str(args.get("component") or "").strip().lower()
    if name:
        entry = DEPENDENCY_GRAPH.get(name)
        if entry is None:
            return {"error": f"unknown component '{name}'",
                    "known_components": sorted(DEPENDENCY_GRAPH)}
        return {"component": name, **entry}
    return {"graph": DEPENDENCY_GRAPH,
            "note": "market_data -> technical; news_worker -> news; "
                    "fundamentals_data -> fundamentals; technical+news+"
                    "fundamentals(+debate) -> evidence -> CIO (with the "
                    "deterministic risk engine); CIO+risk -> execution -> memory"}


@tool("get_agent_errors", "agents",
      "Recent ERROR log entries for one agent.",
      args={"agent": {"type": "string", "description": "agent name", "required": True},
            "limit": {"type": "integer", "description": "max entries (default 15)"}},
      label="Checking agent errors")
def _get_agent_errors(args):
    import main
    agent = str(args.get("agent") or "").strip().lower()
    limit = min(int(args.get("limit") or 15), 60)
    entries = [{"symbol": e.get("symbol"), "message": e.get("message"),
                "timestamp": e.get("timestamp")}
              for e in list(main.agent_logs)
              if e.get("level") == "ERROR" and e.get("agent") == agent][:limit]
    return {"agent": agent, "errors": entries, "count": len(entries)}
