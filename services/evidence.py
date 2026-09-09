"""
services/evidence.py

Evidence-state semantics and the decision-quality gate.

Two concepts that were previously conflated are now explicit:

  EVIDENCE STATE (per analyst agent) — did this agent actually have
  evidence to analyze?
      AVAILABLE   the agent analyzed real evidence and produced a verdict
                  (its verdict may legitimately be NEUTRAL)
      UNAVAILABLE the underlying data or provider was not available
      ERROR       the agent (LLM) failed
      STALE       data existed but is outdated (e.g. the newest daily bar
                  is far in the past)
      OFF         deliberately disabled by configuration (not a failure)

  NEUTRAL is a VERDICT, not a state. It only means: "the agent
  successfully analyzed the available evidence and concluded that the
  evidence is directionally neutral." A failed agent never yields
  NEUTRAL evidence — its stance and confidence are null and its
  evidence state is UNAVAILABLE or ERROR.

  EVIDENCE QUALITY (per symbol, per cycle):
      SUFFICIENT   technical evidence AVAILABLE and nothing else missing
      DEGRADED     technical AVAILABLE but news / fundamentals / debate
                   evidence is missing (trading still allowed — the CIO
                   is told exactly what is missing and must be
                   conservative)
      INSUFFICIENT technical evidence itself is not AVAILABLE (missing,
                   stale or failed) — opening trades is BLOCKED. This is
                   the decision-quality prerequisite, deliberately
                   SEPARATE from portfolio risk: having enough buying
                   power for a $9,999 notional never justifies a trade
                   when the evidence backing it is absent.

Confidence: no numeric confidence formula is invented here (none exists
in the project). Instead, degraded evidence is (a) stated explicitly to
the debate, the risk agent and the CIO, and (b) missing technical
evidence hard-blocks execution.
"""

import logging
from datetime import datetime, timezone

from config import settings

logger = logging.getLogger("evidence_service")

EVIDENCE_STATES = ("AVAILABLE", "UNAVAILABLE", "ERROR", "STALE", "OFF")

# LLM statuses that mean "no evidence could be produced" (data/provider
# side) rather than "the agent itself failed".
_UNAVAILABLE_LLM_STATUSES = {
    "SKIPPED_NO_DATA", "NOT_CONFIGURED", "PROVIDER_QUOTA_EXCEEDED",
    "MODEL_NOT_FOUND", "AUTH_ERROR", "NETWORK_ERROR",
}
_OFF_LLM_STATUSES = {"SKIPPED_DISABLED", "SKIPPED_DETERMINISTIC"}


def state_for_llm_status(llm_status) -> str:
    """Maps an LLM router status to an evidence state for agent reports.

    Provider-side unavailability (no key, quota, model gone, auth, network)
    is UNAVAILABLE — the agent could not reach its reasoning service. Any
    other failure (provider error, unparseable response) is ERROR.
    """
    if llm_status in _UNAVAILABLE_LLM_STATUSES:
        return "UNAVAILABLE"
    if llm_status in _OFF_LLM_STATUSES:
        return "OFF"
    return "ERROR"

# The deterministic technical analytics are the mandatory evidence
# backbone of every decision.
_CRITICAL_AGENTS = ("technical",)
# Analyst agents whose absence degrades (but does not block) decisions.
_OPTIONAL_AGENTS = ("news", "fundamentals", "debate")


def _state_from_report(report) -> str:
    """Derives an agent's evidence state from its report."""
    if not report:
        return "OFF"
    explicit = report.get("evidence_status")
    if explicit in EVIDENCE_STATES:
        # STALE is computed centrally (needs the bar date); an agent never
        # claims it directly.
        if explicit != "STALE":
            return explicit
    if not report.get("error"):
        return "AVAILABLE"
    # Deliberately unconfigured fundamentals (FUNDAMENTALS_PROVIDER=none)
    # is configuration, not a failure — checked before the generic
    # unavailability mapping on llm_status.
    if "NO_PROVIDER_CONFIGURED" in str(report.get("error", "")):
        return "OFF"
    llm_status = report.get("llm_status")
    if llm_status in _UNAVAILABLE_LLM_STATUSES:
        return "UNAVAILABLE"
    if llm_status in _OFF_LLM_STATUSES:
        return "OFF"
    return "ERROR"


def _bar_age_days(indicators: dict):
    """Age of the newest daily bar, in whole calendar days, or None."""
    if not indicators:
        return None
    raw = indicators.get("last_bar_date")
    if not raw:
        return None
    try:
        # pandas timestamps stringify like "2026-09-08 00:00:00+00:00";
        # accept a plain date prefix too.
        text = str(raw).split("+")[0].strip()
        bar_dt = datetime.fromisoformat(text)
        if bar_dt.tzinfo is None:
            bar_dt = bar_dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - bar_dt).total_seconds() / 86400.0
    except (ValueError, TypeError):
        return None


def _technical_stale(indicators: dict):
    """(is_stale, age_days) for the technical evidence, from the newest
    daily bar's age. Daily bars legitimately skip weekends/holidays, so
    the threshold (EVIDENCE_STALE_DAYS, default 5 calendar days) is well
    above a long weekend."""
    age = _bar_age_days(indicators)
    if age is None:
        return False, None
    return age > float(settings.EVIDENCE_STALE_DAYS), age


def _stance_key(agent: str) -> str:
    return "sentiment" if agent == "news" else "signal"


def _agent_line(agent: str, report, indicators=None) -> dict:
    """One agent's compact evidence entry: {state, detail}."""
    state = _state_from_report(report)
    detail = None
    if report:
        if report.get("error"):
            detail = str(report["error"])[:200]
        if agent == "technical" and state == "AVAILABLE" and indicators:
            stale, age = _technical_stale(indicators)
            if stale:
                state = "STALE"
                detail = (
                    f"newest daily bar is {age:.1f} days old "
                    f"(> {settings.EVIDENCE_STALE_DAYS}); evidence is outdated"
                )
    return {"state": state, "detail": detail}


def assess(tech_report, news_report, fundamentals_report=None,
           debate_report=None, indicators=None) -> dict:
    """Builds the evidence-quality snapshot for one symbol.

    Args:
        tech_report / news_report / fundamentals_report / debate_report:
            the agent reports from the current cycle (None = the stage
            did not run — treated as OFF, not as missing evidence).
        indicators: the deterministic analytics bundle (for staleness).

    Returns:
        {
          "agents": {"technical": {"state", "detail"}, ...},
          "quality": "SUFFICIENT" | "DEGRADED" | "INSUFFICIENT",
          "available": [...], "missing": [...], "off": [...],
          "trade_allowed": bool,
          "reasons": [str, ...],
          "staleness": {"last_bar_date", "days_old"} | None,
        }
    """
    reports = {
        "technical": tech_report,
        "news": news_report,
        "fundamentals": fundamentals_report,
        "debate": debate_report,
    }
    agents = {name: _agent_line(name, report, indicators if name == "technical" else None)
              for name, report in reports.items()}

    available = [n for n, e in agents.items() if e["state"] == "AVAILABLE"]
    missing = [n for n, e in agents.items() if e["state"] in ("UNAVAILABLE", "ERROR", "STALE")]
    off = [n for n, e in agents.items() if e["state"] == "OFF"]

    reasons = []
    quality = "SUFFICIENT"

    # 1. Critical evidence (technical) must be AVAILABLE — else no trade.
    for name in _CRITICAL_AGENTS:
        entry = agents[name]
        if entry["state"] == "OFF":
            reasons.append(f"{name} evidence not produced (stage disabled/unconfigured)")
        elif entry["state"] != "AVAILABLE":
            label = entry["state"]
            detail = f" ({entry['detail']})" if entry.get("detail") else ""
            reasons.append(f"{name} evidence {label}{detail}")
        if entry["state"] != "AVAILABLE":
            quality = "INSUFFICIENT"

    # 2. Optional evidence missing degrades the decision but does not
    #    block it (the CIO is told exactly what is missing).
    for name in _OPTIONAL_AGENTS:
        entry = agents[name]
        if entry["state"] in ("UNAVAILABLE", "ERROR", "STALE"):
            label = entry["state"]
            detail = f" ({entry['detail']})" if entry.get("detail") else ""
            reasons.append(f"{name} evidence {label}{detail}")
            if quality == "SUFFICIENT":
                quality = "DEGRADED"

    stale, age = _technical_stale(indicators) if indicators else (False, None)
    staleness = None
    if indicators and indicators.get("last_bar_date"):
        staleness = {"last_bar_date": indicators.get("last_bar_date"),
                     "days_old": round(age, 2) if age is not None else None}

    trade_allowed = quality != "INSUFFICIENT"
    return {
        "agents": agents,
        "quality": quality,
        "available": available,
        "missing": missing,
        "off": off,
        "trade_allowed": trade_allowed,
        "reasons": reasons,
        "staleness": staleness,
    }


def context_lines(evidence: dict) -> str:
    """Human-readable evidence block for LLM prompts (debate / risk /
    CIO). Tells the model exactly which evidence is real and which is
    missing — unavailable evidence is MISSING information, never neutral
    (and never quietly bullish/bearish)."""
    if not evidence:
        return ""
    lines = []
    for name, entry in (evidence.get("agents") or {}).items():
        label = name.upper()
        state = entry.get("state", "UNKNOWN")
        detail = entry.get("detail")
        lines.append(f"{label}: {state}" + (f" — {detail}" if detail else ""))
    lines.append(f"Evidence quality: {evidence.get('quality')}")
    if evidence.get("missing"):
        lines.append(
            "Missing evidence is MISSING INFORMATION — it is NOT neutral, NOT "
            "bearish and NOT bullish. Argue/decide only from the evidence "
            "marked AVAILABLE, and be explicit about what is not known."
        )
    return "\n".join(lines)
