"""
agents/risk_agent.py

Risk Agent. ALL financial arithmetic — position percentage, max position
validation, position sizing, buying power, concentration caps — is computed
DETERMINISTICALLY by services/risk_gate.py. The LLM (routed via
services/llm_service.py, LLM_RISK_PROVIDER / GROQ_RISK_MODEL by default)
only adds qualitative risk reasoning, and may VETO a trade; it can never
approve a trade or a size the deterministic gate did not already allow.

If no risk model is configured, the agent runs in pure deterministic mode
(llm_status=SKIPPED_DETERMINISTIC) — the gate still works.
If the LLM call fails, the deterministic result stands (reported honestly
with the error) and the cycle degrades to PARTIAL_ERROR — never a fake
verdict, never an unvalidated trade.
"""

import json
import logging
import time

from config import settings
from services import llm_service, risk_gate

logger = logging.getLogger("risk_agent")

SYSTEM_INSTRUCTIONS = """You are a strict portfolio risk manager for a paper trading account.
You will be given a proposed trade, the portfolio state, and a DETERMINISTIC
risk assessment that has ALREADY computed: position sizing limits, position
percentage, buying power, and the maximum notional allowed under the
portfolio's concentration cap. Those computed limits are hard constraints —
you cannot raise them.

Your job is to review the qualitative risk: volatility, drawdown, concentration,
and anything the numbers suggest. You may approve an amount UP TO the computed
maximum, or veto the trade entirely (approved=false). Be conservative when
data is incomplete or signals conflict.

PORTFOLIO RISK vs DECISION QUALITY: the computed maximum notional only states
what the portfolio could afford — it is never an endorsement of the trade. If
the decision-quality report shows the evidence behind the trade is unavailable,
stale or failed, veto (approved=false) or sharply reduce the amount, even when
the portfolio-level math alone would permit it.

Respond ONLY with a single valid JSON object, no markdown fences, no preamble, in this exact shape:
{
  "approved": true | false,
  "max_notional_usd": <float, dollar amount approved for this trade, 0 if not approved>,
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "reasoning": "<one to two sentence explanation>",
  "risk_factors": ["<concise factor, from the data only>"],
  "warnings": ["<condition worth flagging, from the data only>"],
  "recommendations": ["<actionable, conservative recommendation>"]
}
"""


def _risk_status(gate_approved: bool, llm_vetoed: bool, side: str) -> str:
    """Honest risk_status for the structured output:
    APPROVED / VETOED / NOT_PROPOSED (monitoring) / BLOCKED."""
    if side is None:
        return "MONITORING"
    if not gate_approved:
        return "BLOCKED"
    if llm_vetoed:
        return "VETOED"
    return "APPROVED"


def _risk_score(gate: dict, indicators: dict, evidence: dict) -> int:
    """Deterministic 0-100 risk score (higher = riskier), documented:
      +40 base when the deterministic gate does not approve the proposal
         (concentration/cash/buying-power constraint active)
      +0..30 volatility: annualized volatility scaled to 30 at 0.60+
      +0..20 drawdown: 1y max drawdown scaled to 20 at -40%+
      +10 existing-position concentration >= 80% of the position cap
      +10 evidence quality not SUFFICIENT (analysis risk)
    The score never influences any hard check — it is explanatory only."""
    score = 0
    if not gate.get("approved"):
        score += 40
    vol = gate.get("volatility")
    if isinstance(vol, (int, float)) and vol > 0:
        score += int(min(30.0, (float(vol) / 0.60) * 30.0))
    dd = gate.get("max_drawdown")
    if isinstance(dd, (int, float)) and dd < 0:
        score += int(min(20.0, (abs(float(dd)) / 0.40) * 20.0))
    max_pct = float(gate.get("max_position_pct") or 0.0)
    pos_pct = float(gate.get("position_pct") or 0.0)
    if max_pct > 0 and pos_pct >= 0.8 * max_pct:
        score += 10
    if evidence and str(evidence.get("quality") or "").upper() != "SUFFICIENT":
        score += 10
    elif not evidence:
        score += 5   # no evidence snapshot supplied: mild uncertainty
    return max(0, min(100, score))


def _extract_json(text: str) -> dict:
    """Tolerant JSON extraction — kept for local use/testing; the live path
    parses through services.llm_service (same logic)."""
    return llm_service.extract_json(text)


def assess_risk(symbol: str, proposed_side: str, account_summary: dict,
                 existing_position: dict = None, indicators: dict = None,
                 evidence: dict = None) -> dict:
    """
    Args:
        symbol: ticker symbol
        proposed_side: "buy" or "sell"
        account_summary: dict from alpaca_service.get_account_summary()
        existing_position: dict for this symbol from get_open_positions(), or None
        indicators: deterministic analytics (volatility/drawdown evidence)
        evidence: evidence-quality snapshot (services/evidence.py) — PORTFOLIO
            RISK and DECISION QUALITY are separate concepts: this agent
            computes/defends the former and is merely INFORMED about the
            latter (it may veto when evidence quality is insufficient, but
            evidence quality is enforced as its own execution prerequisite
            in the cycle, not inside the portfolio-risk math)

    Returns:
        {
          "agent": "risk",
          "symbol": symbol,
          "approved": bool,           # deterministic gate, possibly vetoed by LLM
          "max_notional_usd": float,  # <= deterministic cap, never above
          "risk_level": "LOW"/"MEDIUM"/"HIGH",
          "reasoning": str,
          "error": str | None,
          "deterministic": {...risk_gate.assess result...},
          "provider": str, "model": str, "llm_status": str, "latency_ms": float | null
        }
    """
    route = llm_service.route_info("risk")
    proposed_side = str(proposed_side).lower() if proposed_side else None
    monitoring = proposed_side is None

    # 1. Deterministic gate — always runs, never depends on any LLM.
    #    Monitoring mode (no proposed order): the gate still computes the
    #    exposure facts for the EXISTING position; nothing is proposed.
    gate = risk_gate.assess(
        symbol, proposed_side or "buy", account_summary,
        existing_position, indicators)

    equity = float(gate.get("equity") or 0.0)
    existing_value = float(gate.get("existing_position_value") or 0.0)
    max_pct = float(gate.get("max_position_pct") or 0.0)
    proposed_notional = (float(gate.get("max_notional_usd") or 0.0)
                         if not monitoring else 0.0)
    post_trade_pct = ((existing_value + proposed_notional) / equity
                      if equity > 0 else 0.0)
    evidence_sources = [
        {"source": "alpaca_account", "fact": "equity/cash/buying_power"},
        {"source": "alpaca_positions", "fact": "existing position exposure"},
        {"source": "deterministic_indicators",
         "fact": "volatility/drawdown/staleness",
         "fresh": bool(not (indicators or {}).get("error"))},
    ]
    if evidence:
        evidence_sources.append({
            "source": "evidence_service",
            "fact": "agent evidence states",
            "quality": evidence.get("quality"),
        })

    risk_factors = list(gate.get("reasons") or [])
    warnings = []
    vol = gate.get("volatility")
    dd = gate.get("max_drawdown")
    if isinstance(vol, (int, float)) and float(vol) > 0.35:
        warnings.append(f"elevated volatility ({float(vol):.2f} annualized)")
    if isinstance(dd, (int, float)) and float(dd) < -0.20:
        warnings.append(f"meaningful drawdown ({float(dd)*100:.1f}% over 1y)")
    if monitoring:
        risk_factors.insert(0, "monitoring existing position (no order proposed)")

    base_result = {
        "agent": "risk",
        "symbol": symbol,
        "approved": (False if monitoring else gate["approved"]),
        "max_notional_usd": (0.0 if monitoring else gate["max_notional_usd"]),
        "risk_level": gate["risk_level"],
        "reasoning": ("; ".join(gate["reasons"]) or (
            f"deterministic gate: within {gate['max_position_pct']*100:.1f}% "
            f"concentration cap and available cash"
        )) if not monitoring else (
            f"position monitoring: existing exposure "
            f"{float(gate.get('position_pct') or 0.0)*100:.1f}% of equity "
            f"(cap {max_pct*100:.1f}%); no order proposed"
        ),
        "error": None,
        "deterministic": gate,
        # --- structured risk output (explanatory; hard checks stay in
        #     services.risk_engine / risk_gate and are NEVER overridden) ---
        "risk_status": _risk_status(gate["approved"], False,
                                    None if monitoring else proposed_side),
        "risk_score": _risk_score(gate, indicators or {}, evidence),
        "risk_factors": risk_factors,
        "position_exposure": {
            "existing_position_value": round(existing_value, 2),
            "existing_position_pct_of_equity": gate.get("position_pct"),
            "max_position_pct": max_pct,
            "monitoring_only": monitoring,
        },
        "proposed_order_exposure": (None if monitoring else {
            "proposed_notional_usd": round(proposed_notional, 2),
            "post_trade_pct_of_equity": round(post_trade_pct, 4),
            "max_position_pct": max_pct,
        }),
        "warnings": warnings,
        "blocking_reasons": ([] if monitoring or gate["approved"]
                             else list(gate.get("reasons") or [])),
        "recommendations": [],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_sources": evidence_sources,
        "provider": route["provider"],
        "model": route["model"],
        "llm_status": "SKIPPED_DETERMINISTIC",
        "latency_ms": None,
    }

    # 2. Optional LLM reasoning (provider/model from config; empty model =
    # pure deterministic mode). A missing key or provider failure comes back
    # from the router as a non-OK status — the deterministic gate stands.
    if not route["model"]:
        base_result["summary_mode"] = "deterministic-only"
        return base_result

    if monitoring:
        context_text = (
            f"Symbol: {symbol}\n"
            f"Mode: POSITION MONITORING (no order is proposed; evaluate the "
            f"existing position's exposure and exit risks)\n"
        )
    else:
        context_text = (
            f"Symbol: {symbol}\n"
            f"Proposed side: {proposed_side}\n"
            f"Portfolio equity: ${gate['equity']:.2f}\n"
            f"Available cash: ${gate['cash']:.2f}\n"
            f"Buying power: ${gate['buying_power']:.2f}\n"
            f"Max position percent allowed: {gate['max_position_pct'] * 100:.1f}%\n"
            f"Existing position in {symbol}: {json.dumps(existing_position) if existing_position else 'None'}\n"
            f"Existing position value: ${gate.get('existing_position_value', 0):.2f} "
            f"({gate['position_pct'] * 100:.1f}% of equity)\n"
            f"Annualized volatility: {gate.get('volatility')}\n"
            f"Max drawdown (1y): {gate.get('max_drawdown')}\n"
            f"--- DETERMINISTIC RISK GATE (hard constraints, cannot be raised) ---\n"
            f"Gate approved: {gate['approved']}\n"
            f"Gate max notional: ${gate['max_notional_usd']:.2f}\n"
            f"Gate risk level: {gate['risk_level']}\n"
            f"Gate checks: {json.dumps(gate['checks'])}\n"
        )
    if evidence:
        from services import evidence as evidence_service
        context_text += (
            f"--- DECISION QUALITY (separate from portfolio risk; the gate above "
            f"only says what the portfolio can afford) ---\n"
            f"{evidence_service.context_lines(evidence)}\n"
        )

    result = llm_service.call_json(
        "risk",
        system=SYSTEM_INSTRUCTIONS,
        user=context_text,
        temperature=0.1,
        max_tokens=500,
        symbol=symbol,
        reason="risk_reasoning",)

    base_result["llm_status"] = result.status
    base_result["model"] = result.model
    base_result["latency_ms"] = result.latency_ms

    if not result.ok:
        # Provider/model failure: deterministic gate stands; error surfaced,
        # never a fake verdict, never a silent pass.
        logger.error(f"Risk agent LLM reasoning failed for {symbol}: {result.error}")
        base_result["error"] = result.error
        base_result["reasoning"] = (
            (f"position monitoring (existing exposure "
             f"{float(gate.get('position_pct') or 0.0)*100:.1f}% of equity); "
             if monitoring else "") +
            f"deterministic gate applied (LLM reasoning unavailable: {result.status})"
        )
        return base_result

    parsed = result.parsed

    # LLM may veto or shrink — never exceed the deterministic cap.
    llm_approved = bool(parsed.get("approved", False))
    llm_notional = float(parsed.get("max_notional_usd", 0.0))
    final_notional = max(0.0, min(llm_notional, gate["max_notional_usd"]))
    approved = (False if monitoring
                else gate["approved"] and llm_approved and final_notional > 0)

    base_result["approved"] = approved
    base_result["max_notional_usd"] = 0.0 if monitoring else round(final_notional, 2)
    base_result["risk_status"] = _risk_status(
        gate["approved"], not llm_approved and not monitoring,
        None if monitoring else proposed_side)
    for key in ("risk_factors", "warnings", "recommendations"):
        extra = parsed.get(key)
        if isinstance(extra, list) and extra:
            merged = list(base_result.get(key) or [])
            merged.extend(str(x) for x in extra if str(x).strip())
            base_result[key] = merged
    llm_risk_level = str(parsed.get("risk_level", "")).upper()
    if llm_risk_level in ("LOW", "MEDIUM", "HIGH"):
        # Conservative: never report a LOWER risk level than either layer says.
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        base_result["risk_level"] = max((gate["risk_level"], llm_risk_level), key=lambda l: order[l])
    base_result["reasoning"] = str(parsed.get("reasoning", ""))
    return base_result
