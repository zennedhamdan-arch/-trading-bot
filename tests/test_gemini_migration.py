"""
tests/test_gemini_migration.py

Verifies the Gemini 3.6 Flash / Interactions API migration without making
real network calls (the genai client is stubbed).

Run:  .venv/bin/python tests/test_gemini_migration.py
Exits non-zero on any failure. No external test dependencies.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}")


# ---------------------------------------------------------------------------
# 1. Config: model default + env override
# ---------------------------------------------------------------------------
print("config:")
import importlib
import config

check("default model is gemini-3.6-flash", config.settings.GEMINI_MODEL == "gemini-3.6-flash")
check("key comes from env", config.settings.GEMINI_API_KEY == os.getenv("GEMINI_API_KEY", ""))

os.environ["GEMINI_MODEL"] = "models/gemini-3.6-flash"
importlib.reload(config)
check("GEMINI_MODEL env var is honored", config.settings.GEMINI_MODEL == "models/gemini-3.6-flash")
del os.environ["GEMINI_MODEL"]
importlib.reload(config)
check("default restored", config.settings.GEMINI_MODEL == "gemini-3.6-flash")

# ---------------------------------------------------------------------------
# Stub machinery for the genai client
# ---------------------------------------------------------------------------
from services import gemini_service


class FakeInteraction:
    def __init__(self, output_text, errors=None):
        self.output_text = output_text
        self.errors = errors


class FakeInteractions:
    def __init__(self, result=None, raise_exc=None):
        self.result = result
        self.raise_exc = raise_exc
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_exc:
            raise self.raise_exc
        return self.result


def install_fake(result=None, raise_exc=None):
    fake = FakeInteractions(result=result, raise_exc=raise_exc)
    gemini_service._client = FakeClient(fake)  # client.interactions.create(...)
    return fake


class FakeClient:
    def __init__(self, fake_interactions):
        self.interactions = fake_interactions


def install_fake_client(fake_interactions):
    gemini_service._client = FakeClient(fake_interactions)
    return fake_interactions


def reset(**overrides):
    """Reset gemini_service state with configurable settings."""
    gemini_service._client = None
    saved = {k: getattr(config.settings, k) for k in ("GEMINI_API_KEY", "GEMINI_MODEL")}
    config.settings.GEMINI_API_KEY = overrides.get("key", "test-key")
    config.settings.GEMINI_MODEL = overrides.get("model", "gemini-3.6-flash")
    return saved


def restore(saved):
    for k, v in saved.items():
        setattr(config.settings, k, v)
    gemini_service._client = None


# ---------------------------------------------------------------------------
# 2. gemini_service: Interactions API call contract
# ---------------------------------------------------------------------------
print("gemini_service (Interactions API):")

saved = reset()
fake = install_fake(FakeInteraction('{"signal": "BULLISH", "confidence": 0.8}'))
out = gemini_service.generate_json("YOU ARE A TEST.", "data: 123")
check("returns parsed JSON", out == {"signal": "BULLISH", "confidence": 0.8})
call = fake.calls[0]
check("model passed", call.get("model") == "gemini-3.6-flash")
check("input passed", call.get("input") == "data: 123")
check("system_instruction passed", call.get("system_instruction") == "YOU ARE A TEST.")
check("store=False (no server-side retention)", call.get("store") is False)
check("legacy generate_content not used", "contents" not in call)

# models/ prefix stripped
reset(model="models/gemini-3.6-flash")
install_fake(FakeInteraction('{"a": 1}'))
gemini_service.generate_json("s", "i")
check("models/ prefix stripped",
      gemini_service._client.interactions.calls[-1]["model"] == "gemini-3.6-flash")

# fenced JSON tolerated
reset()
install_fake(FakeInteraction('```json\n{"a": 42, "b": "text"}\n```'))
check("markdown fences tolerated", gemini_service.generate_json("s", "i") == {"a": 42, "b": "text"})

# empty response -> ValueError
reset()
install_fake(FakeInteraction(""))
try:
    gemini_service.generate_json("s", "i")
    check("empty response raises", False)
except ValueError:
    check("empty response raises ValueError", True)

# interaction-level errors surfaced
reset()
install_fake(FakeInteraction("", errors=[{"code": 500, "message": "boom"}]))
try:
    gemini_service.generate_json("s", "i")
    check("interaction errors raise", False)
except RuntimeError as e:
    check("interaction errors raise RuntimeError", "boom" in str(e))

# SDK exceptions propagate
reset()
install_fake(None, raise_exc=RuntimeError("api down"))
try:
    gemini_service.generate_json("s", "i")
    check("SDK errors propagate", False)
except RuntimeError as e:
    check("SDK errors propagate", "api down" in str(e))

# missing key -> RuntimeError before any client construction
reset(key="")
try:
    gemini_service.generate_json("s", "i")
    check("missing key raises", False)
except RuntimeError as e:
    check("missing key raises RuntimeError", "GEMINI_API_KEY" in str(e))
restore(saved)

# ---------------------------------------------------------------------------
# 3. news_agent: prompts, mapping, and fail-safes preserved
# ---------------------------------------------------------------------------
print("news_agent:")
from agents import news_agent

saved = reset()
# no key -> legacy fail-safe
config.settings.GEMINI_API_KEY = ""
r = news_agent.analyze_news("AAPL", ["Apple beats earnings"])
check("no key -> null stance + UNAVAILABLE evidence (never a fake NEUTRAL)",
      r["sentiment"] is None and r["confidence"] is None and r["evidence_status"] == "UNAVAILABLE"
      and r["error"] == "GEMINI_API_KEY not configured.")
check("no key -> disabled summary", r["summary"] == "News agent disabled: missing API key — no news verdict.")

# no headlines -> NEUTRAL, no LLM call
config.settings.GEMINI_API_KEY = "test-key"
gemini_service._client = None
r = news_agent.analyze_news("AAPL", [])
check("no headlines -> null stance + UNAVAILABLE evidence, no LLM call",
      r["sentiment"] is None and r["evidence_status"] == "UNAVAILABLE"
      and r["summary"] == "No recent headlines available — no news verdict."
      and r["llm_status"] == "SKIPPED_NO_DATA")

# happy path via stubbed Interactions client
fake = install_fake_client(FakeInteractions(FakeInteraction(
    '{"sentiment": "BULLISH", "confidence": 0.74, "summary": "Positive earnings coverage.", "key_headline": "Apple beats"}')))
r = news_agent.analyze_news("NVDA", ["NVDA beats", "NVDA rallies", "x", "y", "z"])
check("happy path maps fields", r["sentiment"] == "BULLISH" and abs(r["confidence"] - 0.74) < 1e-9
      and r["key_headline"] == "Apple beats" and r["error"] is None
      and r["evidence_status"] == "AVAILABLE")
call = fake.calls[0]
check("system prompt sent as system_instruction", call["system_instruction"] == news_agent.SYSTEM_INSTRUCTIONS)
check("ticker+headlines sent as input", "Ticker: NVDA" in call["input"] and "- NVDA beats" in call["input"])
check("headlines capped at 15", call["input"].count("\n- ") <= 15)

# LLM failure -> graceful fail-safe
install_fake_client(FakeInteractions(None, raise_exc=RuntimeError("quota exceeded")))
r = news_agent.analyze_news("AAPL", ["h1"])
check("LLM error -> null stance + ERROR evidence (never a fake NEUTRAL)",
      r["sentiment"] is None and r["confidence"] is None and r["evidence_status"] == "ERROR"
      and "quota exceeded" in r["error"]
      and r["summary"] == "News agent encountered an error — no news verdict.")
restore(saved)

# ---------------------------------------------------------------------------
# 4. fundamentals_agent: prompts, mapping, and fail-safes preserved
# ---------------------------------------------------------------------------
print("fundamentals_agent:")
from agents import fundamentals_agent

saved = reset()
# disabled via config
config.settings.ENABLE_FUNDAMENTALS_AGENT = False
r = fundamentals_agent.analyze_fundamentals("AAPL", {"error": None})
check("disabled -> fail-safe", r["summary"] == "Fundamentals agent disabled via config.")
config.settings.ENABLE_FUNDAMENTALS_AGENT = True

# fundamentals fetch error
r = fundamentals_agent.analyze_fundamentals("AAPL", {"symbol": "AAPL", "error": "provider down"})
check("data error -> null signal + UNAVAILABLE evidence (never a fake NEUTRAL)",
      r["signal"] is None and r["confidence"] is None and r["evidence_status"] == "UNAVAILABLE"
      and r["error"] == "DATA_UNAVAILABLE: provider down")

# no key
config.settings.GEMINI_API_KEY = ""
gemini_service._client = None
r = fundamentals_agent.analyze_fundamentals("AAPL", {"symbol": "AAPL", "pe_ratio": 30})
check("no key -> fail-safe", r["error"] == "GEMINI_API_KEY not configured."
      and r["evidence_status"] == "UNAVAILABLE"
      and r["summary"] == "Fundamentals agent disabled: missing API key — no fundamentals verdict.")

# happy path
config.settings.GEMINI_API_KEY = "test-key"
fake = install_fake_client(FakeInteractions(FakeInteraction(
    '{"signal": "BEARISH", "confidence": 0.61, "summary": "Rich valuation."}')))
r = fundamentals_agent.analyze_fundamentals("MSFT", {
    "symbol": "MSFT", "pe_ratio": 35.0, "forward_pe": 30.1, "revenue_growth": 0.12,
    "profit_margin": 0.35, "debt_to_equity": 0.4, "return_on_equity": 0.2, "error": None})
check("happy path maps fields", r["signal"] == "BEARISH" and abs(r["confidence"] - 0.61) < 1e-9
      and r["summary"] == "Rich valuation." and r["error"] is None
      and r["evidence_status"] == "AVAILABLE")
call = fake.calls[0]
check("system prompt sent as system_instruction", call["system_instruction"] == fundamentals_agent.SYSTEM_INSTRUCTIONS)
check("metrics sent as input", "P/E ratio: 35.0" in call["input"] and "Debt-to-equity: 0.4" in call["input"])

# LLM failure
install_fake_client(FakeInteractions(None, raise_exc=ValueError("No JSON object found in model response.")))
r = fundamentals_agent.analyze_fundamentals("AAPL", {"symbol": "AAPL", "pe_ratio": 30})
check("LLM error -> null signal + ERROR evidence (never a fake NEUTRAL)",
      r["signal"] is None and r["evidence_status"] == "ERROR"
      and r["summary"] == "Fundamentals agent encountered an error — no fundamentals verdict.")
restore(saved)

# ---------------------------------------------------------------------------
# 5. Stale references: nothing in the repo mentions the old model
# ---------------------------------------------------------------------------
print("repository references:")
import subprocess
res = subprocess.run(
    ["grep", "-rn", "gemini-2.5", "--include=*.py", "--include=*.txt", "--include=*.md",
     "--include=*.html", "--include=*.js", "--include=*.example", "."],
    capture_output=True, text=True,
)
hits = [l for l in res.stdout.splitlines() if ".venv/" not in l and "__pycache__" not in l and "test_gemini_migration" not in l]
check("no gemini-2.5 references remain", not hits)
if hits:
    for h in hits:
        print("      stale:", h)

res2 = subprocess.run(
    ["grep", "-rn", "generate_content(", "--include=*.py", "."],
    capture_output=True, text=True,
)
hits2 = [l for l in res2.stdout.splitlines() if ".venv/" not in l and "__pycache__" not in l and "test_gemini_migration" not in l]
check("no legacy generate_content calls remain", not hits2)
if hits2:
    for h in hits2:
        print("      stale:", h)

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("ALL GEMINI MIGRATION TESTS PASSED")
