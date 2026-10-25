"""
tests/test_gemini_migration.py

Verifies the Gemini 3.6 Flash / Interactions API migration without making
real network calls (the genai client is stubbed).

Run:  .venv/bin/python tests/test_gemini_migration.py
Exits non-zero on any failure. No external test dependencies.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Isolated SQLite for this run (news intelligence shares the memory DB).
os.environ["MEMORY_DB_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="gemmig-"), "test-memory.db")

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
# 3. news intelligence V2: Gemini-routed batch analysis + cache reader agent
# ---------------------------------------------------------------------------
print("news_agent (V2 cache reader + worker):")
from agents import news_agent
from services import llm_service, news_intelligence
import json

saved = reset()

# no cached intelligence -> no LLM call, null stance + UNAVAILABLE evidence
config.settings.GEMINI_API_KEY = "test-key"
config.settings.LLM_NEWS_PROVIDER = "gemini"
llm_service.reset_all_state()
gemini_service._client = None
r = news_agent.analyze_news("ZZZZ")
check("no cached intelligence -> null stance + UNAVAILABLE, no LLM call",
      r["sentiment"] is None and r["confidence"] is None
      and r["evidence_status"] == "UNAVAILABLE"
      and r["llm_status"] == "SKIPPED_NO_DATA")

# worker path via stubbed Interactions client (V2 batch analysis shape)
BATCH = json.dumps({
    "articles": [{"index": 0, "sentiment": "BULLISH", "confidence": 0.74,
                  "importance": "HIGH", "impact_horizon": "SHORT_TERM"}],
    "overall_sentiment": "BULLISH", "overall_confidence": 0.74,
    "summary": "Positive earnings coverage.",
})
fake = install_fake_client(FakeInteractions(FakeInteraction(BATCH)))
ARTS = [{"id": "gm-1", "headline": "NVDA beats earnings expectations",
         "summary": "", "source": "Reuters", "published_at": "2026-09-10T10:00:00Z"}]
news_intelligence.refresh_symbol("NVDA", fetch_articles=lambda s: ARTS)
check("batch prompt routed through Gemini (one interactions call)",
      len(fake.calls) == 1)
call = fake.calls[0]
check("system prompt sent as system_instruction",
      call["system_instruction"] == news_intelligence._ANALYSIS_INSTRUCTIONS)
check("ticker + articles sent as input",
      "Ticker: NVDA" in call["input"] and "NVDA beats earnings" in call["input"])
r = news_agent.analyze_news("NVDA")
check("agent maps the cached intelligence (sentiment/confidence/provider)",
      r["sentiment"] == "BULLISH" and abs(r["confidence"] - 0.74) < 1e-9
      and r["error"] is None and r["evidence_status"] == "AVAILABLE"
      and r["provider"] == "gemini" and r["cached"] is True)

# same articles again -> duplicate detection, NO new Gemini call
news_intelligence.refresh_symbol("NVDA", fetch_articles=lambda s: ARTS)
check("duplicate articles -> no second Gemini call",
      len(fake.calls) == 1)

# LLM failure in the worker -> honest deterministic fallback, never fake AI
install_fake_client(FakeInteractions(None, raise_exc=RuntimeError("quota exceeded")))
NEW_ARTS = [{"id": "gm-2", "headline": "NVDA faces lawsuit investigation",
             "summary": "", "source": "Bloomberg", "published_at": "2026-09-10T11:00:00Z"}]
news_intelligence.refresh_symbol("NVDA", fetch_articles=lambda s: NEW_ARTS)
r = news_agent.analyze_news("NVDA")
check("LLM failure -> deterministic fallback labeled (mixed history honest)",
      r["evidence_status"] == "AVAILABLE" and r["llm_status"] == "DETERMINISTIC_FALLBACK"
      and "deterministic_fallback" in (r["provider"] or ""))
arts = {a["article_id"]: a for a in news_intelligence.get_recent_articles("NVDA")}
check("fallback article verdict honestly sourced (deterministic_fallback)",
      arts["gm-2"]["analysis"]["source"] == "deterministic_fallback")
restore(saved)

# ---------------------------------------------------------------------------
# 4. fundamentals_agent: prompts, mapping, and fail-safes preserved
# ---------------------------------------------------------------------------
print("fundamentals_agent:")
from agents import fundamentals_agent

# This suite exercises the GEMINI transport; route fundamentals to it
# explicitly (the V2 default route is the UnoRouter chain).
config.settings.LLM_FUNDAMENTALS_PROVIDER = "gemini"

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
