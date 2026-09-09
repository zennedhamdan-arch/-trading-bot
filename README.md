# AI Trader — Autonomous Multi-Agent Trading Bot

A 24/7 paper-trading system: multiple AI agents (technical, news, fundamentals,
bull/bear debate, risk, CIO, execution, memory) analyze a stock universe on a
scheduled cycle and execute paper trades through Alpaca. Ships with a premium
operations dashboard for monitoring everything the system does.

**Paper trading only — this application is hardcoded to the Alpaca paper
endpoint and never touches live money.**

## Quick start

```bash
./run.sh                 # creates the virtualenv, installs deps, serves on :8000
```

Open http://localhost:8000 — the dashboard is live immediately. Without API
keys it will honestly show configuration warnings and unavailable data states;
every control (start/stop bot, run cycle, filters, drawers) still works.

## Make the bot actually trade

1. Copy `.env.example` → `.env`
2. Fill in the keys:
   - **Alpaca** (required) — paper account keys from https://app.alpaca.markets
   - **Groq** — technical/debate/CIO agents (free tier)
   - **Gemini** — news/fundamentals agents (free)
   - **OpenRouter** — risk agent (free models)
3. Optionally set the model ids (defaults verified Sept 2026; validated
   against each provider's live catalog at startup — see "Model & quota
   configuration" below):
   - `GROQ_TECH_MODEL` / `GROQ_DEBATE_MODEL` / `GROQ_CIO_MODEL`
   - `OPENROUTER_RISK_MODEL`
   - `GEMINI_MODEL`
4. Restart: `./run.sh`
4. In the dashboard header, press **Start** — the scheduler runs a full agent
   cycle every `CYCLE_INTERVAL_MINUTES` (default 15). Press **Run Cycle** any
   time to force one immediately.

Everything is observable: decisions and agent reasoning (AI Intelligence),
per-cycle execution records (Cycles), risk verdicts (Risk), agent accuracy
(Agent Performance), and the full audit trail (Activity).

## Dashboard design preview

`http://localhost:8000/?demo=1` renders the complete UI with clearly-labeled
sample data (hatched amber banner on every screen) — useful for evaluating the
design without a running bot. It never mixes with live data.

## Model & quota configuration

All agent→model routing lives in `services/llm_service.py` and is driven by
environment variables — no model ids are hardcoded in agent files.

| Agent | Provider | Env var | Default (verified Sept 2026) |
|---|---|---|---|
| Technical | Groq | `GROQ_TECH_MODEL` | `openai/gpt-oss-20b` |
| Debate | Groq | `GROQ_DEBATE_MODEL` | `openai/gpt-oss-20b` |
| CIO | Groq | `GROQ_CIO_MODEL` | `openai/gpt-oss-120b` |
| Risk | OpenRouter | `OPENROUTER_RISK_MODEL` | `openai/gpt-oss-20b:free` |
| News | Gemini | `GEMINI_MODEL` | `gemini-3.6-flash` |
| Fundamentals | Gemini | `GEMINI_MODEL` | `gemini-3.6-flash` |

- **Validation**: at startup every configured model id is checked against the
  provider's live `/models` catalog. A missing model is reported on the
  dashboard (Health page) and every request for it fails with a clear
  `MODEL_NOT_FOUND` status — never a fabricated result.
- **Quota handling**: each provider has a local rolling-24h request budget
  (`*_DAILY_REQUEST_LIMIT`; Gemini defaults to 20/day, its free-tier limit).
  When the budget is hit — or the provider returns a quota-exhausted 429 —
  further requests short-circuit with `PROVIDER_QUOTA_EXCEEDED` and the
  affected agents are marked UNAVAILABLE for that cycle. Quota exhaustion is
  never retried and never falls back to another model; transient rate limits
  (429 with a short retry delay) get at most one bounded retry.
- **Call reduction**: the Gemini-backed news/fundamentals agents reuse their
  previous LLM analysis while the underlying headlines/metrics are unchanged
  (`NEWS_ANALYSIS_CACHE_TTL_MINUTES`, `FUNDAMENTALS_ANALYSIS_CACHE_TTL_MINUTES`),
  and the bull/bear debate is one structured call per symbol (not two). A
  full 5-symbol cycle sends 15 Groq, 5 OpenRouter and at most 10 Gemini
  requests — and typically 0 Gemini requests once analyses are cached.
- **Fallbacks**: disabled by default. A fallback model is used only when
  explicitly configured (`*_FALLBACK_MODEL`) AND verified in the provider's
  live catalog at startup — never for quota errors, never silently.
- **Per-cycle accounting**: every cycle record (`/api/cycles`, Cycles page →
  cycle detail) carries `llm_usage`: exact request counts per provider, per
  agent and per symbol, with statuses. Nothing is hidden.

## Operational notes

- **Market data**: defaults to the **IEX** feed (`ALPACA_DATA_FEED=IEX`) because
  the free Alpaca paper subscription cannot query SIP. Set `ALPACA_DATA_FEED=SIP`
  only on a paid subscription.
- **Fundamentals**: yfinance responses are cached per symbol (60 min default),
  live calls are paced, and HTTP 429s trigger a global backoff — during
  unavailability the agent reports `DATA_UNAVAILABLE` and the cycle is honestly
  marked `PARTIAL_ERROR`, never faked.
- **Cycle integrity**: every cycle records per-symbol, per-agent status
  (`OK` / `ERROR` / `UNAVAILABLE` / `SKIPPED`) for all 9 stages. The cycle
  status is computed from those records — the scheduler completing a job never
  implies the trading cycle was healthy. See the cycle drawer's
  "Agent Execution" matrix.

## Running the tests

```bash
.venv/bin/python tests/test_gemini_migration.py   # Gemini 3.6 / Interactions API
.venv/bin/python tests/test_production_fixes.py   # httpx/proxies, IEX, history, 429, cycle integrity
.venv/bin/python tests/test_full_pipeline.py      # full cycle, real agents, stubbed transports
```

## Repository layout

```
main.py                  FastAPI server, scheduler, trading cycle, REST API
config.py                Environment/settings loader (single source of truth)
agents/                  technical · news · fundamentals · debate · risk · cio
services/                alpaca_service (broker) · memory_service (SQLite)
static/                  dashboard (bespoke, zero-dependency: css/ + js/)
DASHBOARD.md             Dashboard architecture, endpoints, states, a11y notes
.env.example             All supported environment variables, documented
```

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/portfolio` | Account, positions, orders, equity history, bot state |
| `GET /api/logs` | Live agent event buffer (200 entries) |
| `GET /api/agent-accuracy` | Learning-loop hit rates and weights |
| `GET /api/config` | Active configuration + warnings |
| `GET /api/cycles` | Per-cycle execution records |
| `GET /api/orders?limit=` | Broker order history |
| `GET /api/history?period=` | Equity curve (1D…ALL) |
| `POST /api/bot/start` · `stop` · `run-now` | Bot control |

## Safety notes

- The risk agent is a hard constraint: the CIO may not BUY without its
  approval, and never above its approved notional.
- Agent accuracy weighting is in-context learning, not fine-tuning; treat it
  as a transparency tool, not a guarantee of edge.
