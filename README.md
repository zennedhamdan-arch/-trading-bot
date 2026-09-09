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
3. Restart: `./run.sh`
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
