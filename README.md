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
   - **Groq** — technical/debate/risk/CIO reasoning (free tier)
   - **Gemini** — news/fundamentals interpretation (free; quota-protected)
   - **OpenRouter / NVIDIA** — optional LLM providers (only used when
     explicitly configured and verified)
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

## Architecture

    MARKET DATA (Alpaca primary: bars/quotes/snapshots/news/account/clock)
        ↓
    NORMALIZED DATA LAYER (market_data_service; per-symbol data_quality)
        ↓
    DETERMINISTIC ANALYTICS (Python: RSI/SMA/EMA/MACD/ATR/volatility/
        drawdown/returns, position sizing, exposure — never an LLM)
        ↓
    AI REASONING (LLMRouter: technical/news/fundamentals interpretation,
        bull-vs-bear debate, CIO synthesis)
        ↓
    VALIDATION + RISK GATE (deterministic; AI cannot bypass)
        ↓
    ALPACA PAPER EXECUTION (paper trading only, always)

Key properties:
- **Alpaca is the primary market-data source.** The configured feed
  (`ALPACA_DATA_FEED=IEX`) is labeled everywhere; a feed the subscription
  does not permit fails honestly with `DATA_UNAVAILABLE
  (SUBSCRIPTION_FEED_UNAVAILABLE)` — never a silent switch.
- **Real-time layer** (optional): Alpaca WebSockets (trades/quotes/minute
  bars/news) feed a normalized state exposed at `/api/realtime` and the
  dashboard's Live Market strip. Ticks NEVER trigger LLM calls.
- **No yfinance.** Fundamentals come from a pluggable `FundamentalsProvider`
  (default `none` → explicit `DATA_UNAVAILABLE`). No Yahoo scraping, no
  cookies/crumbs, no trading decision depends on Yahoo.
- **Deterministic analytics:** all financial math (indicators, volatility,
  drawdown, position %, sizing, buying power, limits) is computed once per
  symbol in Python and handed to the AI as evidence.
- **Deterministic risk gate:** `services/risk_gate.py` computes approval,
  max notional and validates every order (action/symbol/qty/buying power/
  position limits) before execution. The LLM may veto or shrink — never
  approve beyond the deterministic cap.
- **Cycle error aggregation:** every failed operation lands in the cycle's
  structured `errors` list (`{provider, type, agent, symbol, message}`),
  alongside `agent_results` (e.g. `technical: 5/5, news: 0/5`) and
  `provider_results`. A `PARTIAL_ERROR` cycle never has an empty error list.
- **Startup health check** logs and exposes (`/api/health`) the state of
  Alpaca, market data, the fundamentals provider and every LLM provider.

## Model & provider configuration

All agent→model routing lives in `services/llm_service.py` (the LLMRouter)
and is driven by environment variables — no model ids are hardcoded in
agent files. Defaults (verified against live catalogs, Sept 2026):

| Task | Provider env | Default provider | Model env | Default model |
|---|---|---|---|---|
| Technical | `LLM_TECH_PROVIDER` | groq | `GROQ_TECH_MODEL` | `openai/gpt-oss-20b` |
| Debate | `LLM_DEBATE_PROVIDER` | groq | `GROQ_DEBATE_MODEL` | `openai/gpt-oss-20b` |
| Risk | `LLM_RISK_PROVIDER` | groq | `GROQ_RISK_MODEL` | `openai/gpt-oss-20b` |
| CIO | `LLM_CIO_PROVIDER` | groq | `GROQ_CIO_MODEL` | `openai/gpt-oss-120b` |
| News | `LLM_NEWS_PROVIDER` | gemini | `GEMINI_MODEL` | `gemini-3.6-flash` |
| Fundamentals | `LLM_FUNDAMENTALS_PROVIDER` | gemini | `GEMINI_MODEL` | `gemini-3.6-flash` |

- **OpenRouter is optional**: it is only used when `OPENROUTER_MODEL` is
  explicitly set AND verified available for the account at startup. Its old
  free risk model was withdrawn (`404 MODEL_NOT_FOUND`), which is why risk
  now defaults to Groq.
- **NVIDIA is an optional secondary provider** (`NVIDIA_API_KEY`,
  `NVIDIA_MODEL`): usable as the provider for any task
  (`LLM_*_PROVIDER=nvidia`) or as the global fallback
  (`LLM_FALLBACK_PROVIDER=nvidia` + `LLM_FALLBACK_MODEL=...`). The app boots
  fine without it.
- **Validation**: at startup every configured model id is checked against
  the provider's live `/models` catalog; a missing model is reported on the
  dashboard and each request fails with `MODEL_NOT_FOUND` — never a fake
  result.
- **Circuit breakers**: a 429 quota-exhaustion opens the provider circuit
  (no retries, no per-symbol storms; server `retry_after` honored); a 404
  opens a per-model circuit; an auth failure pauses the provider. States
  (READY / DEGRADED / QUOTA_EXHAUSTED / MODEL_UNAVAILABLE / AUTH_ERROR /
  NETWORK_ERROR / NOT_CONFIGURED) are shown on the Health page.
- **Call budget**: one 5-symbol cycle = 20 Groq + 5 Gemini requests max
  (news analysis is cached while headlines are unchanged → 0 Gemini on
  repeat cycles; fundamentals uses 0 LLM calls with the default `none`
  provider). Local rolling-24h budgets (`*_DAILY_REQUEST_LIMIT`) protect
  the Gemini free tier (20/day).
- **Fallbacks**: the global fallback route (`LLM_FALLBACK_PROVIDER` +
  `LLM_FALLBACK_MODEL`) is used only when explicitly configured AND
  verified live — never for quota errors, never silently.
- **Per-cycle accounting**: every cycle record carries `llm_usage` (exact
  counts per provider/agent/symbol), shown in the cycle drawer.

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
