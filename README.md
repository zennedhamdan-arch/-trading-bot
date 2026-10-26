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

## Fundamentals data providers

Fundamentals DATA (facts) and fundamentals INTERPRETATION (the LLM) are
strictly separate: `FUNDAMENTALS_PROVIDER` selects the data source,
`LLM_FUNDAMENTALS_PROVIDER` the interpreter. An LLM is never a data source.

| Provider | Select with | Needs | Free tier |
|---|---|---|---|
| none (default) | `FUNDAMENTALS_PROVIDER=none` | — | — |
| Alpha Vantage | `FUNDAMENTALS_PROVIDER=alphavantage` | `ALPHAVANTAGE_API_KEY` | 25 req/day |
| FMP | `FUNDAMENTALS_PROVIDER=fmp` | `FMP_API_KEY` | 250 req/day |

`FUNDAMENTALS_FALLBACK_PROVIDER` is used only when the primary fails.
With `FUNDAMENTALS_CROSS_VALIDATION=true` AND a fallback configured, BOTH
providers are fetched and cross-validated per metric (agreement within
documented tolerances, reporting-period and currency checks, per-metric
`AGREED` / `CONFLICTED` / `SINGLE_SOURCE` / `NOT_COMPARABLE` status, named
confidence levels — never percentages, never averaging). Cross-validation
spends requests on BOTH quotas; it is off by default.

**FMP (Financial Modeling Prep)** uses the official Stable API
(`FMP_BASE_URL=https://financialmodelingprep.com/stable`). The trading path
fetches the minimal endpoint set per symbol (`/profile`, `/income-statement`,
`/ratios`, `/key-metrics`, with `/balance-sheet-statement` only as a labeled
derivation fallback and `/cash-flow-statement` never on the trading path).
Every metric carries provenance (source field, endpoint, reporting date,
fiscal year, period, currency, retrieval time). Protections: per-symbol +
per-endpoint caching (statements 6h default, price-dependent profile 1h),
in-flight deduplication, a local daily request budget (`FMP_DAILY_REQUEST_BUDGET`,
250 default — never bypassing the vendor quota), bounded exponential backoff
after 429s, and restricted endpoints reported as subscription limitations
(never bypassed). Health checks and dashboard polling NEVER issue FMP
requests — they read the in-memory endpoint capability registry.

Endpoint access varies by subscription tier. To verify what YOUR plan
actually allows, run the operator diagnostic (on demand, never at startup):

```
operator: "probe FMP connectivity for AAPL"     # tool: fmp_connectivity
```

It probes all six documented endpoints once, classifies each
(AVAILABLE / EMPTY_DATA / AUTHENTICATION_FAILED / ACCESS_RESTRICTED /
RATE_LIMITED / PROVIDER_ERROR / TIMEOUT / INVALID_RESPONSE — HTTP 200 alone
is NOT "available"; payloads are structurally validated) and records the
results in the capability registry shown on the dashboard Health page.

## Deploying to Render (FMP checklist)

1. **Confirm `FMP_API_KEY` exists** in the correct Render service
   (Environment tab). The key is read server-side only, never sent to the
   frontend, never logged, never committed.
2. **Confirm the deployed service reads it**: after deploy, `/api/config`
   must list `fmp` under `fundamentals.available` (it lists registered
   providers, never values of secrets).
3. **Deploy the intended branch** and verify the deployed commit hash in
   the Render dashboard matches the pushed HEAD.
4. **Check startup logs** — the aligned `startup health check` table must
   show FUNDAMENTALS (with `FUNDAMENTALS_PROVIDER=alphavantage` unchanged,
   FMP stays PENDING: key configured, no fetch yet — no request is sent at
   boot).
5. **Run the FMP connectivity diagnostic** (operator tool
   `fmp_connectivity`, e.g. via the Trading Partner) for AAPL and record
   per-endpoint access on your plan.
6. **Switch the active provider only when intended**:
   `FUNDAMENTALS_PROVIDER=fmp` (optionally
   `FUNDAMENTALS_FALLBACK_PROVIDER=alphavantage` +
   `FUNDAMENTALS_CROSS_VALIDATION=true`).
7. **Verify symbols**: AAPL, MSFT, NVDA normalize with reporting dates and
   currencies preserved (cycle drawer → fundamentals agent result); test one
   symbol with incomplete financial data and confirm missing fields stay
   empty rather than zero.
8. **Inspect `/api/health`** (fundamentals provider rows) and the dashboard
   Health page ("Fundamentals Providers" section: config/auth status,
   endpoint availability, cache, daily usage, rate-limit state, latest
   reporting date).
9. **Inspect `/api/config` for secret leakage** — no key values anywhere.
10. **Confirm the Alpaca paper account remains connected** (Overview page +
    `/api/health` ALPACA row).

## Operational notes

- **Market data**: defaults to the **IEX** feed (`ALPACA_DATA_FEED=IEX`) because
  the free Alpaca paper subscription cannot query SIP. Set `ALPACA_DATA_FEED=SIP`
  only on a paid subscription.
- **Fundamentals**: data comes from the `FundamentalsProvider` abstraction
  (`none` by default; `alphavantage` and `fmp` are registered). Responses are
  cached per symbol AND per endpoint, live calls are paced/throttled, and
  rate limits trigger bounded backoffs — during unavailability the agent
  reports `DATA_UNAVAILABLE` and the cycle is honestly marked
  `PARTIAL_ERROR`, never faked. See "Fundamentals data providers" below.
- **Cycle integrity**: every cycle records per-symbol, per-agent status
  (`OK` / `ERROR` / `UNAVAILABLE` / `SKIPPED`) for all 9 stages. The cycle
  status is computed from those records — the scheduler completing a job never
  implies the trading cycle was healthy. See the cycle drawer's
  "Agent Execution" matrix.
- **Conditional, credit-efficient pipeline**: the technical engine gates every
  expensive step. With **no setup**, news/fundamentals/debate/risk/CIO LLM
  calls do not happen at all — downstream stage statuses record
  `SKIPPED_NO_SETUP` with the reason, and a deterministic HOLD decision is
  persisted (no fabricated agent events). With a setup, the Bull/Bear debate
  (when eligible) and the risk agent run **before** the hard risk engine, so
  their records exist in the live buffer even when the engine rejects the
  trade. Debate eligibility fails closed: the deterministic technical engine
  emits no confidence, so the documented high-confidence requirement cannot be
  verified and debate is skipped (`SKIPPED_DEBATE_NOT_ELIGIBLE`) unless LLM
  technical interpretation provides a verifiable confidence.
- **Debate & risk-agent visibility**: a hard rejection produces a deterministic
  CIO HOLD (no LLM CIO review — an LLM can never unblock what the engine
  blocked); `MARKET_CLOSED` defers only the *order*
  (`DEFERRED_MARKET_CLOSED`, next session recorded) while analysis and the CIO
  review still run. Held positions with no new setup get a risk `MONITORING`
  report every cycle — including outside market hours — and that monitoring is
  **purely deterministic** (`risk_agent.monitor_position`): no per-symbol,
  per-cycle LLM call, full position facts (quantity, entry, current price,
  market value, equity, exposure, data freshness, stop/take availability),
  and structured `missing_inputs` reasons instead of zero-filled values. A
  monitoring result carries no `approved`/`max_notional` verdict — it is never
  a rejected trade. Existing-position exposure is reported separately from
  proposed-order sizing (`existing_position_ok` vs
  `proposed_order_sizing_ok`, with the binding limit and exact calculated
  values), so a position is never flagged invalid because a *new* order would
  exceed the cap. Every persisted event carries `cycle_id`, and agent events
  carry `duration_ms` / `execution_status` / `error_reason` where they apply.
  Timestamps and durations are metadata, never evidence: they are kept out of
  the reports downstream LLM prompts are built from (`_stable_view`), so
  identical cycles hit the LLM response cache instead of re-billing.

## Running the tests

```bash
.venv/bin/python tests/test_gemini_migration.py   # Gemini 3.6 / Interactions API
.venv/bin/python tests/test_production_fixes.py   # httpx/proxies, IEX, history, 429, cycle integrity
.venv/bin/python tests/test_full_pipeline.py      # full cycle, real agents, stubbed transports
.venv/bin/python tests/test_debate_risk_pipeline.py  # debate/risk/monitoring/deferral/sizing contracts
.venv/bin/python tests/test_conditional_pipeline.py  # conditional execution, zero-LLM no-setup path, gating
.venv/bin/python tests/test_alphavantage_fundamentals.py   # AV provider (mocked HTTP)
.venv/bin/python tests/test_fmp_fundamentals.py   # FMP provider + evidence merger (mocked HTTP)
.venv/bin/python tests/test_external_request_hardening.py  # no request storms, rate-limit semantics
NODE_PATH=/path/to/jsdom/node_modules node tests/ui_regression.js   # 12 dashboard pages
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
