# Updates: features adapted from TauricResearch/TradingAgents

This adds three things to the existing 4-agent pipeline (news, technical,
risk, CIO), all achievable at **$0** using APIs you already have keys for
plus one free library (`yfinance`, no key needed).

## What's new

### 1. Fundamentals Agent (`agents/fundamentals_agent.py`)
Pulls P/E ratio, revenue growth, profit margin, debt-to-equity, and ROE via
`yfinance` (free, no signup), then has your existing Gemini key interpret
them into a BULLISH/BEARISH/NEUTRAL signal, same shape as the other agents.

### 2. Bull vs Bear Debate (`agents/debate_agent.py`)
Adapted from TradingAgents' researcher-debate step. Two independent LLM
calls (using your existing Groq key) build the strongest possible case
FOR and AGAINST the trade using the same tech/news/fundamentals data. The
CIO sees both arguments plus a net "edge" score. This is the single
highest-value addition: it catches cases where the individual agents
nominally lean bullish but the actual reasoning underneath is weak.

### 3. Persistent Memory + Adaptive Agent Weighting (`services/memory_service.py`)
A local SQLite file (`data/memory.db`) now survives restarts and does two things:
- **Decision log**: every BUY decision is recorded with which agents said
  what. When that position later closes, realized P&L is computed and
  attached automatically (see `_detect_closed_positions` in `main.py`).
- **Agent accuracy weighting**: each agent's hit rate over its last N
  closed calls (default 20) becomes a weight (0.5–1.5) shown to the CIO,
  so it leans more on agents that have recently been right and less on
  ones that haven't.

This is **not** model fine-tuning -- it's in-context learning: the CIO's
prompt grows richer with real track-record data every cycle, without ever
touching model weights. Being honest about the limits: with a handful of
trades a week you'll have maybe 20-50 data points a month, which isn't a
lot to learn from reliably. Treat the weighting as a transparency/sanity
tool more than a guarantee of improving edge.

## New environment variables
All optional, all default to `true` / sensible values -- see updated `.env.example`:
```
ENABLE_FUNDAMENTALS_AGENT=true
ENABLE_DEBATE=true
ENABLE_MEMORY=true
MEMORY_DB_PATH=data/memory.db
AGENT_ACCURACY_LOOKBACK=20
```
No new API keys needed -- fundamentals reuses `GEMINI_API_KEY`, debate reuses `GROQ_API_KEY`.

## New dependency
`yfinance==0.2.44` added to `requirements.txt` (free, no key).

## New API endpoint
`GET /api/agent-accuracy` -- shows each agent's current hit rate and weight,
so you (or the dashboard) can see the learning loop's state directly.

## What changed in existing files
- `agents/cio_agent.py` -- `make_decision()` now accepts optional
  `fundamentals_report`, `debate_report`, `agent_weights`, and
  `memory_summary` params. Backward compatible: omit them and it behaves
  as before.
- `config.py` -- added the 5 new settings above.
- `main.py` -- cycle now calls fundamentals + debate agents, detects
  closed positions each cycle to feed the learning loop, and records
  every BUY decision to memory.
- `requirements.txt` -- added `yfinance`.

## What did NOT change
- `agents/news_agent.py`, `agents/tech_agent.py`, `agents/risk_agent.py`,
  `services/alpaca_service.py`, `static/index.html` are untouched.
- Still hardcoded to Alpaca **paper trading only** -- no live-money risk
  was introduced by any of this.

## Before merging
Test locally first:
```bash
pip install -r requirements.txt
python main.py
```
Watch the console/logs for the new `fundamentals` and `debate` log entries
each cycle, and check `GET /api/agent-accuracy` once a few trades have
closed to confirm weights are populating.
