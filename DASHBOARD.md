# AI Trader — Dashboard

A premium, production-ready operations dashboard for the autonomous multi-agent
AI stock-trading bot. Institutional dark theme, information-dense, mobile-first.
**Paper trading only.**

```
python main.py            # http://localhost:8000
?demo=1                  # design preview with clearly-labeled sample data
```

## Architecture

Zero frontend dependencies — no CDN JS frameworks, no build step. The UI is a
bespoke design system served as static files by the existing FastAPI backend:

```
static/
  index.html            App shell: sidebar, header, mobile nav, overlays, boot
  css/
    tokens.css          Design tokens (surfaces, ink, semantic color, type, metrics)
    components.css      Component library (badges, tables, drawers, pipeline, states…)
    app.css             Shell layout, page grids, responsive breakpoints, mobile
  js/
    icons.js            Inline SVG icon set (stroke-based, currentColor)
    api.js              Typed fetch layer + formatters + status semantics
    app.js              Store, polling, router, overlays, SVG chart engine, demo mode
    components.js       Shared components (tables→cards, consensus, pipeline, debate, feed)
    drawers.js          Detail drawers: position, order, AI decision, cycle, agent, event
    pages.js            The 11 pages + render/bind lifecycle
```

## Backend endpoints used

| Endpoint | Used for |
| --- | --- |
| `GET /api/portfolio` | KPIs, positions, orders, equity history, bot state |
| `GET /api/logs` | Agent activity, decisions, errors, warnings (200-entry buffer) |
| `GET /api/agent-accuracy` | Agent hit rates / weights (honest empty state at n=0) |
| `GET /api/config` | Read-only config page + configuration warnings |
| `GET /api/cycles` *(new)* | Real per-cycle records: duration, decisions, orders, warnings, errors |
| `GET /api/orders?limit=` *(new)* | Full order history |
| `GET /api/history?period=` *(new)* | Equity curve per chart range (1D…ALL) |
| `POST /api/bot/start` / `stop` / `run-now` | Bot control (stop requires a confirmation modal) |

`/api/cycles`, `/api/orders`, `/api/history` are small, additive changes to
`main.py`/`alpaca_service.py` — they only report what actually happened; nothing
is fabricated.

## Data honesty rules

- Every number on screen comes from a live API response. Nothing is invented.
- Unavailable metrics render explicit **“Data unavailable”** states, never zeros
  dressed up as data.
- Agent accuracy with zero closed samples shows “No data — collecting”.
- API failures surface as error banners + offline status, not broken pages.
- `?demo=1` is a **clearly-labeled** design preview (hatched amber banner on
  every screen). It never mixes with live data.

## States designed

Loading skeletons · empty · error · partial data · API down · bot stopped ·
cycle running/completed/failed/partial · no positions · no orders · no accuracy
data · no history for range · agent disabled via config.

## Interaction model

Position → detail drawer (P&L, AI context, agent opinions, orders, timeline) ·
Order → detail drawer (fills, linked AI decision) · Decision → full reasoning
drawer (verdicts, consensus, debate, expandable reasoning) · Cycle → execution
timeline · Agent → performance + events · Event → structured detail ·
Symbol → filtered activity · `Ctrl+K` / `/` → command palette.

## Accessibility

WCAG AA contrast, visible focus states, keyboard navigation on all interactive
rows (`tabindex` + Enter via role=button), semantic status badges (never color
alone), chart data table fallback, `prefers-reduced-motion` support, skip link,
aria labels on live regions and dialogs.
