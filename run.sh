#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# AI Trader — one-command launcher
# Creates/repairs the virtualenv on first run, then serves the dashboard.
# ---------------------------------------------------------------------------
set -e
cd "$(dirname "$0")"

VENV=".venv"
PY="$VENV/bin/python"

if [ ! -x "$PY" ] || ! "$PY" -c "import fastapi, uvicorn, apscheduler, pandas" 2>/dev/null; then
  echo "[run.sh] preparing virtualenv (first run only)…"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet -r requirements.txt
fi

if [ ! -f .env ]; then
  echo "[run.sh] NOTE: no .env found — the bot will run but cannot trade."
  echo "         Copy .env.example to .env and add your API keys, then restart."
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
echo "[run.sh] AI Trader dashboard → http://$HOST:$PORT  (paper trading only)"
exec "$PY" -m uvicorn main:app --host "$HOST" --port "$PORT"
