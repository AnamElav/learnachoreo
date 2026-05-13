#!/usr/bin/env bash
# Start Redis, Celery worker, LearnAChoreo MCP history server (background), and FastAPI (foreground).
# Run from repo root or choreo-ai-api. Requires Redis installed (brew install redis / apt redis-server).

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$SCRIPT_DIR/.celery_dev.pid"
MCP_PID_FILE="$SCRIPT_DIR/.mcp_dev.pid"

# Use repo venv binaries so celery/uvicorn are found (no reliance on activation)
VENV_BIN=""
if [ -d "$REPO_ROOT/.venv/bin" ]; then
  VENV_BIN="$REPO_ROOT/.venv/bin"
elif [ -d "$SCRIPT_DIR/.venv/bin" ]; then
  VENV_BIN="$SCRIPT_DIR/.venv/bin"
fi
if [ -n "$VENV_BIN" ]; then
  export PATH="$VENV_BIN:$PATH"
  PYTHON="$VENV_BIN/python"
else
  echo "Warning: no .venv found. From repo root run: python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt -r choreo-ai-api/requirements.txt"
  PYTHON=python3
fi

export PYTHONPATH="$SCRIPT_DIR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Load env files into the shell so uvicorn/celery/MCP see SUPABASE_* and other secrets.
# Background jobs inherit this process's exported environment (set -a while sourcing).
# Prefer repo root `.env`, then choreo-ai-api `.env`. (Matches python-dotenv in `app/main.py`.)
set -a
[[ -f "$REPO_ROOT/.env" ]] && . "$REPO_ROOT/.env" || true
[[ -f "$SCRIPT_DIR/.env" ]] && . "$SCRIPT_DIR/.env" || true
set +a
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export MCP_SERVER_URL="${MCP_SERVER_URL:-http://localhost:8001/mcp}"
export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
export OUTPUTS_DIR="${OUTPUTS_DIR:-$REPO_ROOT/data/outputs}"
mkdir -p "$DATA_DIR" "$OUTPUTS_DIR"

# Start Redis (Mac vs Linux)
if command -v brew &>/dev/null; then
  brew services start redis 2>/dev/null || true
else
  sudo service redis-server start 2>/dev/null || true
fi

# Clean up background workers on exit
cleanup() {
  if [ -f "$MCP_PID_FILE" ]; then
    mcp_pid=$(cat "$MCP_PID_FILE")
    kill "$mcp_pid" 2>/dev/null || true
    rm -f "$MCP_PID_FILE"
  fi
  if [ -f "$PID_FILE" ]; then
    pid=$(cat "$PID_FILE")
    kill "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
  fi
}
trap cleanup EXIT INT TERM

# Celery worker in background (inherits env from above)
cd "$SCRIPT_DIR"
"$PYTHON" -m celery -A app.celery_app worker --loglevel=info &
echo $! > "$PID_FILE"

# MCP server (streamable HTTP) — same shell env as Celery; must match MCP_SERVER_URL
"$PYTHON" -m uvicorn app.mcp_server:mcp_app --port 8001 &
echo $! > "$MCP_PID_FILE"
echo "Started MCP history server (PID $(cat "$MCP_PID_FILE")) at ${MCP_SERVER_URL}"

# FastAPI in foreground (Ctrl+C will trigger cleanup)
"$PYTHON" -m uvicorn app.main:app --reload --port 8000
