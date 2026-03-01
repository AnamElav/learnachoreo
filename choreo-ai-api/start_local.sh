#!/usr/bin/env bash
# Start Redis, Celery worker (background), and FastAPI server for local development.
# Run from repo root or choreo-ai-api. Requires Redis installed (brew install redis / apt redis-server).

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$SCRIPT_DIR/.celery_dev.pid"

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

export PYTHONPATH="$REPO_ROOT"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
export OUTPUTS_DIR="${OUTPUTS_DIR:-$REPO_ROOT/data/outputs}"
mkdir -p "$DATA_DIR" "$OUTPUTS_DIR"

# Start Redis (Mac vs Linux)
if command -v brew &>/dev/null; then
  brew services start redis 2>/dev/null || true
else
  sudo service redis-server start 2>/dev/null || true
fi

# Clean up Celery on exit
cleanup() {
  if [ -f "$PID_FILE" ]; then
    pid=$(cat "$PID_FILE")
    kill "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
  fi
}
trap cleanup EXIT INT TERM

# Celery worker in background
cd "$SCRIPT_DIR"
"$PYTHON" -m celery -A app.celery_app worker --loglevel=info &
echo $! > "$PID_FILE"

# FastAPI in foreground (Ctrl+C will trigger cleanup)
"$PYTHON" -m uvicorn app.main:app --reload --port 8000
