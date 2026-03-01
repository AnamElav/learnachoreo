#!/usr/bin/env bash
# Stop the Celery worker and Uvicorn server started by start_local.sh.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/.celery_dev.pid"

if [ -f "$PID_FILE" ]; then
  pid=$(cat "$PID_FILE")
  kill "$pid" 2>/dev/null && echo "Stopped Celery worker (PID $pid)" || true
  rm -f "$PID_FILE"
fi

pkill -f "uvicorn app.main:app" 2>/dev/null && echo "Stopped Uvicorn" || true
