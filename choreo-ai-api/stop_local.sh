#!/usr/bin/env bash
# Stop Celery, MCP history server, and Uvicorn started by start_local.sh.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/.celery_dev.pid"
MCP_PID_FILE="$SCRIPT_DIR/.mcp_dev.pid"

if [ -f "$MCP_PID_FILE" ]; then
  mcp_pid=$(cat "$MCP_PID_FILE")
  kill "$mcp_pid" 2>/dev/null && echo "Stopped MCP server (PID $mcp_pid)" || true
  rm -f "$MCP_PID_FILE"
fi

if [ -f "$PID_FILE" ]; then
  pid=$(cat "$PID_FILE")
  kill "$pid" 2>/dev/null && echo "Stopped Celery worker (PID $pid)" || true
  rm -f "$PID_FILE"
fi

pkill -f "uvicorn app.main:app" 2>/dev/null && echo "Stopped Uvicorn" || true
