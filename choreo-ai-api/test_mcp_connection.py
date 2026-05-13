#!/usr/bin/env python3
"""
Verify Claude can reach the local LearnAChoreo MCP server and invoke history tools.

Prerequisites:
  1. MCP server running (e.g. `PYTHONPATH=. python -m app.mcp_server` or `./start_local.sh`).
  2. ANTHROPIC_API_KEY and Supabase env vars set (see `.env`).

The MCP connector requires the beta Messages API (mcp-client) and an mcp_toolset entry;
see https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import anthropic

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

if load_dotenv:
    load_dotenv(_REPO_ROOT / ".env")
    load_dotenv(_SCRIPT_DIR / ".env")

MCP_BETA = "mcp-client-2025-11-20"
MCP_SERVER_NAME = "learnachoreo-history"
MCP_URL = (os.getenv("MCP_SERVER_URL") or "https://gargle-sublevel-region.ngrok-free.dev/mcp").strip()
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def main() -> None:
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Using MCP URL: {MCP_URL!r}", flush=True)

    client = anthropic.Anthropic(api_key=api_key)

    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=1000,
        betas=[MCP_BETA],
        mcp_servers=[
            {
                "type": "url",
                "url": MCP_URL,
                "name": MCP_SERVER_NAME,
            }
        ],
        tools=[{"type": "mcp_toolset", "mcp_server_name": MCP_SERVER_NAME}],
        messages=[
            {
                "role": "user",
                "content": (
                    "Use the get_persistent_errors tool with "
                    "user_id='750406f8-3dd9-4f6d-b778-43bead716c02' and "
                    "video_id='9TWj9I3CKzg' and tell me what you find."
                ),
            }
        ],
        timeout=120.0,
    )

    print(f"stop_reason: {response.stop_reason!r}", flush=True)
    print("--- response.content ---", flush=True)
    print(response.content, flush=True)


if __name__ == "__main__":
    main()
