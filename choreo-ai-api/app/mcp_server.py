"""
MCP server exposing LearnAChoreo user history from Supabase for the coaching agent.

Run (streamable HTTP):  PYTHONPATH=<choreo-ai-api> uvicorn app.mcp_server:mcp_app --host 0.0.0.0 --port 8001
Also:                     PYTHONPATH=. python -m app.mcp_server  (uses FastMCP.run streamable-http)
Smoke test:   PYTHONPATH=. python -m app.mcp_server --test-supabase
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# Running as `python app/mcp_server.py` leaves `app` off sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.supabase_client import get_supabase_client

mcp = FastMCP(
    "LearnAChoreo History",
    instructions="Tools to read user joint history, phrase attempts, and coaching trends from Supabase.",
    host="0.0.0.0",
    port=8001,
)


def _norm_joint(joint_name: str) -> str:
    return (joint_name or "").strip().upper()


def _session_ids_for_user_video(sb: Any, user_id: str, video_id: str) -> list[str]:
    res = (
        sb.table("user_sessions")
        .select("id")
        .eq("user_id", user_id)
        .eq("video_id", video_id)
        .execute()
    )
    rows = res.data or []
    return [str(r["id"]) for r in rows if r.get("id") is not None]


def _coaching_note_text(notes_field: Any) -> str | None:
    if notes_field is None:
        return None
    if isinstance(notes_field, dict):
        t = notes_field.get("note_text")
        return str(t).strip() if t else None
    if isinstance(notes_field, list):
        parts: list[str] = []
        for item in notes_field:
            if isinstance(item, dict):
                t = item.get("note_text")
                if t:
                    parts.append(str(t).strip())
        return "\n".join(parts) if parts else None
    return None


def _trend_from_scores_chronological(scores: list[float]) -> str:
    """
    overall_score is average absolute joint difference (degrees); lower is better.
    improving  -> scores decrease over time
    regressing -> scores increase over time
    stuck      -> flat / ambiguous
    """
    n = len(scores)
    if n < 2:
        return "stuck"
    span = max(scores) - min(scores)
    per_step = (scores[-1] - scores[0]) / (n - 1)
    eps_step = 0.35
    eps_span = 0.5
    if span < eps_span:
        return "stuck"
    if per_step < -eps_step:
        return "improving"
    if per_step > eps_step:
        return "regressing"
    return "stuck"


@mcp.tool()
def get_user_joint_history(
    user_id: str,
    video_id: str,
    joint_name: str,
) -> dict[str, Any] | None:
    """Return avg_diff, attempt_count, last_seen_at for one joint, or None if no row exists."""
    sb = get_supabase_client()
    jn = _norm_joint(joint_name)
    res = (
        sb.table("user_joint_history")
        .select("avg_diff, attempt_count, last_seen_at")
        .eq("user_id", user_id)
        .eq("video_id", video_id)
        .eq("joint_name", jn)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return None
    row = rows[0]
    return {
        "avg_diff": float(row.get("avg_diff") or 0.0),
        "attempt_count": int(row.get("attempt_count") or 0),
        "last_seen_at": row.get("last_seen_at"),
    }


@mcp.tool()
def get_persistent_errors(
    user_id: str,
    video_id: str,
    threshold_deg: float = 30.0,
    min_attempts: int = 2,
) -> list[dict[str, Any]]:
    """
    Joints where avg_diff is above threshold and the user has attempted enough times.
    Ordered by avg_diff descending (most problematic first).
    """
    sb = get_supabase_client()
    res = (
        sb.table("user_joint_history")
        .select("joint_name, avg_diff, attempt_count, last_seen_at")
        .eq("user_id", user_id)
        .eq("video_id", video_id)
        .gt("avg_diff", threshold_deg)
        .gte("attempt_count", min_attempts)
        .order("avg_diff", desc=True)
        .execute()
    )
    return list(res.data or [])


@mcp.tool()
def get_phrase_attempts(
    user_id: str,
    video_id: str,
    segment_id: str | int,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Last N phrase attempts for a phrase (segment), joined with coaching_notes when present.
    """
    sb = get_supabase_client()
    session_ids = _session_ids_for_user_video(sb, user_id, video_id)
    if not session_ids:
        return []

    lim = max(1, min(int(limit), 50))
    seg_key: str | int = int(segment_id) if isinstance(segment_id, (int, float)) else int(str(segment_id).strip())
    res = (
        sb.table("phrase_attempts")
        .select("attempt_number, overall_score, coaching_notes(note_text)")
        .in_("session_id", session_ids)
        .eq("segment_id", seg_key)
        .order("id", desc=True)
        .limit(lim)
        .execute()
    )
    out: list[dict[str, Any]] = []
    for row in res.data or []:
        out.append(
            {
                "attempt_number": int(row.get("attempt_number") or 0),
                "overall_score": float(row.get("overall_score") or 0.0),
                "coaching_note": _coaching_note_text(row.get("coaching_notes")),
            }
        )
    return out


@mcp.tool()
def get_improvement_trend(
    user_id: str,
    video_id: str,
    joint_name: str,
) -> dict[str, Any]:
    """
    Last 5 phrase attempts (most recent first in query) that have a coaching note for this joint.
    overall_score is avg abs joint diff (degrees): lower is better.
    Returns chronological overall_scores and trend: improving | stuck | regressing.
    """
    sb = get_supabase_client()
    jn = _norm_joint(joint_name)
    session_ids = _session_ids_for_user_video(sb, user_id, video_id)
    if not session_ids:
        return {"trend": "stuck", "overall_scores": []}

    res = (
        sb.table("phrase_attempts")
        .select("id, overall_score, coaching_notes!inner(joint_name)")
        .in_("session_id", session_ids)
        .eq("coaching_notes.joint_name", jn)
        .order("id", desc=True)
        .limit(25)
        .execute()
    )
    rows_raw = list(res.data or [])
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for r in rows_raw:
        pid = str(r.get("id"))
        if pid in seen:
            continue
        seen.add(pid)
        rows.append(r)
        if len(rows) >= 5:
            break
    rows.reverse()
    scores = [float(r.get("overall_score") or 0.0) for r in rows]
    return {
        "trend": _trend_from_scores_chronological(scores),
        "overall_scores": scores,
        "joint_name": jn,
    }


# ASGI app for `uvicorn app.mcp_server:mcp_app` (e.g. start_local.sh).
mcp_app = mcp.streamable_http_app()


if __name__ == "__main__":
    # Smoke test: verify Supabase queries (set env vars first).
    _TEST_USER_ID = "00000000-0000-0000-0000-000000000001"
    _TEST_VIDEO_ID = os.getenv("MCP_TEST_VIDEO_ID", "test-video-placeholder")

    if "--test-supabase" in sys.argv:
        print(
            f"get_persistent_errors(user_id={_TEST_USER_ID!r}, video_id={_TEST_VIDEO_ID!r})",
            flush=True,
        )
        try:
            r = get_persistent_errors(_TEST_USER_ID, _TEST_VIDEO_ID)
            print(r, flush=True)
        except RuntimeError as exc:
            if "SUPABASE_URL" in str(exc) or "SUPABASE_SERVICE_ROLE_KEY" in str(exc):
                print(exc, flush=True)
                sys.exit(2)
            raise
    else:
        # Host/port are FastMCP constructor settings; `run()` only selects transport.
        mcp.run(transport="streamable-http")
