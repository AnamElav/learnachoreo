"""FastAPI app: POST /process, GET /status/{job_id}, GET /video/{job_id}, POST /coaching."""
import json
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load .env: try choreo-ai-api first, then repo root (so choreo-ai-api wins)
_app_dir = Path(__file__).resolve().parent
_choreo_api_dir = _app_dir.parent
_repo_root = _choreo_api_dir.parent
load_dotenv(_repo_root / ".env")
load_dotenv(_choreo_api_dir / ".env")

import redis
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import REDIS_URL, OUTPUTS_DIR, SKELETONS_DIR
from app.tasks import process_video, set_job_status, JOB_KEY_PREFIX

app = FastAPI(title="Choreo AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)

# Loose YouTube URL check
YOUTUBE_PATTERN = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+",
    re.IGNORECASE,
)


class ProcessRequest(BaseModel):
    youtube_url: str


def _get_redis():
    return redis.from_url(REDIS_URL, decode_responses=True)


def _validate_youtube_url(url: str) -> None:
    if not url or not isinstance(url, str):
        raise HTTPException(status_code=400, detail="youtube_url must be a non-empty string")
    url = url.strip()
    if not YOUTUBE_PATTERN.match(url):
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL; expected format like https://www.youtube.com/watch?v=... or https://youtu.be/...",
        )


class CoachingRequest(BaseModel):
    segment_id: int
    reference_angle_summary: dict[str, float]
    user_angles: dict[str, float]
    user_joint_confidence: dict[str, float] | None = None
    valid_joints: list[str] | None = None
    match_level: str  # "good", "developing", "needs_work"
    skill_level: str  # e.g. "beginner"
    style: str | None = None  # optional choreo style, e.g. "contemporary"


class CoachingResponse(BaseModel):
    note: str


ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


def _get_anthropic_api_key() -> str:
    """Read and normalize API key at request time (avoids stale/env issues)."""
    raw = os.getenv("ANTHROPIC_API_KEY") or ""
    return raw.strip().replace("\r", "")


# Log at startup so you can confirm the key is loaded (length only, no leak)
_key_len = len(_get_anthropic_api_key())
print(f"[coaching] ANTHROPIC_API_KEY loaded: length={_key_len} (expected ~100+)")


@app.post("/process")
def process_request(body: ProcessRequest):
    """Accept a YouTube URL, validate it, enqueue a pipeline job, return job_id."""
    url = body.youtube_url
    _validate_youtube_url(url)
    job_id = str(uuid.uuid4())
    set_job_status(job_id, "pending")
    process_video.delay(job_id, url.strip())
    return {"job_id": job_id}


@app.post("/coaching", response_model=CoachingResponse)
async def generate_coaching_note(body: CoachingRequest):
    """
    Generate a short coaching note using Claude based on pose differences.

    Expects:
      - segment_id: current phrase
      - reference_angle_summary: per-joint reference angles for the phrase
      - user_angles: current median user joint angles
      - match_level: 'good' | 'developing' | 'needs_work'
      - skill_level: e.g. 'beginner'
    """
    api_key = _get_anthropic_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not configured on the server.")

    # Map canonical joint keys to the names we expect in angle_summary / user_angles
    joint_keys = {
        "LEFT_ELBOW": "Left elbow",
        "RIGHT_ELBOW": "Right elbow",
        "LEFT_SHOULDER": "Left shoulder",
        "RIGHT_SHOULDER": "Right shoulder",
        "LEFT_KNEE": "Left knee",
        "RIGHT_KNEE": "Right knee",
    }

    def _get(body_map: dict, key: str):
        # Accept uppercase and lowercase (pipeline uses lowercase)
        return body_map.get(key) or body_map.get(key.lower())

    def get_angle(joint_key: str) -> tuple[float | None, float | None, float | None]:
        # Pipeline uses lowercase (left_elbow); frontend sends uppercase (LEFT_ELBOW). Accept both.
        ref_angle = _get(body.reference_angle_summary, joint_key)
        user_angle = _get(body.user_angles, joint_key)
        if ref_angle is None or user_angle is None:
            return None, None, None
        diff = user_angle - ref_angle  # signed: positive = over-extended, negative = under-extended
        return user_angle, ref_angle, diff

    def get_confidence(joint_key: str) -> float | None:
        if not body.user_joint_confidence:
            return None
        v = _get(body.user_joint_confidence, joint_key)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    # Determine which joints are valid to discuss
    valid_keys: list[str] = []
    if body.valid_joints:
        # Normalize provided list (support lower/upper)
        provided = set([k.upper() for k in body.valid_joints if isinstance(k, str)])
        for k in joint_keys.keys():
            if k in provided:
                valid_keys.append(k)
    else:
        for k in joint_keys.keys():
            user_angle, ref_angle, _ = get_angle(k)
            conf = get_confidence(k)
            if user_angle is None or ref_angle is None:
                continue
            if float(user_angle) == 0.0:
                continue
            if conf is not None and conf < 0.3:
                continue
            valid_keys.append(k)

    lines: list[str] = []
    for key in valid_keys:
        label = joint_keys[key]
        user_angle, ref_angle, diff = get_angle(key)
        if user_angle is None or ref_angle is None or diff is None:
            continue
        lines.append(f"- {label}: {user_angle:.1f} vs {ref_angle:.1f} = {diff:.1f}° off")

    diffs_text = "\n".join(lines)

    style = body.style or "contemporary"

    valid_labels = [joint_keys[k] for k in valid_keys]
    valid_note = f"Note: only the following joints had valid detection data: {valid_labels}"

    prompt = (
        f"You are a supportive dance teacher giving real-time feedback to a {body.skill_level} dancer.\n\n"
        f"They are practicing phrase {body.segment_id} of a {style} choreography.\n\n"
        f"{valid_note}\n\n"
        "Here are the joint angle differences between what they're doing and the reference "
        "(positive means they're over-extending, negative means under-extending):\n"
        f"{diffs_text if diffs_text else '(no valid joint angles available)'}\n\n"
        "Only mention joints where the difference is greater than 15°. "
        "If all joints are within 15°, give positive reinforcement about what they're doing well.\n\n"
        "Give exactly 1–2 sentences of specific, actionable feedback. "
        "Name the body part and describe what to do differently in plain movement language — not angles or numbers. "
        "Sound like a human dance teacher, warm and direct. "
        "Never give feedback about joints not listed in the Note line."
    )

    # Debug: print structured joint comparison before calling Claude
    print("JOINT COMPARISON:")
    for key, label in joint_keys.items():
        user_angle, ref_angle, diff = get_angle(key)
        key_snake = label.lower().replace(" ", "_")
        conf = get_confidence(key)
        conf_str = f"{conf:.2f}" if conf is not None else "n/a"
        if user_angle is None or ref_angle is None or diff is None:
            print(f"{key_snake}:  user=n/a   ref=n/a   diff=0.0° conf={conf_str}")
        else:
            print(f"{key_snake}:  user={user_angle:.1f}° ref={ref_angle:.1f}° diff={diff:.1f}° conf={conf_str}")
    print(f"VALID JOINTS: {valid_labels}")

    # Log prompt for debugging
    print("=== Claude coaching prompt ===")
    print(prompt)
    print("=== End prompt ===")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": 150,
                    "system": "You are a supportive dance teacher. Always respond with 1–2 short sentences of coaching.",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                },
            )

        # Log raw response text for debugging (even on non-2xx to help diagnose)
        print("=== Claude raw response ===")
        print(resp.text)
        print("=== End response ===")

        resp.raise_for_status()
        data = resp.json()

        # Anthropic messages API: content is a list of blocks; we expect the first to be text
        note = ""
        try:
            blocks = data.get("content") or []
            if blocks and isinstance(blocks, list):
                first = blocks[0]
                # Support either {"type":"text","text": "..."} or plain string just in case
                if isinstance(first, dict) and "text" in first:
                    note = str(first["text"]).strip()
                else:
                    note = str(first).strip()
        except Exception:
            note = ""

        if not note:
            note = "Great effort on this phrase—keep breathing through the movement and stay connected to your lines."

        return CoachingResponse(note=note)

    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Error calling Claude API: {exc}") from exc


@app.get("/status/{job_id}")
def get_status(job_id: str):
    """Return job status and choreo_data if complete."""
    r = _get_redis()
    key = f"{JOB_KEY_PREFIX}{job_id}"
    raw = r.get(key)
    if not raw:
        raise HTTPException(status_code=404, detail="Job not found")
    data = json.loads(raw)
    status = data.get("status", "unknown")
    out = {"job_id": job_id, "status": status}
    if data.get("error"):
        out["error"] = data["error"]
    if data.get("video_id"):
        out["video_id"] = data["video_id"]
    if status == "complete":
        choreo_file = OUTPUTS_DIR / job_id / "choreo_data.json"
        if choreo_file.is_file():
            out["choreo_data"] = json.loads(choreo_file.read_text(encoding="utf-8"))
        else:
            out["error"] = "choreo_data.json not found"
    return out


@app.get("/skeleton/{video_id}")
def get_skeleton(video_id: str):
    """Return per-frame skeleton landmarks for the video. Used for overlay sync."""
    if not video_id or "/" in video_id or ".." in video_id:
        raise HTTPException(status_code=400, detail="Invalid video_id")
    path = SKELETONS_DIR / f"{video_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Skeleton not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/video/{job_id}")
def get_video(job_id: str, request: Request):
    """Stream the video file for a job with HTTP range request support for seeking."""
    if not job_id or "/" in job_id or ".." in job_id:
        raise HTTPException(status_code=400, detail="Invalid job_id")

    video_path = OUTPUTS_DIR / job_id / "video.mp4"
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        # Parse range header: "bytes=start-end"
        range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if not range_match:
            raise HTTPException(status_code=416, detail="Invalid range header")

        start = int(range_match.group(1))
        end_str = range_match.group(2)
        end = int(end_str) if end_str else file_size - 1

        if start >= file_size or end >= file_size or start > end:
            raise HTTPException(status_code=416, detail="Range not satisfiable")

        content_length = end - start + 1

        def iter_file_range():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                chunk_size = 64 * 1024  # 64KB chunks
                while remaining > 0:
                    read_size = min(chunk_size, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_file_range(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            },
        )
    else:
        # No range header, return full file
        def iter_file():
            with open(video_path, "rb") as f:
                chunk_size = 64 * 1024
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    yield data

        return StreamingResponse(
            iter_file(),
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            },
        )
