"""FastAPI app: POST /process, GET /status/{job_id}."""
import json
import re
import uuid

import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import REDIS_URL, OUTPUTS_DIR
from app.tasks import process_video, set_job_status, JOB_KEY_PREFIX

app = FastAPI(title="Choreo AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
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


@app.post("/process")
def process_request(body: ProcessRequest):
    """Accept a YouTube URL, validate it, enqueue a pipeline job, return job_id."""
    url = body.youtube_url
    _validate_youtube_url(url)
    job_id = str(uuid.uuid4())
    set_job_status(job_id, "pending")
    process_video.delay(job_id, url.strip())
    return {"job_id": job_id}


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
