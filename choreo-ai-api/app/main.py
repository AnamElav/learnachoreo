"""FastAPI app: POST /process, GET /status/{job_id}, GET /video/{job_id}."""
import json
import os
import re
import uuid

import redis
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
