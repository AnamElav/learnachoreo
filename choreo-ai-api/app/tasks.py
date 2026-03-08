"""Background tasks: run pipeline and save output to job output dir."""
import json
import os
import sys
from pathlib import Path

import redis

from app.celery_app import celery_app
from app.config import OUTPUTS_DIR, DATA_DIR, REDIS_URL

JOB_KEY_PREFIX = "job:"
JOB_TTL_SECONDS = 86400 * 7  # 7 days


def _get_redis():
    return redis.from_url(REDIS_URL, decode_responses=True)


def set_job_status(job_id: str, status: str, error: str | None = None, video_id: str | None = None) -> None:
    r = _get_redis()
    key = f"{JOB_KEY_PREFIX}{job_id}"
    payload = {"status": status}
    if error:
        payload["error"] = error
    if video_id:
        payload["video_id"] = video_id
    r.set(key, json.dumps(payload), ex=JOB_TTL_SECONDS)


@celery_app.task(bind=True)
def process_video(self, job_id: str, youtube_url: str) -> None:
    """Run the choreo pipeline for the given URL and save choreo_data + video to OUTPUTS_DIR/job_id/."""
    set_job_status(job_id, "processing")

    # Ensure pipeline package is importable (run from repo root in Docker).
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Create output directory for this job
    out_dir = OUTPUTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        os.environ["DATA_DIR"] = DATA_DIR
        from pipeline import run_pipeline

        # Pass output_dir so pipeline writes choreo_data.json and video.mp4 directly there
        choreo_path = run_pipeline(youtube_url, output_dir=out_dir)
    except Exception as e:
        set_job_status(job_id, "failed", error=str(e))
        raise

    # Extract video_id from choreo_data
    choreo_data = json.loads(choreo_path.read_text(encoding="utf-8"))
    video_id = choreo_data.get("video_id", "")

    set_job_status(job_id, "complete", video_id=video_id)
