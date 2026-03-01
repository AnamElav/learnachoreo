# Choreo AI API

FastAPI app that accepts a YouTube URL and runs the choreo pipeline in the background via Celery, then exposes job status and result.

## Endpoints

- **POST /process** — Body: `{ "youtube_url": "https://www.youtube.com/watch?v=..." }`. Validates the URL, enqueues a job, returns `{ "job_id": "<uuid>" }`.
- **GET /status/{job_id}** — Returns `{ "job_id", "status": "pending"|"processing"|"complete"|"failed", "choreo_data"?: {...}, "error"?: "..." }`.

## Local development (recommended)

Use the scripts so you don’t need Docker. **Docker is for production deployment only**; for day-to-day dev use `start_local.sh`.

**Prerequisites:** Redis (`brew install redis` on Mac, or `apt install redis-server` on Linux) and a venv with **both** requirement files (the worker runs the full pipeline, which needs the main repo’s deps too):

```bash
cd /path/to/learnachoreo
source .venv/bin/activate
pip install -r requirements.txt
pip install -r choreo-ai-api/requirements.txt
```

**Start backend (Redis + Celery worker + FastAPI):**

```bash
cd /path/to/learnachoreo/choreo-ai-api
./start_local.sh
```

- API: http://localhost:8000  
- Docs: http://localhost:8000/docs  

Press Ctrl+C to stop the server; the script will stop the Celery worker as well.

**Stop worker and server from another terminal:**

```bash
cd /path/to/learnachoreo/choreo-ai-api
./stop_local.sh
```

Pipeline outputs go to `learnachoreo/data/outputs/{job_id}/choreo_data.json`.

---

## Production: run with Docker Compose

From the **repo root** (learnachoreo):

```bash
docker compose -f choreo-ai-api/docker-compose.yml up --build
```

- API: http://localhost:8000  
- Docs: http://localhost:8000/docs  
- Redis: localhost:6379  

Pipeline outputs are written under `/data` in the containers (volume `pipeline_and_outputs`). Completed jobs: `/data/outputs/{job_id}/choreo_data.json`.
