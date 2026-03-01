# Choreo AI Web

Next.js frontend for the Choreo AI pipeline. Paste a YouTube URL to analyze choreography and see quality score and segment count.

## Setup

1. `.env.local` sets `NEXT_PUBLIC_API_URL=http://localhost:8000`. Edit it if your API runs elsewhere.
2. Ensure the API is running (e.g. `./choreo-ai-api/start_local.sh` from the repo root).

## Run

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Enter a YouTube URL, click **Analyze**, and wait for the result (polling runs every 3 seconds).
