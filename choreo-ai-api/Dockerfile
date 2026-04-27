# Build from repo root (learnachoreo). docker-compose uses context: .. and dockerfile: choreo-ai-api/Dockerfile
# Minimal runtime image: python:3.11-slim + ffmpeg + only required pip packages.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runtime dependencies from requirements file.
COPY choreo-ai-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code (respects .dockerignore in context)
COPY . .

WORKDIR /app/choreo-ai-api
ENV PYTHONPATH=/app
ENV DATA_DIR=/data/pipeline
ENV OUTPUTS_DIR=/data/outputs

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
