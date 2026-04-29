"""App configuration from environment."""
import os
from pathlib import Path

REDIS_URL: str = (
    os.environ.get("REDIS_URL")
    or os.environ.get("REDIS_PRIVATE_URL")
    or os.environ.get("REDIS_PUBLIC_URL")
    or "redis://localhost:6379/0"
)
OUTPUTS_DIR: Path = Path(os.environ.get("OUTPUTS_DIR", "/data/outputs")).resolve()

# Pipeline runs from repo root and uses DATA_DIR for raw/skeletons/choreo_data
DATA_DIR: str = os.environ.get("DATA_DIR", "/data/pipeline")
SKELETONS_DIR: Path = Path(DATA_DIR) / "skeletons"
