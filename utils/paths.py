"""Path helpers for data directories. Respects DATA_DIR from environment."""

import os
from pathlib import Path


def get_data_dir() -> Path:
    """Base data directory (e.g. ./data or DATA_DIR env)."""
    return Path(os.environ.get("DATA_DIR", "data")).resolve()


def get_raw_dir() -> Path:
    """Directory for downloaded raw videos."""
    return get_data_dir() / "raw"


def get_frames_dir(video_id: str) -> Path:
    """Directory for extracted frames for a given video_id."""
    return get_data_dir() / "frames" / video_id


def get_models_dir() -> Path:
    """Directory for downloaded models (e.g. pose_landmarker_lite.task)."""
    return get_data_dir() / "models"


def get_skeletons_dir() -> Path:
    """Directory for pose skeleton JSON outputs."""
    return get_data_dir() / "skeletons"


def get_skeleton_path(video_id: str) -> Path:
    """JSON path for a video's pose skeleton timeline."""
    return get_skeletons_dir() / f"{video_id}.json"


def get_segments_dir() -> Path:
    """Directory for phrase/segment metadata."""
    return get_data_dir() / "segments"


def get_segments_path(video_id: str) -> Path:
    """JSON path for a video's phrase segmentation."""
    return get_segments_dir() / f"{video_id}.segments.json"
