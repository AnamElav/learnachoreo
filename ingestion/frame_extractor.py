"""
Extract video frames using FFmpeg.

- extract_frames(): decode to JPEGs on disk (legacy; uses significant space).
- stream_frames(): decode one frame at a time via FFmpeg stdout; no frame files
  on disk. Use this for low-disk pipeline runs.
"""

import re
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np

from utils.paths import get_frames_dir


class FrameExtractionError(Exception):
    """Raised when frame extraction fails (missing file, FFmpeg error, etc.)."""

    pass


def _video_id_from_path(video_path: str | Path) -> str:
    """Derive a safe video_id from the video file path (stem, sanitized)."""
    stem = Path(video_path).resolve().stem
    # Keep only alphanumeric, underscore, hyphen
    safe = re.sub(r"[^\w\-]", "_", stem)
    return safe or "video"


def extract_frames(
    video_path: str | Path,
    video_id: str | None = None,
    fps: float = 10,
    out_dir: Path | None = None,
) -> tuple[list[str], int]:
    """
    Extract frames from a video file as JPEGs using FFmpeg.

    Args:
        video_path: Path to the video file (e.g. from download_youtube_video).
        video_id: Identifier used for the output subdir (data/frames/{video_id}/).
            If None, derived from the video filename stem (sanitized).
        fps: Frames per second to extract. Default 30.
        out_dir: Base directory for frame output. Defaults to data/frames (see utils.paths).
            Frames are written to out_dir / video_id /.

    Returns:
        (frame_paths, count) where frame_paths is a list of absolute paths to
        JPEG files in order (frame_0001.jpg, frame_0002.jpg, ...), and count is
        the total number of frames.

    Raises:
        FrameExtractionError: If the video file is missing, unreadable, or
            FFmpeg fails (e.g. not installed or invalid format).
    """
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FrameExtractionError(f"Video file not found: {video_path}")

    video_id = video_id or _video_id_from_path(video_path)
    if out_dir is None:
        frames_dir = get_frames_dir(video_id)
    else:
        frames_dir = Path(out_dir).resolve() / video_id
    frames_dir = Path(frames_dir).resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Output pattern: frame_0001.jpg, frame_0002.jpg, ...
    out_pattern = str(frames_dir / "frame_%04d.jpg")

    # Clamp to 10fps to keep extraction/processing fast.
    if fps > 10:
        fps = 10

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        out_pattern,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except FileNotFoundError:
        raise FrameExtractionError(
            "FFmpeg is not installed or not on PATH. Install it (e.g. brew install ffmpeg on macOS)."
        ) from None
    except subprocess.TimeoutExpired:
        raise FrameExtractionError("Frame extraction timed out.") from None

    if result.returncode != 0:
        stderr = (result.stderr or "")[-500:]
        raise FrameExtractionError(
            f"FFmpeg failed (exit code {result.returncode}). {stderr}"
        )

    # Collect frame paths in order
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"), key=lambda p: p.name)
    paths_str = [str(p.resolve()) for p in frame_paths]
    return paths_str, len(frame_paths)


def _get_video_dimensions(video_path: Path) -> tuple[int, int]:
    """Return (width, height) via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise FrameExtractionError(
            f"ffprobe failed: {(result.stderr or result.stdout or '').strip()}"
        )
    line = (result.stdout or "").strip()
    if not line:
        raise FrameExtractionError("ffprobe returned no stream dimensions")
    parts = line.split(",")
    if len(parts) != 2:
        raise FrameExtractionError(f"Unexpected ffprobe output: {line}")
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError:
        raise FrameExtractionError(f"Invalid dimensions: {line}")
    if w <= 0 or h <= 0:
        raise FrameExtractionError(f"Invalid dimensions: {w}x{h}")
    return w, h


def stream_frames(
    video_path: str | Path,
    fps: float = 10,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """
    Yield frames from a video one at a time via FFmpeg stdout (no frame files on disk).

    Args:
        video_path: Path to the video file.
        fps: Target frames per second (FFmpeg -vf fps filter).

    Yields:
        (frame_index, timestamp_ms, rgb_array) where rgb_array is (height, width, 3) uint8.
    """
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FrameExtractionError(f"Video file not found: {video_path}")

    # Clamp to 10fps to keep extraction/processing fast.
    if fps > 10:
        fps = 10

    width, height = _get_video_dimensions(video_path)
    frame_bytes = width * height * 3

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-an", "-sn",
        "-",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=frame_bytes,
        )
    except FileNotFoundError:
        raise FrameExtractionError(
            "FFmpeg is not installed or not on PATH. Install it (e.g. brew install ffmpeg on macOS)."
        ) from None

    try:
        frame_index = 0
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) != frame_bytes:
                break
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            timestamp_ms = (frame_index / fps) * 1000.0
            yield frame_index, timestamp_ms, arr
            frame_index += 1
    finally:
        proc.terminate()
        proc.wait(timeout=5)
