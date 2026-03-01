"""
Phrase detection from skeleton + audio.

Given a skeleton JSON (from pose_extraction.pose_estimator) and the
corresponding raw video, this module:

- Uses librosa to detect musical beats from the video's audio.
- Uses skeleton landmarks to detect "pose reset" moments where overall
  movement velocity drops (neutral positions between phrases).
- Combines these signals to produce a segments JSON with an array of:
  {
    "segment_id": int,
    "start_frame": int,
    "end_frame": int,
    "start_time_ms": float,
    "end_time_ms": float,
    "beat_count": int,
  }

Segments aim for roughly 8 beats each where the music allows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, TypedDict

import librosa
import numpy as np

from utils.paths import get_raw_dir, get_segments_dir, get_segments_path


class SegmentationError(Exception):
    """Raised when phrase detection fails (missing files, malformed JSON, etc.)."""

    pass


class Landmark(TypedDict):
    name: str
    x: float
    y: float
    z: float
    visibility: float


class Frame(TypedDict, total=False):
    frame_number: int
    timestamp_ms: float
    landmarks: List[Landmark]
    overall_confidence: float
    angles: dict


class SegmentDict(TypedDict):
    segment_id: int
    start_frame: int
    end_frame: int
    start_time_ms: float
    end_time_ms: float
    beat_count: int


@dataclass
class Segment:
    segment_id: int
    start_frame: int
    end_frame: int
    start_time_ms: float
    end_time_ms: float
    beat_count: int


def _frame_times_sec(frames: List[Frame]) -> np.ndarray:
    """Return per-frame timestamps in seconds (aligned to frames list order)."""
    return np.array([float(f.get("timestamp_ms", 0.0)) for f in frames], dtype=float) / 1000.0


def _load_skeleton(path: Path) -> List[Frame]:
    if not path.is_file():
        raise SegmentationError(f"Skeleton JSON not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise SegmentationError(f"Failed to parse skeleton JSON: {e}") from e
    if not isinstance(data, list):
        raise SegmentationError("Skeleton JSON must be a list of frame objects.")
    return data  # type: ignore[return-value]


def _find_video_path(video_id: str) -> Path:
    """Infer the raw video path from its ID in data/raw/."""
    raw_dir = get_raw_dir()
    if not raw_dir.is_dir():
        raise SegmentationError(f"Raw video directory not found: {raw_dir}")
    candidates = [p for p in raw_dir.iterdir() if p.is_file() and p.stem == video_id]
    if not candidates:
        raise SegmentationError(f"No raw video found for video_id={video_id} in {raw_dir}")
    # Prefer mp4 if multiple candidates exist.
    mp4s = [p for p in candidates if p.suffix.lower() == ".mp4"]
    if mp4s:
        return mp4s[0]
    return candidates[0]


def _detect_beats(video_path: Path) -> np.ndarray:
    """Return beat times in seconds using librosa."""
    try:
        y, sr = librosa.load(str(video_path), mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return beat_times
    except Exception as e:  # noqa: BLE001
        raise SegmentationError(f"Failed to analyze audio for beats: {e}") from e


def _map_beats_to_frames(beat_times_sec: np.ndarray, frames: List[Frame]) -> List[int]:
    """Map each beat time to the index of the closest frame."""
    if len(frames) == 0 or len(beat_times_sec) == 0:
        return []
    frame_times = np.array([float(f.get("timestamp_ms", 0.0)) for f in frames]) / 1000.0
    beat_indices: List[int] = []
    for t in beat_times_sec:
        idx = int(np.argmin(np.abs(frame_times - t)))
        beat_indices.append(idx)
    return beat_indices


def _pose_velocities(frames: List[Frame]) -> np.ndarray:
    """
    Approximate body movement velocity between consecutive frames.

    We average positions of a set of key joints and take the L2 distance between
    consecutive mean positions.
    """
    if len(frames) < 2:
        return np.zeros(0, dtype=float)

    key_joint_names = {
        "NOSE",
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
    }

    centers: List[np.ndarray | None] = []
    for frame in sorted(frames, key=lambda f: f.get("frame_number", 0)):
        lms = frame.get("landmarks") or []
        pts = [
            np.array([lm["x"], lm["y"], lm["z"]], dtype=float)
            for lm in lms
            if lm["name"] in key_joint_names
        ]
        if not pts:
            centers.append(None)
        else:
            centers.append(np.mean(pts, axis=0))

    vels: List[float] = []
    for prev, curr in zip(centers, centers[1:]):
        if prev is None or curr is None:
            vels.append(0.0)
        else:
            vels.append(float(np.linalg.norm(curr - prev)))
    return np.array(vels, dtype=float)


def _pose_reset_boundaries(
    frames: List[Frame],
    velocity_threshold: float = 0.01,
    min_separation_sec: float = 1.0,
) -> List[int]:
    """
    Return frame indices that look like "pose reset" moments: low velocity minima.
    """
    vels = _pose_velocities(frames)
    if vels.size == 0:
        return []

    boundaries: List[int] = []
    times = _frame_times_sec(frames)
    last_t = -1e9
    # vels has length len(frames) - 1; velocity at transition i is between frame i and i+1.
    for i in range(1, len(vels) - 1):
        if vels[i] < velocity_threshold and vels[i] <= vels[i - 1] and vels[i] <= vels[i + 1]:
            b = i + 1  # boundary at frame after the low-velocity transition
            t = float(times[b]) if b < len(times) else None
            if t is None:
                continue
            if t - last_t >= min_separation_sec:
                boundaries.append(b)
                last_t = t
    return boundaries


def _nearest_frame_index(frame_times_sec: np.ndarray, t_sec: float) -> int:
    return int(np.argmin(np.abs(frame_times_sec - t_sec)))


def _beats_near_pose_resets(
    beat_times_sec: np.ndarray,
    pose_reset_times_sec: np.ndarray,
    window_sec: float = 0.5,
) -> np.ndarray:
    """
    Return beat times that have a pose reset within +/- window_sec.
    """
    if beat_times_sec.size == 0 or pose_reset_times_sec.size == 0:
        return np.array([], dtype=float)
    selected: List[float] = []
    for rt in pose_reset_times_sec:
        i = int(np.argmin(np.abs(beat_times_sec - rt)))
        if abs(float(beat_times_sec[i] - rt)) <= window_sec:
            selected.append(float(beat_times_sec[i]))
    if not selected:
        return np.array([], dtype=float)
    return np.array(sorted(set(selected)), dtype=float)


def _build_segments_with_rules(
    frames: List[Frame],
    frame_times_sec: np.ndarray,
    beat_times_sec: np.ndarray,
    candidate_boundary_times_sec: np.ndarray,
    min_seg_sec: float = 4.0,
    target_low_sec: float = 8.0,
    target_high_sec: float = 16.0,
    force_break_after_sec: float = 20.0,
) -> List[Segment]:
    """
    Greedy segmentation:
    - Prefer boundaries (beat+reset) that yield 8-16s segments.
    - Enforce min 4s.
    - If no pose-reset boundary for >20s, force break at nearest beat.
    """
    n = len(frames)
    if n == 0:
        return []

    start_idx = 0
    start_t = float(frame_times_sec[start_idx])
    end_t = float(frame_times_sec[-1])
    segments: List[Segment] = []
    seg_id = 0

    # Boundaries are times; ensure sorted and within range.
    cand = np.array([t for t in candidate_boundary_times_sec if start_t < t < end_t], dtype=float)
    cand.sort()

    beats = np.array([t for t in beat_times_sec if start_t < t < end_t], dtype=float)
    beats.sort()

    while start_idx < n - 1:
        start_t = float(frame_times_sec[start_idx])

        # Ideal end window.
        ideal_lo = start_t + target_low_sec
        ideal_hi = start_t + target_high_sec
        min_end = start_t + min_seg_sec
        force_t = start_t + force_break_after_sec

        # Candidate boundaries satisfying min duration.
        cand_after_min = cand[cand >= min_end]

        # Pick a candidate within ideal window if possible.
        within = cand_after_min[(cand_after_min >= ideal_lo) & (cand_after_min <= ideal_hi)]
        if within.size > 0:
            # Choose closest to the middle of target window.
            target_mid = start_t + (target_low_sec + target_high_sec) / 2.0
            end_t_candidate = float(within[np.argmin(np.abs(within - target_mid))])
        else:
            # No good candidate: if we've exceeded the force-break window, force at nearest beat.
            if force_t <= end_t and beats.size > 0:
                # Choose nearest beat to force_t, but still respect min duration.
                i = int(np.argmin(np.abs(beats - force_t)))
                end_t_candidate = float(beats[i])
                if end_t_candidate < min_end:
                    # pick the next beat after min_end
                    later = beats[beats >= min_end]
                    end_t_candidate = float(later[0]) if later.size else min(end_t, min_end)
            else:
                # Otherwise, take the earliest candidate after ideal_lo if it exists,
                # else take the earliest after min_end, else finish.
                later_ideal = cand[cand >= ideal_lo]
                if later_ideal.size > 0:
                    end_t_candidate = float(later_ideal[0])
                else:
                    later_min = cand_after_min
                    if later_min.size > 0:
                        end_t_candidate = float(later_min[0])
                    else:
                        end_t_candidate = end_t

        end_idx = _nearest_frame_index(frame_times_sec, end_t_candidate)
        if end_idx <= start_idx:
            break

        # Build segment.
        start_frame = int(frames[start_idx].get("frame_number", start_idx))
        end_frame = int(frames[end_idx].get("frame_number", end_idx))
        start_time_ms = float(frames[start_idx].get("timestamp_ms", start_t * 1000.0))
        end_time_ms = float(frames[end_idx].get("timestamp_ms", end_t_candidate * 1000.0))

        # Beat count based on beat times within segment.
        if beats.size > 0:
            beat_count = int(((beats >= start_t) & (beats <= float(frame_times_sec[end_idx]))).sum())
        else:
            beat_count = 0

        segments.append(
            Segment(
                segment_id=seg_id,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                beat_count=beat_count,
            )
        )
        seg_id += 1

        # Advance.
        start_idx = end_idx

    return segments


def detect_phrases(
    skeleton_path: str | Path,
    pose_reset_threshold: float = 0.01,
) -> str:
    """
    Detect learnable phrases from a skeleton JSON and its source audio.

    Segmentation rules:
    - Minimum segment duration: 4 seconds
    - Target segment duration: 8–16 seconds
    - Segment boundaries are only created when BOTH:
      - a beat boundary exists, AND
      - a pose reset exists within +/- 0.5 seconds of that beat
    - If no pose-reset boundary is found for >20 seconds, force a break at the nearest beat.

    Args:
        skeleton_path: Path to skeleton JSON (list of frames).
        pose_reset_threshold: Velocity threshold for pose resets.

    Returns:
        Absolute path to the written segments JSON file.
    """
    skeleton_path = Path(skeleton_path).resolve()
    frames = _load_skeleton(skeleton_path)
    if not frames:
        raise SegmentationError("Skeleton contains no frames.")

    frames = sorted(frames, key=lambda f: f.get("frame_number", 0))
    video_id = skeleton_path.stem
    video_path = _find_video_path(video_id)

    # Beat detection from audio.
    beat_times_sec = _detect_beats(video_path)
    frame_times_sec = _frame_times_sec(frames)

    # Pose reset boundaries -> times.
    pose_bounds = _pose_reset_boundaries(frames, velocity_threshold=pose_reset_threshold, min_separation_sec=1.0)
    pose_reset_times_sec = np.array(
        [float(frame_times_sec[i]) for i in pose_bounds if 0 <= i < len(frame_times_sec)],
        dtype=float,
    )

    # Candidate boundaries must have BOTH a beat and a pose reset within 0.5s.
    candidate_boundary_times_sec = _beats_near_pose_resets(
        beat_times_sec=beat_times_sec,
        pose_reset_times_sec=pose_reset_times_sec,
        window_sec=0.5,
    )

    # Build segments with rules (min 4s, target 8-16s, force at nearest beat after 20s).
    segments = _build_segments_with_rules(
        frames=frames,
        frame_times_sec=frame_times_sec,
        beat_times_sec=beat_times_sec,
        candidate_boundary_times_sec=candidate_boundary_times_sec,
        min_seg_sec=4.0,
        target_low_sec=8.0,
        target_high_sec=16.0,
        force_break_after_sec=20.0,
    )

    # Ensure segments directory exists and write segments JSON.
    get_segments_dir().mkdir(parents=True, exist_ok=True)
    segments_path = get_segments_path(video_id)
    payload: List[SegmentDict] = [asdict(s) for s in segments]  # type: ignore[assignment]
    segments_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print average segment duration for verification.
    if segments:
        durations_sec = [(s.end_time_ms - s.start_time_ms) / 1000.0 for s in segments]
        avg_sec = float(np.mean(durations_sec))
        print(f"       Avg segment duration: {avg_sec:.2f}s")

    return str(segments_path.resolve())

