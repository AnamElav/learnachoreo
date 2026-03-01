"""
Assess video/skeleton quality before treating choreography as learnable content.

Given a skeleton JSON (output of pose_extraction.pose_estimator), this module
computes:
- confidence_score: average overall_confidence across all frames
- continuity_score: how continuous the motion is (penalizes large jumps that
  look like camera cuts, based on a key landmark trajectory)
- coverage_score: fraction of original frames that passed the confidence
  threshold, estimated from frame_number gaps
- overall_score: simple average of the three scores, in [0, 1]
- flags: list containing any of:
  - "low_lighting" (low confidence)
  - "camera_cuts_detected" (sudden large jumps)
  - "low_coverage" (few frames passed confidence threshold)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, TypedDict

import numpy as np


class VideoQualityError(Exception):
    """Raised when video quality scoring fails (e.g., malformed skeleton JSON)."""

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
    angles: Dict[str, float]


class QualityAssessment(TypedDict):
    overall_score: float
    confidence_score: float
    continuity_score: float
    coverage_score: float
    flags: List[str]


def _load_skeleton(skeleton_path: Path) -> List[Frame]:
    if not skeleton_path.is_file():
        raise VideoQualityError(f"Skeleton JSON not found: {skeleton_path}")
    try:
        data = json.loads(skeleton_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise VideoQualityError(f"Failed to parse skeleton JSON: {e}") from e
    if not isinstance(data, list):
        raise VideoQualityError("Skeleton JSON must be a list of frame objects.")
    return data  # type: ignore[return-value]


def _confidence_score(frames: List[Frame]) -> float:
    if not frames:
        return 0.0
    vals = [float(frame.get("overall_confidence", 0.0)) for frame in frames]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _coverage_score(frames: List[Frame]) -> float:
    if not frames:
        return 0.0
    # Estimate total original frames from frame_number range.
    numbers = [int(frame.get("frame_number", 0)) for frame in frames]
    max_idx = max(numbers)
    total_estimated = max_idx + 1  # frame_number is 0-based
    if total_estimated <= 0:
        return 0.0
    return float(len(frames) / total_estimated)


def _continuity_score(frames: List[Frame], jump_threshold: float = 0.15) -> tuple[float, bool]:
    """
    Compute continuity_score and whether camera cuts were detected.

    We track the 2D trajectory of a key landmark (NOSE, or fallback) and mark a
    "cut" whenever the normalized distance between consecutive frames exceeds
    jump_threshold.
    """
    if len(frames) < 2:
        return 1.0, False

    key_names = ("NOSE", "MID_HIP", "PELVIS")
    positions: list[np.ndarray | None] = []

    for frame in sorted(frames, key=lambda f: f.get("frame_number", 0)):
        lms = frame.get("landmarks") or []
        lm_map: Dict[str, Landmark] = {lm["name"]: lm for lm in lms}
        lm = None
        for name in key_names:
            lm = lm_map.get(name)
            if lm is not None:
                break
        if lm is None and lms:
            # Fallback to first landmark
            lm = lms[0]
        if lm is None:
            positions.append(None)
        else:
            positions.append(np.array([float(lm["x"]), float(lm["y"])], dtype=float))

    jumps = 0
    transitions = 0
    for prev, curr in zip(positions, positions[1:]):
        if prev is None or curr is None:
            continue
        transitions += 1
        dist = float(np.linalg.norm(curr - prev))
        if dist > jump_threshold:
            jumps += 1

    if transitions == 0:
        return 1.0, False

    continuity = float(max(0.0, 1.0 - jumps / transitions))
    return continuity, jumps > 0


def score_video_quality(skeleton_path: str | Path) -> QualityAssessment:
    """
    Compute a quality assessment dictionary for a skeleton JSON.

    Returns:
        {
          "overall_score": float in [0, 1],
          "confidence_score": float in [0, 1],
          "continuity_score": float in [0, 1],
          "coverage_score": float in [0, 1],
          "flags": ["low_lighting", "camera_cuts_detected", "low_coverage", ...]
        }
    """
    path = Path(skeleton_path).resolve()
    frames = _load_skeleton(path)

    confidence = _confidence_score(frames)
    coverage = _coverage_score(frames)
    continuity, cuts_detected = _continuity_score(frames)

    # Overall score: simple average of the three.
    overall = float((confidence + coverage + continuity) / 3.0) if frames else 0.0

    flags: List[str] = []
    if confidence < 0.7:
        flags.append("low_lighting")
    if cuts_detected or continuity < 0.8:
        flags.append("camera_cuts_detected")
    if coverage < 0.7:
        flags.append("low_coverage")

    return {
        "overall_score": overall,
        "confidence_score": confidence,
        "continuity_score": continuity,
        "coverage_score": coverage,
        "flags": flags,
    }

