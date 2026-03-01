"""
Compute joint angles from a pose skeleton JSON produced by pose_estimator.

For each frame, this module:
- Reads landmarks (name, x, y, z, visibility)
- Computes the following angles (in degrees):
  - left_elbow: angle at LEFT_ELBOW from LEFT_SHOULDER-LEFT_ELBOW-LEFT_WRIST
  - right_elbow: angle at RIGHT_ELBOW from RIGHT_SHOULDER-RIGHT_ELBOW-RIGHT_WRIST
  - left_knee: angle at LEFT_KNEE from LEFT_HIP-LEFT_KNEE-LEFT_ANKLE
  - right_knee: angle at RIGHT_KNEE from RIGHT_HIP-RIGHT_KNEE-RIGHT_ANKLE
  - left_shoulder: angle at LEFT_SHOULDER from LEFT_ELBOW-LEFT_SHOULDER-LEFT_HIP
  - right_shoulder: angle at RIGHT_SHOULDER from RIGHT_ELBOW-RIGHT_SHOULDER-RIGHT_HIP
  - hip_tilt: magnitude of the angle (in the image plane) of the vector
    RIGHT_HIP -> LEFT_HIP relative to the horizontal axis

Angles are added to each frame object under an ``"angles"`` key.
"""

from __future__ import annotations

import json
from math import atan2, degrees
from pathlib import Path
from typing import Dict, List, TypedDict

import numpy as np


class JointAngleError(Exception):
    """Raised when joint angle computation fails (e.g., malformed skeleton JSON)."""

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


def angle_at_point(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the angle in degrees at point B formed by points A-B-C.

    Args:
        a, b, c: numpy arrays of shape (3,) representing (x, y, z) coordinates.

    Returns:
        Angle at B in degrees in [0, 180]. Returns 0.0 if vectors are degenerate.
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cosang = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cosang = max(min(cosang, 1.0), -1.0)
    return float(np.degrees(np.arccos(cosang)))


def _landmark_map(landmarks: List[Landmark]) -> Dict[str, Landmark]:
    """Map landmark name -> landmark dict."""
    return {lm["name"]: lm for lm in landmarks}


def _vec_from_lm(lm: Landmark) -> np.ndarray:
    return np.array([lm["x"], lm["y"], lm["z"]], dtype=float)


def _compute_frame_angles(landmarks: List[Landmark]) -> Dict[str, float]:
    lm = _landmark_map(landmarks)
    angles: Dict[str, float] = {}

    def have(*names: str) -> bool:
        return all(name in lm for name in names)

    # Elbows: shoulder-elbow-wrist
    if have("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"):
        angles["left_elbow"] = angle_at_point(
            _vec_from_lm(lm["LEFT_SHOULDER"]),
            _vec_from_lm(lm["LEFT_ELBOW"]),
            _vec_from_lm(lm["LEFT_WRIST"]),
        )
    if have("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"):
        angles["right_elbow"] = angle_at_point(
            _vec_from_lm(lm["RIGHT_SHOULDER"]),
            _vec_from_lm(lm["RIGHT_ELBOW"]),
            _vec_from_lm(lm["RIGHT_WRIST"]),
        )

    # Knees: hip-knee-ankle
    if have("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"):
        angles["left_knee"] = angle_at_point(
            _vec_from_lm(lm["LEFT_HIP"]),
            _vec_from_lm(lm["LEFT_KNEE"]),
            _vec_from_lm(lm["LEFT_ANKLE"]),
        )
    if have("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"):
        angles["right_knee"] = angle_at_point(
            _vec_from_lm(lm["RIGHT_HIP"]),
            _vec_from_lm(lm["RIGHT_KNEE"]),
            _vec_from_lm(lm["RIGHT_ANKLE"]),
        )

    # Shoulders: elbow-shoulder-hip
    if have("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"):
        angles["left_shoulder"] = angle_at_point(
            _vec_from_lm(lm["LEFT_ELBOW"]),
            _vec_from_lm(lm["LEFT_SHOULDER"]),
            _vec_from_lm(lm["LEFT_HIP"]),
        )
    if have("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"):
        angles["right_shoulder"] = angle_at_point(
            _vec_from_lm(lm["RIGHT_ELBOW"]),
            _vec_from_lm(lm["RIGHT_SHOULDER"]),
            _vec_from_lm(lm["RIGHT_HIP"]),
        )

    # Hip tilt: orientation of vector RIGHT_HIP -> LEFT_HIP in the image plane (x, y).
    if have("LEFT_HIP", "RIGHT_HIP"):
        left = lm["LEFT_HIP"]
        right = lm["RIGHT_HIP"]
        dx = float(left["x"] - right["x"])
        dy = float(left["y"] - right["y"])
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            # Angle relative to horizontal axis; take magnitude as tilt.
            raw_angle = degrees(atan2(dy, dx))
            angles["hip_tilt"] = float(abs(raw_angle))
        else:
            angles["hip_tilt"] = 0.0

    return angles


def add_joint_angles_to_skeleton(
    skeleton_path: str | Path,
    output_path: str | Path | None = None,
) -> str:
    """
    Load a skeleton JSON file, compute joint angles for each frame, and write it back.

    Args:
        skeleton_path: Path to the JSON produced by pose_estimator. Expected to be
            a list of frame objects.
        output_path: Optional output path. If None, the input file is overwritten
            in-place.

    Returns:
        Absolute path to the written JSON file (output_path or skeleton_path).

    Raises:
        JointAngleError: If the skeleton JSON is malformed or cannot be processed.
    """
    src = Path(skeleton_path).resolve()
    if not src.is_file():
        raise JointAngleError(f"Skeleton JSON not found: {src}")

    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise JointAngleError(f"Failed to parse skeleton JSON: {e}") from e

    if not isinstance(data, list):
        raise JointAngleError("Skeleton JSON must be a list of frame objects.")

    frames: List[Frame] = data  # type: ignore[assignment]

    for frame in frames:
        landmarks = frame.get("landmarks") or []
        if not isinstance(landmarks, list):
            continue
        frame["angles"] = _compute_frame_angles(landmarks)  # type: ignore[assignment]

    if output_path is None:
        dst = src
    else:
        dst = Path(output_path).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)

    dst.write_text(json.dumps(frames, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(dst.resolve())

