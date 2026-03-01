"""
Run MediaPipe Pose on frame images and persist a skeleton timeline.

Uses the task-based Pose Landmarker API (mediapipe.tasks.vision.PoseLandmarker)
with the pose_landmarker_lite.task model. Same output format as before:
33 named landmarks with x, y, z, visibility per frame.

- estimate_poses_for_frames(): read frames from a directory (legacy).
- estimate_poses_from_video(): stream frames from FFmpeg, no frame files on disk.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, TypedDict

import cv2
import mediapipe as mp
import numpy as np

from utils.paths import get_models_dir, get_skeleton_path, get_skeletons_dir


# 33 pose landmark names in index order (same as legacy PoseLandmark enum).
POSE_LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]

POSE_LANDMARKER_LITE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/"
    "float16/latest/pose_landmarker_lite.task"
)


class PoseEstimationError(Exception):
    """Raised when pose estimation fails (e.g., missing frames directory)."""

    pass


class LandmarkDict(TypedDict):
    name: str
    x: float
    y: float
    z: float
    visibility: float


class FrameDict(TypedDict):
    frame_number: int
    timestamp_ms: float
    landmarks: List[LandmarkDict]
    overall_confidence: float


@dataclass
class FrameResult:
    frame_number: int
    timestamp_ms: float
    landmarks: list[LandmarkDict]
    overall_confidence: float


def _get_pose_landmarker_model_path() -> Path:
    """Return path to pose_landmarker_lite.task, downloading it if missing."""
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "pose_landmarker_lite.task"
    if not path.is_file():
        try:
            urllib.request.urlretrieve(POSE_LANDMARKER_LITE_URL, path)
        except Exception as e:
            raise PoseEstimationError(
                f"Failed to download pose_landmarker_lite.task from {POSE_LANDMARKER_LITE_URL}: {e}"
            ) from e
    return path


def _landmarks_from_task_result(
    pose_landmarks_list: list,
) -> list[LandmarkDict]:
    """Convert task API pose_landmarks (list of NormalizedLandmark) to our format."""
    out: list[LandmarkDict] = []
    for idx, lm in enumerate(pose_landmarks_list):
        name = POSE_LANDMARK_NAMES[idx] if idx < len(POSE_LANDMARK_NAMES) else f"LANDMARK_{idx}"
        vis = getattr(lm, "visibility", None)
        visibility = float(vis) if vis is not None else 1.0
        out.append(
            {
                "name": name,
                "x": float(lm.x) if lm.x is not None else 0.0,
                "y": float(lm.y) if lm.y is not None else 0.0,
                "z": float(lm.z) if lm.z is not None else 0.0,
                "visibility": visibility,
            }
        )
    return out


def _overall_confidence(landmarks: list[LandmarkDict]) -> float:
    if not landmarks:
        return 0.0
    return float(sum(lm["visibility"] for lm in landmarks) / len(landmarks))


def _downscale_rgb(rgb: np.ndarray, max_width: int = 640) -> np.ndarray:
    """Downscale RGB frame for faster inference; preserve aspect ratio."""
    h, w = rgb.shape[:2]
    if w <= max_width or max_width <= 0:
        return rgb
    scale = max_width / float(w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _iter_frame_files(frames_dir: Path) -> Iterable[Path]:
    """Yield frame image files in a stable, sorted order."""
    if not frames_dir.is_dir():
        raise PoseEstimationError(f"Frames directory not found: {frames_dir}")

    candidates = sorted(
        [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name,
    )
    if not candidates:
        raise PoseEstimationError(f"No frame images found in directory: {frames_dir}")
    return candidates


def estimate_poses_for_frames(
    frames_dir: str | Path,
    video_id: str,
    fps: float = 30.0,
    confidence_threshold: float = 0.6,
    output_path: str | Path | None = None,
) -> str:
    """
    Run MediaPipe Pose Landmarker on all frames in a directory and write skeleton JSON.
    """
    from mediapipe.tasks.python.core import base_options
    from mediapipe.tasks.python.vision import pose_landmarker
    from mediapipe.tasks.python.vision.core import vision_task_running_mode

    frames_dir_path = Path(frames_dir).resolve()
    frame_files = list(_iter_frame_files(frames_dir_path))

    if output_path is None:
        get_skeletons_dir().mkdir(parents=True, exist_ok=True)
        json_path = get_skeleton_path(video_id)
    else:
        json_path = Path(output_path).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = str(_get_pose_landmarker_model_path())
    base_opts = base_options.BaseOptions(model_asset_path=model_path)
    options = pose_landmarker.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    results_list: list[FrameResult] = []

    with pose_landmarker.PoseLandmarker.create_from_options(options) as landmarker:
        for idx, frame_path in enumerate(frame_files):
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = landmarker.detect(mp_image)

            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                continue
            landmarks = _landmarks_from_task_result(result.pose_landmarks[0])
            conf = _overall_confidence(landmarks)
            if conf < confidence_threshold:
                continue

            frame_number = idx
            timestamp_ms = (frame_number / fps) * 1000.0
            results_list.append(
                FrameResult(
                    frame_number=frame_number,
                    timestamp_ms=timestamp_ms,
                    landmarks=landmarks,
                    overall_confidence=conf,
                )
            )

    frames_payload: list[FrameDict] = [asdict(r) for r in results_list]  # type: ignore[assignment]
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(frames_payload, f, ensure_ascii=False, indent=2)

    return str(json_path.resolve())


def estimate_poses_from_video(
    video_path: str | Path,
    video_id: str,
    fps: float = 30.0,
    confidence_threshold: float = 0.6,
    output_path: str | Path | None = None,
) -> str:
    """
    Run MediaPipe Pose Landmarker on a video by streaming frames from FFmpeg.
    Uses VIDEO running mode and pose_landmarker_lite; downscales frames to 640px.
    """
    from mediapipe.tasks.python.core import base_options
    from mediapipe.tasks.python.vision import pose_landmarker
    from mediapipe.tasks.python.vision.core import vision_task_running_mode

    from ingestion.frame_extractor import stream_frames

    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise PoseEstimationError(f"Video file not found: {video_path}")

    if output_path is None:
        get_skeletons_dir().mkdir(parents=True, exist_ok=True)
        json_path = get_skeleton_path(video_id)
    else:
        json_path = Path(output_path).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = str(_get_pose_landmarker_model_path())
    base_opts = base_options.BaseOptions(model_asset_path=model_path)
    options = pose_landmarker.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    results_list: list[FrameResult] = []

    with pose_landmarker.PoseLandmarker.create_from_options(options) as landmarker:
        for frame_number, timestamp_ms, rgb in stream_frames(video_path, fps=fps):
            rgb_small = _downscale_rgb(rgb, max_width=640)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
            result = landmarker.detect_for_video(mp_image, int(round(timestamp_ms)))

            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                continue
            landmarks = _landmarks_from_task_result(result.pose_landmarks[0])
            conf = _overall_confidence(landmarks)
            if conf < confidence_threshold:
                continue
            results_list.append(
                FrameResult(
                    frame_number=frame_number,
                    timestamp_ms=timestamp_ms,
                    landmarks=landmarks,
                    overall_confidence=conf,
                )
            )

    frames_payload: list[FrameDict] = [asdict(r) for r in results_list]  # type: ignore[assignment]
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(frames_payload, f, ensure_ascii=False, indent=2)

    return str(json_path.resolve())
