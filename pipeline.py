"""
End-to-end pipeline for choreo ingestion and analysis.

Usage:
    python pipeline.py "https://www.youtube.com/watch?v=..."

Steps:
    1) Download video (yt-dlp)
    2) Run pose estimation (MediaPipe) — frames streamed from FFmpeg, no frame files on disk
    3) Compute joint angles
    4) Score video quality
    5) Detect learnable phrases
    6) Write choreo_data JSON
    7) Cleanup: delete raw video file (keep only derived JSONs)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from ingestion import (
    DownloadError,
    QualityAssessment,
    VideoQualityError,
    download_youtube_video,
    score_video_quality,
)
from pose_extraction import (
    JointAngleError,
    PoseEstimationError,
    add_joint_angles_to_skeleton,
    estimate_poses_from_video,
)
from segmentation import SegmentationError, detect_phrases
from utils.paths import get_data_dir


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_segment_angles(
    skeleton_frames: List[dict],
    start_frame: int,
    end_frame: int,
) -> Dict[str, float]:
    """
    Compute mean angle per joint over frames in [start_frame, end_frame].
    """
    # Map frame_number -> frame dict for quick lookup.
    frames_by_number: Dict[int, dict] = {
        int(f.get("frame_number", i)): f for i, f in enumerate(skeleton_frames)
    }

    angle_sums: Dict[str, float] = {}
    angle_counts: Dict[str, int] = {}

    for fn in range(start_frame, end_frame + 1):
        frame = frames_by_number.get(fn)
        if not frame:
            continue
        angles = frame.get("angles") or {}
        if not isinstance(angles, dict):
            continue
        for name, val in angles.items():
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            angle_sums[name] = angle_sums.get(name, 0.0) + v
            angle_counts[name] = angle_counts.get(name, 0) + 1

    summary: Dict[str, float] = {}
    for name, s in angle_sums.items():
        count = angle_counts.get(name, 0)
        if count > 0:
            summary[name] = float(s / count)
    return summary


def run_pipeline(source_url: str) -> Path:
    """
    Run the full pipeline for a single YouTube URL.

    Returns:
        Path to the written choreo_data JSON file.
    """
    stage_times: List[Tuple[str, float]] = []

    print(f"[1/7] Downloading video from URL: {source_url}")
    t0 = time.perf_counter()
    try:
        video_path_str = download_youtube_video(source_url)
    except DownloadError as e:
        print(f"Download failed: {e}", file=sys.stderr)
        raise
    stage_times.append(("Download", time.perf_counter() - t0))

    video_path = Path(video_path_str)
    video_id = video_path.stem
    print(f"       Downloaded to: {video_path} (video_id={video_id})")
    print(f"       Stage took: {stage_times[-1][1]:.1f}s")

    print("[2/7] Running pose estimation (MediaPipe, 10 fps)...")
    t0 = time.perf_counter()
    try:
        skeleton_path_str = estimate_poses_from_video(
            video_path=video_path,
            video_id=video_id,
            fps=10.0,
            confidence_threshold=0.6,
        )
    except PoseEstimationError as e:
        print(f"Pose estimation failed: {e}", file=sys.stderr)
        raise
    stage_times.append(("Pose estimation", time.perf_counter() - t0))
    skeleton_path = Path(skeleton_path_str)
    print(f"       Skeleton JSON written to: {skeleton_path}")
    print(f"       Stage took: {stage_times[-1][1]:.1f}s")

    print("[3/7] Computing joint angles...")
    t0 = time.perf_counter()
    try:
        skeleton_with_angles_path_str = add_joint_angles_to_skeleton(skeleton_path)
    except JointAngleError as e:
        print(f"Joint angle computation failed: {e}", file=sys.stderr)
        raise
    stage_times.append(("Joint angles", time.perf_counter() - t0))
    skeleton_with_angles_path = Path(skeleton_with_angles_path_str)
    print(f"       Skeleton updated with angles: {skeleton_with_angles_path}")
    print(f"       Stage took: {stage_times[-1][1]:.1f}s")

    print("[4/7] Scoring video quality...")
    t0 = time.perf_counter()
    try:
        quality: QualityAssessment = score_video_quality(skeleton_with_angles_path)
    except VideoQualityError as e:
        print(f"Quality scoring failed: {e}", file=sys.stderr)
        raise
    stage_times.append(("Quality scoring", time.perf_counter() - t0))
    print(f"       Quality overall_score={quality['overall_score']:.3f}")
    print(f"       Stage took: {stage_times[-1][1]:.1f}s")

    print("[5/7] Detecting phrases / segments...")
    t0 = time.perf_counter()
    try:
        segments_path_str = detect_phrases(skeleton_with_angles_path)
    except SegmentationError as e:
        print(f"Segmentation failed: {e}", file=sys.stderr)
        raise
    stage_times.append(("Segmentation", time.perf_counter() - t0))
    segments_path = Path(segments_path_str)
    segments_raw: List[dict] = _load_json(segments_path)
    print(f"       Segments JSON written to: {segments_path} (segments={len(segments_raw)})")
    print(f"       Stage took: {stage_times[-1][1]:.1f}s")

    print("[6/7] Building final choreo_data JSON...")
    t0 = time.perf_counter()
    skeleton_frames: List[dict] = _load_json(skeleton_with_angles_path)

    segments_with_angles: List[dict] = []
    for seg in segments_raw:
        start_frame = int(seg.get("start_frame", 0))
        end_frame = int(seg.get("end_frame", start_frame))
        angle_summary = _summarize_segment_angles(skeleton_frames, start_frame, end_frame)
        seg_out = dict(seg)
        seg_out["angle_summary"] = angle_summary
        segments_with_angles.append(seg_out)

    choreo_data = {
        "video_id": video_id,
        "source_url": source_url,
        "quality_assessment": quality,
        "segments": segments_with_angles,
    }

    output_dir = get_data_dir() / "choreo_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    choreo_path = output_dir / f"{video_id}.choreo_data.json"
    choreo_path.write_text(json.dumps(choreo_data, ensure_ascii=False, indent=2), encoding="utf-8")
    stage_times.append(("Build choreo_data", time.perf_counter() - t0))
    print(f"       Stage took: {stage_times[-1][1]:.1f}s")

    # Cleanup: delete raw video to free space; keep only derived JSONs
    print("[7/7] Cleanup: removing raw video file...")
    t0 = time.perf_counter()
    if video_path.is_file():
        try:
            video_path.unlink()
            print(f"       Deleted: {video_path}")
        except OSError as e:
            print(f"       Warning: could not delete raw video: {e}", file=sys.stderr)
    else:
        print(f"       (raw video already absent)")
    stage_times.append(("Cleanup", time.perf_counter() - t0))
    print(f"       Stage took: {stage_times[-1][1]:.1f}s")

    # Final summary
    flags = quality.get("flags", [])
    if segments_with_angles:
        avg_seg_sec = float(
            sum((s["end_time_ms"] - s["start_time_ms"]) for s in segments_with_angles) / len(segments_with_angles) / 1000.0
        )
    else:
        avg_seg_sec = 0.0
    total_sec = sum(t for _, t in stage_times)
    print("\n=== Pipeline Summary ===")
    print(f"Video ID: {video_id}")
    print(f"Source URL: {source_url}")
    print(f"Quality overall_score: {quality['overall_score']:.3f}")
    print(f"Segments detected: {len(segments_with_angles)}")
    print(f"Avg segment duration: {avg_seg_sec:.2f}s")
    print(f"Flags: {', '.join(flags) if flags else 'none'}")
    print(f"Choreo data written to: {choreo_path}")
    print("\n--- Time by stage ---")
    for name, sec in stage_times:
        pct = (sec / total_sec * 100) if total_sec else 0
        print(f"  {name}: {sec:.1f}s ({pct:.0f}%)")
    print(f"  Total: {total_sec:.1f}s")

    return choreo_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run full choreo ingestion pipeline on a YouTube URL.")
    parser.add_argument("url", help="YouTube video URL")
    args = parser.parse_args(argv)

    try:
        run_pipeline(args.url)
    except Exception:
        # Errors should already have been printed with context.
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

