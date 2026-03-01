"""Ingestion module: video and audio acquisition for the dance learning platform."""

from ingestion.downloader import DownloadError, download_youtube_video
from ingestion.frame_extractor import FrameExtractionError, extract_frames
from ingestion.video_quality_scorer import QualityAssessment, VideoQualityError, score_video_quality

__all__ = [
    "download_youtube_video",
    "DownloadError",
    "extract_frames",
    "FrameExtractionError",
    "score_video_quality",
    "VideoQualityError",
    "QualityAssessment",
]
