"""Segmentation module: temporal and structural segmentation of dance content."""

from segmentation.phrase_detector import SegmentationError, detect_phrases

__all__ = [
    "detect_phrases",
    "SegmentationError",
]
