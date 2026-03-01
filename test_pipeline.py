"""
Batch test for the choreo pipeline.

Runs the full pipeline against several representative video types and outputs
an aggregated markdown report with quality scores, segment counts, and flags.

Usage:
    python test_pipeline.py

NOTE: Replace the placeholder URLs below with real test videos before running.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pipeline import run_pipeline
from utils.paths import get_data_dir


@dataclass
class TestVideo:
    key: str
    description: str
    url: str


TEST_VIDEOS: List[TestVideo] = [
    TestVideo(
        key="well_lit_tutorial",
        description="Well-lit single-dancer tutorial video",
        url="https://www.youtube.com/watch?v=bY3IC-U3e1c",
    ),
    TestVideo(
        key="live_performance_multi",
        description="Live performance with multiple people",
        url="https://www.youtube.com/watch?v=18fe5rgmvYI",
    ),
    TestVideo(
        key="fast_hip_hop",
        description="Fast hip-hop routine",
        url="https://www.youtube.com/watch?v=nkeh7Bx3GzY",
    ),
    TestVideo(
        key="slow_contemporary",
        description="Slow contemporary piece",
        url="https://www.youtube.com/watch?v=ATdYddT00Yg",
    ),
    TestVideo(
        key="multi_camera_cuts",
        description="Video with multiple camera cuts",
        url="https://www.youtube.com/watch?v=RvTugU80Td4",
    ),
]
 
def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_score(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "N/A"


def main() -> int:
    results = []

    for tv in TEST_VIDEOS:
        print(f"\n=== Running pipeline for: {tv.description} ===")
        print(f"URL: {tv.url}")
        try:
            choreo_path = run_pipeline(tv.url)
        except Exception as e:  # noqa: BLE001
            print(f"Pipeline failed for {tv.key}: {e}")
            results.append(
                {
                    "key": tv.key,
                    "description": tv.description,
                    "url": tv.url,
                    "error": str(e),
                }
            )
            continue

        choreo_data = _load_json(choreo_path)
        qa = choreo_data.get("quality_assessment", {})
        segments = choreo_data.get("segments", []) or []

        results.append(
            {
                "key": tv.key,
                "description": tv.description,
                "url": tv.url,
                "video_id": choreo_data.get("video_id"),
                "overall_score": qa.get("overall_score"),
                "confidence_score": qa.get("confidence_score"),
                "continuity_score": qa.get("continuity_score"),
                "coverage_score": qa.get("coverage_score"),
                "flags": qa.get("flags", []),
                "segment_count": len(segments),
            }
        )

    # Build markdown report.
    lines: List[str] = []
    lines.append("# Pipeline Comparison Report")
    lines.append("")
    lines.append("This report summarizes how the ingestion pipeline behaves on a few representative video types.")
    lines.append("")

    for r in results:
        lines.append(f"## {r['description']} (`{r['key']}`)")
        lines.append("")
        lines.append(f"- **URL**: {r['url']}")
        if "error" in r:
            lines.append(f"- **Status**: ❌ Error: `{r['error']}`")
            lines.append("")
            continue
        lines.append(f"- **Video ID**: `{r.get('video_id')}`")
        lines.append(f"- **Overall quality score**: `{_fmt_score(r.get('overall_score'))}`")
        lines.append(f"- **Confidence score**: `{_fmt_score(r.get('confidence_score'))}`")
        lines.append(f"- **Continuity score**: `{_fmt_score(r.get('continuity_score'))}`")
        lines.append(f"- **Coverage score**: `{_fmt_score(r.get('coverage_score'))}`")
        lines.append(f"- **Segments detected**: `{r.get('segment_count')}`")
        flags = r.get("flags") or []
        if flags:
            lines.append(f"- **Flags**: `{', '.join(flags)}`")
        else:
            lines.append("- **Flags**: `none`")
        lines.append("")

    report_md = "\n".join(lines)

    reports_dir = get_data_dir() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "pipeline_comparison.md"
    report_path.write_text(report_md, encoding="utf-8")

    print("\n=== Comparison report ===")
    print(report_md)
    print(f"\nMarkdown report written to: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

