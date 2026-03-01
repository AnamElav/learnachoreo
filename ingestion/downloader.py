"""
Download a YouTube video to local storage using yt-dlp.

Returns the local file path. Handles invalid URLs, private videos,
age-restricted content, and other common failures with clear errors.
"""

from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError as YtDlpDownloadError
from yt_dlp.utils import ExtractorError, GeoRestrictedError

from utils.paths import get_raw_dir


class DownloadError(Exception):
    """Raised when video download fails (private, age-restricted, invalid URL, etc.)."""

    pass


def _classify_error(msg: str) -> str:
    """Map yt-dlp error message to a short, user-facing description."""
    msg_lower = msg.lower()
    if "private" in msg_lower or "sign in if you've been granted" in msg_lower:
        return "Video is private. Only the owner or invited users can access it."
    if "age" in msg_lower or "age-restricted" in msg_lower or "confirm your age" in msg_lower:
        return "Video is age-restricted and cannot be downloaded without authentication."
    if "unavailable" in msg_lower or "video unavailable" in msg_lower:
        return "Video is unavailable (deleted, region-locked, or otherwise restricted)."
    if "invalid" in msg_lower and "url" in msg_lower:
        return "Invalid or unsupported URL."
    if "georestrict" in msg_lower or "not available in your country" in msg_lower:
        return "Video is not available in your region."
    if "copyright" in msg_lower or "blocked" in msg_lower:
        return "Video has been blocked (e.g. copyright or policy)."
    if "login" in msg_lower or "sign in" in msg_lower:
        return "Video requires sign-in or membership to access."
    return msg or "Download failed for an unknown reason."


def download_youtube_video(url: str, out_dir: Path | None = None) -> str:
    """
    Download a YouTube video to a local directory and return its file path.

    Args:
        url: YouTube (or other yt-dlp–supported) video URL.
        out_dir: Directory to save the video. Defaults to DATA_DIR/raw (see utils.paths).

    Returns:
        Absolute path to the downloaded video file (e.g. .mp4, .webm).

    Raises:
        DownloadError: On invalid URL, private video, age-restricted content,
            region block, or any other yt-dlp failure. Message is user-friendly.
    """
    out_dir = out_dir or get_raw_dir()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_tmpl = str(out_dir / "%(id)s.%(ext)s")

    opts = {
        "outtmpl": out_tmpl,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                raise DownloadError("Could not get video information; the URL may be invalid.")
            video_id = info.get("id")
            ext = info.get("ext", "mp4")
            path = out_dir / f"{video_id}.{ext}"
            if not path.exists():
                # yt-dlp might use a different ext; find the file we just wrote
                candidates = list(out_dir.glob(f"{video_id}.*"))
                if not candidates:
                    raise DownloadError("Download reported success but no output file was found.")
                path = candidates[0]
            return str(path.resolve())
    except ExtractorError as e:
        msg = getattr(e, "msg", str(e))
        raise DownloadError(_classify_error(str(msg))) from e
    except GeoRestrictedError as e:
        msg = getattr(e, "msg", str(e))
        raise DownloadError(_classify_error(str(msg))) from e
    except YtDlpDownloadError as e:
        raise DownloadError(_classify_error(str(e))) from e
    except Exception as e:
        raise DownloadError(_classify_error(str(e))) from e
