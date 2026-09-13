"""Downloader — download YouTube source video using yt-dlp."""

import json
import logging
import subprocess
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)


# ── Quality presets ──────────────────────────────────────────────────

_QUALITY_PRESETS: dict[str, dict] = {
    "360p": {
        "format": (
            "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/"
            "best[height<=360][ext=mp4]/best"
        ),
        # Android client is fine for 360p — it returns a single combined stream.
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    },
    "1080p": {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        # Default client — exposes video-only HD streams (vp9, avc1, av01)
        # that the android client hides due to YouTube's SABR experiment.
        "extractor_args": {},
    },
}


def probe_resolution(path: Path) -> tuple[int, int]:
    """Return (width, height) of a video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    try:
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except (json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(f"Could not parse video dimensions: {exc}") from exc


def download_video(url: str, output_path: Path, *, quality: str = "360p") -> Path:
    """Download the source video to *output_path* and return the final file path.

    *quality* selects the target resolution — ``"360p"`` (default, used by
    ``main.py`` for lightweight clip previews) or ``"1080p"`` (used by
    ``process.py`` for Full-HD vertical processing).

    Uses yt-dlp to grab an MP4-compatible format.  Raises RuntimeError on
    failure with a clear message.
    """
    preset = _QUALITY_PRESETS.get(quality)
    if preset is None:
        raise ValueError(
            f"Unsupported quality '{quality}'. "
            f"Choose from: {', '.join(_QUALITY_PRESETS)}"
        )

    # Ensure parent directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove extension — yt-dlp appends it based on the chosen format.
    template = str(output_path.with_suffix(""))

    opts: dict = {
        "format": preset["format"],
        "merge_output_format": "mp4",
        "outtmpl": template + ".%(ext)s",
        "quiet": True,
        "no_warnings": True,
    }

    if preset["extractor_args"]:
        opts["extractor_args"] = preset["extractor_args"]

    logger.info("Downloading video from %s (quality=%s)", url, quality)

    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            raise RuntimeError(f"Video download failed: {exc}") from exc

    if not info:
        raise RuntimeError("yt-dlp returned no info after download.")

    # yt-dlp may adjust the extension; find the actual file.
    final = Path(template + ".mp4")
    if final.is_file():
        return final

    # Fallback: check with the extension yt-dlp reported.
    ext = info.get("ext", "mp4")
    fallback = Path(f"{template}.{ext}")
    if fallback.is_file():
        return fallback

    raise RuntimeError(
        f"Download appeared to succeed but output file not found. "
        f"Expected: {final}"
    )

