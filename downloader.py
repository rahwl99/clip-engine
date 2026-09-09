"""Downloader — download YouTube source video using yt-dlp."""

import logging
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)


def download_video(url: str, output_path: Path) -> Path:
    """Download the source video to *output_path* and return the final file path.

    Uses yt-dlp to grab an MP4-compatible format.  Raises RuntimeError on
    failure with a clear message.
    """
    # Ensure parent directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove extension — yt-dlp appends it based on the chosen format.
    template = str(output_path.with_suffix(""))

    opts: dict = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": template + ".%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }

    logger.info("Downloading video from %s", url)

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
