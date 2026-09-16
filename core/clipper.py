"""Clipper — extract MP4 clips from a source video using FFmpeg."""

import logging
from pathlib import Path

from core.exceptions import FFmpegError
from core.ffmpeg import check_ffmpeg, run_ffmpeg

logger = logging.getLogger(__name__)


def generate_clip(
    source: Path,
    start: float,
    end: float,
    output: Path,
) -> None:
    """Cut a single clip from *source* between *start* and *end* seconds.

    Re-encodes to H.264 + AAC for reliable, independently playable MP4 files.
    Raises FFmpegError if FFmpeg fails or the output file is not created.
    """
    duration = round(end - start, 3)

    if duration <= 0:
        raise FFmpegError(
            f"Invalid clip duration: start={start}, end={end}, duration={duration}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-ss", f"{start:.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",               # overwrite if regenerating
        str(output),
    ]

    run_ffmpeg(cmd, output_file=output)
