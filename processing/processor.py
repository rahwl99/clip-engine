"""Processor — generate vertical 9:16 clips from a source video using FFmpeg.

Supports dynamic face-tracking crop and burned-in ASS subtitles.
"""

import logging
from pathlib import Path

from core.exceptions import FFmpegError, ProcessingError
from core.ffmpeg import probe_video_dimensions, run_ffmpeg

logger = logging.getLogger(__name__)


def _probe_dimensions(source: Path) -> tuple[int, int]:
    """Return (width, height) of *source* using ffprobe.

    Maintained for backward compatibility and test mock compatibility.
    """
    return probe_video_dimensions(source)


def generate_vertical_clip(
    source: Path,
    start: float,
    end: float,
    output: Path,
    *,
    crop_x_expr: str | None = None,
    crop_w: int | None = None,
    crop_h: int | None = None,
    ass_path: Path | None = None,
    vertical: bool = True,
) -> None:
    """Generate a clip from *source* between *start*–*end*.

    When *vertical* is True (default), crops and scales to 1080×1920 (9:16).
    Supports dynamic camera crop trajectory (via *crop_x_expr*) or centered crop.
    When *vertical* is False, preserves the source video resolution and aspect ratio.

    Burned-in ASS subtitles are applied when *ass_path* is provided.
    Audio is preserved (H.264 + AAC).
    Raises ProcessingError / FFmpegError on failure.
    """
    duration = round(end - start, 3)

    if duration <= 0:
        raise ProcessingError(
            f"Invalid clip duration: start={start}, end={end}, duration={duration}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    src_w, src_h = _probe_dimensions(source)

    vf_filters: list[str] = []

    if vertical:
        # Compute default 9:16 crop dimensions if not provided
        if crop_w is None or crop_h is None:
            crop_h = src_h
            crop_w = int(src_h * 9 / 16)
            if crop_w % 2 != 0:
                crop_w += 1
            if crop_w > src_w:
                crop_w = src_w
                crop_h = int(src_w * 16 / 9)

        crop_y = (src_h - crop_h) // 2

        # If no dynamic expression provided, use centered crop
        if not crop_x_expr:
            crop_x = (src_w - crop_w) // 2
            crop_x_expr = str(crop_x)

        vf_filters.extend([
            f"crop=w={crop_w}:h={crop_h}:x='{crop_x_expr}':y={crop_y}",
            "scale=1080:1920",
        ])

    # Add subtitle filter if ASS file exists
    if ass_path and ass_path.is_file():
        # Escape colons and backslashes for FFmpeg libass filter syntax on Windows
        escaped_ass = ass_path.resolve().as_posix().replace(":", r"\:")
        vf_filters.append(f"subtitles='{escaped_ass}'")

    cmd = [
        "ffmpeg",
        "-ss", f"{start:.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
    ]

    if vf_filters:
        cmd.extend(["-vf", ",".join(vf_filters)])

    cmd.extend([
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",
        str(output),
    ])

    run_ffmpeg(cmd, output_file=output)


# Alias for general clip processing
generate_processed_clip = generate_vertical_clip


def process_video(*args, **kwargs):
    """Process clips for a video ID into vertical/subtitled format."""
    from core.pipeline import process_video as _process_video
    return _process_video(*args, **kwargs)
