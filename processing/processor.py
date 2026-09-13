"""Processor — generate vertical 9:16 clips from a source video using FFmpeg.

Supports dynamic face-tracking crop and burned-in ASS subtitles.
"""

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _probe_dimensions(source: Path) -> tuple[int, int]:
    """Return (width, height) of *source* using ffprobe.

    Raises RuntimeError if ffprobe fails or the dimensions cannot be parsed.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(source),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {source.name}: {result.stderr.strip()}"
        )

    try:
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except (json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(
            f"Could not parse video dimensions from ffprobe output: {exc}"
        ) from exc


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
) -> None:
    """Generate a single 1080×1920 (9:16) clip from *source* between *start*–*end*.

    Supports dynamic camera crop trajectory (via *crop_x_expr*) and burned-in ASS
    subtitles (via *ass_path*).

    Audio is preserved (H.264 + AAC).
    Raises RuntimeError on failure.
    """
    duration = round(end - start, 3)

    if duration <= 0:
        raise RuntimeError(
            f"Invalid clip duration: start={start}, end={end}, duration={duration}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    src_w, src_h = _probe_dimensions(source)

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

    vf_filters = [
        f"crop=w={crop_w}:h={crop_h}:x='{crop_x_expr}':y={crop_y}",
        "scale=1080:1920",
    ]

    # Add subtitle filter if ASS file exists
    if ass_path and ass_path.is_file():
        # Escape colons and backslashes for FFmpeg libass filter syntax on Windows
        escaped_ass = ass_path.resolve().as_posix().replace(":", r"\:")
        vf_filters.append(f"subtitles='{escaped_ass}'")

    vf = ",".join(vf_filters)

    cmd = [
        "ffmpeg",
        "-ss", f"{start:.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-vf", vf,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",
        str(output),
    ]

    logger.debug("FFmpeg vertical command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        stderr_lines = result.stderr.strip().splitlines()
        detail = "\n".join(stderr_lines[-5:]) if stderr_lines else "(no output)"
        raise RuntimeError(f"FFmpeg failed for {output.name}:\n{detail}")

    if not output.is_file():
        raise RuntimeError(
            f"FFmpeg exited successfully but {output.name} was not created."
        )
