"""FFmpeg and ffprobe execution helpers and diagnostics."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from core.exceptions import FFmpegError

logger = logging.getLogger(__name__)


def check_ffmpeg() -> None:
    """Verify that FFmpeg is available on the system PATH.

    Raises FFmpegError with installation guidance if not found.
    """
    if shutil.which("ffmpeg") is None:
        raise FFmpegError(
            "FFmpeg was not found.\n"
            "Please install FFmpeg and make sure it is available in your system PATH.\n"
            "  Windows:  winget install FFmpeg\n"
            "  macOS:    brew install ffmpeg\n"
            "  Linux:    sudo apt install ffmpeg"
        )
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        raise FFmpegError(
            "FFmpeg was not found.\n"
            "Please install FFmpeg and make sure it is available in your system PATH."
        )
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(f"FFmpeg returned an error: {exc.stderr.decode('utf-8', errors='replace')}") from exc


def probe_video_dimensions(source: Path) -> tuple[int, int]:
    """Return (width, height) of a video file using ffprobe.

    Raises FFmpegError on failure.
    """
    if shutil.which("ffprobe") is None:
        raise FFmpegError("ffprobe was not found in system PATH.")

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
        raise FFmpegError(f"ffprobe failed for {source.name}: {result.stderr.strip()}")

    try:
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except (json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
        raise FFmpegError(f"Could not parse video dimensions from {source.name}: {exc}") from exc


def run_ffmpeg(cmd: list[str], output_file: Path | None = None) -> None:
    """Execute an FFmpeg command safely, verifying non-zero exit code and output file existence.

    Raises FFmpegError on failure.
    """
    logger.debug("FFmpeg command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        stderr_lines = result.stderr.strip().splitlines()
        detail = "\n".join(stderr_lines[-5:]) if stderr_lines else "(no output)"
        name = output_file.name if output_file else "command"
        raise FFmpegError(f"FFmpeg failed for {name}:\n{detail}")

    if output_file is not None and not output_file.is_file():
        raise FFmpegError(
            f"FFmpeg exited successfully but {output_file.name} was not created."
        )
