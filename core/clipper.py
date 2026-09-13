"""Clipper — extract MP4 clips from a source video using FFmpeg."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def check_ffmpeg() -> None:
    """Verify that FFmpeg is available on the system PATH.

    Raises RuntimeError with install instructions if it cannot be found.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg was not found.\n"
            "Please install FFmpeg and make sure it is available in your system PATH.\n"
            "  Windows:  winget install FFmpeg\n"
            "  macOS:    brew install ffmpeg\n"
            "  Linux:    sudo apt install ffmpeg"
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg returned an error: {exc.stderr}") from exc


def generate_clip(
    source: Path,
    start: float,
    end: float,
    output: Path,
) -> None:
    """Cut a single clip from *source* between *start* and *end* seconds.

    Re-encodes to H.264 + AAC for reliable, independently playable MP4 files.
    Raises RuntimeError if FFmpeg fails or the output file is not created.
    """
    duration = round(end - start, 3)

    if duration <= 0:
        raise RuntimeError(
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

    logger.debug("FFmpeg command: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Extract the last few lines of stderr — usually the most useful.
        stderr_lines = result.stderr.strip().splitlines()
        detail = "\n".join(stderr_lines[-5:]) if stderr_lines else "(no output)"
        raise RuntimeError(f"FFmpeg failed for {output.name}:\n{detail}")

    if not output.is_file():
        raise RuntimeError(
            f"FFmpeg exited successfully but {output.name} was not created."
        )
