"""Clipt — identify and generate high-potential short-form clips from YouTube videos."""

import sys
from pathlib import Path

# Ensure Unicode output works on Windows consoles.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from core.config import DEFAULT_MAX_DURATION, DEFAULT_MIN_DURATION, DEFAULT_OUTPUT_DIR
from core.exceptions import CliptError
from core.pipeline import analyze_video

OUTPUT_DIR = DEFAULT_OUTPUT_DIR

# ── Video Length Configuration (Edit Here) ────────────────────────────
# Set your desired clip duration in seconds (highlighted for easy editing):
# Examples:
#   Short clips (Shorts, Reels, TikTok):  MIN_DURATION = 30,  MAX_DURATION = 90
#   Medium clips (highlights, topics):   MIN_DURATION = 90,  MAX_DURATION = 300   (1.5 to 5 mins)
#   Longer deep-dives / chapters:        MIN_DURATION = 300, MAX_DURATION = 900   (5 to 15 mins)
# ─────────────────────────────────────────────────────────────────────
MIN_DURATION: int = DEFAULT_MIN_DURATION   # Minimum clip duration in seconds
MAX_DURATION: int = DEFAULT_MAX_DURATION   # Maximum clip duration in seconds


def main() -> None:
    """CLI entry point for video analysis and preview clip generation."""
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python main.py <youtube-url>")
        sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 1)

    url = sys.argv[1]

    print()
    print("Clipt")
    print("─────")
    print()

    try:
        result = analyze_video(
            url,
            output_dir=OUTPUT_DIR,
            min_duration=MIN_DURATION,
            max_duration=MAX_DURATION,
            progress_callback=print,
        )
    except CliptError as exc:
        print(f"\nError: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        sys.exit(1)

    video_output_dir = OUTPUT_DIR / result.video_id
    success_count = len([c for c in result.clips if c.filename])
    total = len(result.clips)

    print()
    print(f"Done \u2014 {success_count}/{total} clips generated.")
    print()
    print("Output:")
    print(f"  {video_output_dir}/")
    print()


if __name__ == "__main__":
    main()
