"""Clipt — generate vertical 9:16 versions of previously identified clips.

Usage:
    python process.py <video_id> [--force]

Features:
    - 1080p high-quality source video download (reusable cache support)
    - Local OpenCV face detection & deterministic subject tracking
    - Smooth 9:16 vertical camera path with inertia
    - Reused YouTube transcript converted to clip-relative ASS subtitles
    - Burned-in subtitles & dynamic crop rendered via FFmpeg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure Unicode and line-buffered output on Windows consoles.
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from core.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SUBTITLE_STYLE,
    ENABLE_FACE_TRACKING,
    ENABLE_SUBTITLES,
    ENABLE_VERTICAL,
)
from core.exceptions import CliptError
from core.pipeline import process_video
from core.transcript import extract_video_id
from processing.subtitles import SUBTITLE_STYLES

OUTPUT_DIR: Path = DEFAULT_OUTPUT_DIR

# ── Feature Toggles (Bools in Code) ──────────────────────────────────
ENABLE_VERTICAL: bool = ENABLE_VERTICAL        # Convert to 9:16 vertical video
ENABLE_FACE_TRACKING: bool = ENABLE_FACE_TRACKING   # Local OpenCV YuNet face tracking
ENABLE_SUBTITLES: bool = ENABLE_SUBTITLES      # Burn-in animated ASS subtitles

# Choose subtitle style: "highlight_yellow" (default), "highlight_cyan", "highlight_green", "highlight_magenta", "classic"
SUBTITLE_STYLE: str = DEFAULT_SUBTITLE_STYLE


def parse_args() -> argparse.Namespace:
    """Parse and validate CLI arguments for process.py."""
    parser = argparse.ArgumentParser(
        description="Clipt — repurpose long form content into shorts."
    )
    parser.add_argument("video_id", help="The YouTube video ID to process")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process and overwrite existing clips in processedFiles",
    )
    parser.add_argument(
        "--vertical",
        "--crop",
        "--9-16",
        dest="vertical",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_VERTICAL,
        help=f"Enable/disable 9:16 vertical conversion (default: {ENABLE_VERTICAL}). Use --no-vertical to keep original aspect ratio.",
    )
    parser.add_argument(
        "--face-tracking",
        "--face-track",
        dest="face_tracking",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_FACE_TRACKING,
        help=f"Enable/disable OpenCV YuNet face tracking (default: {ENABLE_FACE_TRACKING}). Use --no-face-tracking for centered crop.",
    )
    parser.add_argument(
        "--subtitles",
        "--subs",
        dest="subtitles",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_SUBTITLES,
        help=f"Enable/disable burned-in ASS subtitles (default: {ENABLE_SUBTITLES}). Use --no-subtitles to disable.",
    )
    parser.add_argument(
        "--subtitle-style",
        default=SUBTITLE_STYLE,
        help=f"Subtitle visual style (choices: {', '.join(SUBTITLE_STYLES.keys())}, classic, yellow, cyan, green). Default: {SUBTITLE_STYLE}",
    )
    return parser.parse_args()


def process_clips(
    video_id: str,
    *,
    vertical: bool = ENABLE_VERTICAL,
    face_tracking: bool = ENABLE_FACE_TRACKING,
    subtitles: bool = ENABLE_SUBTITLES,
    force: bool = False,
    subtitle_style: str = SUBTITLE_STYLE,
) -> dict[str, Any]:
    """Process clips for *video_id* according to the specified feature toggles.

    Parameters:
        video_id: The YouTube video identifier or URL.
        vertical: If True, crops and scales to 9:16 (1080×1920). If False, keeps source aspect ratio.
        face_tracking: If True (and vertical=True), runs YuNet face tracking. If False, uses centered crop.
        subtitles: If True, generates ASS captions and burns them in. If False, skips subtitles.
        force: If True, re-renders already-existing clips in processedFiles.
        subtitle_style: Visual style preset for ASS subtitles.

    Returns:
        Summary dict containing success_count, fail_count, skip_count, and output_dir.
    """
    clean_id = extract_video_id(video_id)

    print()
    print("Clipt Video Processor")
    print("─" * 22)
    print()
    print(f"Video ID: {clean_id}")
    print("Configuration:")
    print(f"  • 9:16 Vertical Conversion : {'Enabled (1080x1920)' if vertical else 'Disabled (Original aspect ratio)'}")
    if vertical:
        print(f"  • Face Tracking            : {'Enabled (YuNet)' if face_tracking else 'Disabled (Centered crop)'}")
    else:
        print("  • Face Tracking            : Bypassed (Non-vertical)")
    print(f"  • Subtitles                : {f'Enabled ({subtitle_style})' if subtitles else 'Disabled'}")
    print()

    return process_video(
        clean_id,
        output_dir=OUTPUT_DIR,
        vertical=vertical,
        face_tracking=face_tracking,
        subtitles=subtitles,
        force=force,
        subtitle_style=subtitle_style,
        progress_callback=print,
    )


def main() -> None:
    """CLI entry point for video processing."""
    args = parse_args()
    try:
        process_clips(
            args.video_id,
            vertical=args.vertical,
            face_tracking=args.face_tracking,
            subtitles=args.subtitles,
            force=args.force,
            subtitle_style=args.subtitle_style or SUBTITLE_STYLE,
        )
    except CliptError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
