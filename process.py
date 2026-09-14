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

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure Unicode and line-buffered output on Windows consoles.
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from core.clipper import check_ffmpeg
from core.downloader import download_video, probe_resolution
from processing.face_tracker import track_face_for_clip
from processing.processor import generate_vertical_clip
from processing.subtitles import (
    DEFAULT_SUBTITLE_STYLE,
    SUBTITLE_STYLES,
    build_clip_subtitles,
    generate_ass_file,
    load_or_fetch_transcript,
)

OUTPUT_DIR = Path("output")

# ── Feature Toggles (Bools in Code) ──────────────────────────────────
ENABLE_VERTICAL: bool = True        # Convert to 9:16 vertical video
ENABLE_FACE_TRACKING: bool = True   # Local OpenCV YuNet face tracking
ENABLE_SUBTITLES: bool = True      # Burn-in animated ASS subtitles

# Choose subtitle style: "highlight_yellow" (default), "highlight_cyan", "highlight_green", "highlight_magenta", "classic"
SUBTITLE_STYLE: str = DEFAULT_SUBTITLE_STYLE


def parse_args() -> argparse.Namespace:
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
) -> dict:
    """Process clips for *video_id* according to the specified feature toggles.

    Parameters:
        video_id: The YouTube video identifier.
        vertical: If True, crops and scales to 9:16 (1080×1920). If False, keeps source aspect ratio.
        face_tracking: If True (and vertical=True), runs YuNet face tracking. If False, uses centered crop.
        subtitles: If True, generates ASS captions and burns them in. If False, skips subtitles.
        force: If True, re-renders already-existing clips in processedFiles.
        subtitle_style: Visual style preset for ASS subtitles.

    Returns:
        Summary dict containing success_count, fail_count, skip_count, and output_dir.
    """
    print()
    print("Clipt Video Processor")
    print("─" * 22)
    print()
    print(f"Video ID: {video_id}")
    print("Configuration:")
    print(f"  • 9:16 Vertical Conversion : {'Enabled (1080x1920)' if vertical else 'Disabled (Original aspect ratio)'}")
    if vertical:
        print(f"  • Face Tracking            : {'Enabled (YuNet)' if face_tracking else 'Disabled (Centered crop)'}")
    else:
        print("  • Face Tracking            : Bypassed (Non-vertical)")
    print(f"  • Subtitles                : {f'Enabled ({subtitle_style})' if subtitles else 'Disabled'}")
    print()

    # ── Check output directory exists ─────────────────────────────────
    video_dir = OUTPUT_DIR / video_id

    if not video_dir.is_dir():
        print(f"Error: No Clipt project found for video ID '{video_id}'.")
        print("Run python main.py <youtube-url> first.")
        sys.exit(1)

    # ── Load clips.json ───────────────────────────────────────────────
    clips_json_path = video_dir / "clips.json"

    if not clips_json_path.is_file():
        print(f"Error: No clips.json found for video ID '{video_id}'.")
        print()
        print("Run:")
        print("  python main.py <youtube-url>")
        print()
        print("before running:")
        print(f"  python process.py {video_id}")
        sys.exit(1)

    print("[1/3] Loading clips.json...")

    try:
        data = json.loads(clips_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  Error: Could not read clips.json: {exc}")
        sys.exit(1)

    clips = data.get("clips", [])
    source_url = data.get("source_url")

    if not clips:
        print("  Error: clips.json contains no clips.")
        sys.exit(1)

    if not source_url:
        print("  Error: clips.json does not contain a source_url.")
        print("  Re-run python main.py <youtube-url> to regenerate clips.json.")
        sys.exit(1)

    print(f"  ✓ Found {len(clips)} clips")
    print()

    # ── Check FFmpeg ──────────────────────────────────────────────────
    try:
        check_ffmpeg()
    except RuntimeError as exc:
        print(f"  Error: {exc}")
        sys.exit(1)

    # ── Download or load high-quality source video ────────────────────
    print("[2/3] Downloading high-quality source...")
    print(f"  Source URL: {source_url}")

    cached_source = video_dir / "source.mp4"
    tmpdir: str | None = None

    if cached_source.is_file():
        source_video = cached_source
        print(f"  ✓ Found existing source video: {cached_source.name}")
    else:
        tmpdir = tempfile.mkdtemp(prefix="clipt_process_")
        source_path = Path(tmpdir) / f"{video_id}.mp4"
        try:
            source_video = download_video(source_url, source_path, quality="1080p")
        except RuntimeError as exc:
            print(f"  Error: {exc}")
            shutil.rmtree(tmpdir, ignore_errors=True)
            sys.exit(1)

    # Verify actual source resolution.
    try:
        src_w, src_h = probe_resolution(source_video)
    except RuntimeError:
        src_w, src_h = 0, 0

    if src_h >= 1080:
        print(f"  ✓ Downloaded source: {src_w}x{src_h}")
    elif src_h > 0:
        print(f"  ⚠ 1080p source unavailable.")
        print(f"    Highest available source: {src_w}x{src_h}")
    else:
        print("  ✓ Source video ready (resolution unknown)")

    print()

    # ── Load or fetch transcript ──────────────────────────────────────
    transcript_segments = []
    if subtitles:
        transcript_segments = load_or_fetch_transcript(video_dir, source_url)
        if transcript_segments:
            print(f"  ✓ Loaded transcript ({len(transcript_segments)} segments)")
        else:
            print("  ⚠ No transcript available — clips will be generated without subtitles")
    else:
        print("  → Subtitles disabled (skipping transcript fetch)")

    print()

    # ── Generate processed clips ──────────────────────────────────────
    action_label = "vertical 9:16 clips" if vertical else "clips (original aspect ratio)"
    print(f"[3/3] Generating {action_label}...")
    print()

    processed_dir = video_dir / "processedFiles"
    processed_dir.mkdir(parents=True, exist_ok=True)

    subs_dir = processed_dir / "subtitles"
    if subtitles:
        subs_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0
    skip_count = 0

    for idx, clip in enumerate(clips, start=1):
        filename = clip.get("filename")
        if not filename:
            fail_count += 1
            print(f"Processing clip {idx}/{len(clips)}: (unknown) — missing filename")
            continue

        output_path = processed_dir / filename

        # Skip already-processed files unless --force is specified
        if output_path.is_file() and not force:
            skip_count += 1
            print(f"Processing clip {idx}/{len(clips)}: {filename} already exists — skipping")
            continue

        start = clip.get("start")
        end = clip.get("end")

        if start is None or end is None:
            fail_count += 1
            print(f"Processing clip {idx}/{len(clips)}: {filename} — missing start/end timestamps")
            continue

        print(f"Processing clip {idx}/{len(clips)} ({filename})")

        # ── 1. Face tracking ──
        crop_x_expr = None
        crop_w = None
        crop_h = None

        if vertical:
            if face_tracking:
                print("  Detecting faces...")
                try:
                    track_res = track_face_for_clip(
                        source_video,
                        float(start),
                        float(end),
                        video_id,
                        filename,
                    )
                    crop_w = track_res.crop_w
                    crop_h = track_res.crop_h
                    crop_x_expr = track_res.crop_x_expr

                    if not track_res.is_tracked or track_res.face_count == 0:
                        print("  ⚠ No face detected")
                        print("  → Using centered crop")
                    elif track_res.face_count == 1:
                        print("  ✓ 1 face detected")
                        print("  → Tracking subject")
                        print("  ✓ Generating camera path")
                    else:
                        print(f"  ✓ {track_res.face_count} faces detected")
                        print(f"  ✓ Selected face {track_res.selected_face_idx}")
                        print("  → Tracking subject")
                        print("  ✓ Generating camera path")

                except Exception as exc:
                    print(f"  ⚠ Face tracking encountered an issue: {exc}")
                    print("  → Falling back to centered crop")
            else:
                print("  → Face tracking disabled (using centered crop)")
        else:
            # When vertical conversion is disabled, face tracking crop is bypassed
            pass

        # ── 2. Subtitle generation ──
        ass_path = None
        if subtitles:
            if transcript_segments:
                clip_subs = build_clip_subtitles(transcript_segments, float(start), float(end))
                if clip_subs:
                    ass_filename = Path(filename).with_suffix(".ass").name
                    temp_ass_path = subs_dir / ass_filename
                    try:
                        v_w = 1080 if vertical else (src_w if src_w > 0 else 1920)
                        v_h = 1920 if vertical else (src_h if src_h > 0 else 1080)
                        generate_ass_file(
                            clip_subs,
                            temp_ass_path,
                            style=subtitle_style,
                            video_width=v_w,
                            video_height=v_h,
                        )
                        ass_path = temp_ass_path
                        print(f"  ✓ Generating subtitles ({subtitle_style} style)")
                    except Exception as exc:
                        print(f"  ⚠ Could not generate ASS subtitles: {exc}")
                else:
                    print("  ⚠ No transcript segments overlapping this clip")
        else:
            print("  → Subtitles disabled")

        # ── 3. FFmpeg rendering ──
        try:
            if vertical:
                print("  ✓ Rendering 1080x1920 (9:16)")
            else:
                res_str = f"{src_w}x{src_h}" if src_w and src_h else "source resolution"
                print(f"  ✓ Rendering {res_str} (original aspect ratio)")

            generate_vertical_clip(
                source_video,
                float(start),
                float(end),
                output_path,
                crop_x_expr=crop_x_expr,
                crop_w=crop_w,
                crop_h=crop_h,
                ass_path=ass_path,
                vertical=vertical,
            )
            success_count += 1
            print(f"  ✓ {filename}")
        except RuntimeError as exc:
            fail_count += 1
            print(f"  ✗ {filename} — {exc}")

        # Clean up temporary ASS file
        if ass_path and ass_path.is_file():
            try:
                ass_path.unlink()
            except OSError:
                pass

        print()

    # Clean up empty subtitles dir if leftover
    if subs_dir.is_dir():
        try:
            subs_dir.rmdir()
        except OSError:
            pass

    # ── Clean up temp files ───────────────────────────────────────────
    if tmpdir:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    # ── Summary ───────────────────────────────────────────────────────
    total = success_count + fail_count + skip_count
    parts = []
    if success_count:
        parts.append(f"{success_count} generated")
    if skip_count:
        parts.append(f"{skip_count} skipped")
    if fail_count:
        parts.append(f"{fail_count} failed")

    print(f"Processing complete. {', '.join(parts)} ({total} total)")
    print()
    print("Output:")
    print(f"  {processed_dir}/")
    print()

    return {
        "success_count": success_count,
        "fail_count": fail_count,
        "skip_count": skip_count,
        "output_dir": processed_dir,
    }


def main() -> None:
    args = parse_args()
    process_clips(
        args.video_id,
        vertical=args.vertical,
        face_tracking=args.face_tracking,
        subtitles=args.subtitles,
        force=args.force,
        subtitle_style=args.subtitle_style or SUBTITLE_STYLE,
    )


if __name__ == "__main__":
    main()

