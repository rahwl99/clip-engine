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

# Choose subtitle style: "highlight_yellow" (default), "highlight_cyan", "highlight_green", "highlight_magenta", "classic"
SUBTITLE_STYLE: str = DEFAULT_SUBTITLE_STYLE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clipt — generate vertical 9:16 clips with face tracking and subtitles."
    )
    parser.add_argument("video_id", help="The YouTube video ID to process")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process and overwrite existing clips in processedFiles",
    )
    parser.add_argument(
        "--subtitle-style",
        default=SUBTITLE_STYLE,
        help=f"Subtitle visual style (choices: {', '.join(SUBTITLE_STYLES.keys())}, classic, yellow, cyan, green). Default: {SUBTITLE_STYLE}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_id = args.video_id
    force = args.force
    subtitle_style = args.subtitle_style or SUBTITLE_STYLE

    print()
    print("Clipt Video Processor")
    print("─" * 22)
    print()
    print(f"Video ID: {video_id}")
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
        print(f"  ✓ Source video ready (resolution unknown)")

    print()

    # ── Load or fetch transcript ──────────────────────────────────────
    transcript_segments = load_or_fetch_transcript(video_dir, source_url)
    if transcript_segments:
        print(f"  ✓ Loaded transcript ({len(transcript_segments)} segments)")
    else:
        print("  ⚠ No transcript available — clips will be generated without subtitles")

    print()

    # ── Generate vertical clips ───────────────────────────────────────
    print("[3/3] Generating vertical clips...")
    print()

    processed_dir = video_dir / "processedFiles"
    processed_dir.mkdir(parents=True, exist_ok=True)

    subs_dir = processed_dir / "subtitles"
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
        print("  Detecting faces...")

        # ── 1. Face tracking ──
        crop_x_expr = None
        crop_w = None
        crop_h = None
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
                print("  ✓ Tracking subject")
                print("  ✓ Generating camera path")

        except Exception as exc:
            print(f"  ⚠ Face tracking encountered an issue: {exc}")
            print("  → Falling back to centered crop")

        # ── 2. Subtitle generation ──
        ass_path = None
        if transcript_segments:
            clip_subs = build_clip_subtitles(transcript_segments, float(start), float(end))
            if clip_subs:
                ass_filename = Path(filename).with_suffix(".ass").name
                temp_ass_path = subs_dir / ass_filename
                try:
                    generate_ass_file(clip_subs, temp_ass_path, style=subtitle_style)
                    ass_path = temp_ass_path
                    print(f"  ✓ Generating subtitles ({subtitle_style} style)")
                except Exception as exc:
                    print(f"  ⚠ Could not generate ASS subtitles: {exc}")
            else:
                print("  ⚠ No transcript segments overlapping this clip")

        # ── 3. FFmpeg rendering ──
        try:
            print("  ✓ Rendering 1080x1920")
            generate_vertical_clip(
                source_video,
                float(start),
                float(end),
                output_path,
                crop_x_expr=crop_x_expr,
                crop_w=crop_w,
                crop_h=crop_h,
                ass_path=ass_path,
            )
            success_count += 1
            print(f"  ✓ {filename}")
        except RuntimeError as exc:
            fail_count += 1
            print(f"  ✗ {filename} — {exc}")

        # Clean up temporary ASS file per Section 14 / Section 21
        if ass_path and ass_path.is_file():
            try:
                ass_path.unlink()
            except OSError:
                pass

        print()

    # Clean up empty subtitles dir if leftover
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


if __name__ == "__main__":
    main()
