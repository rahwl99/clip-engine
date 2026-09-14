"""Clipt — identify and generate high-potential short-form clips from YouTube videos."""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Ensure Unicode output works on Windows consoles.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from core.transcript import get_transcript
from core.analyzer import analyze_transcript
from core.downloader import download_video
from core.clipper import check_ffmpeg, generate_clip
from core.models import Clip, ClipResult

OUTPUT_DIR = Path("output")

# ── Video Length Configuration (Edit Here) ────────────────────────────
# Set your desired clip duration in seconds (highlighted for easy editing):
# Examples:
#   Short clips (Shorts, Reels, TikTok):  MIN_DURATION = 30,  MAX_DURATION = 90
#   Medium clips (highlights, topics):   MIN_DURATION = 90,  MAX_DURATION = 300   (1.5 to 5 mins)
#   Longer deep-dives / chapters:        MIN_DURATION = 300, MAX_DURATION = 900   (5 to 15 mins)
# ─────────────────────────────────────────────────────────────────────
MIN_DURATION: int = 30   # Minimum clip duration in seconds
MAX_DURATION: int = 90   # Maximum clip duration in seconds


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <youtube-url>")
        sys.exit(1)

    url = sys.argv[1]

    print()
    print("Clipt")
    print("─────")
    print()

    # ── Step 1: Extract video info ────────────────────────────────────
    print("[1/5] Extracting video information...")
    try:
        from core.transcript import extract_video_id
        video_id = extract_video_id(url)
    except ValueError as exc:
        print(f"  Error: {exc}")
        sys.exit(1)

    print(f"  Video ID: {video_id}")

    # ── Step 2: Fetch transcript ──────────────────────────────────────
    print("[2/5] Fetching transcript...")
    try:
        _, segments = get_transcript(url)
    except (ValueError, RuntimeError) as exc:
        print(f"  Error: {exc}")
        sys.exit(1)

    print(f"  {len(segments)} segments loaded.")

    # ── Step 3: Gemini analysis ───────────────────────────────────────
    print(f"[3/5] Finding high-potential clips with Gemini ({MIN_DURATION}s–{MAX_DURATION}s target)...")
    try:
        clip_dicts = analyze_transcript(
            segments,
            min_duration=MIN_DURATION,
            max_duration=MAX_DURATION,
        )
    except RuntimeError as exc:
        print(f"  Error: {exc}")
        sys.exit(1)

    clips = [Clip.model_validate(d) for d in clip_dicts]
    print(f"  {len(clips)} clip(s) found.")

    # ── Step 4: Download source video ─────────────────────────────────
    print("[4/5] Downloading source video...")

    # Fail fast if FFmpeg is missing — don't waste time downloading.
    try:
        check_ffmpeg()
    except RuntimeError as exc:
        print(f"  Error: {exc}")
        sys.exit(1)

    tmpdir = tempfile.mkdtemp(prefix="clipt_")
    source_path = Path(tmpdir) / f"{video_id}.mp4"

    try:
        source_video = download_video(url, source_path, quality="360p")
    except RuntimeError as exc:
        print(f"  Error: {exc}")
        sys.exit(1)

    print("  Download complete.")

    # ── Step 5: Generate clips ────────────────────────────────────────
    print("[5/5] Generating clips...")
    print()

    video_output_dir = OUTPUT_DIR / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for i, clip in enumerate(clips):
        filename = f"clip_{i + 1:02d}.mp4"
        clip_output = video_output_dir / filename

        try:
            generate_clip(source_video, clip.start, clip.end, clip_output)
            clip.filename = filename
            success_count += 1
            print(f"  \u2713 {filename}")
        except RuntimeError as exc:
            fail_count += 1
            print(f"  \u2717 {filename} \u2014 {exc}")

    # ── Clean up temp files ───────────────────────────────────────────
    try:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass

    # ── Save results ──────────────────────────────────────────────────
    print()

    if success_count == 0 and fail_count > 0:
        print("Error: No clips were generated successfully.")
        sys.exit(1)

    result = ClipResult(
        video_id=video_id,
        source_url=url,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        clips=clips,
    )

    clips_json_path = video_output_dir / "clips.json"
    clips_json_path.write_text(
        result.model_dump_json(indent=2, exclude_none=True),
        encoding="utf-8",
    )

    total = success_count + fail_count
    print(f"Done \u2014 {success_count}/{total} clips generated.")
    print()
    print("Output:")
    print(f"  {video_output_dir}/")
    print()


if __name__ == "__main__":
    main()
