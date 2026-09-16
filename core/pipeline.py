"""Core processing engine pipelines for Clipt.

Provides reusable, job-isolated programmatic interfaces for clip analysis and production rendering:
- analyze_video: phase 1 editorial analysis and proxy clip generation
- process_video: phase 2 production 9:16 vertical rendering with face tracking and subtitles
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from core.analyzer import analyze_transcript
from core.clipper import generate_clip
from core.config import (
    DEFAULT_MAX_DURATION,
    DEFAULT_MIN_DURATION,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SUBTITLE_STYLE,
    ENABLE_FACE_TRACKING,
    ENABLE_SUBTITLES,
    ENABLE_VERTICAL,
    MAX_CLIPS,
    get_clips_json_path,
    get_processed_dir,
    get_source_path,
    get_transcript_path,
    get_video_dir,
)
from core.downloader import download_video, probe_resolution
from core.exceptions import ProcessingError
from core.ffmpeg import check_ffmpeg
from core.models import Clip, ClipResult
from core.transcript import extract_video_id, get_transcript
from processing.face_tracker import track_face_for_clip
from processing.processor import generate_vertical_clip
from processing.subtitles import (
    build_clip_subtitles,
    generate_ass_file,
    load_or_fetch_transcript,
)

logger = logging.getLogger(__name__)


def analyze_video(
    url: str,
    *,
    output_dir: Path | str | None = None,
    min_duration: int = DEFAULT_MIN_DURATION,
    max_duration: int = DEFAULT_MAX_DURATION,
    max_clips: int = MAX_CLIPS,
    api_key: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> ClipResult:
    """Analyze a YouTube video, identify viral clips using Gemini, and generate preview MP4 clips.

    Parameters:
        url: YouTube video URL or 11-character video ID.
        output_dir: Base directory for output (defaults to DEFAULT_OUTPUT_DIR).
        min_duration: Minimum duration for clip candidates (seconds).
        max_duration: Maximum duration for clip candidates (seconds).
        max_clips: Maximum number of clips to return.
        api_key: Optional Gemini API key.
        progress_callback: Optional callable receiving progress status messages.

    Returns:
        ClipResult containing video metadata and validated Clip candidates.
    """
    base_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR

    # 1. System checks
    check_ffmpeg()

    # 2. Extract video ID
    if progress_callback:
        progress_callback("[1/5] Extracting video information...")
    video_id = extract_video_id(url)
    if progress_callback:
        progress_callback(f"  Video ID: {video_id}")

    # Standardize URL
    source_url = url if url.startswith("http") else f"https://www.youtube.com/watch?v={video_id}"

    # 3. Fetch transcript
    if progress_callback:
        progress_callback("[2/5] Fetching transcript...")
    _, segments = get_transcript(source_url)
    if progress_callback:
        progress_callback(f"  {len(segments)} segments loaded.")

    video_output_dir = get_video_dir(video_id, base_dir=base_dir)
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Cache transcript locally for phase 2
    transcript_cache_path = get_transcript_path(video_id, base_dir=base_dir)
    try:
        transcript_cache_path.write_text(
            json.dumps(segments, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not cache transcript.json: %s", exc)

    # 4. Analyze transcript with Gemini
    if progress_callback:
        progress_callback(f"[3/5] Finding high-potential clips with Gemini ({min_duration}s–{max_duration}s target)...")

    clip_dicts = analyze_transcript(
        segments,
        min_duration=min_duration,
        max_duration=max_duration,
        max_clips=max_clips,
        api_key=api_key,
    )
    clips = [Clip.model_validate(d) for d in clip_dicts]
    if progress_callback:
        progress_callback(f"  {len(clips)} clip(s) found.")

    # 5. Download lightweight 360p proxy video for preview clipping
    if progress_callback:
        progress_callback("[4/5] Downloading source video...")

    tmpdir = tempfile.mkdtemp(prefix=f"clipt_{video_id}_")
    source_path = Path(tmpdir) / f"{video_id}.mp4"

    try:
        source_video = download_video(source_url, source_path, quality="360p")
        if progress_callback:
            progress_callback("  Download complete.")

        # 6. Generate horizontal preview clips
        if progress_callback:
            progress_callback("[5/5] Generating clips...\n")

        success_count = 0
        fail_count = 0

        for i, clip in enumerate(clips):
            filename = f"clip_{i + 1:02d}.mp4"
            clip_output = video_output_dir / filename

            try:
                generate_clip(source_video, clip.start, clip.end, clip_output)
                clip.filename = filename
                success_count += 1
                if progress_callback:
                    progress_callback(f"  \u2713 {filename}")
            except Exception as exc:
                fail_count += 1
                if progress_callback:
                    progress_callback(f"  \u2717 {filename} \u2014 {exc}")

        if success_count == 0 and fail_count > 0:
            raise ProcessingError("No clips were generated successfully.")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # 7. Write manifest
    result = ClipResult(
        video_id=video_id,
        source_url=source_url,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        clips=clips,
    )

    clips_json_path = get_clips_json_path(video_id, base_dir=base_dir)
    clips_json_path.write_text(
        result.model_dump_json(indent=2, exclude_none=True),
        encoding="utf-8",
    )

    return result


def process_video(
    video_id: str,
    *,
    output_dir: Path | str | None = None,
    vertical: bool = ENABLE_VERTICAL,
    face_tracking: bool = ENABLE_FACE_TRACKING,
    subtitles: bool = ENABLE_SUBTITLES,
    force: bool = False,
    subtitle_style: str = DEFAULT_SUBTITLE_STYLE,
    model_path: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Process clips for *video_id* into production formats (9:16 vertical, face tracking, ASS subtitles).

    Parameters:
        video_id: YouTube video ID or URL.
        output_dir: Base directory for output (defaults to DEFAULT_OUTPUT_DIR).
        vertical: If True, crops and scales to 9:16 (1080×1920). If False, keeps source aspect ratio.
        face_tracking: If True (and vertical=True), runs YuNet face tracking.
        subtitles: If True, generates ASS captions and burns them in.
        force: If True, re-renders already-existing clips in processedFiles.
        subtitle_style: Visual style preset for ASS subtitles.
        model_path: Optional path to YuNet ONNX face detection model.
        progress_callback: Optional callable receiving progress status messages.

    Returns:
        Summary dict containing success_count, fail_count, skip_count, and output_dir.
    """
    base_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    check_ffmpeg()

    # Normalize video ID
    clean_id = extract_video_id(video_id)
    video_dir = get_video_dir(clean_id, base_dir=base_dir)

    if not video_dir.is_dir():
        raise ProcessingError(
            f"No Clipt project found for video ID '{clean_id}' at {video_dir}.\n"
            "Run python main.py <youtube-url> first."
        )

    clips_json_path = get_clips_json_path(clean_id, base_dir=base_dir)
    if not clips_json_path.is_file():
        raise ProcessingError(
            f"No clips.json found for video ID '{clean_id}'.\n\n"
            "Run:\n"
            "  python main.py <youtube-url>\n\n"
            f"before running:\n"
            f"  python process.py {clean_id}"
        )

    if progress_callback:
        progress_callback("[1/3] Loading clips.json...")

    try:
        manifest = json.loads(clips_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ProcessingError(f"Could not read clips.json: {exc}") from exc

    clips = manifest.get("clips", [])
    source_url = manifest.get("source_url")

    if not clips:
        raise ProcessingError("clips.json contains no clips.")
    if not source_url:
        raise ProcessingError(
            "clips.json does not contain a source_url.\n"
            "Re-run python main.py <youtube-url> to regenerate clips.json."
        )

    if progress_callback:
        progress_callback(f"  \u2713 Found {len(clips)} clips\n")

    # High-quality source download or cached source
    if progress_callback:
        progress_callback("[2/3] Downloading high-quality source...")
        progress_callback(f"  Source URL: {source_url}")

    cached_source = get_source_path(clean_id, base_dir=base_dir)
    tmpdir: str | None = None

    if cached_source.is_file():
        source_video = cached_source
        if progress_callback:
            progress_callback(f"  \u2713 Found existing source video: {cached_source.name}")
    else:
        tmpdir = tempfile.mkdtemp(prefix=f"clipt_process_{clean_id}_")
        source_target = Path(tmpdir) / f"{clean_id}.mp4"
        try:
            source_video = download_video(source_url, source_target, quality="1080p")
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise

    # Probe actual source resolution
    try:
        src_w, src_h = probe_resolution(source_video)
    except Exception:
        src_w, src_h = 0, 0

    if progress_callback:
        if src_h >= 1080:
            progress_callback(f"  \u2713 Downloaded source: {src_w}x{src_h}")
        elif src_h > 0:
            progress_callback("  \u26a0 1080p source unavailable.")
            progress_callback(f"    Highest available source: {src_w}x{src_h}")
        else:
            progress_callback("  \u2713 Source video ready (resolution unknown)")
        progress_callback("")

    # Load transcript for subtitles
    transcript_segments = []
    if subtitles:
        transcript_segments = load_or_fetch_transcript(video_dir, source_url)
        if progress_callback:
            if transcript_segments:
                progress_callback(f"  \u2713 Loaded transcript ({len(transcript_segments)} segments)")
            else:
                progress_callback("  \u26a0 No transcript available — clips will be generated without subtitles")
    else:
        if progress_callback:
            progress_callback("  \u2192 Subtitles disabled (skipping transcript fetch)")

    if progress_callback:
        progress_callback("")

    # Generate processed clips
    action_label = "vertical 9:16 clips" if vertical else "clips (original aspect ratio)"
    if progress_callback:
        progress_callback(f"[3/3] Generating {action_label}...\n")

    processed_dir = get_processed_dir(clean_id, base_dir=base_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    subs_dir = processed_dir / "subtitles"
    if subtitles:
        subs_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0
    skip_count = 0
    resolved_model_path = model_path or DEFAULT_MODEL_PATH

    try:
        for idx, clip in enumerate(clips, start=1):
            filename = clip.get("filename")
            if not filename:
                fail_count += 1
                if progress_callback:
                    progress_callback(f"Processing clip {idx}/{len(clips)}: (unknown) \u2014 missing filename")
                continue

            output_path = processed_dir / filename

            if output_path.is_file() and not force:
                skip_count += 1
                if progress_callback:
                    progress_callback(f"Processing clip {idx}/{len(clips)}: {filename} already exists \u2014 skipping")
                continue

            start = clip.get("start")
            end = clip.get("end")
            if start is None or end is None:
                fail_count += 1
                if progress_callback:
                    progress_callback(f"Processing clip {idx}/{len(clips)}: {filename} \u2014 missing start/end timestamps")
                continue

            if progress_callback:
                progress_callback(f"Processing clip {idx}/{len(clips)} ({filename})")

            # 1. Face tracking
            crop_x_expr = None
            crop_w = None
            crop_h = None

            if vertical:
                if face_tracking:
                    if progress_callback:
                        progress_callback("  Detecting faces...")
                    try:
                        track_res = track_face_for_clip(
                            source_video,
                            float(start),
                            float(end),
                            clean_id,
                            filename,
                            model_path=resolved_model_path,
                        )
                        crop_w = track_res.crop_w
                        crop_h = track_res.crop_h
                        crop_x_expr = track_res.crop_x_expr
                        if progress_callback:
                            if not track_res.is_tracked or track_res.face_count == 0:
                                progress_callback("  \u26a0 No face detected")
                                progress_callback("  \u2192 Using centered crop")
                            elif track_res.face_count == 1:
                                progress_callback("  \u2713 1 face detected")
                                progress_callback("  \u2192 Tracking subject")
                                progress_callback("  \u2713 Generating camera path")
                            else:
                                progress_callback(f"  \u2713 {track_res.face_count} faces detected")
                                progress_callback(f"  \u2713 Selected face {track_res.selected_face_idx}")
                                progress_callback("  \u2192 Tracking subject")
                                progress_callback("  \u2713 Generating camera path")
                    except Exception as exc:
                        logger.warning("Face tracking fallback for %s: %s", filename, exc)
                        if progress_callback:
                            progress_callback(f"  \u26a0 Face tracking encountered an issue: {exc}")
                            progress_callback("  \u2192 Falling back to centered crop")
                else:
                    if progress_callback:
                        progress_callback("  \u2192 Face tracking disabled (using centered crop)")
            else:
                pass

            # 2. Subtitle generation
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
                            if progress_callback:
                                progress_callback(f"  \u2713 Generating subtitles ({subtitle_style} style)")
                        except Exception as exc:
                            logger.warning("ASS subtitle generation failed: %s", exc)
                            if progress_callback:
                                progress_callback(f"  \u26a0 Could not generate ASS subtitles: {exc}")
                    else:
                        if progress_callback:
                            progress_callback("  \u26a0 No transcript segments overlapping this clip")
            else:
                if progress_callback:
                    progress_callback("  \u2192 Subtitles disabled")

            # 3. FFmpeg rendering
            try:
                if progress_callback:
                    if vertical:
                        progress_callback("  \u2713 Rendering 1080x1920 (9:16)")
                    else:
                        res_str = f"{src_w}x{src_h}" if src_w and src_h else "source resolution"
                        progress_callback(f"  \u2713 Rendering {res_str} (original aspect ratio)")

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
                if progress_callback:
                    progress_callback(f"  \u2713 {filename}")
            except Exception as exc:
                fail_count += 1
                if progress_callback:
                    progress_callback(f"  \u2717 {filename} \u2014 {exc}")

            # Clean up temp ASS
            if ass_path and ass_path.is_file():
                try:
                    ass_path.unlink()
                except OSError:
                    pass

            if progress_callback:
                progress_callback("")

    finally:
        # Clean up empty subtitles dir
        if subs_dir.is_dir():
            try:
                subs_dir.rmdir()
            except OSError:
                pass

        # Clean up temp download directory if any
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    total = success_count + fail_count + skip_count
    parts = []
    if success_count:
        parts.append(f"{success_count} generated")
    if skip_count:
        parts.append(f"{skip_count} skipped")
    if fail_count:
        parts.append(f"{fail_count} failed")

    if progress_callback:
        progress_callback(f"Processing complete. {', '.join(parts)} ({total} total)\n")
        progress_callback("Output:")
        progress_callback(f"  {processed_dir}/\n")

    return {
        "video_id": clean_id,
        "success_count": success_count,
        "fail_count": fail_count,
        "skip_count": skip_count,
        "output_dir": processed_dir,
    }
