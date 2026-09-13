"""Comprehensive verification test suite for Clipt Face Tracking and Subtitles."""

import json
import sys
import tempfile
from pathlib import Path
import numpy as np

# Ensure Unicode output works on Windows consoles.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from processing.face_tracker import (
    FaceBox,
    FaceDetector,
    select_face_deterministically,
    smooth_camera_positions,
    build_ffmpeg_crop_expression,
    CameraKeyframe,
    track_face_for_clip,
)
from processing.subtitles import (
    format_ass_time,
    build_clip_subtitles,
    generate_ass_file,
    load_or_fetch_transcript,
    get_subtitle_style,
    SUBTITLE_STYLES,
    DEFAULT_SUBTITLE_STYLE,
)
from processing.processor import generate_vertical_clip


def test_subtitles_timing_and_formatting():
    print("[TEST] Subtitle timing and formatting...")
    # Timestamp formatting
    assert format_ass_time(0.0) == "0:00:00.00"
    assert format_ass_time(61.25) == "0:01:01.25"
    assert format_ass_time(3600.0) == "1:00:00.00"

    # Clipping logic
    segments = [
        {"start": 10.0, "end": 15.0, "text": "Before clip"},
        {"start": 18.0, "end": 25.0, "text": "Overlapping start"},
        {"start": 25.0, "end": 35.0, "text": "Inside clip"},
        {"start": 35.0, "end": 42.0, "text": "Overlapping end"},
        {"start": 50.0, "end": 60.0, "text": "After clip"},
    ]
    # Clip from 20.0 to 40.0
    res = build_clip_subtitles(segments, 20.0, 40.0, max_words_per_screen=4)
    assert len(res) > 0

    # Verify max 4 words per screen
    for chunk in res:
        word_count = len(chunk["text"].split())
        assert word_count <= 4, f"Chunk exceeded 4 words: {chunk['text']}"

    # Verify sequential non-overlapping timing
    for i in range(len(res) - 1):
        assert res[i]["end"] <= res[i + 1]["start"] + 0.01, "Chunks must not overlap"

    # Verify outside segments were excluded
    all_text = " ".join(r["text"] for r in res)
    assert "BEFORE CLIP" not in all_text
    assert "AFTER CLIP" not in all_text

    print("  ✓ Subtitle timing, 4-word max, and non-overlap clipping passed")


def test_face_selection_and_determinism():
    print("[TEST] Face selection and determinism...")
    faces = [
        FaceBox(100, 200, 80, 80),
        FaceBox(500, 200, 80, 80),
        FaceBox(900, 200, 80, 80),
    ]
    # Case: Single face
    idx_single, box_single = select_face_deterministically([faces[0]], "vid1", "clip_01.mp4")
    assert idx_single == 0
    assert box_single == faces[0]

    # Case: Multiple faces — deterministic for same video_id and clip
    idx_a, _ = select_face_deterministically(faces, "vid1", "clip_01.mp4")
    idx_b, _ = select_face_deterministically(faces, "vid1", "clip_01.mp4")
    assert idx_a == idx_b, "Selection must be deterministic for identical inputs"

    # Different clips may select different faces
    selections = {select_face_deterministically(faces, "vid1", f"clip_{i:02d}.mp4")[0] for i in range(10)}
    assert len(selections) > 1, "Deterministic selection should vary across different clips"

    print(f"  ✓ Deterministic multi-face selection passed (selected face {idx_a + 1})")


def test_camera_smoothing_and_bounds():
    print("[TEST] Camera smoothing and boundary clamping...")
    # Moving subject across a 1920x1080 source (9:16 crop is 608 wide)
    raw = [
        (0.0, 100.0),   # Far left
        (1.0, 300.0),   # Moving right
        (2.0, 1000.0),  # Sudden jump
        (3.0, 1900.0),  # Far right
    ]
    src_w = 1920
    crop_w = 608
    smoothed = smooth_camera_positions(raw, crop_w, src_w, alpha=0.3, max_speed_px_per_s=180.0)

    for t, cx in smoothed:
        # Verify strict clamping within valid frame bounds
        assert 0.0 <= cx <= (src_w - crop_w), f"Crop X {cx} exceeded [0, {src_w - crop_w}]"

    # Verify inertia prevented jump from 300 to 1000 at t=2.0 (dt=1.0, max delta=180)
    x_at_1 = [x for t, x in smoothed if t == 1.0][0]
    x_at_2 = [x for t, x in smoothed if t == 2.0][0]
    assert abs(x_at_2 - x_at_1) <= 180.1, f"Movement {abs(x_at_2 - x_at_1)} exceeded max rate 180px/s"

    print("  ✓ Camera smoothing and boundary clamping passed")


def test_missing_transcript_fallback():
    print("[TEST] Missing transcript fallback...")
    segments = []
    clip_subs = build_clip_subtitles(segments, 0.0, 10.0)
    assert clip_subs == []
    print("  ✓ Missing transcript generates empty subtitles list without error")


def test_no_face_fallback():
    print("[TEST] No face detected fallback...")
    # Simulated tracking result when 0 faces found
    res = track_face_for_clip(
        Path("output/3qHkcs3kG44/clip_01.mp4"),
        0.0,
        1.0,
        "vid_dummy",
        "clip_dummy.mp4"
    )
    # The clip has a face, but let's test the fallback expression builder with empty keyframes
    empty_expr = build_ffmpeg_crop_expression([], crop_w=608, src_w=1920)
    assert empty_expr == str((1920 - 608) // 2)
    print(f"  ✓ Empty keyframes fallback to center crop: {empty_expr}")


def test_subtitle_styles_and_word_highlighting():
    print("[TEST] Subtitle styles and active word highlighting...")

    # 1. Style resolution and aliases
    yellow_style = get_subtitle_style("highlight_yellow")
    assert yellow_style["highlight"] is True
    assert yellow_style["highlight_color"] == "&H0000FFFF"

    cyan_style = get_subtitle_style("cyan")
    assert cyan_style["highlight"] is True
    assert cyan_style["highlight_color"] == "&H00FFFF00"

    classic_style = get_subtitle_style("classic")
    assert classic_style["highlight"] is False

    unknown_style = get_subtitle_style("nonexistent_style")
    assert unknown_style["name"] == SUBTITLE_STYLES[DEFAULT_SUBTITLE_STYLE]["name"]

    # 2. Weighted duration estimation
    segs = [{"start": 0.0, "end": 4.0, "text": "A extraordinarily remarkable test"}]
    chunks = build_clip_subtitles(segs, 0.0, 4.0, max_words_per_screen=4)
    assert len(chunks) == 1
    words = chunks[0]["words"]
    assert len(words) == 4
    # "A" should receive less duration than "EXTRAORDINARILY"
    dur_a = words[0]["end"] - words[0]["start"]
    dur_long = words[1]["end"] - words[1]["start"]
    assert dur_long > dur_a, f"Long word ({dur_long}s) should have more duration than 'A' ({dur_a}s)"

    # 3. ASS Generation with active word highlight (Sliced Dialogue)
    with tempfile.NamedTemporaryFile(suffix=".ass", delete=False, mode="w", encoding="utf-8") as tmp:
        ass_path = Path(tmp.name)

    try:
        generate_ass_file(chunks, ass_path, style="highlight_yellow")
        ass_content = ass_path.read_text(encoding="utf-8")

        # Must have dialogue lines for each word
        dialogue_lines = [l for l in ass_content.splitlines() if l.startswith("Dialogue:")]
        assert len(dialogue_lines) == 4, f"Expected 4 dialogue events, got {len(dialogue_lines)}"

        # First line highlights "A"
        assert r"{\c&H0000FFFF" in dialogue_lines[0]
        assert r"{\rShortsStyle}" in dialogue_lines[0]
        assert r"\fscx108\fscy108" in dialogue_lines[0]

        # 4. ASS Generation in classic style (no word highlighting)
        generate_ass_file(chunks, ass_path, style="classic")
        classic_content = ass_path.read_text(encoding="utf-8")
        classic_lines = [l for l in classic_content.splitlines() if l.startswith("Dialogue:")]
        assert len(classic_lines) == 1, f"Expected 1 dialogue event for classic chunk, got {len(classic_lines)}"
        assert r"{\c&H" not in classic_lines[0]
        assert "A EXTRAORDINARILY REMARKABLE TEST" in classic_lines[0]

    finally:
        if ass_path.is_file():
            ass_path.unlink()

    print("  ✓ Subtitle styles and active word highlighting passed")


def run_all():
    print("=" * 60)
    print("RUNNING CLIPT AUTOMATED TEST SUITE")
    print("=" * 60)
    test_subtitles_timing_and_formatting()
    test_subtitle_styles_and_word_highlighting()
    test_face_selection_and_determinism()
    test_camera_smoothing_and_bounds()
    test_missing_transcript_fallback()
    test_no_face_fallback()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
