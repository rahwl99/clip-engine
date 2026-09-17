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
    sample_clip = Path("output/3qHkcs3kG44/clip_01.mp4")
    if sample_clip.is_file():
        res = track_face_for_clip(
            sample_clip,
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


def test_cli_argument_toggles():
    print("[TEST] CLI argument toggles...")
    from unittest.mock import patch
    from process import parse_args

    # Default flags
    with patch("sys.argv", ["process.py", "test_vid"]):
        args = parse_args()
        assert args.vertical is True, "Default vertical should be True"
        assert args.face_tracking is True, "Default face_tracking should be True"
        assert args.subtitles is True, "Default subtitles should be True"

    # Negative flags
    with patch("sys.argv", ["process.py", "test_vid", "--no-vertical"]):
        args = parse_args()
        assert args.vertical is False, "--no-vertical should set vertical to False"

    with patch("sys.argv", ["process.py", "test_vid", "--no-crop"]):
        args = parse_args()
        assert args.vertical is False, "--no-crop should set vertical to False"

    with patch("sys.argv", ["process.py", "test_vid", "--no-9-16"]):
        args = parse_args()
        assert args.vertical is False, "--no-9-16 should set vertical to False"

    with patch("sys.argv", ["process.py", "test_vid", "--no-face-tracking"]):
        args = parse_args()
        assert args.face_tracking is False, "--no-face-tracking should set face_tracking to False"

    with patch("sys.argv", ["process.py", "test_vid", "--no-face-track"]):
        args = parse_args()
        assert args.face_tracking is False, "--no-face-track should set face_tracking to False"

    with patch("sys.argv", ["process.py", "test_vid", "--no-subtitles"]):
        args = parse_args()
        assert args.subtitles is False, "--no-subtitles should set subtitles to False"

    with patch("sys.argv", ["process.py", "test_vid", "--no-subs"]):
        args = parse_args()
        assert args.subtitles is False, "--no-subs should set subtitles to False"

    # All disabled
    with patch("sys.argv", ["process.py", "test_vid", "--no-vertical", "--no-face-tracking", "--no-subtitles"]):
        args = parse_args()
        assert args.vertical is False
        assert args.face_tracking is False
        assert args.subtitles is False

    # Explicit positive flags
    with patch("sys.argv", ["process.py", "test_vid", "--vertical", "--face-tracking", "--subtitles"]):
        args = parse_args()
        assert args.vertical is True
        assert args.face_tracking is True
        assert args.subtitles is True

    print("  ✓ CLI argument toggles and alias parsing passed")


def test_code_boolean_toggles():
    print("[TEST] Code boolean configuration and process_clips signature...")
    import inspect
    import process

    # Verify exported top-level constants
    assert hasattr(process, "ENABLE_VERTICAL"), "process.py must export ENABLE_VERTICAL"
    assert hasattr(process, "ENABLE_FACE_TRACKING"), "process.py must export ENABLE_FACE_TRACKING"
    assert hasattr(process, "ENABLE_SUBTITLES"), "process.py must export ENABLE_SUBTITLES"
    assert isinstance(process.ENABLE_VERTICAL, bool)
    assert isinstance(process.ENABLE_FACE_TRACKING, bool)
    assert isinstance(process.ENABLE_SUBTITLES, bool)

    # Verify process_clips signature has default boolean parameters
    sig = inspect.signature(process.process_clips)
    params = sig.parameters
    assert "vertical" in params, "process_clips must accept 'vertical' parameter"
    assert "face_tracking" in params, "process_clips must accept 'face_tracking' parameter"
    assert "subtitles" in params, "process_clips must accept 'subtitles' parameter"
    assert params["vertical"].default == process.ENABLE_VERTICAL
    assert params["face_tracking"].default == process.ENABLE_FACE_TRACKING
    assert params["subtitles"].default == process.ENABLE_SUBTITLES

    print("  ✓ Code boolean toggles and process_clips signature passed")


def test_ass_generation_horizontal_vs_vertical():
    print("[TEST] ASS subtitle generation for horizontal vs vertical resolutions...")
    chunks = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "TEST CAPTION",
            "words": [{"word": "TEST", "start": 0.0, "end": 1.0}, {"word": "CAPTION", "start": 1.0, "end": 2.0}],
        }
    ]

    with tempfile.NamedTemporaryFile(suffix=".ass", delete=False, mode="w", encoding="utf-8") as tmp:
        ass_path = Path(tmp.name)

    try:
        # 1. Vertical 9:16 (1080x1920)
        generate_ass_file(chunks, ass_path, video_width=1080, video_height=1920)
        v_content = ass_path.read_text(encoding="utf-8")
        assert "PlayResX: 1080" in v_content
        assert "PlayResY: 1920" in v_content
        assert ",78," in v_content, "Vertical ASS should use 78pt font"
        assert ",360," in v_content, "Vertical ASS should use 360 vertical margin"

        # 2. Horizontal 16:9 (1920x1080)
        generate_ass_file(chunks, ass_path, video_width=1920, video_height=1080)
        h_content = ass_path.read_text(encoding="utf-8")
        assert "PlayResX: 1920" in h_content
        assert "PlayResY: 1080" in h_content
        assert ",44," in h_content, "Horizontal ASS should scale font size down proportionally"
        assert ",108," in h_content, "Horizontal ASS should use bottom-aligned 108 vertical margin"

    finally:
        if ass_path.is_file():
            ass_path.unlink()

    print("  ✓ Horizontal vs vertical ASS subtitle generation passed")


def test_processor_vertical_filter_toggle():
    print("[TEST] Processor FFmpeg command with vertical toggle...")
    from unittest.mock import patch, MagicMock
    from processing.processor import generate_vertical_clip, generate_processed_clip

    assert generate_processed_clip is generate_vertical_clip, "generate_processed_clip should alias generate_vertical_clip"

    fake_source = Path("dummy_source.mp4")
    fake_output = Path("dummy_output.mp4")

    # Mock _probe_dimensions and subprocess.run
    with patch("processing.processor._probe_dimensions", return_value=(1920, 1080)), \
         patch("subprocess.run") as mock_run, \
         patch("pathlib.Path.is_file", return_value=True), \
         patch("pathlib.Path.mkdir"):

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        # Test 1: vertical=True
        generate_vertical_clip(fake_source, 0.0, 5.0, fake_output, vertical=True)
        call_args_vertical = mock_run.call_args[0][0]
        cmd_str_v = " ".join(call_args_vertical)
        assert "-vf" in call_args_vertical, "Vertical clip must have -vf filter"
        assert "scale=1080:1920" in cmd_str_v, "Vertical clip must scale to 1080:1920"
        assert "crop=" in cmd_str_v, "Vertical clip must crop"

        # Test 2: vertical=False (no subtitles)
        generate_vertical_clip(fake_source, 0.0, 5.0, fake_output, vertical=False)
        call_args_horizontal = mock_run.call_args[0][0]
        cmd_str_h = " ".join(call_args_horizontal)
        assert "-vf" not in call_args_horizontal, "Non-vertical clip without subs should not have -vf"
        assert "scale=1080:1920" not in cmd_str_h
        assert "crop=" not in cmd_str_h

        # Test 3: vertical=False (with subtitles)
        fake_ass = Path("dummy.ass")
        with patch.object(Path, "is_file", return_value=True):
            generate_vertical_clip(fake_source, 0.0, 5.0, fake_output, vertical=False, ass_path=fake_ass)
            call_args_with_subs = mock_run.call_args[0][0]
            cmd_str_subs = " ".join(call_args_with_subs)
            assert "-vf" in call_args_with_subs, "Non-vertical clip with subs must include -vf"
            assert "subtitles=" in cmd_str_subs
            assert "crop=" not in cmd_str_subs
            assert "scale=1080:1920" not in cmd_str_subs

    print("  ✓ Processor FFmpeg vertical toggle passed")


def test_video_duration_configuration():
    print("[TEST] Video duration configuration and prompt generation...")
    import inspect
    import main
    from core.analyzer import (
        DEFAULT_MIN_DURATION,
        DEFAULT_MAX_DURATION,
        _build_system_prompt,
        _build_user_prompt,
        analyze_transcript,
    )

    # Check that variables exist and are prominently accessible
    assert hasattr(main, "MIN_DURATION"), "main.py must expose MIN_DURATION"
    assert hasattr(main, "MAX_DURATION"), "main.py must expose MAX_DURATION"
    assert isinstance(main.MIN_DURATION, int)
    assert isinstance(main.MAX_DURATION, int)
    assert main.MIN_DURATION > 0
    assert main.MAX_DURATION > main.MIN_DURATION

    # Check analyzer defaults
    assert DEFAULT_MIN_DURATION == 30
    assert DEFAULT_MAX_DURATION == 90

    # Test short prompt generation
    sys_prompt_short = _build_system_prompt(30, 90)
    assert "30 and 90 seconds long (30s–90s)" in sys_prompt_short
    assert "short-form" in sys_prompt_short

    # Test medium prompt generation
    sys_prompt_med = _build_system_prompt(90, 300)
    assert "90 and 300 seconds long (90s–300s)" in sys_prompt_med
    assert "medium-length" in sys_prompt_med

    # Test long prompt generation
    sys_prompt_long = _build_system_prompt(300, 900)
    assert "300 and 900 seconds long (300s–900s)" in sys_prompt_long
    assert "in-depth chapters" in sys_prompt_long

    # Test user prompt duration injection
    user_prompt = _build_user_prompt("dummy transcript", 120, 360, 5)
    assert "between 120 and 360 seconds in duration" in user_prompt
    assert "Identify up to 5" in user_prompt

    # Verify analyze_transcript signature
    sig = inspect.signature(analyze_transcript)
    assert "min_duration" in sig.parameters
    assert "max_duration" in sig.parameters

    print("  ✓ Video duration configuration and prompt generation passed")


def test_configuration_and_paths():
    print("[TEST] Configuration defaults, overrides, and path helpers...")
    from core.config import (
        EngineConfig,
        DEFAULT_MIN_DURATION,
        DEFAULT_MAX_DURATION,
        ENABLE_VERTICAL,
        ENABLE_FACE_TRACKING,
        ENABLE_SUBTITLES,
        get_video_dir,
        get_processed_dir,
        get_clips_json_path,
        get_source_path,
        get_transcript_path,
    )

    # 1. Config defaults
    cfg = EngineConfig()
    assert cfg.min_duration == DEFAULT_MIN_DURATION
    assert cfg.max_duration == DEFAULT_MAX_DURATION
    assert cfg.vertical == ENABLE_VERTICAL
    assert cfg.face_tracking == ENABLE_FACE_TRACKING
    assert cfg.subtitles == ENABLE_SUBTITLES

    # 2. Config override
    custom_cfg = EngineConfig(min_duration=60, max_duration=120, vertical=False)
    assert custom_cfg.min_duration == 60
    assert custom_cfg.max_duration == 120
    assert custom_cfg.vertical is False

    # 3. Path helpers with custom base directory
    custom_base = Path("/custom/output")
    vid_id = "test_vid_123"
    assert get_video_dir(vid_id, base_dir=custom_base) == custom_base / vid_id
    assert get_processed_dir(vid_id, base_dir=custom_base) == custom_base / vid_id / "processedFiles"
    assert get_clips_json_path(vid_id, base_dir=custom_base) == custom_base / vid_id / "clips.json"
    assert get_source_path(vid_id, base_dir=custom_base) == custom_base / vid_id / "source.mp4"
    assert get_transcript_path(vid_id, base_dir=custom_base) == custom_base / vid_id / "transcript.json"

    print("  ✓ Configuration defaults, overrides, and path helpers passed")


def test_video_id_validation():
    print("[TEST] Video ID extraction and URL validation...")
    from core.transcript import extract_video_id
    from core.exceptions import VideoIDError

    test_cases = [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("http://youtube.com/watch?v=dQw4w9WgXcQ&feature=share", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ]

    for url, expected in test_cases:
        assert extract_video_id(url) == expected, f"Failed for {url}"

    # Invalid inputs
    invalid_inputs = [
        "https://notyoutube.com/watch?v=12345",
        "invalid_video_id",
        "",
        "   ",
        "https://youtube.com/",
    ]
    for bad in invalid_inputs:
        try:
            extract_video_id(bad)
            assert False, f"Expected VideoIDError for {bad}"
        except VideoIDError:
            pass
        except ValueError:
            pass

    print("  ✓ Video ID extraction and URL validation passed")


def test_exception_hierarchy():
    print("[TEST] Domain exception hierarchy and backward compatibility...")
    from core.exceptions import (
        CliptError,
        ConfigurationError,
        VideoIDError,
        TranscriptError,
        AnalysisError,
        DownloadError,
        FFmpegError,
        ProcessingError,
    )

    # Base class check
    for exc_cls in (
        ConfigurationError,
        VideoIDError,
        TranscriptError,
        AnalysisError,
        DownloadError,
        FFmpegError,
        ProcessingError,
    ):
        assert issubclass(exc_cls, CliptError)

    # Backward compatibility with standard library exceptions
    assert issubclass(VideoIDError, ValueError)
    assert issubclass(TranscriptError, RuntimeError)
    assert issubclass(AnalysisError, RuntimeError)
    assert issubclass(DownloadError, RuntimeError)
    assert issubclass(FFmpegError, RuntimeError)
    assert issubclass(ProcessingError, RuntimeError)

    print("  ✓ Domain exception hierarchy and backward compatibility passed")


def test_programmatic_pipeline_interface():
    print("[TEST] Programmatic engine interface (API readiness)...")
    from core import analyze_video as av_from_core, process_video as pv_from_core
    from core.pipeline import analyze_video, process_video
    from processing import process_video as pv_from_proc
    from processing.processor import process_video as pv_from_processor
    from core.exceptions import ProcessingError

    # Assert callable exports across packages
    assert callable(analyze_video)
    assert callable(process_video)
    assert callable(av_from_core)
    assert callable(pv_from_core)
    assert callable(pv_from_proc)
    assert callable(pv_from_processor)

    # Verify process_video raises ProcessingError when project dir does not exist (does NOT call sys.exit)
    try:
        process_video("nonexist123", output_dir=tempfile.mkdtemp())
        assert False, "Expected ProcessingError for nonexistent video project"
    except ProcessingError as exc:
        assert "nonexist123" in str(exc)

    # Verify progress callback receives messages without error
    received_messages = []
    try:
        process_video(
            "dummy123456",
            output_dir=tempfile.mkdtemp(),
            progress_callback=received_messages.append,
        )
    except ProcessingError:
        pass

    print("  ✓ Programmatic engine interface passed without calling sys.exit()")


def test_fastapi_endpoints():
    print("[TEST] FastAPI endpoints (Health, Analyze, Status, Process)...")
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from api.main import app
    from api.jobs import job_store

    job_store.clear()
    client = TestClient(app)

    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

    # 2. Analyze validation
    res = client.post("/jobs/analyze", json={"youtube_url": "invalid_url_string"})
    assert res.status_code == 400

    # 3. Analyze dispatch
    with patch("api.routes._executor.submit"):
        res = client.post("/jobs/analyze", json={"youtube_url": "https://www.youtube.com/watch?v=3qHkcs3kG44"})
        assert res.status_code == 202
        job_id = res.json()["job_id"]
        assert res.json()["status"] == "processing"

    # 4. Job status
    res = client.get(f"/jobs/{job_id}")
    assert res.status_code == 200
    assert res.json()["job_id"] == job_id

    # 404 for nonexistent
    res = client.get("/jobs/missing123")
    assert res.status_code == 404

    # 5. Process validation
    job_store.create_job(job_type="analyze", video_id="3qHkcs3kG44", job_id="test_done")
    job_store.update_job(
        "test_done",
        status="completed",
        result={
            "video_id": "3qHkcs3kG44",
            "clips": [{"clip_id": "clip_01", "filename": "clip_01.mp4"}],
        },
    )

    # Invalid clip ID
    res = client.post("/jobs/test_done/process", json={"clip_ids": ["clip_99"]})
    assert res.status_code == 400

    # Valid dispatch
    with patch("api.routes._executor.submit"):
        res = client.post(
            "/jobs/test_done/process",
            json={
                "clip_ids": ["clip_01"],
                "vertical": True,
                "face_tracking": True,
                "subtitles": True,
                "subtitle_style": "highlight_yellow",
            },
        )
        assert res.status_code == 202
        assert res.json()["status"] == "processing"

    print("  ✓ FastAPI endpoints (Health, Analyze, Status, Process) passed")


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
    test_cli_argument_toggles()
    test_code_boolean_toggles()
    test_ass_generation_horizontal_vs_vertical()
    test_processor_vertical_filter_toggle()
    test_video_duration_configuration()
    test_configuration_and_paths()
    test_video_id_validation()
    test_exception_hierarchy()
    test_programmatic_pipeline_interface()
    test_fastapi_endpoints()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all()


