"""Centralized configuration and filesystem path helpers for Clipt."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Ensure .env is loaded when config is imported
load_dotenv()

# Project root directory (parent of core/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default output directory
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

# Default duration bounds (in seconds)
DEFAULT_MIN_DURATION: int = 30
DEFAULT_MAX_DURATION: int = 90
MAX_CLIPS: int = 10

# Default feature toggles
ENABLE_VERTICAL: bool = True
ENABLE_FACE_TRACKING: bool = True
ENABLE_SUBTITLES: bool = True
DEFAULT_SUBTITLE_STYLE: str = "highlight_yellow"

# Pretrained model path
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"


@dataclass
class EngineConfig:
    """Runtime configuration for Clipt analysis and processing pipelines."""

    output_dir: Path = DEFAULT_OUTPUT_DIR
    min_duration: int = DEFAULT_MIN_DURATION
    max_duration: int = DEFAULT_MAX_DURATION
    max_clips: int = MAX_CLIPS
    vertical: bool = ENABLE_VERTICAL
    face_tracking: bool = ENABLE_FACE_TRACKING
    subtitles: bool = ENABLE_SUBTITLES
    subtitle_style: str = DEFAULT_SUBTITLE_STYLE
    model_path: Path = DEFAULT_MODEL_PATH
    api_key: str | None = None

    def get_api_key(self) -> str:
        """Resolve Gemini API key from explicit config or environment."""
        key = self.api_key or os.environ.get("GEMINI_API_KEY", "")
        return key.strip()


# ── Filesystem Path Helpers ──────────────────────────────────────────

def get_video_dir(video_id: str, base_dir: Path | None = None) -> Path:
    """Return the output directory path for a specific video_id."""
    base = Path(base_dir) if base_dir else DEFAULT_OUTPUT_DIR
    return base / video_id


def get_processed_dir(video_id: str, base_dir: Path | None = None) -> Path:
    """Return the processedFiles directory path for a specific video_id."""
    return get_video_dir(video_id, base_dir) / "processedFiles"


def get_clips_json_path(video_id: str, base_dir: Path | None = None) -> Path:
    """Return the path to clips.json for a specific video_id."""
    return get_video_dir(video_id, base_dir) / "clips.json"


def get_source_path(video_id: str, base_dir: Path | None = None, filename: str = "source.mp4") -> Path:
    """Return the path to the cached source video file for a specific video_id."""
    return get_video_dir(video_id, base_dir) / filename


def get_transcript_path(video_id: str, base_dir: Path | None = None) -> Path:
    """Return the path to cached transcript.json for a specific video_id."""
    return get_video_dir(video_id, base_dir) / "transcript.json"
