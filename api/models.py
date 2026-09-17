"""FastAPI Pydantic request and response schemas."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, field_validator

from core.config import (
    DEFAULT_SUBTITLE_STYLE,
    ENABLE_FACE_TRACKING,
    ENABLE_SUBTITLES,
    ENABLE_VERTICAL,
)
from processing.subtitles import SUBTITLE_STYLES


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = "ok"


class AnalyzeRequest(BaseModel):
    """Request payload for starting a video analysis job."""

    youtube_url: str = Field(
        ...,
        description="YouTube video URL or 11-character video ID",
        examples=["https://www.youtube.com/watch?v=3qHkcs3kG44"],
    )


class JobResponse(BaseModel):
    """Immediate response returning the assigned job_id and initial status."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Current job status (queued, processing)")


class ProcessRequest(BaseModel):
    """Request payload for processing selected clips into 9:16 vertical videos."""

    clip_ids: list[str] = Field(
        default_factory=list,
        description="List of clip IDs or filenames to process (e.g. ['clip_01', 'clip_02']). If empty, processes all clips.",
        examples=[["clip_01", "clip_02"]],
    )
    vertical: bool = Field(
        default=ENABLE_VERTICAL,
        description="Enable/disable 9:16 vertical conversion (default: True).",
    )
    face_tracking: bool = Field(
        default=ENABLE_FACE_TRACKING,
        description="Enable/disable OpenCV YuNet face tracking (default: True).",
    )
    subtitles: bool = Field(
        default=ENABLE_SUBTITLES,
        description="Enable/disable burned-in ASS subtitles (default: True).",
    )
    subtitle_style: str = Field(
        default=DEFAULT_SUBTITLE_STYLE,
        description=f"Subtitle visual style preset (choices: {', '.join(SUBTITLE_STYLES.keys())}).",
    )
    force: bool = Field(
        default=False,
        description="Re-render and overwrite already existing processed clips.",
    )

    @field_validator("subtitle_style")
    @classmethod
    def validate_subtitle_style(cls, v: str) -> str:
        if v not in SUBTITLE_STYLES:
            valid_styles = ", ".join(SUBTITLE_STYLES.keys())
            raise ValueError(
                f"Invalid subtitle_style '{v}'. Must be one of: {valid_styles}"
            )
        return v


class JobStatusResponse(BaseModel):
    """Job status polling response schema."""

    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    stage: str | None = None
    progress: int | None = None
    video_id: str | None = None
    result: Any | None = None
    error: str | None = None

    model_config = {
        "extra": "ignore",
    }
