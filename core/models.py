"""Pydantic models for Clipt clip candidates and output results."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Clip(BaseModel):
    """A single clip candidate. Pydantic validates Gemini's output."""

    start: float = Field(..., ge=0)
    end: float = Field(..., ge=0)
    duration: float = Field(0.0, ge=0)
    score: int = Field(..., ge=0, le=100)
    title: str = Field(..., min_length=1)
    hook: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    categories: list[str] = Field(default_factory=list)
    filename: str | None = Field(default=None)

    @model_validator(mode="after")
    def _check_timing(self) -> "Clip":
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be after start ({self.start})")
        self.duration = round(self.end - self.start, 3)
        return self


class ClipResult(BaseModel):
    """Complete output written to clips.json."""

    video_id: str
    source_url: str
    generated_at: str
    clips: list[Clip]
