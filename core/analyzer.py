"""Analyzer — send transcript to Gemini, parse and validate clip candidates."""

import json
import logging
import os

from google import genai

from core.config import (
    DEFAULT_MAX_DURATION,
    DEFAULT_MIN_DURATION,
    MAX_CLIPS,
)
from core.exceptions import AnalysisError, ConfigurationError
from core.models import Clip

logger = logging.getLogger(__name__)


# ── Prompt Generation ─────────────────────────────────────────────────

def _build_system_prompt(min_duration: int, max_duration: int) -> str:
    """Construct the Gemini system instruction tailored to the desired duration."""
    if max_duration <= 90:
        target_type = "short-form clips (YouTube Shorts, Instagram Reels, or TikTok)"
        pacing_guidance = (
            "Prioritise strong hooks, curiosity, surprising information, "
            "emotional moments, and punchy self-contained context."
        )
    elif max_duration <= 300:
        target_type = f"medium-length standalone video segments ({min_duration / 60:.1f}–{max_duration / 60:.1f} minutes)"
        pacing_guidance = (
            "Prioritise complete topic discussions, engaging explanations, "
            "stories with full arcs, and high-value insights."
        )
    else:
        target_type = f"in-depth chapters and long-form topic breakdowns ({min_duration / 60:.1f}–{max_duration / 60:.1f} minutes)"
        pacing_guidance = (
            "Prioritise comprehensive deep-dives, detailed interview answers, "
            "and full conceptual explanations."
        )

    return f"""\
You are a professional video editor and content strategist.

Your job is to analyse a timestamped video transcript and identify the strongest \
self-contained segments that would perform well as {target_type}.

Rules:
- Only use timestamps that appear in the provided transcript. Never invent timestamps.
- Each clip MUST be between {min_duration} and {max_duration} seconds long ({min_duration}s–{max_duration}s).
- Each clip must be self-contained — a viewer with no prior context should understand it.
- {pacing_guidance}
- Avoid greetings, filler, repetitive sections, clips that require excessive context, \
  and overlapping clips.
- Return ONLY valid JSON. No markdown fences, no commentary outside the JSON.
"""


def _build_user_prompt(
    timestamped_transcript: str,
    min_duration: int,
    max_duration: int,
    max_clips: int,
) -> str:
    """Construct user prompt specifying desired duration range."""
    return f"""\
Below is the full timestamped transcript of a video.

Identify up to {max_clips} of the highest-potential segments between {min_duration} and {max_duration} seconds in duration.

For each clip return a JSON object with exactly these fields:
- "start": clip start time in seconds (float)
- "end": clip end time in seconds (float)
- "score": editorial quality score from 0 to 100 (integer)
- "title": a short, compelling title
- "hook": the opening hook line that would grab a viewer
- "reason": one sentence explaining why this segment works
- "categories": list of content category strings (e.g. "business", "comedy")

Return a JSON array ordered by score descending.

TRANSCRIPT:
{timestamped_transcript}
"""


# ── Gemini call ───────────────────────────────────────────────────────

def analyze_transcript(
    segments: list[dict],
    min_duration: int = DEFAULT_MIN_DURATION,
    max_duration: int = DEFAULT_MAX_DURATION,
    max_clips: int = MAX_CLIPS,
    api_key: str | None = None,
) -> list[dict]:
    """Send transcript segments to Gemini and return validated clip dicts.

    Parameters:
        segments: Timestamped transcript segments.
        min_duration: Minimum clip duration in seconds (default: DEFAULT_MIN_DURATION).
        max_duration: Maximum clip duration in seconds (default: DEFAULT_MAX_DURATION).
        max_clips: Maximum number of clips to return.
        api_key: Optional Gemini API key. If omitted, read from GEMINI_API_KEY environment variable.

    Returns a list of clip dicts ready for JSON serialisation.
    """
    try:
        from core.transcript import format_transcript
    except ImportError:
        from transcript import format_transcript

    resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not resolved_api_key or resolved_api_key == "your-api-key-here":
        raise ConfigurationError(
            "GEMINI_API_KEY is not set. Add it to your .env file or pass it directly."
        )

    timestamped_text = format_transcript(segments)
    system_prompt = _build_system_prompt(min_duration, max_duration)
    user_prompt = _build_user_prompt(timestamped_text, min_duration, max_duration, max_clips)

    client = genai.Client(api_key=resolved_api_key)

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.4,
                automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
            ),
        )
    except Exception as exc:
        raise AnalysisError(f"Gemini API call failed: {exc}") from exc

    text = response.text
    if not text:
        raise AnalysisError("Gemini returned an empty response.")

    clips = _parse_response(text)

    if not clips:
        raise AnalysisError("Gemini returned no valid clips.")

    # Sort by score, remove overlaps, cap at max_clips.
    clips.sort(key=lambda c: c.score, reverse=True)
    clips = _remove_overlapping(clips)
    clips = clips[:max_clips]

    return [c.model_dump() for c in clips]


# ── Response parsing ──────────────────────────────────────────────────

def _parse_response(raw: str) -> list[Clip]:
    """Parse Gemini's JSON into validated Clip objects.

    Skips individual bad clips instead of failing the whole batch.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise AnalysisError(
            f"Could not parse Gemini response as JSON: {exc}\n"
            f"Response (first 500 chars):\n{raw[:500]}"
        ) from exc

    if not isinstance(data, list):
        raise AnalysisError(f"Expected JSON array from Gemini, got {type(data).__name__}.")

    clips: list[Clip] = []
    for i, item in enumerate(data):
        try:
            clips.append(Clip.model_validate(item))
        except Exception as exc:
            logger.warning("Skipping clip %d: %s", i, exc)

    return clips


def _remove_overlapping(clips: list[Clip], threshold: float = 0.5) -> list[Clip]:
    """Remove lower-scored clips that overlap >threshold with a kept clip."""
    kept: list[Clip] = []
    for clip in clips:
        overlaps = False
        for existing in kept:
            overlap_start = max(clip.start, existing.start)
            overlap_end = min(clip.end, existing.end)
            overlap = max(0.0, overlap_end - overlap_start)
            shorter = min(clip.duration, existing.duration)
            if shorter > 0 and (overlap / shorter) > threshold:
                overlaps = True
                break
        if not overlaps:
            kept.append(clip)
    return kept
