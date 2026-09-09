"""Analyzer — send transcript to Gemini, parse and validate clip candidates."""

import json
import logging
import os

from google import genai

from models import Clip

logger = logging.getLogger(__name__)

MAX_CLIPS: int = 10


# ── Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a professional short-form video editor and content strategist.

Your job is to analyse a timestamped video transcript and identify the strongest \
self-contained segments that would perform well as YouTube Shorts, Instagram Reels, \
or TikTok clips.

Rules:
- Only use timestamps that appear in the provided transcript. Never invent timestamps.
- Each clip should be 30–90 seconds long.
- Each clip must be self-contained — a viewer with no prior context should understand it.
- Prioritise strong hooks, curiosity, surprising information, emotional moments, \
  useful insights, controversial opinions, stories, clear payoffs, and standalone context.
- Avoid greetings, filler, repetitive sections, clips that require excessive context, \
  and overlapping clips.
- Return ONLY valid JSON. No markdown fences, no commentary outside the JSON.
"""


def _build_user_prompt(timestamped_transcript: str) -> str:
    return f"""\
Below is the full timestamped transcript of a video.

Identify up to {MAX_CLIPS} of the highest-potential short-form clips.

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

def analyze_transcript(segments: list[dict]) -> list[dict]:
    """Send transcript segments to Gemini and return validated clip dicts.

    Reads GEMINI_API_KEY from environment. Returns a list of clip dicts
    ready for JSON serialisation.
    """
    from transcript import format_transcript

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file."
        )

    timestamped_text = format_transcript(segments)
    user_prompt = _build_user_prompt(timestamped_text)

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
                automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    text = response.text
    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    clips = _parse_response(text)

    if not clips:
        raise RuntimeError("Gemini returned no valid clips.")

    # Sort by score, remove overlaps, cap at MAX_CLIPS.
    clips.sort(key=lambda c: c.score, reverse=True)
    clips = _remove_overlapping(clips)
    clips = clips[:MAX_CLIPS]

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
        raise RuntimeError(
            f"Could not parse Gemini response as JSON: {exc}\n"
            f"Response (first 500 chars):\n{raw[:500]}"
        ) from exc

    if not isinstance(data, list):
        raise RuntimeError(f"Expected JSON array from Gemini, got {type(data).__name__}.")

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
