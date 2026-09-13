"""Subtitles — transcript loading, clip-relative timing, and ASS subtitle generation."""

import json
import logging
import re
from pathlib import Path

from core.transcript import get_transcript

logger = logging.getLogger(__name__)

# ── Preset Subtitle Styles ─────────────────────────────────────────────
# ASS Color format is &HAABBGGRR where AA is alpha (00=opaque), BB=blue, GG=green, RR=red.
SUBTITLE_STYLES: dict[str, dict] = {
    "highlight_yellow": {
        "name": "Social Electric Yellow",
        "description": "Bold white text with punchy electric yellow active word highlight (Shorts/Reels default)",
        "highlight": True,
        "highlight_color": "&H0000FFFF",  # ASS BGR: 00 Blue, FF Green, FF Red = Yellow
        "primary_color": "&H00FFFFFF",    # White
        "outline_color": "&H00000000",    # Black
        "outline_width": 5,
        "shadow_width": 2,
        "font_name": "Arial",
        "font_size": 78,
        "margin_v": 360,
        "pop_scale": 108,                 # 108% scale pop on active word
    },
    "highlight_cyan": {
        "name": "Social Neon Cyan",
        "description": "Bold white text with electric cyan active word highlight",
        "highlight": True,
        "highlight_color": "&H00FFFF00",  # ASS BGR: FF Blue, FF Green, 00 Red = Cyan
        "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "outline_width": 5,
        "shadow_width": 2,
        "font_name": "Arial",
        "font_size": 78,
        "margin_v": 360,
        "pop_scale": 108,
    },
    "highlight_green": {
        "name": "Social Lime Green",
        "description": "Bold white text with vibrant neon green active word highlight",
        "highlight": True,
        "highlight_color": "&H0039FF14",  # ASS BGR: Neon Green
        "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "outline_width": 5,
        "shadow_width": 2,
        "font_name": "Arial",
        "font_size": 78,
        "margin_v": 360,
        "pop_scale": 108,
    },
    "highlight_magenta": {
        "name": "Social Magenta / Pink",
        "description": "Bold white text with hot magenta active word highlight",
        "highlight": True,
        "highlight_color": "&H00FF00FF",  # ASS BGR: Magenta
        "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "outline_width": 5,
        "shadow_width": 2,
        "font_name": "Arial",
        "font_size": 78,
        "margin_v": 360,
        "pop_scale": 108,
    },
    "classic": {
        "name": "Classic White",
        "description": "Static white text with heavy black outline (original style without word highlight)",
        "highlight": False,
        "highlight_color": "&H00FFFFFF",
        "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "outline_width": 5,
        "shadow_width": 2,
        "font_name": "Arial",
        "font_size": 78,
        "margin_v": 360,
        "pop_scale": 100,
    },
}

DEFAULT_SUBTITLE_STYLE = "highlight_yellow"

STYLE_ALIASES = {
    "yellow": "highlight_yellow",
    "gold": "highlight_yellow",
    "cyan": "highlight_cyan",
    "blue": "highlight_cyan",
    "green": "highlight_green",
    "lime": "highlight_green",
    "pink": "highlight_magenta",
    "magenta": "highlight_magenta",
    "white": "classic",
    "old": "classic",
    "default": DEFAULT_SUBTITLE_STYLE,
}


def get_subtitle_style(style_spec: str | dict | None = None) -> dict:
    """Resolve a style name, alias, or custom dictionary into a full style config."""
    if isinstance(style_spec, dict):
        base = dict(SUBTITLE_STYLES[DEFAULT_SUBTITLE_STYLE])
        base.update(style_spec)
        return base

    if not style_spec:
        key = DEFAULT_SUBTITLE_STYLE
    else:
        key = style_spec.lower().strip()
        key = STYLE_ALIASES.get(key, key)

    if key in SUBTITLE_STYLES:
        return dict(SUBTITLE_STYLES[key])

    logger.warning("Unknown subtitle style '%s', falling back to '%s'", style_spec, DEFAULT_SUBTITLE_STYLE)
    return dict(SUBTITLE_STYLES[DEFAULT_SUBTITLE_STYLE])


def format_ass_time(seconds: float) -> str:
    """Format seconds into ASS timestamp format: H:MM:SS.cc (centiseconds)."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        secs += 1
        centis -= 100
        if secs >= 60:
            minutes += 1
            secs -= 60
            if minutes >= 60:
                hours += 1
                minutes -= 60
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def load_or_fetch_transcript(video_dir: Path, source_url: str | None) -> list[dict]:
    """Load transcript segments from local transcript.json or fetch via yt-dlp.

    Returns a list of segment dicts: [{"start": float, "end": float, "text": str}, ...].
    If transcript is unavailable, logs a warning and returns an empty list.
    """
    transcript_file = video_dir / "transcript.json"

    # 1. Try local cache
    if transcript_file.is_file():
        try:
            data = json.loads(transcript_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception as exc:
            logger.warning("Failed to parse cached transcript.json: %s", exc)

    # 2. Fetch via yt-dlp if URL is provided
    if not source_url:
        return []

    try:
        _, segments = get_transcript(source_url)
        if segments:
            # Cache for future runs
            try:
                transcript_file.write_text(
                    json.dumps(segments, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.warning("Could not cache transcript.json: %s", exc)
            return segments
    except Exception as exc:
        logger.warning("Failed to fetch transcript: %s", exc)

    return []


def clean_subtitle_text(text: str) -> str:
    """Clean text for ASS subtitle display.

    Removes excessive whitespace and escapes special ASS characters.
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    # ASS uses curly braces for override tags; escape or sanitize
    cleaned = cleaned.replace("{", "(").replace("}", ")")
    return cleaned


def _calculate_word_weights(words: list[str]) -> list[float]:
    """Calculate relative time weights based on word character length.

    Short grammatical words receive base weight, while longer words are allocated
    proportionally more time to match natural human speech cadence.
    """
    weights = []
    for w in words:
        cleaned = re.sub(r"[^\w]", "", w)
        length = len(cleaned)
        weights.append(max(1.0, min(length, 12) * 0.75))
    return weights


def build_clip_subtitles(
    segments: list[dict],
    clip_start: float,
    clip_end: float,
    *,
    max_words_per_screen: int = 4,
) -> list[dict]:
    """Extract transcript words strictly within [clip_start, clip_end] and group into
    sequential chunks of at most *max_words_per_screen* words appearing one after the other.

    Each chunk contains:
      - 'start': chunk start time (seconds, relative to clip)
      - 'end': chunk end time (seconds, relative to clip)
      - 'text': full chunk text string
      - 'words': list of word dicts [{'word': str, 'start': float, 'end': float}, ...]
    """
    if not segments:
        return []

    # 1. Clean segment boundaries: clamp each segment's end to the next segment's start
    clean_segs = []
    for i, s in enumerate(segments):
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        text = clean_subtitle_text(str(s.get("text", "")))
        if not text or end <= start:
            continue

        if i + 1 < len(segments):
            next_start = float(segments[i + 1].get("start", 0.0))
            if next_start > start:
                end = min(end, next_start)

        seg_entry = {"start": start, "end": end, "text": text}
        if "words" in s:
            seg_entry["words"] = s["words"]
        clean_segs.append(seg_entry)

    # 2. Extract words strictly within clip range
    word_timeline: list[tuple[float, float, str]] = []
    for seg in clean_segs:
        s = seg["start"]
        e = seg["end"]
        if e <= clip_start or s >= clip_end:
            continue

        # Check if segment provides explicit per-word timestamps
        seg_words = seg.get("words")
        if seg_words and isinstance(seg_words, list):
            for w_info in seg_words:
                w_text = clean_subtitle_text(str(w_info.get("word", "")))
                w_s = float(w_info.get("start", s))
                w_e = float(w_info.get("end", e))
                if w_text and w_e > clip_start and w_s < clip_end:
                    word_timeline.append((w_s, w_e, w_text))
            continue

        words = seg["text"].split()
        if not words:
            continue

        dur = e - s
        weights = _calculate_word_weights(words)
        total_weight = sum(weights) or 1.0

        curr_t = s
        for i, w in enumerate(words):
            w_dur = dur * (weights[i] / total_weight)
            w_start = curr_t
            w_end = curr_t + w_dur
            curr_t = w_end
            if w_end > clip_start and w_start < clip_end:
                word_timeline.append((w_start, w_end, w))

    if not word_timeline:
        return []

    # 3. Deduplicate consecutive identical words from rolling captions
    deduped: list[tuple[float, float, str]] = []
    for w_s, w_e, w in word_timeline:
        if (
            deduped
            and deduped[-1][2].lower() == w.lower()
            and abs(w_s - deduped[-1][0]) < 1.8
        ):
            continue
        deduped.append((w_s, w_e, w))

    # 4. Group into sequential chunks of at most max_words_per_screen
    raw_chunks: list[dict] = []
    for i in range(0, len(deduped), max_words_per_screen):
        group = deduped[i : i + max_words_per_screen]
        c_text = " ".join(item[2] for item in group).upper()
        c_start = max(0.0, round(group[0][0] - clip_start, 2))
        c_end = min(round(clip_end - clip_start, 2), round(group[-1][1] - clip_start, 2))

        if c_end > c_start + 0.1:
            chunk_dur = c_end - c_start
            chunk_words_raw = [item[2].upper() for item in group]
            chunk_weights = _calculate_word_weights(chunk_words_raw)
            total_cw = sum(chunk_weights) or 1.0

            words_data: list[dict] = []
            curr_w_start = c_start
            for idx, w_raw in enumerate(chunk_words_raw):
                w_dur = chunk_dur * (chunk_weights[idx] / total_cw)
                w_end = c_end if idx == len(chunk_words_raw) - 1 else round(curr_w_start + w_dur, 2)
                words_data.append({
                    "word": w_raw,
                    "start": curr_w_start,
                    "end": max(curr_w_start + 0.05, w_end),
                })
                curr_w_start = w_end

            raw_chunks.append({
                "start": c_start,
                "end": c_end,
                "text": c_text,
                "words": words_data,
            })

    if not raw_chunks:
        return []

    # 5. Guarantee strictly monotonic non-overlapping sequential display
    chunks: list[dict] = []
    for i, c in enumerate(raw_chunks):
        c_start = c["start"]
        c_end = c["end"]
        if i + 1 < len(raw_chunks):
            next_start = raw_chunks[i + 1]["start"]
            if next_start > c_start:
                c_end = min(c_end, next_start)
            else:
                raw_chunks[i + 1]["start"] = round(c_end, 2)

        if c_end > c_start + 0.1:
            words = c.get("words", [])
            if words:
                chunk_dur = c_end - c_start
                chunk_words_raw = [w["word"] for w in words]
                chunk_weights = _calculate_word_weights(chunk_words_raw)
                total_cw = sum(chunk_weights) or 1.0
                adjusted_words = []
                curr_w_start = c_start
                for idx, w_raw in enumerate(chunk_words_raw):
                    w_dur = chunk_dur * (chunk_weights[idx] / total_cw)
                    w_end = c_end if idx == len(chunk_words_raw) - 1 else round(curr_w_start + w_dur, 2)
                    adjusted_words.append({
                        "word": w_raw,
                        "start": curr_w_start,
                        "end": max(curr_w_start + 0.05, w_end),
                    })
                    curr_w_start = w_end
                words = adjusted_words

            chunks.append({
                "start": c_start,
                "end": c_end,
                "text": c["text"],
                "words": words,
            })

    return chunks


def generate_ass_file(
    subtitles: list[dict],
    output_path: Path,
    *,
    style: str | dict | None = None,
    video_width: int = 1080,
    video_height: int = 1920,
    font_name: str | None = None,
    font_size: int | None = None,
    margin_v: int | None = None,
) -> Path:
    """Generate an ASS subtitle file styled for vertical (9:16) video.

    Styles:
    - Multiple presets: highlight_yellow, highlight_cyan, highlight_green, highlight_magenta, classic
    - Active word highlighting via dialogue slicing for dynamic social retention
    - High contrast: Bold text with heavy outline and drop shadow
    - Safe zone: Positioned in lower-middle area (margin_v=360) away from UI icons
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    style_cfg = get_subtitle_style(style)

    f_name = font_name or style_cfg.get("font_name", "Arial")
    f_size = font_size or style_cfg.get("font_size", 78)
    m_v = margin_v if margin_v is not None else style_cfg.get("margin_v", 360)
    primary_col = style_cfg.get("primary_color", "&H00FFFFFF")
    outline_col = style_cfg.get("outline_color", "&H00000000")
    outline_w = style_cfg.get("outline_width", 5)
    shadow_w = style_cfg.get("shadow_width", 2)
    highlight_enabled = style_cfg.get("highlight", True)
    highlight_col = style_cfg.get("highlight_color", "&H0000FFFF")
    pop_scale = style_cfg.get("pop_scale", 108)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ShortsStyle,{f_name},{f_size},{primary_col},&H0000FFFF,{outline_col},&H80000000,-1,0,0,0,100,100,1,0,1,{outline_w},{shadow_w},2,40,40,{m_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for sub in subtitles:
        words = sub.get("words", [])
        if highlight_enabled and words and len(words) > 0:
            # Generate sliced Dialogue events highlighting each spoken word
            for active_idx, active_word_info in enumerate(words):
                w_start = active_word_info["start"]
                w_end = active_word_info["end"]
                if w_end <= w_start:
                    continue

                start_str = format_ass_time(w_start)
                end_str = format_ass_time(w_end)

                formatted_words = []
                for j, w_info in enumerate(words):
                    w_text = w_info["word"]
                    if j == active_idx:
                        if pop_scale and pop_scale > 100:
                            formatted_words.append(
                                f"{{\\c{highlight_col}\\fscx{pop_scale}\\fscy{pop_scale}}}{w_text}{{\\rShortsStyle}}"
                            )
                        else:
                            formatted_words.append(
                                f"{{\\c{highlight_col}}}{w_text}{{\\rShortsStyle}}"
                            )
                    else:
                        formatted_words.append(w_text)

                line_text = " ".join(formatted_words)
                events.append(
                    f"Dialogue: 0,{start_str},{end_str},ShortsStyle,,0,0,0,,{line_text}"
                )
        else:
            # Classic full chunk display without active word highlight
            start_str = format_ass_time(sub["start"])
            end_str = format_ass_time(sub["end"])
            text = sub["text"]
            events.append(
                f"Dialogue: 0,{start_str},{end_str},ShortsStyle,,0,0,0,,{text}"
            )

    content = header + "\n".join(events) + "\n"
    output_path.write_text(content, encoding="utf-8")
    return output_path
