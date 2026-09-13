"""Transcript retrieval — extract video ID, fetch subtitles via yt-dlp, parse into segments."""

import re

import yt_dlp


# ── URL handling ──────────────────────────────────────────────────────

_YOUTUBE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/watch\?.*v=(?P<id>[a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/shorts/(?P<id>[a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:https?://)?youtu\.be/(?P<id>[a-zA-Z0-9_-]{11})"),
]


def extract_video_id(url: str) -> str:
    """Extract the 11-character video ID from a YouTube URL."""
    url = url.strip()
    for pattern in _YOUTUBE_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group("id")
    raise ValueError(f"Not a valid YouTube URL: {url}")


# ── Transcript fetching ──────────────────────────────────────────────

def get_transcript(url: str) -> tuple[str, list[dict]]:
    """Fetch English subtitles for a YouTube video.

    Returns (video_id, segments) where each segment is:
        {"start": float, "end": float, "text": str}

    Uses yt-dlp's Python API — no subprocess needed.
    """
    video_id = extract_video_id(url)

    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
        "subtitleslangs": ["en"],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as exc:
            raise RuntimeError(f"yt-dlp failed: {exc}") from exc

    if not info:
        raise RuntimeError("yt-dlp returned no video info.")

    segments = _extract_segments(info)

    if not segments:
        raise RuntimeError(f"No usable English transcript found for video {video_id}.")

    return video_id, segments


def _extract_segments(info: dict) -> list[dict]:
    """Find the best English json3 subtitle URL and download + parse it."""
    sub_url = _find_subtitle_url(info)
    if not sub_url:
        return []

    data = _download_json3(sub_url)
    return _parse_json3_events(data)


def _find_subtitle_url(info: dict) -> str | None:
    """Pick the best English json3 subtitle URL from yt-dlp info."""
    for key in ("subtitles", "automatic_captions"):
        subs: dict = info.get(key, {})
        for lang in ("en", "en-US", "en-GB"):
            tracks = subs.get(lang, [])
            for track in tracks:
                if track.get("ext") == "json3" and track.get("url"):
                    return track["url"]
    return None


def _download_json3(url: str) -> dict:
    """Download json3 subtitle data from a URL."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to download subtitle data: {exc}") from exc


def _parse_json3_events(data: dict) -> list[dict]:
    """Parse json3 subtitle events into simple segment dicts."""
    events = data.get("events", [])
    segments: list[dict] = []

    for event in events:
        segs = event.get("segs")
        if not segs:
            continue

        start_ms: int = event.get("tStartMs", 0)
        duration_ms: int = event.get("dDurationMs", 0)

        words = []
        text_parts = []
        for seg in segs:
            cleaned = seg.get("utf8", "").strip()
            cleaned = re.sub(r"\s+", " ", cleaned)
            if cleaned and cleaned != "\n":
                text_parts.append(cleaned)
                offset_ms = seg.get("tOffsetMs")
                if offset_ms is not None:
                    w_start_s = round((start_ms + offset_ms) / 1000.0, 3)
                    w_dur_ms = seg.get("dDurationMs")
                    if w_dur_ms:
                        w_end_s = round((start_ms + offset_ms + w_dur_ms) / 1000.0, 3)
                    else:
                        w_end_s = round((start_ms + duration_ms) / 1000.0, 3)
                    words.append({"word": cleaned, "start": w_start_s, "end": w_end_s})

        text = " ".join(text_parts).strip()
        if not text:
            continue

        seg_entry = {
            "start": round(start_ms / 1000.0, 3),
            "end": round((start_ms + duration_ms) / 1000.0, 3),
            "text": text,
        }
        if words:
            for idx in range(len(words) - 1):
                if words[idx]["end"] > words[idx + 1]["start"]:
                    words[idx]["end"] = words[idx + 1]["start"]
            seg_entry["words"] = words

        segments.append(seg_entry)

    return segments


# ── Formatting ────────────────────────────────────────────────────────

def format_transcript(segments: list[dict]) -> str:
    """Format segments as timestamped lines for the AI prompt.

    Example: [00:12:44.200 → 00:12:47.800] Some spoken text
    """
    lines = []
    for seg in segments:
        s = _fmt_ts(seg["start"])
        e = _fmt_ts(seg["end"])
        lines.append(f"[{s} → {e}] {seg['text']}")
    return "\n".join(lines)


def _fmt_ts(seconds: float) -> str:
    """Seconds → HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
