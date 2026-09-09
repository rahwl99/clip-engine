# Clipt

**Clipt** is a CLI tool that identifies high-potential short-form clip candidates from YouTube videos and generates them as MP4 files.

## How It Works

```
YouTube URL → Transcript (yt-dlp) → AI Analysis (Gemini) → Ranked Clips → Download Video → FFmpeg → MP4 Clips + JSON
```

1. **Transcript extraction** — Uses `yt-dlp` to pull English subtitles (json3 format) and parses them into timestamped segments (`start`, `end`, `text`).
2. **AI analysis** — Sends the full timestamped transcript to **Gemini** with a prompt that acts as a short-form content strategist. Gemini returns up to 10 clip candidates scored 0–100.
3. **Validation** — Clip candidates are validated with **Pydantic**, overlapping clips are removed.
4. **Video download** — Downloads the source video via `yt-dlp` to a temp directory.
5. **Clip generation** — Uses **FFmpeg** (H.264 + AAC re-encode) to cut each clip as a standalone MP4.
6. **Output** — Saves clips and metadata to `output/<video_id>/`.

## Project Structure

```
main.py          — CLI entry point, 5-step pipeline orchestration
transcript.py    — Video ID extraction, yt-dlp subtitle fetching, json3 parsing
analyzer.py      — Gemini prompt, API call, response parsing
models.py        — Pydantic models (Clip, ClipResult)
downloader.py    — yt-dlp source video download
clipper.py       — FFmpeg clip extraction
requirements.txt — yt-dlp, google-genai, pydantic, python-dotenv
.env             — GEMINI_API_KEY (not committed)
output/          — Generated clips per video ID
```

## Output Structure

```
output/<video_id>/
├── clip_01.mp4
├── clip_02.mp4
├── ...
└── clips.json
```

## Clip Output Schema

Each clip contains: `start`, `end`, `duration`, `score`, `title`, `hook`, `reason`, `categories`, `filename`.

The full `clips.json` also includes: `video_id`, `source_url`, `generated_at`.

## Current State

This is **V0.1** — a working prototype that identifies clips and generates MP4 files. No frontend, database, publishing, captions, or 9:16 conversion.

## Tech Stack

- **Python 3.12+**
- **yt-dlp** — transcript retrieval + video download
- **Google Gemini (google-genai)** — AI clip detection
- **Pydantic** — output validation
- **python-dotenv** — env config
- **FFmpeg** (system dependency) — video clip generation

## Prerequisites

- Python 3.12+
- FFmpeg installed and available in system PATH
- `GEMINI_API_KEY` set in `.env`

## Usage

```bash
python main.py <youtube-url>
```
