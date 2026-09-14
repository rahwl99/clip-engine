# Clipt

**Clipt** is a CLI tool that automatically identifies high-potential short-form clip candidates from long-form YouTube videos and generates them as production-ready MP4 files — including vertical 9:16 versions for YouTube Shorts, Instagram Reels, and TikTok with dynamic face tracking and animated word-highlighted captions.

---

## How It Works

```
YouTube URL → Transcript (yt-dlp) → AI Analysis (Gemini) → Ranked Clips → Download Video → FFmpeg → MP4 Clips + JSON
                                                                                               ↓
Vertical 9:16 Processing: Full HD Download → OpenCV YuNet Face Tracking → Camera Smoothing → ASS Subtitles → Final Shorts
```

1. **Transcript extraction** — Uses `yt-dlp` to extract English subtitles (`json3` format) and parses them into timestamped segments and word-level token offsets.
2. **AI analysis** — Sends the timestamped transcript to **Gemini** acting as a viral video strategist. Gemini returns high-retention clip candidates scored 0–100.
3. **Validation & Deduplication** — Candidates are validated with **Pydantic**, and overlapping clip segments are intelligently pruned.
4. **Video download** — Downloads the source video via `yt-dlp` (lightweight 360p proxy for initial fast extraction, Full HD 1080p for production vertical export).
5. **Clip generation** — Uses **FFmpeg** (H.264 + AAC re-encode) to cut each horizontal clip as a standalone preview MP4.
6. **Vertical 9:16 processing** — Tracks speaker faces using **OpenCV YuNet**, computes an inertia-smoothed camera trajectory, generates punchy dialogue-sliced ASS subtitles with real-time active spoken-word highlighting, and renders final vertical videos to `output/<video_id>/processedFiles/`.

---

## Two-Step Workflow

### Step 1 — Analyze and identify clips (Editorial Phase)

```bash
python main.py <youtube-url>
```

Downloads a lightweight (360p) proxy video, identifies high-potential clips via Gemini, and generates preview-quality MP4 clips alongside a `clips.json` manifest.

#### Customizing Clip / Video Length
At the top of [main.py], you can change the target video duration:
```python
# ── Video Length Configuration (Edit Here) ────────────────────────────
MIN_DURATION: int = 30   # Minimum clip duration in seconds
MAX_DURATION: int = 90   # Maximum clip duration in seconds
```
* **Short clips (Shorts/Reels/TikTok)**: `MIN_DURATION = 30`, `MAX_DURATION = 90`
* **Medium segments (highlights/topics)**: `MIN_DURATION = 90`, `MAX_DURATION = 300` (1.5 to 5 mins)
* **Longer deep-dives / chapters**: `MIN_DURATION = 300`, `MAX_DURATION = 900` (5 to 15 mins)

### Step 2 — Generate vertical versions (Production Phase)

```bash
python process.py <video_id>
```

Downloads the source video in Full HD (1080p), reads the existing `clips.json`, applies computer-vision face tracking, burns in animated captions, and generates 9:16 (1080×1920) vertical versions into `processedFiles/`.

Optional flags:
```bash
# Overwrite previously processed clips
python process.py <video_id> --force

# Disable 9:16 vertical crop (keep original aspect ratio / resolution)
python process.py <video_id> --no-vertical
# Aliases: --no-crop, --no-9-16

# Disable face tracking (use static centered crop when 9:16 is enabled)
python process.py <video_id> --no-face-tracking
# Alias: --no-face-track

# Disable burned-in subtitles
python process.py <video_id> --no-subtitles
# Alias: --no-subs

# Combine toggles (e.g. original aspect ratio with subtitles, without face tracking)
python process.py <video_id> --no-vertical --no-face-tracking

# Choose a specific subtitle highlight style
python process.py <video_id> --subtitle-style highlight_cyan
```

### Configuring Feature Toggles via Code (Bools)

You can configure defaults directly in [process.py]:

```python
# ── Feature Toggles (Bools in Code) ──────────────────────────────────
ENABLE_VERTICAL: bool = True        # Convert to 9:16 vertical video
ENABLE_FACE_TRACKING: bool = True   # Local OpenCV YuNet face tracking
ENABLE_SUBTITLES: bool = True      # Burn-in animated ASS subtitles
```

Or invoke programmatically in Python:

```python
from process import process_clips

# Process in original aspect ratio with subtitles
process_clips("3qHkcs3kG44", vertical=False, subtitles=True)

# Process vertical 9:16 with fast centered crop (skip face tracking)
process_clips("3qHkcs3kG44", vertical=True, face_tracking=False)
```

> **Note:** Step 2 requires that Step 1 has already been run for the same video. The `<video_id>` is the 11-character YouTube video identifier (e.g. `3qHkcs3kG44`).

---

## Project Structure

```
clip-engine/
├── core/                              # Core pipeline & editorial modules
│   ├── __init__.py
│   ├── analyzer.py                    # Gemini prompt engineering, API call, & deduplication
│   ├── clipper.py                     # FFmpeg clip extraction & system PATH validator
│   ├── downloader.py                  # Multi-profile yt-dlp source video downloader (360p / 1080p)
│   ├── models.py                      # Pydantic data models (Clip, ClipResult)
│   └── transcript.py                  # YouTube subtitle retrieval, json3 parsing, & word offsets
│
├── processing/                        # Production vertical video repurposing modules
│   ├── __init__.py
│   ├── face_tracker.py                # OpenCV YuNet neural face detection & camera trajectory smoothing
│   ├── processor.py                   # FFmpeg vertical 9:16 dynamic cropping & subtitle filter builder
│   └── subtitles.py                   # ASS subtitle engine with active spoken-word highlighting & styles
│
├── models/
│   └── face_detection_yunet_2023mar.onnx  # Pretrained YuNet ONNX face detection neural network
│
├── main.py                            # CLI entry point — Phase 1 (Editorial & horizontal previews)
├── process.py                         # CLI entry point — Phase 2 (Production vertical 9:16 clips)
├── test_suite.py                      # Comprehensive automated test suite
├── requirements.txt                   # Direct Python dependencies
├── .env                               # Local secrets (GEMINI_API_KEY)
├── .gitignore                         # Git ignore configuration
└── output/                            # Generated output per video ID
```

---

## Output Structure

```
output/<video_id>/
├── source.mp4                 # Cached high-quality source video (from Step 2)
├── transcript.json            # Cached raw transcript segments
├── clips.json                 # Pydantic-validated editorial manifest
├── clip_01.mp4                # Preview clips (360p, horizontal)
├── clip_02.mp4
├── ...
└── processedFiles/
    ├── clip_01.mp4            # Vertical clips (1080×1920, 9:16 with face tracking & subtitles)
    ├── clip_02.mp4
    └── ...
```

---

## Subtitle Styling Presets

Clipt generates modern, high-energy ASS captions with active word highlighting and scale pop. The following presets can be passed via `--subtitle-style`:

| Style Name | Description | Active Color |
| :--- | :--- | :--- |
| `highlight_yellow` *(default)* | Bold white text with electric yellow highlight | `#00FFFF` (Yellow) |
| `highlight_cyan` | High-contrast neon cyan highlight | `#FFFF00` (Cyan) |
| `highlight_green` | Bright neon lime highlight | `#39FF14` (Green) |
| `highlight_magenta` | Electric hot pink / magenta highlight | `#FF00FF` (Magenta) |
| `classic` | Clean solid white subtitles with dark outline | White (no highlight) |

---

## Tech Stack

- **Python 3.12+**
- **yt-dlp** — transcript retrieval & multi-profile video download
- **Google Gemini (`google-genai`)** — AI viral clip candidate detection
- **Pydantic** — schema definition & output validation
- **OpenCV (`opencv-python`)** — YuNet deep neural network face detection & subject tracking
- **python-dotenv** — environment variable management
- **FFmpeg** (system dependency) — video segmentation, dynamic camera crop expressions, & ASS subtitle rendering

---

## Prerequisites

- Python 3.12+
- FFmpeg installed and available in system `PATH`
- `GEMINI_API_KEY` set in `.env`

---

## Setup & Testing

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and set your GEMINI_API_KEY
   ```

3. **Run automated test suite**:
   ```bash
   python test_suite.py
   ```
