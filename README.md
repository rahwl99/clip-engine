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
from core.pipeline import analyze_video, process_video

# Step 1: Editorial analysis & preview generation
clip_result = analyze_video("https://www.youtube.com/watch?v=videoID")

# Step 2: Production 9:16 vertical processing
summary = process_video("videoID", vertical=True, face_tracking=True, subtitles=True)

# Or use the legacy process_clips helper:
from process import process_clips
process_clips("videoID", vertical=False, subtitles=True)
```

> **Note:** Step 2 requires that Step 1 has already been run for the same video. The `<video_id>` is the 11-character YouTube video identifier (e.g. `videoID`).

---

## Project Structure

```
clip-engine/
├── core/                              # Core pipeline & editorial modules
│   ├── __init__.py                    # Lazy package exports (analyze_video, process_video)
│   ├── analyzer.py                    # Gemini prompt engineering, API call, & deduplication
│   ├── clipper.py                     # FFmpeg clip extraction & cut generation
│   ├── config.py                      # Centralized configuration & Path helpers
│   ├── downloader.py                  # Multi-profile yt-dlp source video downloader (360p / 1080p)
│   ├── exceptions.py                  # Domain exception hierarchy (CliptError, FFmpegError, etc.)
│   ├── ffmpeg.py                      # Centralized FFmpeg/ffprobe execution & probe helpers
│   ├── models.py                      # Pydantic data models (Clip, ClipResult)
│   ├── pipeline.py                    # Job-isolated engine pipelines (analyze_video, process_video)
│   └── transcript.py                  # YouTube subtitle retrieval, json3 parsing, & word offsets
│
├── processing/                        # Production vertical video repurposing modules
│   ├── __init__.py                    # Re-exports (generate_vertical_clip, process_video, etc.)
│   ├── face_tracker.py                # OpenCV YuNet neural face detection & camera trajectory smoothing
│   ├── processor.py                   # FFmpeg vertical 9:16 dynamic cropping & subtitle filter builder
│   └── subtitles.py                   # ASS subtitle engine with active spoken-word highlighting & styles
│
├── models/
│   └── face_detection_yunet_2023mar.onnx  # Pretrained YuNet ONNX face detection neural network
│
├── main.py                            # CLI entry point — Phase 1 (Editorial & horizontal previews)
├── process.py                         # CLI entry point — Phase 2 (Production vertical 9:16 clips)
├── test_suite.py                      # Comprehensive automated test suite (15 unit/pipeline tests)
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

---

## Docker Usage

Clipt can be run inside a Linux Docker container with all required system tools (FFmpeg, OpenCV YuNet dependencies, font rendering libraries) preconfigured.

### Build

```bash
docker build -t clipt .
```

### Check Python & Dependencies

```bash
docker run --rm clipt python --version
docker run --rm clipt ffmpeg -version
docker run --rm clipt ffprobe -version
```

You can also run the automated test suite inside Docker:
```bash
docker run --rm clipt python test_suite.py
```

### Run Analysis (Step 1)

Execute the analysis and preview clip generation inside Docker:

```bash
# Linux / macOS
docker run --rm --env-file .env -v "$(pwd)/output:/app/output" clipt python main.py "<YOUTUBE_URL>"

# Windows PowerShell
docker run --rm --env-file .env -v "${PWD}/output:/app/output" clipt python main.py "<YOUTUBE_URL>"
```

### Run Processing (Step 2)

Execute high-quality vertical 9:16 processing, face tracking, and subtitle burn-in:

```bash
# Linux / macOS
docker run --rm --env-file .env -v "$(pwd)/output:/app/output" clipt python process.py "<VIDEO_ID>"

# Windows PowerShell
docker run --rm --env-file .env -v "${PWD}/output:/app/output" clipt python process.py "<VIDEO_ID>"
```

All CLI flags remain supported:
```bash
docker run --rm --env-file .env -v "${PWD}/output:/app/output" clipt python process.py "<VIDEO_ID>" --force --subtitle-style highlight_cyan
```

### Environment Variables

The `.env` file is **not** included inside the Docker image to protect secrets. API keys (such as `GEMINI_API_KEY`) are passed at runtime via `--env-file .env` or `-e GEMINI_API_KEY="your-key"`.

### Output Mounting

Clipt saves all media and manifests under `output/<video_id>/`. Mounting the host `./output` directory to container `/app/output`:

```bash
-v "${PWD}/output:/app/output"
```

ensures that downloaded sources, manifests (`clips.json`), horizontal previews, and final vertical clips (`processedFiles/`) persist on your host machine after the container exits.

---

## FastAPI Server

Clipt provides a lightweight FastAPI application layer wrapping the existing engine for asynchronous, background job-based video repurposing.

### Running the API Server

#### Locally:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Inside Docker:
```bash
# Windows PowerShell
docker run --rm -p 8000:8000 --env-file .env -v "${PWD}/output:/app/output" clipt uvicorn api.main:app --host 0.0.0.0 --port 8000

# Linux / macOS
docker run --rm -p 8000:8000 --env-file .env -v "$(pwd)/output:/app/output" clipt uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Interactive API documentation (Swagger UI) is available at:
`http://localhost:8000/docs`

### API Endpoints

#### 1. Health Check
```http
GET /health
```
Response:
```json
{
  "status": "ok"
}
```

#### 2. Start Video Analysis Job
```http
POST /jobs/analyze
Content-Type: application/json

{
  "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```
Response (`202 Accepted`):
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "processing"
}
```

#### 3. Poll Job Status & Results
```http
GET /jobs/{job_id}
```
- In-progress response:
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "processing",
  "stage": "analyzing_transcript",
  "progress": 55,
  "video_id": "VIDEO_ID"
}
```
- Completed response:
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "completed",
  "stage": "completed",
  "progress": 100,
  "video_id": "VIDEO_ID",
  "result": {
    "video_id": "VIDEO_ID",
    "source_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "generated_at": "2026-09-17T12:00:00",
    "clips": [
      {
        "clip_id": "clip_01",
        "title": "Clip Title",
        "hook": "Hook text",
        "reason": "Why this clip works",
        "score": 95,
        "start": 10.0,
        "end": 35.0,
        "duration": 25.0,
        "categories": ["productivity"],
        "filename": "clip_01.mp4"
      }
    ]
  }
}
```

#### 4. Process Selected Clips (9:16 Vertical Export)
```http
POST /jobs/{job_id}/process
Content-Type: application/json

{
  "clip_ids": ["clip_01"],
  "vertical": true,
  "face_tracking": true,
  "subtitles": true,
  "subtitle_style": "highlight_yellow",
  "force": false
}
```
Response (`202 Accepted`):
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "processing"
}
```

