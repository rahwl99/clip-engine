FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies:
# - ffmpeg / ffprobe: video and audio extraction and transcoding
# - libgl1, libglib2.0-0: OpenCV runtime dependencies
# - fonts-liberation, fontconfig: font rendering for FFmpeg ASS subtitles (Arial-compatible)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    fonts-liberation \
    fontconfig \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models (model is under models/)
COPY . .

# Ensure output directory exists for volume mounting
RUN mkdir -p /app/output

EXPOSE 8000

# Default command displays usage; can be easily overridden at runtime
CMD ["python", "main.py", "--help"]
