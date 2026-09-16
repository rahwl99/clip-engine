"""Custom exceptions for the Clipt processing engine.

All engine-specific errors derive from CliptError to allow consumers
(CLI, web APIs, cloud workers) to catch and handle them appropriately.
"""


class CliptError(Exception):
    """Base exception for all Clipt engine errors."""


class ConfigurationError(CliptError):
    """Raised when required configuration or environment settings are missing or invalid."""


class VideoIDError(CliptError, ValueError):
    """Raised when a video ID or YouTube URL cannot be parsed or is invalid."""


class TranscriptError(CliptError, RuntimeError):
    """Raised when transcript extraction, downloading, or parsing fails."""


class AnalysisError(CliptError, RuntimeError):
    """Raised when AI analysis (Gemini) fails or returns an invalid response."""


class DownloadError(CliptError, RuntimeError):
    """Raised when downloading video content via yt-dlp fails."""


class FFmpegError(CliptError, RuntimeError):
    """Raised when FFmpeg or ffprobe execution fails or binary is missing."""


class ProcessingError(CliptError, RuntimeError):
    """Raised when video processing, face tracking, or subtitle generation fails."""
