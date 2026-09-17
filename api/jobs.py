"""In-memory job state store for Clipt background execution."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class JobState:
    """Represents the lifecycle state of a Clipt analysis or processing job."""

    job_id: str
    status: str = "queued"  # "queued", "processing", "completed", "failed"
    job_type: str = "analyze"  # "analyze" or "process"
    video_id: str | None = None
    stage: str | None = None
    progress: int | None = None
    result: Any | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert job state to dictionary representation."""
        data: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
        }
        if self.stage is not None:
            data["stage"] = self.stage
        if self.progress is not None:
            data["progress"] = self.progress
        if self.video_id is not None:
            data["video_id"] = self.video_id
        if self.result is not None:
            data["result"] = self.result
        if self.error is not None:
            data["error"] = self.error
        return data


class JobStore:
    """Thread-safe in-memory registry for managing background job states.

    Designed to be easily replaced with persistent storage (e.g. Redis, DB) in the future.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        job_type: str = "analyze",
        video_id: str | None = None,
        job_id: str | None = None,
    ) -> JobState:
        """Create and register a new job."""
        with self._lock:
            jid = job_id or uuid.uuid4().hex[:12]
            job = JobState(job_id=jid, job_type=job_type, video_id=video_id)
            self._jobs[jid] = job
            return job

    def get_job(self, job_id: str) -> JobState | None:
        """Retrieve a job by its ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: int | None = None,
        video_id: str | None = None,
        result: Any | None = None,
        error: str | None = None,
    ) -> JobState | None:
        """Update job fields thread-safely."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if status is not None:
                job.status = status
            if stage is not None:
                job.stage = stage
            if progress is not None:
                job.progress = progress
            if video_id is not None:
                job.video_id = video_id
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            job.updated_at = _utcnow()
            return job

    def clear(self) -> None:
        """Clear all stored jobs (useful for testing)."""
        with self._lock:
            self._jobs.clear()


# Global singleton instance
job_store = JobStore()
