"""FastAPI route handlers for Clipt API."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, HTTPException, status

from api.jobs import job_store
from api.models import (
    AnalyzeRequest,
    HealthResponse,
    JobResponse,
    JobStatusResponse,
    ProcessRequest,
)
from core.exceptions import CliptError, VideoIDError
from core.pipeline import analyze_video, process_video
from core.transcript import extract_video_id

logger = logging.getLogger(__name__)

router = APIRouter()

# ThreadPoolExecutor for local background task execution
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="clipt_job_")


# ── Background Task Workers ──────────────────────────────────────────

def _run_analyze(job_id: str, youtube_url: str) -> None:
    """Background worker executing the analyze_video pipeline."""
    job_store.update_job(job_id, status="processing", stage="starting", progress=5)

    def on_progress(msg: str) -> None:
        stage = "analyzing"
        progress = 20
        if "[1/5]" in msg:
            stage = "extracting_video_info"
            progress = 10
        elif "[2/5]" in msg:
            stage = "fetching_transcript"
            progress = 30
        elif "[3/5]" in msg:
            stage = "analyzing_transcript"
            progress = 55
        elif "[4/5]" in msg:
            stage = "downloading_preview_source"
            progress = 75
        elif "[5/5]" in msg:
            stage = "generating_preview_clips"
            progress = 90
        job_store.update_job(job_id, stage=stage, progress=progress)

    try:
        result = analyze_video(youtube_url, progress_callback=on_progress)
        res_dict = result.model_dump()
        job_store.update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            video_id=result.video_id,
            result=res_dict,
        )
    except CliptError as exc:
        logger.error("Analysis job %s failed with CliptError: %s", job_id, exc)
        job_store.update_job(
            job_id,
            status="failed",
            stage="failed",
            error=str(exc),
        )
    except Exception as exc:
        logger.exception("Analysis job %s encountered unexpected error: %s", job_id, exc)
        job_store.update_job(
            job_id,
            status="failed",
            stage="failed",
            error=f"Unexpected error during analysis: {exc}",
        )


def _run_process(
    job_id: str,
    video_id: str,
    clip_ids: list[str] | None,
    vertical: bool,
    face_tracking: bool,
    subtitles: bool,
    subtitle_style: str,
    force: bool,
) -> None:
    """Background worker executing the process_video pipeline."""
    job_store.update_job(job_id, status="processing", stage="starting_processing", progress=5)

    def on_progress(msg: str) -> None:
        stage = "processing"
        progress = 20
        if "[1/3]" in msg:
            stage = "loading_clips_manifest"
            progress = 15
        elif "[2/3]" in msg:
            stage = "downloading_high_res_source"
            progress = 35
        elif "[3/3]" in msg:
            stage = "rendering_vertical_clips"
            progress = 65
        job_store.update_job(job_id, stage=stage, progress=progress)

    try:
        proc_result = process_video(
            video_id,
            clip_ids=clip_ids if clip_ids else None,
            vertical=vertical,
            face_tracking=face_tracking,
            subtitles=subtitles,
            subtitle_style=subtitle_style,
            force=force,
            progress_callback=on_progress,
        )
        if "output_dir" in proc_result:
            proc_result["output_dir"] = str(proc_result["output_dir"])

        job = job_store.get_job(job_id)
        current_res: dict[str, Any] = (
            job.result.copy() if (job and isinstance(job.result, dict)) else {}
        )
        current_res["processing"] = proc_result

        job_store.update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            result=current_res,
        )
    except CliptError as exc:
        logger.error("Processing job %s failed with CliptError: %s", job_id, exc)
        job_store.update_job(
            job_id,
            status="failed",
            stage="failed",
            error=str(exc),
        )
    except Exception as exc:
        logger.exception("Processing job %s encountered unexpected error: %s", job_id, exc)
        job_store.update_job(
            job_id,
            status="failed",
            stage="failed",
            error=f"Unexpected error during clip processing: {exc}",
        )


# ── HTTP Endpoints ───────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
def health_check() -> HealthResponse:
    """Lightweight health check returning 'ok'. Does not invoke the Clipt engine."""
    return HealthResponse(status="ok")


@router.post(
    "/jobs/analyze",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Analyze video and generate preview clips",
    tags=["Jobs"],
)
def analyze_job(request: AnalyzeRequest) -> JobResponse:
    """Validate YouTube URL, register an analysis job, and run analyze_video asynchronously."""
    try:
        video_id = extract_video_id(request.youtube_url)
    except VideoIDError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid YouTube URL: {exc}",
        ) from exc

    job = job_store.create_job(job_type="analyze", video_id=video_id)
    job_store.update_job(job.job_id, status="processing", stage="queued", progress=0)

    _executor.submit(_run_analyze, job.job_id, request.youtube_url)

    return JobResponse(job_id=job.job_id, status="processing")


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    response_model_exclude_none=True,
    summary="Get job status and results",
    tags=["Jobs"],
)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Retrieve the current state, progress, results, or error for a job."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found",
        )

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        video_id=job.video_id,
        result=job.result,
        error=job.error,
    )


@router.post(
    "/jobs/{job_id}/process",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Process selected clips into production 9:16 vertical videos",
    tags=["Jobs"],
)
def process_job_clips(job_id: str, request: ProcessRequest) -> JobResponse:
    """Verify analysis has completed, validate requested clip IDs, and start vertical 9:16 rendering."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found",
        )

    if job.status != "completed" or not job.result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot process clips: Job '{job_id}' is in '{job.status}' status (must be 'completed').",
        )

    if not job.video_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job '{job_id}' does not have an associated video ID.",
        )

    # Validate requested clip IDs against analysis results
    if request.clip_ids:
        raw_clips = job.result.get("clips", [])
        available_ids: set[str] = set()
        for c in raw_clips:
            if isinstance(c, dict):
                if c.get("clip_id"):
                    available_ids.add(c["clip_id"])
                if c.get("filename"):
                    available_ids.add(c["filename"].removesuffix(".mp4"))

        missing = [cid for cid in request.clip_ids if cid.removesuffix(".mp4") not in available_ids]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Requested clip ID(s) not found in analysis manifest: {', '.join(missing)}",
            )

    # Transition job back to processing state
    job_store.update_job(
        job_id,
        status="processing",
        stage="queued_for_processing",
        progress=0,
        error=None,
    )

    _executor.submit(
        _run_process,
        job_id,
        job.video_id,
        request.clip_ids,
        request.vertical,
        request.face_tracking,
        request.subtitles,
        request.subtitle_style,
        request.force,
    )

    return JobResponse(job_id=job.job_id, status="processing")
