"""Unit and integration tests for Clipt FastAPI layer."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.jobs import job_store
from api.main import app
from core.exceptions import AnalysisError, ProcessingError
from core.models import Clip, ClipResult


class TestCliptAPI(unittest.TestCase):
    """Test suite for FastAPI endpoints."""

    def setUp(self) -> None:
        job_store.clear()
        self.client = TestClient(app)

    # ── Health Endpoint ──────────────────────────────────────────────

    def test_health_check(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    # ── Analyze Endpoint ─────────────────────────────────────────────

    def test_analyze_invalid_url(self) -> None:
        response = self.client.post("/jobs/analyze", json={"youtube_url": "invalid_url_string"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Not a valid YouTube URL", response.json()["detail"])

    def test_analyze_valid_url_dispatches_job(self) -> None:
        with patch("api.routes._executor.submit") as mock_submit:
            response = self.client.post(
                "/jobs/analyze",
                json={"youtube_url": "https://www.youtube.com/watch?v=3qHkcs3kG44"},
            )
            self.assertEqual(response.status_code, 202)
            data = response.json()
            self.assertIn("job_id", data)
            self.assertEqual(data["status"], "processing")
            mock_submit.assert_called_once()

            # Verify registered in job_store
            job = job_store.get_job(data["job_id"])
            self.assertIsNotNone(job)
            self.assertEqual(job.video_id, "3qHkcs3kG44")

    # ── Job Status Endpoint ──────────────────────────────────────────

    def test_get_nonexistent_job_returns_404(self) -> None:
        response = self.client.get("/jobs/nonexistent123")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())

    def test_get_job_status_in_progress(self) -> None:
        job = job_store.create_job(job_type="analyze", video_id="test_vid", job_id="job_prog")
        job_store.update_job("job_prog", status="processing", stage="analyzing_gemini", progress=50)

        response = self.client.get("/jobs/job_prog")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["job_id"], "job_prog")
        self.assertEqual(data["status"], "processing")
        self.assertEqual(data["stage"], "analyzing_gemini")
        self.assertEqual(data["progress"], 50)
        self.assertEqual(data["video_id"], "test_vid")

    def test_get_job_status_completed(self) -> None:
        dummy_result = {
            "video_id": "test_vid",
            "source_url": "https://www.youtube.com/watch?v=test_vid",
            "generated_at": "2026-09-17T12:00:00",
            "clips": [
                {
                    "start": 10.0,
                    "end": 30.0,
                    "duration": 20.0,
                    "score": 90,
                    "title": "Clip Title",
                    "hook": "Hook text",
                    "reason": "Reason text",
                    "categories": ["category1"],
                    "filename": "clip_01.mp4",
                    "clip_id": "clip_01",
                }
            ],
        }
        job = job_store.create_job(job_type="analyze", video_id="test_vid", job_id="job_done")
        job_store.update_job("job_done", status="completed", stage="completed", progress=100, result=dummy_result)

        response = self.client.get("/jobs/job_done")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["result"]["clips"][0]["clip_id"], "clip_01")

    def test_get_job_status_failed(self) -> None:
        job = job_store.create_job(job_type="analyze", video_id="test_vid", job_id="job_err")
        job_store.update_job("job_err", status="failed", stage="failed", error="Transcription failed")

        response = self.client.get("/jobs/job_err")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "failed")
        self.assertEqual(data["error"], "Transcription failed")

    # ── Process Endpoint ─────────────────────────────────────────────

    def test_process_nonexistent_job_returns_404(self) -> None:
        response = self.client.post("/jobs/nonexistent123/process", json={"clip_ids": ["clip_01"]})
        self.assertEqual(response.status_code, 404)

    def test_process_uncompleted_job_returns_400(self) -> None:
        job_store.create_job(job_type="analyze", video_id="test_vid", job_id="job_pending")
        job_store.update_job("job_pending", status="processing")

        response = self.client.post("/jobs/job_pending/process", json={"clip_ids": ["clip_01"]})
        self.assertEqual(response.status_code, 400)
        self.assertIn("must be 'completed'", response.json()["detail"])

    def test_process_invalid_clip_id_returns_400(self) -> None:
        dummy_result = {
            "video_id": "test_vid",
            "clips": [
                {"clip_id": "clip_01", "filename": "clip_01.mp4"},
                {"clip_id": "clip_02", "filename": "clip_02.mp4"},
            ],
        }
        job_store.create_job(job_type="analyze", video_id="test_vid", job_id="job_done")
        job_store.update_job("job_done", status="completed", result=dummy_result)

        response = self.client.post("/jobs/job_done/process", json={"clip_ids": ["clip_99"]})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Requested clip ID(s) not found", response.json()["detail"])

    def test_process_invalid_subtitle_style_returns_422(self) -> None:
        response = self.client.post(
            "/jobs/job_done/process",
            json={"clip_ids": ["clip_01"], "subtitle_style": "unknown_style"},
        )
        self.assertEqual(response.status_code, 422)

    def test_process_valid_clips_dispatches_job(self) -> None:
        dummy_result = {
            "video_id": "test_vid",
            "clips": [
                {"clip_id": "clip_01", "filename": "clip_01.mp4"},
                {"clip_id": "clip_02", "filename": "clip_02.mp4"},
            ],
        }
        job_store.create_job(job_type="analyze", video_id="test_vid", job_id="job_done")
        job_store.update_job("job_done", status="completed", result=dummy_result)

        with patch("api.routes._executor.submit") as mock_submit:
            response = self.client.post(
                "/jobs/job_done/process",
                json={
                    "clip_ids": ["clip_01"],
                    "vertical": True,
                    "face_tracking": True,
                    "subtitles": True,
                    "subtitle_style": "highlight_yellow",
                },
            )
            self.assertEqual(response.status_code, 202)
            data = response.json()
            self.assertEqual(data["job_id"], "job_done")
            self.assertEqual(data["status"], "processing")
            mock_submit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
