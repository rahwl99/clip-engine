"""Face tracker — detect faces, deterministically select subjects, and generate smooth 9:16 camera paths."""

from __future__ import annotations

import hashlib
import logging
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

# Default path for YuNet face detector model (checks project root or local models/)
_root_model = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
_local_model = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_MODEL_PATH = _root_model if _root_model.is_file() else _local_model


@dataclass
class FaceBox:
    """Bounding box for a detected face (x, y, width, height)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return float(self.w * self.h)

    def iou(self, other: FaceBox) -> float:
        """Calculate Intersection over Union (IoU) with another box."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection <= 0:
            return 0.0

        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def distance_to(self, other: FaceBox) -> float:
        """Euclidean distance between box centers."""
        return math.hypot(self.center_x - other.center_x, self.center_y - other.center_y)


@dataclass
class CameraKeyframe:
    """A crop window position at a specific time offset within the clip."""

    time: float  # clip-relative seconds (0.0 .. duration)
    crop_x: float  # top-left X coordinate of the 9:16 crop window
    crop_y: float  # top-left Y coordinate of the 9:16 crop window


@dataclass
class TrackingResult:
    """Result of tracking a face across a clip."""

    face_count: int  # Number of distinct faces detected in initial detection
    selected_face_idx: int | None  # 1-based index of selected face, or None
    is_tracked: bool  # True if a face was successfully tracked
    crop_w: int
    crop_h: int
    keyframes: list[CameraKeyframe]
    crop_x_expr: str  # FFmpeg evaluation expression for crop X


class FaceDetector:
    """Headless OpenCV face detector using YuNet (FaceDetectorYN)."""

    def __init__(self, model_path: Path | None = None, input_size: tuple[int, int] = (640, 360)) -> None:
        import cv2

        self._cv2 = cv2
        self.model_path = model_path or DEFAULT_MODEL_PATH

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Face detection model not found at {self.model_path}. "
                "Ensure models/face_detection_yunet_2023mar.onnx is present."
            )

        self.input_size = input_size
        self._detector = cv2.FaceDetectorYN.create(
            model=str(self.model_path),
            config="",
            input_size=self.input_size,
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )

    def detect(self, bgr_frame) -> list[FaceBox]:
        """Detect faces in a BGR image frame.

        Returns list of FaceBox sorted left-to-right.
        """
        h, w = bgr_frame.shape[:2]
        if self.input_size != (w, h):
            self._detector.setInputSize((w, h))
            self.input_size = (w, h)

        _, raw_faces = self._detector.detect(bgr_frame)
        boxes: list[FaceBox] = []

        if raw_faces is not None:
            for f in raw_faces:
                bx, by, bw, bh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                if bw > 10 and bh > 10:
                    boxes.append(FaceBox(bx, by, bw, bh))

        # Sort left-to-right for consistent subject indexing
        boxes.sort(key=lambda b: b.x)
        return boxes


def select_face_deterministically(
    candidates: Sequence[FaceBox],
    video_id: str,
    clip_filename: str,
) -> tuple[int, FaceBox]:
    """Deterministically select one face index (0-based) from candidates using video_id + filename seed.

    Returns (selected_index, selected_box).
    """
    if not candidates:
        raise ValueError("Cannot select from empty candidates")

    if len(candidates) == 1:
        return 0, candidates[0]

    seed_str = f"{video_id}_{clip_filename}"
    digest = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
    seed_int = int(digest[:8], 16)
    rng = random.Random(seed_int)
    idx = rng.randint(0, len(candidates) - 1)
    return idx, candidates[idx]


def smooth_camera_positions(
    raw_positions: list[tuple[float, float]],
    crop_w: int,
    src_w: int,
    alpha: float = 0.25,
    max_speed_px_per_s: float = 180.0,
) -> list[tuple[float, float]]:
    """Apply exponential moving average and maximum speed limiting to crop X positions.

    raw_positions: list of (time, center_x)
    Returns: list of (time, crop_x) clamped to [0, src_w - crop_w]
    """
    if not raw_positions:
        return []

    max_x = max(0, src_w - crop_w)
    half_crop = crop_w / 2.0

    smoothed: list[tuple[float, float]] = []

    # Initial position
    t0, cx0 = raw_positions[0]
    initial_crop_x = max(0.0, min(float(max_x), cx0 - half_crop))
    smoothed.append((t0, initial_crop_x))

    current_x = initial_crop_x

    for i in range(1, len(raw_positions)):
        t_prev, _ = raw_positions[i - 1]
        t_curr, cx_curr = raw_positions[i]
        dt = max(0.001, t_curr - t_prev)

        target_crop_x = max(0.0, min(float(max_x), cx_curr - half_crop))

        # Exponential moving average filter for inertia
        new_x = current_x + alpha * (target_crop_x - current_x)

        # Rate limiter to avoid sudden jerky camera pans
        max_delta = max_speed_px_per_s * dt
        delta = new_x - current_x
        if abs(delta) > max_delta:
            new_x = current_x + math.copysign(max_delta, delta)

        new_x = max(0.0, min(float(max_x), new_x))
        current_x = new_x
        smoothed.append((t_curr, current_x))

    return smoothed


def build_ffmpeg_crop_expression(
    keyframes: list[CameraKeyframe],
    crop_w: int,
    src_w: int,
) -> str:
    """Build an FFmpeg crop 'x' expression that interpolates keyframes over time 't'.

    Returns a safe expression string like:
        min(max(0, if(lt(t, 1.0), 100 + (t-0.0)*(120-100)/1.0, ...)), max_x)
    """
    max_x = max(0, src_w - crop_w)

    if not keyframes:
        default_x = (src_w - crop_w) // 2
        return str(default_x)

    if len(keyframes) == 1:
        return str(int(round(keyframes[0].crop_x)))

    diffs = [abs(k.crop_x - keyframes[0].crop_x) for k in keyframes]
    if max(diffs) < 5.0:
        # Stationary: avoid complex expression
        avg_x = sum(k.crop_x for k in keyframes) / len(keyframes)
        return str(int(round(avg_x)))

    # Subsample keyframes to keep expression compact (~1 keyframe per 0.5s)
    reduced: list[CameraKeyframe] = [keyframes[0]]
    for k in keyframes[1:]:
        if k.time - reduced[-1].time >= 0.5:
            reduced.append(k)
    if reduced[-1].time < keyframes[-1].time:
        reduced.append(keyframes[-1])

    # Build piecewise linear interpolation expression
    expr = f"{round(reduced[-1].crop_x, 1)}"
    for i in range(len(reduced) - 2, -1, -1):
        k0 = reduced[i]
        k1 = reduced[i + 1]
        dt = max(0.01, k1.time - k0.time)
        dx = k1.crop_x - k0.crop_x
        segment = f"({round(k0.crop_x, 1)}+(t-{round(k0.time, 2)})*{round(dx / dt, 2)})"
        expr = f"if(lt(t,{round(k1.time, 2)}),{segment},{expr})"

    return f"min(max(0,{expr}),{max_x})"


def _probe_dimensions(video_path: Path) -> tuple[int, int]:
    """Get video width and height using ffprobe."""
    import json
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {res.stderr.strip()}")
    info = json.loads(res.stdout)
    stream = info["streams"][0]
    return int(stream["width"]), int(stream["height"])


def track_face_for_clip(
    source_video: Path,
    clip_start: float,
    clip_end: float,
    video_id: str,
    clip_filename: str,
    *,
    sample_fps: float = 2.5,
) -> TrackingResult:
    """Sample video frames across [clip_start, clip_end] to detect and track a face.

    Uses high-speed FFmpeg pipe decoding combined with YuNet for fast, accurate tracking.

    Follows rules:
    - 0 faces detected: returns TrackingResult with is_tracked=False (centered crop fallback)
    - 1 face detected: tracks that face throughout clip
    - >=2 faces detected: deterministically selects one face, tracks that identity throughout clip
    """
    import numpy as np

    duration = clip_end - clip_start
    if duration <= 0:
        raise ValueError(f"Invalid clip duration: {duration}")

    src_w, src_h = _probe_dimensions(source_video)

    # Compute 9:16 crop window dimensions
    crop_h = src_h
    crop_w = int(src_h * 9 / 16)
    if crop_w % 2 != 0:
        crop_w += 1
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(src_w * 16 / 9)

    # Scale to 640px width proxy for fast, lightweight inference
    proxy_w = 640
    proxy_h = int(round(640 * src_h / src_w))
    if proxy_h % 2 != 0:
        proxy_h += 1

    scale_x = src_w / proxy_w
    scale_y = src_h / proxy_h
    frame_bytes = proxy_w * proxy_h * 3

    # Launch FFmpeg pipe to decode sampled frames directly into memory
    ffmpeg_cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{clip_start:.3f}",
        "-i", str(source_video),
        "-t", f"{duration:.3f}",
        "-r", str(sample_fps),
        "-s", f"{proxy_w}x{proxy_h}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]

    detector = FaceDetector(input_size=(proxy_w, proxy_h))

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    initial_face_count = 0
    selected_idx: int | None = None
    current_target: FaceBox | None = None
    raw_centers: list[tuple[float, float]] = []  # (rel_time, center_x)

    frame_idx = 0
    dt = 1.0 / sample_fps

    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            rel_t = min(duration, round(frame_idx * dt, 3))
            frame_idx += 1

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((proxy_h, proxy_w, 3))
            proxy_detections = detector.detect(frame)

            # Map proxy coordinates back to original source dimensions
            detections = [
                FaceBox(
                    int(round(d.x * scale_x)),
                    int(round(d.y * scale_y)),
                    int(round(d.w * scale_x)),
                    int(round(d.h * scale_y)),
                )
                for d in proxy_detections
            ]

            # Step 1: Initialize target face on first frame where faces appear
            if current_target is None:
                if detections:
                    initial_face_count = len(detections)
                    s_idx, chosen_box = select_face_deterministically(
                        detections, video_id, clip_filename
                    )
                    selected_idx = s_idx + 1  # 1-based index for user logging
                    current_target = chosen_box
                    raw_centers.append((rel_t, chosen_box.center_x))
            else:
                # Step 2: Track selected face — match candidate closest to previous target
                if detections:
                    best_match = None
                    best_score = float("inf")

                    for cand in detections:
                        iou_val = current_target.iou(cand)
                        dist = current_target.distance_to(cand)

                        # Prioritize overlap + close distance
                        score = dist - (iou_val * 300.0)
                        if score < best_score:
                            best_score = score
                            best_match = cand

                    # If match is reasonably close (within 40% of frame width), update
                    if best_match is not None and current_target.distance_to(best_match) < (src_w * 0.4):
                        current_target = best_match

                raw_centers.append((rel_t, current_target.center_x))

    finally:
        proc.communicate()

    # Check if any face was tracked
    if not raw_centers or current_target is None:
        # Case A: No faces detected
        default_x = (src_w - crop_w) // 2
        default_y = (src_h - crop_h) // 2
        return TrackingResult(
            face_count=0,
            selected_face_idx=None,
            is_tracked=False,
            crop_w=crop_w,
            crop_h=crop_h,
            keyframes=[CameraKeyframe(time=0.0, crop_x=float(default_x), crop_y=float(default_y))],
            crop_x_expr=str(default_x),
        )

    # Smooth camera positions
    smoothed_positions = smooth_camera_positions(raw_centers, crop_w, src_w)
    crop_y = (src_h - crop_h) // 2

    keyframes = [
        CameraKeyframe(time=t, crop_x=round(cx, 1), crop_y=float(crop_y))
        for t, cx in smoothed_positions
    ]

    crop_x_expr = build_ffmpeg_crop_expression(keyframes, crop_w, src_w)

    return TrackingResult(
        face_count=initial_face_count,
        selected_face_idx=selected_idx,
        is_tracked=True,
        crop_w=crop_w,
        crop_h=crop_h,
        keyframes=keyframes,
        crop_x_expr=crop_x_expr,
    )
