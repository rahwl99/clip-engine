"""Processing package for Clipt."""

from processing.processor import generate_processed_clip, generate_vertical_clip
from processing.face_tracker import track_face_for_clip
from processing.subtitles import generate_ass_file, build_clip_subtitles


def process_video(*args, **kwargs):
    from core.pipeline import process_video as _pv
    return _pv(*args, **kwargs)


__all__ = [
    "generate_vertical_clip",
    "generate_processed_clip",
    "track_face_for_clip",
    "generate_ass_file",
    "build_clip_subtitles",
    "process_video",
]
