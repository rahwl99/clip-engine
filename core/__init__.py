"""Core package for Clipt."""

__all__ = ["analyze_video", "process_video"]


def __getattr__(name: str):
    if name in ("analyze_video", "process_video"):
        from core.pipeline import analyze_video, process_video
        return {"analyze_video": analyze_video, "process_video": process_video}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
