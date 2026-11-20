"""Shared FFmpeg setup for move-debugger MP4 exports."""

from pathlib import Path

from matplotlib import animation


def mp4_writer(path: str | Path, *, fps: int) -> tuple[str, animation.FFMpegWriter]:
    """Validate an output path and return a Matplotlib FFmpeg writer."""
    if fps < 1:
        raise ValueError("fps must be at least 1")
    output = Path(path)
    if output.suffix.lower() != ".mp4":
        raise ValueError("to_mp4 must name an .mp4 file")
    if output.exists():
        raise FileExistsError(f"MP4 output already exists: {output}")
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("MP4 export requires FFmpeg on PATH")
    return str(output), animation.FFMpegWriter(fps=fps, codec="h264")


def hold_frames(seconds: float, fps: int) -> int:
    """Keep even a zero-duration debugger step visible for one frame."""
    if seconds < 0:
        raise ValueError("pause_time must be non-negative")
    return max(1, round(seconds * fps))
