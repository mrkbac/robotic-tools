from __future__ import annotations

from portable_ffmpeg import add_to_path


def ensure_ffmpeg(worker_id: str | None) -> None:
    """Initialize FFmpeg once in the controller before xdist workers spawn."""
    if worker_id is None:
        add_to_path()
