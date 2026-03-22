"""Shared protocols and data types for video decompression backends.

This module has no external dependencies — only stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

# ---------------------------------------------------------------------------
# Decompression
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecompressedFrame:
    """Result of decompressing a single video frame."""

    data: bytes
    """JPEG bytes (when ``is_jpeg=True``) or raw RGB24 bytes."""
    width: int
    height: int
    is_jpeg: bool


class VideoDecompressorProtocol(Protocol):
    """Decompresses H.264/H.265 video packets to image data."""

    def decompress(self, video_data: bytes, codec: str) -> DecompressedFrame | None:
        """Decompress a single video packet.

        Returns a ``DecompressedFrame``, or ``None`` if the decoder
        needs more data (e.g. waiting for a keyframe).
        """
        ...

    def flush(self) -> list[DecompressedFrame]:
        """Flush any buffered frames from the decoder."""
        ...
