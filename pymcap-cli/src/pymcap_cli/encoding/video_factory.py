"""Factory functions for selecting video decompression backends.

Uses lazy imports so that importing this module does not pull in PyAV or
trigger ffmpeg subprocess calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pymcap_cli.encoding.encoder_common import EncoderMode

if TYPE_CHECKING:
    from pymcap_cli.encoding.video_protocols import VideoDecompressorProtocol


def create_video_decompressor(
    video_format: Literal["compressed", "raw"] = "compressed",
    jpeg_quality: int = 90,
    *,
    mode: EncoderMode = EncoderMode.AUTO,
) -> VideoDecompressorProtocol:
    """Create a video decompressor using the requested backend."""
    if mode == EncoderMode.PYAV:
        from pymcap_cli.encoding.video_pyav import PyAVVideoDecompressor  # noqa: PLC0415

        return PyAVVideoDecompressor(video_format=video_format, jpeg_quality=jpeg_quality)

    if mode == EncoderMode.FFMPEG_CLI:
        from pymcap_cli.encoding.video_ffmpeg import FFmpegVideoDecompressor  # noqa: PLC0415

        return FFmpegVideoDecompressor(video_format=video_format, jpeg_quality=jpeg_quality)

    # AUTO: try PyAV, fall back to ffmpeg CLI.
    try:
        from pymcap_cli.encoding.video_pyav import PyAVVideoDecompressor  # noqa: PLC0415

        return PyAVVideoDecompressor(video_format=video_format, jpeg_quality=jpeg_quality)
    except ImportError:
        from pymcap_cli.encoding.video_ffmpeg import (  # noqa: PLC0415
            FFmpegVideoDecompressor,
            find_ffmpeg,
        )

        if find_ffmpeg():
            return FFmpegVideoDecompressor(video_format=video_format, jpeg_quality=jpeg_quality)
        raise
