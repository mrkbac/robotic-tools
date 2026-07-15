"""Convenience composition for decoding ROS2 MCAP messages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from mcap_ros2_support_fast.decoder import DecoderFactory

    from mcap_codec_support.pointcloud import PointCloudDecompressFactory
    from mcap_codec_support.video import EncoderMode, VideoDecompressFactory

VideoBackend: TypeAlias = Literal["auto", "pyav", "ffmpeg-cli", "gstreamer"]


def create_decoder_factories(
    *,
    video_format: Literal["compressed", "raw"] = "raw",
    jpeg_quality: int = 90,
    video_backend: EncoderMode | VideoBackend = "auto",
) -> list[VideoDecompressFactory | PointCloudDecompressFactory | DecoderFactory]:
    """Create decoders for compressed video, point clouds, and regular ROS2 CDR.

    The specialized factories must precede the general ROS2 factory so their
    schemas are decoded and decompressed before the CDR fallback is considered.
    """
    from mcap_ros2_support_fast.decoder import DecoderFactory  # noqa: PLC0415

    from mcap_codec_support.pointcloud import PointCloudDecompressFactory  # noqa: PLC0415
    from mcap_codec_support.video import (  # noqa: PLC0415
        EncoderMode,
        VideoDecompressFactory,
    )

    return [
        VideoDecompressFactory(
            video_format=video_format,
            jpeg_quality=jpeg_quality,
            backend=EncoderMode(video_backend),
        ),
        PointCloudDecompressFactory(),
        DecoderFactory(),
    ]
