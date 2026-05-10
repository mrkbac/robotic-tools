"""Decoder factory for decompressing CompressedVideo topics.

Provides ``VideoDecompressFactory`` for use with ``read_message_decoded``.
Uses ``VideoDecompressorProtocol`` — no direct ``av`` or ``subprocess`` imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from mcap_codec_support.video.common import EncoderMode
from mcap_codec_support.video.schemas import COMPRESSED_VIDEO_SCHEMA

if TYPE_CHECKING:
    from collections.abc import Callable

    from small_mcap import Channel, Schema

    from mcap_codec_support._messages import Header
    from mcap_codec_support._protocols import VideoDecompressorProtocol
    from mcap_codec_support.video._messages import CompressedImageDict, ImageDict
    from mcap_codec_support.video.common import DecompressedFrame


# ---------------------------------------------------------------------------
# VideoDecompressFactory
# ---------------------------------------------------------------------------


class VideoDecompressFactory:
    """Channel-aware decoder factory: CompressedVideo → CompressedImage or Image.

    Creates a separate ``VideoDecompressorProtocol`` per channel for proper
    P-frame handling. No direct ``av`` or ``subprocess`` imports.
    """

    channel_aware = True

    def __init__(
        self,
        *,
        video_format: Literal["compressed", "raw"] = "compressed",
        jpeg_quality: int = 90,
        backend: EncoderMode = EncoderMode.AUTO,
    ) -> None:
        self._video_format = video_format
        self._jpeg_quality = jpeg_quality
        self._backend = backend

        from mcap_ros2_support_fast.decoder import DecoderFactory  # noqa: PLC0415

        self._cdr_factory = DecoderFactory()
        self._decompressors: dict[int, VideoDecompressorProtocol] = {}

    def flush_all(self) -> list[DecompressedFrame]:
        """Flush all decompressors and return remaining frames."""
        return [frame for _, frame in self.flush_all_by_channel()]

    def flush_all_by_channel(self) -> list[tuple[int, DecompressedFrame]]:
        """Flush all decompressors and keep channel ownership for each frame."""
        frames: list[tuple[int, DecompressedFrame]] = []
        for channel_id, decompressor in self._decompressors.items():
            frames.extend((channel_id, frame) for frame in decompressor.flush())
        return frames

    def _get_decompressor(self, channel_id: int) -> VideoDecompressorProtocol:
        if channel_id not in self._decompressors:
            from mcap_codec_support.video.compression import (  # noqa: PLC0415
                create_video_decompressor,
            )

            self._decompressors[channel_id] = create_video_decompressor(
                video_format=self._video_format,
                jpeg_quality=self._jpeg_quality,
                mode=self._backend,
            )
        return self._decompressors[channel_id]

    def decoder_for(
        self,
        message_encoding: str,
        schema: Schema | None,
        channel: Channel,
    ) -> Callable[[bytes | memoryview], CompressedImageDict | ImageDict | None] | None:
        if schema is None or schema.name != COMPRESSED_VIDEO_SCHEMA:
            return None

        cdr_decoder = self._cdr_factory.decoder_for(message_encoding, schema)
        if cdr_decoder is None:
            return None

        decompressor = self._get_decompressor(channel.id)

        def _decode(data: bytes | memoryview) -> CompressedImageDict | ImageDict | None:
            decoded = cdr_decoder(data)
            codec = decoded.format
            video_data = decoded.data
            if isinstance(video_data, memoryview):
                video_data = bytes(video_data)

            frame: DecompressedFrame | None = decompressor.decompress(video_data, codec)
            if frame is None:
                return None

            timestamp = decoded.timestamp
            header: Header = {
                "stamp": {"sec": timestamp.sec, "nanosec": timestamp.nanosec},
                "frame_id": decoded.frame_id,
            }

            if frame.is_jpeg:
                return {"header": header, "format": "jpeg", "data": frame.data}

            return {
                "header": header,
                "height": frame.height,
                "width": frame.width,
                "encoding": "rgb8",
                "is_bigendian": 0,
                "step": frame.width * 3,
                "data": frame.data,
            }

        return _decode
