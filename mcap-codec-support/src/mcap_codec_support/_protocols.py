"""Shared structural Protocols for encoder, decoder, and ROS message inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from av import VideoFrame
    from small_mcap import DecodedMessage

    from mcap_codec_support.video.common import DecompressedFrame, EncoderConfig


class RawImageMessage(Protocol):
    """Structural shape of a ROS ``sensor_msgs/Image`` message."""

    width: int
    height: int
    encoding: str
    data: bytes


class CompressedImageMsg(Protocol):
    """Structural shape of a ROS ``sensor_msgs/CompressedImage`` message."""

    @property
    def data(self) -> bytes | bytearray | memoryview: ...


class VideoEncoderProtocol(Protocol):
    """Structural interface shared by VideoEncoder and FFmpegVideoEncoder."""

    config: EncoderConfig

    def encode(self, frame: VideoFrame) -> bytes | None: ...

    def flush_packets(self) -> list[bytes]: ...


class VideoDecompressorProtocol(Protocol):
    """Decompresses H.264/H.265 video packets to image data."""

    def decompress(self, video_data: bytes, codec: str) -> DecompressedFrame | None:
        """Decompress a single video packet."""
        ...

    def flush(self) -> list[DecompressedFrame]:
        """Flush any buffered frames from the decoder."""
        ...


class VideoCompressionBackend(Protocol):
    """Backend used by roscompress for CompressedVideo output."""

    label: str
    prefetch_supported: bool

    def test_encoder(self, encoder_name: str) -> bool: ...

    def resolve_encoder(self, codec: str) -> str: ...

    def decode_compressed(self, data: bytes) -> tuple[VideoFrame, int, int]: ...

    def decode_image(
        self, msg: DecodedMessage, schema_name: str
    ) -> tuple[VideoFrame, int, int]: ...

    def create_encoder(
        self,
        width: int,
        height: int,
        codec_name: str,
        quality: int,
        *,
        input_pix_fmt: str | None = None,
        scale: tuple[int, int] | None = None,
    ) -> VideoEncoderProtocol: ...

    def get_pix_fmt(self, topic: str) -> str | None: ...


class VideoFileStrategy(Protocol):
    """Strategy contract used by the lazy MP4 file writer."""

    config: EncoderConfig

    def write_compressed(self, data: bytes, log_time_ns: int) -> None: ...

    def write_raw(self, data: bytes, log_time_ns: int) -> None: ...

    def write_rgb(self, rgb: np.ndarray, log_time_ns: int) -> None: ...

    def close(self) -> int: ...
