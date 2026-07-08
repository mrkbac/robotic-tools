"""Shared structural Protocols for encoder, decoder, and ROS message inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeVar

if TYPE_CHECKING:
    from av import VideoFrame
    from small_mcap import DecodedMessage

    from mcap_codec_support.video.common import DecompressedFrame, EncoderConfig

# The frame representation differs by backend: PyAV works on ``av.VideoFrame``,
# the ffmpeg-CLI backend on raw ``bytes``. Parameterizing keeps decode→encode
# paired per backend instead of pretending they share one frame type.
FrameT = TypeVar("FrameT")


class RawImageMessage(Protocol):
    """Structural shape of a ROS ``sensor_msgs/Image`` message."""

    width: int
    height: int
    encoding: str
    step: int
    data: bytes


class CompressedImageMsg(Protocol):
    """Structural shape of a ROS ``sensor_msgs/CompressedImage`` message."""

    @property
    def data(self) -> bytes | bytearray | memoryview: ...


class VideoEncoderProtocol(Protocol[FrameT]):
    """Encoder interface; ``FrameT`` is the per-backend frame type."""

    config: EncoderConfig

    def encode(self, frame: FrameT) -> bytes | None: ...

    def flush_packets(self) -> list[bytes]: ...

    def close(self) -> None:
        """Release the encoder's native context / subprocess."""
        ...


class VideoDecompressorProtocol(Protocol):
    """Decompresses H.264/H.265 video packets to image data."""

    def decompress(self, video_data: bytes, codec: str) -> DecompressedFrame | None:
        """Decompress a single video packet."""
        ...

    def flush(self) -> list[DecompressedFrame]:
        """Flush any buffered frames from the decoder."""
        ...


class VideoCompressionBackend(Protocol[FrameT]):
    """Backend used by roscompress for CompressedVideo output.

    ``FrameT`` ties ``decode_*`` output to the frame type ``create_encoder``'s
    encoder consumes, so a backend can't decode to one frame type and encode
    another.
    """

    label: str
    prefetch_supported: bool

    def test_encoder(self, encoder_name: str) -> bool: ...

    def resolve_encoder(self, codec: str) -> str: ...

    def decode_compressed(self, data: bytes) -> tuple[FrameT, int, int]: ...

    def decode_image(self, msg: DecodedMessage, schema_name: str) -> tuple[FrameT, int, int]: ...

    def create_encoder(
        self,
        width: int,
        height: int,
        codec_name: str,
        quality: int,
        *,
        input_pix_fmt: str | None = None,
        scale: tuple[int, int] | None = None,
    ) -> VideoEncoderProtocol[FrameT]: ...

    def get_pix_fmt(self, topic: str) -> str | None: ...


# A backend chosen at runtime is either the PyAV (VideoFrame) or ffmpeg-CLI
# (bytes) flavor; this union is the honest type at that dynamic boundary.
AnyVideoBackend: TypeAlias = "VideoCompressionBackend[VideoFrame] | VideoCompressionBackend[bytes]"
