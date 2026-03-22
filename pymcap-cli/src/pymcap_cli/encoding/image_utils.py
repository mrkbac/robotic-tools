"""Shared image decoding and encoder utilities for video/roscompress commands."""

import threading
from fractions import Fraction
from io import BytesIO
from typing import TYPE_CHECKING, Protocol, cast

import av
import av.error
import numpy as np
from av import Packet, VideoFrame
from numpy.typing import NDArray

from pymcap_cli.encoding.encoder_common import (
    EncoderConfig,
    VideoEncoderError,
    build_encoder_options,
)
from pymcap_cli.encoding.encoder_common import (
    resolve_encoder as _resolve_encoder,
)
from pymcap_cli.encoding.encoder_common import (
    resolve_encoder_for_backend as _resolve_encoder_for_backend,
)

if TYPE_CHECKING:
    from av.container import InputContainer
    from av.video.codeccontext import VideoCodecContext


class CompressedImageMsg(Protocol):
    """Protocol for ROS CompressedImage messages."""

    @property
    def data(self) -> bytes | bytearray | memoryview: ...


class ImageMsg(Protocol):
    """Protocol for ROS Image messages."""

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def encoding(self) -> str: ...

    @property
    def data(self) -> bytes | bytearray | memoryview: ...


def test_encoder(encoder_name: str) -> bool:
    """Test if an encoder is available on this system."""
    try:
        av.CodecContext.create(encoder_name, "w")
    except (av.error.FFmpegError, ValueError):
        return False
    else:
        return True


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------


class _DecoderLocal(threading.local):
    """Thread-local persistent codec contexts for JPEG and PNG decoding."""

    mjpeg_ctx: "VideoCodecContext | None" = None
    png_ctx: "VideoCodecContext | None" = None


_decoder_local = _DecoderLocal()


def _get_mjpeg_ctx() -> "VideoCodecContext":
    """Get or create thread-local persistent MJPEG decoder context."""
    ctx = _decoder_local.mjpeg_ctx
    if ctx is None:
        ctx = cast("VideoCodecContext", av.CodecContext.create("mjpeg", "r"))
        ctx.open()
        _decoder_local.mjpeg_ctx = ctx
    return ctx


def _get_png_ctx() -> "VideoCodecContext":
    """Get or create thread-local persistent PNG decoder context."""
    ctx = _decoder_local.png_ctx
    if ctx is None:
        ctx = av.CodecContext.create("png", "r")
        ctx.open()
        _decoder_local.png_ctx = ctx
    return ctx


def _detect_image_format(data: bytes) -> str:
    """Detect image format from magic bytes."""
    if data[:2] == b"\xff\xd8":
        return "mjpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return "unknown"


def _decode_via_container(data: bytes) -> VideoFrame:
    """Fallback: decode an image by opening a full av container."""
    try:
        with cast("InputContainer", av.open(BytesIO(data))) as container:
            for frame in container.decode(video=0):
                return frame
    except av.error.FFmpegError as exc:
        raise VideoEncoderError(f"Failed to decode compressed image: {exc}") from exc
    raise VideoEncoderError("Decoder produced no frames")


def decode_compressed_frame(compressed_data: bytes) -> VideoFrame:
    """Decode a compressed image (JPEG/PNG) to a VideoFrame.

    Uses persistent thread-local codec contexts to avoid the overhead of
    creating a new av.open() container per frame. Falls back to a full
    container open for unrecognised formats.
    """
    fmt = _detect_image_format(compressed_data)
    if fmt == "unknown":
        return _decode_via_container(compressed_data)

    ctx = _get_mjpeg_ctx() if fmt == "mjpeg" else _get_png_ctx()
    try:
        for frame in ctx.decode(Packet(compressed_data)):
            return frame
    except av.error.FFmpegError as exc:
        raise VideoEncoderError(f"Failed to decode compressed image: {exc}") from exc

    raise VideoEncoderError("Decoder produced no frames")


def raw_image_to_array(message: ImageMsg) -> NDArray[np.uint8]:
    """Convert a ROS Image message to an RGB numpy array."""
    if not hasattr(message, "data") or not message.data:
        raise VideoEncoderError("Image has no data")
    if not hasattr(message, "width") or not hasattr(message, "height"):
        raise VideoEncoderError("Image missing width/height")
    if not hasattr(message, "encoding"):
        raise VideoEncoderError("Image missing encoding")

    width = message.width
    height = message.height
    encoding = str(message.encoding).lower()
    data = bytes(message.data)

    if encoding in {"rgb", "rgb8"}:
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        return array.copy()
    if encoding in {"bgr", "bgr8"}:
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        return array[..., ::-1].copy()
    if encoding in {"mono", "mono8", "8uc1"}:
        mono_array = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        return np.repeat(mono_array[:, :, None], 3, axis=2)

    raise VideoEncoderError(f"Unsupported image encoding: {message.encoding}")


# ---------------------------------------------------------------------------
# Encoder resolution (delegates to encoder_common with PyAV test function)
# ---------------------------------------------------------------------------


def resolve_encoder(codec: str, *, use_hardware: bool = True) -> str:
    """Pick the best available encoder for *codec* using PyAV to probe."""
    return _resolve_encoder(codec, test_fn=test_encoder, use_hardware=use_hardware)


def resolve_encoder_for_backend(codec: str, backend: str) -> str:
    """Pick the encoder for *codec* using the specified *backend* (PyAV probe)."""
    return _resolve_encoder_for_backend(codec, backend, test_fn=test_encoder)


class VideoEncoder:
    """PyAV-based video encoder for converting images to compressed video."""

    # Pixel formats that are plane-compatible with yuv420p (only color range differs)
    # and can be passed directly to the encoder without an expensive sws_scale reformat.
    _YUV420P_COMPAT = frozenset({"yuv420p", "yuvj420p"})

    def __init__(
        self,
        width: int,
        height: int,
        codec_name: str,
        quality: int = 28,
        target_fps: float = 30.0,
        gop_size: int = 30,
        *,
        preset: str | None = None,
    ) -> None:
        self.config = EncoderConfig(width=width, height=height, codec_name=codec_name)
        self._target_fps = max(target_fps, 1.0)
        self._frame_index = 0
        self._quality = quality
        self._gop_size = gop_size

        try:
            self._context: VideoCodecContext = cast(
                "VideoCodecContext", av.CodecContext.create(codec_name, "w")
            )
        except av.error.FFmpegError as exc:
            raise VideoEncoderError(f"Failed to create encoder {codec_name}: {exc}") from exc

        # Configure encoder
        fps_int = max(round(self._target_fps), 1)
        self._context.width = width
        self._context.height = height
        self._context.pix_fmt = "yuv420p"
        self._context.time_base = Fraction(1, fps_int)
        self._context.framerate = Fraction(fps_int, 1)
        self._context.gop_size = gop_size
        self._context.max_b_frames = 0  # Ensure every frame produces immediate output

        # Set codec-specific options
        options, bit_rate = build_encoder_options(codec_name, quality, width, height, preset=preset)
        if bit_rate is not None:
            self._context.bit_rate = bit_rate
        if options:
            self._context.options = options

        try:
            self._context.open()
        except av.error.FFmpegError as exc:
            raise VideoEncoderError(f"Failed to open encoder {codec_name}: {exc}") from exc

    def encode(self, frame: VideoFrame) -> bytes | None:
        """Encode a single frame and return compressed video bytes, or None if buffered."""
        # Only reformat when dimensions differ or the format is truly incompatible.
        # yuvj420p (full-range) is plane-compatible with yuv420p; FFmpeg handles the
        # color-range flag internally so we can skip the costly sws_scale conversion.
        needs_resize = frame.width != self.config.width or frame.height != self.config.height
        needs_fmt = frame.format.name not in self._YUV420P_COMPAT
        if needs_resize or needs_fmt:
            frame = frame.reformat(
                width=self.config.width, height=self.config.height, format=self._context.pix_fmt
            )
        frame.pts = self._frame_index
        self._frame_index += 1

        try:
            packets = list(self._context.encode(frame))
        except av.error.FFmpegError as exc:
            raise VideoEncoderError(f"Encoding error: {exc}") from exc

        if not packets:
            return None

        return b"".join(bytes(packet) for packet in packets)

    def flush(self) -> bytes | None:
        """Flush remaining buffered frames from the encoder."""
        try:
            packets = list(self._context.encode(None))
        except av.error.FFmpegError:
            return None
        if not packets:
            return None
        return b"".join(bytes(packet) for packet in packets)
