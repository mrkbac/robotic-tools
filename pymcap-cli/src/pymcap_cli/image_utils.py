"""Shared image decoding and encoder utilities for video/roscompress commands."""

import threading
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

import av
import av.error
import numpy as np
from av import Packet, VideoFrame
from numpy.typing import NDArray

if TYPE_CHECKING:
    from av.container import InputContainer
    from av.video.codeccontext import VideoCodecContext

COMPRESSED_SCHEMAS = {"sensor_msgs/msg/CompressedImage", "sensor_msgs/CompressedImage"}
RAW_SCHEMAS = {"sensor_msgs/msg/Image", "sensor_msgs/Image"}
IMAGE_SCHEMAS = COMPRESSED_SCHEMAS | RAW_SCHEMAS


class VideoEncoderError(Exception):
    """Raised when encoding fails."""


def test_encoder(encoder_name: str) -> bool:
    """Test if an encoder is available on this system."""
    try:
        av.CodecContext.create(encoder_name, "w")
    except (av.error.FFmpegError, ValueError):
        return False
    else:
        return True


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


def raw_image_to_array(message: Any) -> NDArray[np.uint8]:
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
