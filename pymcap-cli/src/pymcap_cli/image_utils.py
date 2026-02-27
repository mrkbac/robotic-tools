"""Shared image decoding and encoder utilities for video/roscompress commands."""

import io
from typing import TYPE_CHECKING, Any, cast

import av
import av.error
import numpy as np
from av import VideoFrame
from numpy.typing import NDArray

if TYPE_CHECKING:
    from av.container import InputContainer

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


def decode_compressed_frame(compressed_data: bytes) -> VideoFrame:
    """Decode a compressed image (JPEG/PNG) to a VideoFrame."""
    try:
        container = cast("InputContainer", av.open(io.BytesIO(compressed_data), format="image2"))
        for frame in container.decode(video=0):
            container.close()
            return frame
    except Exception as exc:
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
