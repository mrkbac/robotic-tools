"""Transform JPEG CompressedImage to CompressedVideo using pluggable backends."""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Literal

from PIL import Image

from . import Transformer, TransformError
from .image_to_video_backends import (
    EncodingConfig,
    FFmpegBackend,
    GStreamerBackend,
    ImageToVideoBackend,
    ImageToVideoBackendError,
    is_ffmpeg_available,
    is_gstreamer_available,
)

logger = logging.getLogger(__name__)


class ImageToVideoTransformer(Transformer):
    """Transform sensor_msgs/CompressedImage (JPEG) to foxglove_msgs/CompressedVideo (H.264)."""

    def __init__(
        self,
        codec: str = "h264",
        quality: int = 23,  # CRF value for libx264 (lower = better quality)
        preset: str = "fast",
        use_hardware: bool = True,
        max_dimension: int = 480,  # Maximum dimension (width or height) for downscaling
        backend: Literal["auto", "ffmpeg", "gstreamer"] = "auto",
        timeout_s: float = 30.0,
    ) -> None:
        """Initialize the transformer.

        Args:
            codec: Video codec to use ('h264', 'h265', 'vp9', 'av1')
            quality: Encoding quality (CRF for x264: 0-51, lower is better)
            preset: Encoding preset for libx264 ('ultrafast', 'fast', 'medium', 'slow')
            use_hardware: Whether to try hardware acceleration
            max_dimension: Maximum dimension (width or height) in pixels. Images larger
                          than this will be downscaled proportionally (default: 720 for HD)
            backend: Backend implementation to use ('ffmpeg', 'gstreamer', or 'auto')
            timeout_s: Maximum time in seconds to wait for an encode operation
        """
        self.codec = codec
        self.quality = quality
        self.preset = preset
        self.use_hardware = use_hardware
        self.max_dimension = max_dimension
        self._config = EncodingConfig(
            codec=codec,
            quality=quality,
            preset=preset,
            use_hardware=use_hardware,
            timeout_s=timeout_s,
        )
        self._backend = self._initialise_backend(backend)
        logger.info("ImageToVideoTransformer backend selected: %s", type(self._backend).__name__)

    def get_input_schema(self) -> str:
        """Get the input message schema name."""
        return "sensor_msgs/msg/CompressedImage"

    def get_output_schema(self) -> str:
        """Get the output message schema name."""
        return "foxglove_msgs/CompressedVideo"

    def _calculate_downscale_dimensions(self, width: int, height: int) -> tuple[int, int]:
        """Calculate target dimensions for downscaling while maintaining aspect ratio.

        Args:
            width: Original image width
            height: Original image height

        Returns:
            Tuple of (target_width, target_height). Returns original dimensions
            if no downscaling is needed.
        """

        def ensure_even(value: int) -> int:
            if value % 2 == 0:
                return value
            # Prefer shrinking by one pixel; if that would hit zero, bump to 2 instead
            adjusted = value - 1
            return adjusted if adjusted >= 2 else 2

        # If both dimensions are within max_dimension, no scaling needed besides ensuring even size
        if width <= self.max_dimension and height <= self.max_dimension:
            return ensure_even(width), ensure_even(height)

        # Calculate aspect ratio
        aspect_ratio = width / height

        # Determine which dimension to constrain
        if width > height:
            # Width is the limiting factor
            new_width = self.max_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            # Height is the limiting factor
            new_height = self.max_dimension
            new_width = int(new_height * aspect_ratio)

        # Ensure dimensions are even (required for most video codecs)
        new_width = ensure_even(new_width)
        new_height = ensure_even(new_height)

        return new_width, new_height

    def _initialise_backend(
        self, backend_choice: Literal["auto", "ffmpeg", "gstreamer"]
    ) -> ImageToVideoBackend:
        if backend_choice not in {"auto", "ffmpeg", "gstreamer"}:
            raise ValueError(f"Unsupported backend selection: {backend_choice}")

        candidates: list[tuple[str, type[ImageToVideoBackend]]] = []

        if backend_choice in {"auto", "gstreamer"} and is_gstreamer_available():
            candidates.append(("gstreamer", GStreamerBackend))

        if backend_choice in {"auto", "ffmpeg"} and is_ffmpeg_available():
            candidates.append(("ffmpeg", FFmpegBackend))

        if not candidates:
            if backend_choice == "auto":
                raise RuntimeError(
                    "No encoding backends detected. Install ffmpeg or GStreamer with the required plugins."
                )
            raise RuntimeError(
                f"Requested backend '{backend_choice}' is unavailable on this system."
            )

        errors: list[str] = []
        for name, backend_cls in candidates:
            try:
                return backend_cls(self._config)
            except ImageToVideoBackendError as err:
                logger.warning("Failed to initialise %s backend: %s", name, err)
                errors.append(f"{name}: {err}")

        if errors:
            detail = "; ".join(errors)
            raise RuntimeError(f"Unable to initialise requested backend(s): {detail}")

        raise RuntimeError("Failed to initialise any image-to-video backend")

    def transform(self, message: Any) -> dict[str, Any]:
        """Transform CompressedImage to CompressedVideo.

        Args:
            message: Decoded CompressedImage message

        Returns:
            CompressedVideo message dict

        Raises:
            TransformError: If transformation fails
        """
        try:
            if not message.data:
                raise TransformError(f"Empty image data, {message}")

            # Verify it's JPEG (for now we only support JPEG)
            if message.format and "jpeg" not in message.format.lower():
                raise TransformError(f"Unsupported format: {message.format}")

            # Get image dimensions using PIL (faster than ffprobe)
            try:
                img = Image.open(BytesIO(message.data))
                width, height = img.size
            except Exception as e:
                raise TransformError(f"Failed to read image: {e}") from e

            # Calculate target dimensions (may downscale if too large)
            target_width, target_height = self._calculate_downscale_dimensions(width, height)

            if (target_width, target_height) != (width, height):
                logger.debug(f"Downscaling from {width}x{height} to {target_width}x{target_height}")

            # Encode JPEG frame with selected backend
            h264_data = self._backend.encode(
                message.data, width, height, target_width, target_height
            )

            # Build output message
            output_message = {
                "timestamp": {
                    "sec": message.header.stamp.sec,
                    "nanosec": message.header.stamp.nanosec,
                },
                "frame_id": message.header.frame_id,
                "data": list(h264_data),  # Convert bytes to list of uint8
                "format": self.codec,
            }

            return output_message

        except ImageToVideoBackendError as err:
            raise TransformError(f"Encoding backend failed: {err}") from err
        except TransformError:
            raise
        except Exception as e:
            raise TransformError(f"Transform failed: {e}") from e


__all__ = ["ImageToVideoTransformer"]
