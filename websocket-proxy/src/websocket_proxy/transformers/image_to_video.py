"""Transform CompressedImage to CompressedVideo using a persistent PyAV encoder."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pymcap_cli.image_utils import (
    FOXGLOVE_COMPRESSED_VIDEO,
    EncoderConfig,
    VideoEncoder,
    VideoEncoderError,
    calculate_downscale_dimensions,
    decode_compressed_frame,
    get_software_encoder,
    resolve_encoder,
)

from . import Transformer, TransformError

if TYPE_CHECKING:
    from types import SimpleNamespace

logger = logging.getLogger(__name__)


class ImageToVideoTransformer(Transformer):
    """Transform sensor_msgs/CompressedImage to foxglove_msgs/CompressedVideo via PyAV."""

    def __init__(
        self,
        codec: str = "h264",
        quality: int = 23,
        preset: str = "fast",
        use_hardware: bool = True,
        max_dimension: int = 480,
        gop_size: int = 15,
        target_fps: float = 30.0,
    ) -> None:
        self.codec = codec.lower()
        self.quality = quality
        self.preset = preset
        self.use_hardware = use_hardware
        self.max_dimension = max_dimension
        self.gop_size = max(1, gop_size)
        self.target_fps = target_fps

        try:
            self.encoder_name = resolve_encoder(self.codec, use_hardware=self.use_hardware)
        except ValueError as exc:
            raise TransformError(str(exc)) from exc
        logger.info("Using encoder: %s", self.encoder_name)

        self._stream_config: EncoderConfig | None = None
        self._stream_encoder: VideoEncoder | None = None

    def get_input_schema(self) -> str:
        return "sensor_msgs/msg/CompressedImage"

    def get_output_schema(self) -> str:
        return "foxglove_msgs/CompressedVideo"

    def get_output_schema_definition(self) -> str:
        return FOXGLOVE_COMPRESSED_VIDEO

    def transform(self, message: SimpleNamespace) -> dict[str, Any]:
        if not message.data:
            raise TransformError(f"Empty image data, {message}")

        frame, target_width, target_height = self._prepare_frame(message.data)

        config = EncoderConfig(
            width=target_width, height=target_height, codec_name=self.encoder_name
        )
        encoder = self._ensure_stream_encoder(config)

        try:
            encoded = encoder.encode(frame)
        except VideoEncoderError as exc:
            self._fallback_to_software(exc)
            config = EncoderConfig(
                width=target_width, height=target_height, codec_name=self.encoder_name
            )
            encoder = self._ensure_stream_encoder(config)
            try:
                encoded = encoder.encode(frame)
            except VideoEncoderError as inner:
                raise TransformError(str(inner)) from inner

        if encoded is None:
            raise TransformError("Frame buffered by encoder")

        return {
            "timestamp": {
                "sec": message.header.stamp.sec,
                "nanosec": message.header.stamp.nanosec,
            },
            "frame_id": message.header.frame_id,
            "data": list(encoded),
            "format": self.codec,
        }

    # ------------------------------------------------------------------
    # Encoder management helpers
    # ------------------------------------------------------------------

    def _create_encoder(self, config: EncoderConfig) -> VideoEncoder:
        return VideoEncoder(
            width=config.width,
            height=config.height,
            codec_name=config.codec_name,
            quality=self.quality,
            target_fps=self.target_fps,
            gop_size=self.gop_size,
            preset=self.preset,
        )

    def _fallback_to_software(self, exc: Exception) -> None:
        """Switch to software encoder after a hardware encoder failure.

        Raises TransformError if already using the software encoder.
        """
        try:
            software_encoder = get_software_encoder(self.codec)
        except ValueError:
            raise TransformError(str(exc)) from exc
        if self.encoder_name == software_encoder:
            raise TransformError(str(exc)) from exc
        logger.warning(
            "Encoder %s failed (%s); falling back to %s",
            self.encoder_name,
            exc,
            software_encoder,
        )
        self.encoder_name = software_encoder
        self._stream_encoder = None
        self._stream_config = None

    def _ensure_stream_encoder(self, config: EncoderConfig) -> VideoEncoder:
        if self._stream_encoder and self._stream_config == config:
            return self._stream_encoder

        self._stream_encoder = None
        self._stream_config = None
        try:
            self._stream_encoder = self._create_encoder(config)
        except VideoEncoderError as exc:
            self._fallback_to_software(exc)
            fallback_config = EncoderConfig(
                width=config.width,
                height=config.height,
                codec_name=self.encoder_name,
            )
            self._stream_encoder = self._create_encoder(fallback_config)
            self._stream_config = fallback_config
            return self._stream_encoder

        self._stream_config = config
        return self._stream_encoder

    # ------------------------------------------------------------------
    # Image preparation helpers
    # ------------------------------------------------------------------

    def _prepare_frame(self, image_data: bytes) -> tuple:
        try:
            frame = decode_compressed_frame(image_data)
        except VideoEncoderError as exc:
            raise TransformError(f"Failed to decode image bytes: {exc}") from exc

        target_width, target_height = calculate_downscale_dimensions(
            frame.width, frame.height, self.max_dimension
        )
        if (target_width, target_height) != (frame.width, frame.height):
            frame = frame.reformat(width=target_width, height=target_height)
        return frame, target_width, target_height


__all__ = ["ImageToVideoTransformer"]
