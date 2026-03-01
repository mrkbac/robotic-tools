"""Transform CompressedImage to CompressedVideo using a persistent PyAV encoder."""

from __future__ import annotations

import logging
from typing import Any

from pymcap_cli.image_utils import (
    EncoderConfig,
    VideoEncoder,
    VideoEncoderError,
    calculate_downscale_dimensions,
    decode_compressed_frame,
    detect_encoder,
)

from . import Transformer, TransformError

logger = logging.getLogger(__name__)

# Software codec mapping for codecs that don't use detect_encoder()
_SOFTWARE_CODEC_MAP: dict[str, str] = {
    "h264": "libx264",
    "h265": "libx265",
    "vp9": "libvpx-vp9",
    "av1": "libaom-av1",
}


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

        self.encoder_name = self._resolve_encoder()
        logger.info("Using encoder: %s", self.encoder_name)

        self._stream_config: EncoderConfig | None = None
        self._stream_encoder: VideoEncoder | None = None

    def get_input_schema(self) -> str:
        return "sensor_msgs/msg/CompressedImage"

    def get_output_schema(self) -> str:
        return "foxglove_msgs/CompressedVideo"

    def transform(self, message: Any) -> dict[str, Any]:
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
            software_encoder = self._get_software_encoder()
            if self.encoder_name != software_encoder:
                logger.warning(
                    "Encoder %s failed (%s); falling back to %s",
                    self.encoder_name,
                    exc,
                    software_encoder,
                )
                self.encoder_name = software_encoder
                self._stream_encoder = None
                self._stream_config = None
                fallback_config = EncoderConfig(
                    width=target_width,
                    height=target_height,
                    codec_name=self.encoder_name,
                )
                encoder = self._ensure_stream_encoder(fallback_config)
                try:
                    encoded = encoder.encode(frame)
                except VideoEncoderError as inner:
                    raise TransformError(str(inner)) from inner
            else:
                raise TransformError(str(exc)) from exc

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

    def _ensure_stream_encoder(self, config: EncoderConfig) -> VideoEncoder:
        if self._stream_encoder and self._stream_config == config:
            return self._stream_encoder

        self._stream_encoder = None
        self._stream_config = None
        try:
            self._stream_encoder = VideoEncoder(
                width=config.width,
                height=config.height,
                codec_name=config.codec_name,
                quality=self.quality,
                target_fps=self.target_fps,
                gop_size=self.gop_size,
                preset=self.preset,
            )
        except VideoEncoderError as exc:
            software_encoder = self._get_software_encoder()
            if config.codec_name != software_encoder:
                logger.warning(
                    "Encoder %s unavailable (%s); falling back to %s",
                    config.codec_name,
                    exc,
                    software_encoder,
                )
                self.encoder_name = software_encoder
                fallback_config = EncoderConfig(
                    width=config.width,
                    height=config.height,
                    codec_name=software_encoder,
                )
                self._stream_encoder = VideoEncoder(
                    width=fallback_config.width,
                    height=fallback_config.height,
                    codec_name=fallback_config.codec_name,
                    quality=self.quality,
                    target_fps=self.target_fps,
                    gop_size=self.gop_size,
                    preset=self.preset,
                )
                self._stream_config = fallback_config
                return self._stream_encoder
            self._stream_encoder = None
            self._stream_config = None
            raise TransformError(str(exc)) from exc

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

    # ------------------------------------------------------------------
    # Encoder detection logic
    # ------------------------------------------------------------------

    def _resolve_encoder(self) -> str:
        if self.codec in {"h264", "h265"} and self.use_hardware:
            try:
                return detect_encoder(self.codec)  # type: ignore[arg-type]
            except ValueError:
                pass
        return self._get_software_encoder()

    def _get_software_encoder(self) -> str:
        encoder = _SOFTWARE_CODEC_MAP.get(self.codec)
        if not encoder:
            raise TransformError(
                f"Unsupported codec '{self.codec}'. Supported: {', '.join(_SOFTWARE_CODEC_MAP)}"
            )
        return encoder


__all__ = ["ImageToVideoTransformer"]
