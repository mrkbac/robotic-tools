"""Transform CompressedImage to CompressedVideo using a persistent PyAV encoder."""

from __future__ import annotations

import io
import logging
import platform
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal, cast

import av
import av.error

if TYPE_CHECKING:
    from av.video.codeccontext import VideoCodecContext

from . import Transformer, TransformError

logger = logging.getLogger(__name__)
# Extended from av/video/__init__.pyi to include all encoders used in this module
_VideoCodecName = Literal[
    "gif",
    "h264",
    "hevc",
    "libvpx",
    "libx264",
    "libx265",
    "libvpx-vp9",
    "libaom-av1",
    "h264_videotoolbox",
    "hevc_videotoolbox",
    "h264_nvenc",
    "hevc_nvenc",
    "h264_vaapi",
    "hevc_vaapi",
    "mpeg4",
    "png",
    "qtrle",
]


@dataclass(frozen=True, slots=True)
class _EncoderConfig:
    """Immutable configuration for the active PyAV encoder."""

    width: int
    height: int
    codec_name: _VideoCodecName


class _PyAVEncoder:
    """Wrapper around a PyAV codec context for streaming video encoding."""

    def __init__(
        self,
        config: _EncoderConfig,
        quality: int,
        preset: str,
        gop_size: int,
        target_fps: float,
    ) -> None:
        self.config = config
        self._target_fps = max(target_fps, 1.0)
        self._frame_index = 0
        try:
            self._context: VideoCodecContext = cast(
                "VideoCodecContext", av.CodecContext.create(self.config.codec_name, "w")
            )
        except av.error.FFmpegError as exc:  # pragma: no cover - construction should rarely fail
            raise TransformError(
                f"Failed to initialise codec {self.config.codec_name}: {exc}"
            ) from exc

        fps_int = max(round(self._target_fps), 1)
        self._context.width = self.config.width
        self._context.height = self.config.height
        self._context.pix_fmt = "yuv420p"
        self._context.time_base = Fraction(1, fps_int)
        self._context.framerate = Fraction(fps_int, 1)
        self._context.gop_size = gop_size
        self._apply_options(quality, preset)

        try:
            self._context.open()
        except av.error.FFmpegError as exc:  # pragma: no cover
            raise TransformError(f"Failed to open codec {self.config.codec_name}: {exc}") from exc

    def _apply_options(self, quality: int, preset: str) -> None:
        options: dict[str, str] = {}
        if self.config.codec_name in {"libx264", "h264_videotoolbox"}:
            options["preset"] = preset
            options["crf"] = str(quality)
            if self.config.codec_name == "libx264":
                options["tune"] = "zerolatency"
        elif self.config.codec_name in {"libx265", "hevc_videotoolbox"}:
            options["preset"] = preset
            options["crf"] = str(quality)
        elif self.config.codec_name == "libvpx-vp9":
            options["cpu-used"] = "4"
            options["crf"] = str(quality)
        elif self.config.codec_name == "libaom-av1":
            options["cpu-used"] = "6"
            options["crf"] = str(quality)
        if options:
            self._context.options = options

    def encode(self, frame: av.VideoFrame) -> bytes:
        if (
            frame.width != self.config.width
            or frame.height != self.config.height
            or frame.format.name != "rgb24"
        ):
            frame = frame.reformat(
                width=self.config.width, height=self.config.height, format="rgb24"
            )

        frame = frame.reformat(format=self._context.pix_fmt)
        frame.pts = self._frame_index
        self._frame_index += 1

        try:
            packets = list(self._context.encode(frame))
        except av.error.FFmpegError as exc:  # pragma: no cover
            raise TransformError(f"Encoding error: {exc}") from exc

        if not packets:
            # Some codecs buffer internally; try flushing a packet without closing the stream
            try:
                packets = list(self._context.encode(None))
            except av.error.FFmpegError as exc:  # pragma: no cover
                raise TransformError(f"Encoder flush error: {exc}") from exc

        data = b"".join(bytes(packet) for packet in packets)
        if not data:
            raise TransformError("Encoder produced empty packet")
        return data

    def close(self) -> None:
        pass


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

        self.encoder_name: _VideoCodecName = self._detect_encoder()
        logger.info("Using encoder: %s", self.encoder_name)

        self._stream_config: _EncoderConfig | None = None
        self._stream_encoder: _PyAVEncoder | None = None

    def get_input_schema(self) -> str:
        return "sensor_msgs/msg/CompressedImage"

    def get_output_schema(self) -> str:
        return "foxglove_msgs/CompressedVideo"

    def transform(self, message: Any) -> dict[str, Any]:
        if not message.data:
            raise TransformError(f"Empty image data, {message}")

        frame, target_width, target_height = self._prepare_frame(message.data)

        config = _EncoderConfig(
            width=target_width, height=target_height, codec_name=self.encoder_name
        )
        encoder = self._ensure_stream_encoder(config)

        try:
            encoded = encoder.encode(frame)
        except TransformError as exc:
            software_encoder = self._get_software_encoder()
            if self.encoder_name != software_encoder:
                logger.warning(
                    "Encoder %s failed (%s); falling back to %s",
                    self.encoder_name,
                    exc,
                    software_encoder,
                )
                self.encoder_name = software_encoder
                self._close_stream_encoder()
                fallback_config = _EncoderConfig(
                    width=target_width,
                    height=target_height,
                    codec_name=self.encoder_name,
                )
                encoder = self._ensure_stream_encoder(fallback_config)
                encoded = encoder.encode(frame)
            else:
                self._close_stream_encoder()
                raise

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

    def _ensure_stream_encoder(self, config: _EncoderConfig) -> _PyAVEncoder:
        if self._stream_encoder and self._stream_config == config:
            return self._stream_encoder

        self._close_stream_encoder()
        try:
            self._stream_encoder = _PyAVEncoder(
                config=config,
                quality=self.quality,
                preset=self.preset,
                gop_size=self.gop_size,
                target_fps=self.target_fps,
            )
        except TransformError as exc:
            software_encoder = self._get_software_encoder()
            if config.codec_name != software_encoder:
                logger.warning(
                    "Encoder %s unavailable (%s); falling back to %s",
                    config.codec_name,
                    exc,
                    software_encoder,
                )
                self.encoder_name = software_encoder
                fallback_config = _EncoderConfig(
                    width=config.width,
                    height=config.height,
                    codec_name=software_encoder,
                )
                self._stream_encoder = _PyAVEncoder(
                    config=fallback_config,
                    quality=self.quality,
                    preset=self.preset,
                    gop_size=self.gop_size,
                    target_fps=self.target_fps,
                )
                self._stream_config = fallback_config
                return self._stream_encoder
            self._stream_encoder = None
            self._stream_config = None
            raise

        self._stream_config = config
        return self._stream_encoder

    def _close_stream_encoder(self) -> None:
        if self._stream_encoder:
            self._stream_encoder.close()
        self._stream_encoder = None
        self._stream_config = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        self._close_stream_encoder()

    # ------------------------------------------------------------------
    # Image preparation helpers
    # ------------------------------------------------------------------

    def _prepare_frame(self, image_data: bytes) -> tuple[av.VideoFrame, int, int]:
        frame = self._decode_image(image_data)
        target_width, target_height = self._calculate_downscale_dimensions(
            frame.width, frame.height
        )
        if (target_width, target_height) != (frame.width, frame.height):
            frame = frame.reformat(width=target_width, height=target_height)
        return frame, target_width, target_height

    def _decode_image(self, image_data: bytes) -> av.VideoFrame:
        try:
            with av.open(io.BytesIO(image_data), mode="r", format="image2") as container:
                for frame in container.decode(video=0):
                    return frame.reformat(format="rgb24")
        except (av.error.FFmpegError, ValueError) as exc:
            raise TransformError(f"Failed to decode image bytes: {exc}") from exc

        raise TransformError("Decoder did not produce any frames")

    def _calculate_downscale_dimensions(self, width: int, height: int) -> tuple[int, int]:
        def ensure_even(value: int) -> int:
            return value if value % 2 == 0 else max(value - 1, 2)

        if width <= self.max_dimension and height <= self.max_dimension:
            return ensure_even(width), ensure_even(height)

        aspect_ratio = width / height
        if width > height:
            new_width = self.max_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = self.max_dimension
            new_width = int(new_height * aspect_ratio)

        return ensure_even(new_width), ensure_even(new_height)

    # ------------------------------------------------------------------
    # Encoder detection logic
    # ------------------------------------------------------------------

    def _detect_encoder(self) -> _VideoCodecName:
        if self.codec in {"h264", "h265"} and self.use_hardware:
            system = platform.system()
            if system == "Darwin":
                preferred: _VideoCodecName = (
                    "h264_videotoolbox" if self.codec == "h264" else "hevc_videotoolbox"
                )
                if self._test_encoder(preferred):
                    return preferred
            elif system == "Linux":
                preferred_hw: list[_VideoCodecName] = (
                    ["h264_nvenc", "h264_vaapi"]
                    if self.codec == "h264"
                    else ["hevc_nvenc", "hevc_vaapi"]
                )
                for encoder in preferred_hw:
                    if self._test_encoder(encoder):
                        return encoder

        return self._get_software_encoder()

    def _get_software_encoder(self) -> _VideoCodecName:
        codec_map: dict[str, _VideoCodecName] = {
            "h264": "libx264",
            "h265": "libx265",
            "vp9": "libvpx-vp9",
            "av1": "libaom-av1",
        }
        encoder = codec_map.get(self.codec)
        if not encoder:
            raise TransformError(
                f"Unsupported codec '{self.codec}'. Supported: {', '.join(codec_map)}"
            )
        return encoder

    def _test_encoder(self, encoder: _VideoCodecName) -> bool:
        try:
            av.CodecContext.create(encoder, "w")
        except (av.error.FFmpegError, ValueError):
            return False
        return True


__all__ = ["ImageToVideoTransformer"]
