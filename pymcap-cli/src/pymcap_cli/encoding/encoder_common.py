"""Shared encoder types, config, and option builders — no PyAV dependency."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# ROS schema sets
# ---------------------------------------------------------------------------

COMPRESSED_SCHEMAS = {"sensor_msgs/msg/CompressedImage", "sensor_msgs/CompressedImage"}
RAW_SCHEMAS = {"sensor_msgs/msg/Image", "sensor_msgs/Image"}
IMAGE_SCHEMAS = COMPRESSED_SCHEMAS | RAW_SCHEMAS

FOXGLOVE_COMPRESSED_VIDEO = """builtin_interfaces/Time timestamp
string frame_id
uint8[] data
string format

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""

# ---------------------------------------------------------------------------
# Errors / enums
# ---------------------------------------------------------------------------


class VideoEncoderError(Exception):
    """Raised when encoding fails."""


class VideoCodec(str, Enum):
    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    AV1 = "av1"


class EncoderBackend(str, Enum):
    AUTO = "auto"
    SOFTWARE = "software"
    VIDEOTOOLBOX = "videotoolbox"
    NVENC = "nvenc"
    VAAPI = "vaapi"


class EncoderMode(str, Enum):
    """Which video encoder backend to use."""

    AUTO = "auto"
    PYAV = "pyav"
    FFMPEG_CLI = "ffmpeg-cli"


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Configuration for a video encoder."""

    width: int
    height: int
    codec_name: str


class VideoEncoderProtocol(Protocol):
    """Structural interface shared by VideoEncoder and SubprocessVideoEncoder."""

    config: EncoderConfig

    def encode(self, frame: Any) -> bytes | None: ...

    def flush(self) -> bytes | None: ...


# Mapping from short codec name to software (CPU) encoder name.
SOFTWARE_CODEC_MAP: dict[str, str] = {
    "h264": "libx264",
    "h265": "libx265",
    "vp9": "libvpx-vp9",
    "av1": "libaom-av1",
}

# Mapping from short codec name to hardware encoder names per backend.
HARDWARE_CODEC_MAP: dict[str, dict[str, str]] = {
    "h264": {
        "videotoolbox": "h264_videotoolbox",
        "nvenc": "h264_nvenc",
        "vaapi": "h264_vaapi",
    },
    "h265": {
        "videotoolbox": "hevc_videotoolbox",
        "nvenc": "hevc_nvenc",
        "vaapi": "hevc_vaapi",
    },
}


def get_software_encoder(codec: str) -> str:
    """Return the software encoder name for *codec*.

    Raises:
        ValueError: If *codec* is not in SOFTWARE_CODEC_MAP.
    """
    sw = SOFTWARE_CODEC_MAP.get(codec)
    if not sw:
        raise ValueError(f"Unsupported codec '{codec}'. Supported: {', '.join(SOFTWARE_CODEC_MAP)}")
    return sw


def build_encoder_options(
    codec_name: str,
    quality: int,
    width: int,
    height: int,
    *,
    preset: str | None = None,
) -> tuple[dict[str, str], int | None]:
    """Build codec-specific encoder options from user-facing quality (CRF).

    Returns (options_dict, bit_rate_or_none).
    """
    if codec_name == "libx264":
        return {
            "crf": str(quality),
            "preset": preset or "superfast",
            "tune": "zerolatency",
        }, None
    if codec_name == "libx265":
        return {
            "crf": str(quality),
            "preset": preset or "superfast",
        }, None
    if codec_name in {"h264_videotoolbox", "hevc_videotoolbox"}:
        pixel_scale = (width * height) / (1920 * 1080)
        bit_rate = int(5_000_000 * (2 ** ((28 - quality) / 6)) * pixel_scale)
        return {}, bit_rate
    if codec_name in {"h264_nvenc", "hevc_nvenc"}:
        return {"rc": "vbr", "cq": str(quality)}, None
    if codec_name in {"h264_vaapi", "hevc_vaapi"}:
        return {"qp": str(quality)}, None
    if codec_name == "libvpx-vp9":
        opts: dict[str, str] = {"cpu-used": "4", "crf": str(quality)}
        if preset:
            opts["deadline"] = preset
        return opts, None
    if codec_name == "libaom-av1":
        opts = {"cpu-used": "6", "crf": str(quality)}
        if preset:
            opts["usage"] = preset
        return opts, None
    return {}, None


def get_encoder_options(codec: VideoCodec, encoder_name: str) -> dict[str, str]:
    """Return encoder preset options for bitrate-mode encoding (e.g. file output)."""
    options: dict[str, str] = {}
    if "nvenc" in encoder_name:
        options["preset"] = "p4"
    elif (
        codec in (VideoCodec.H264, VideoCodec.H265) and "libx264" in encoder_name
    ) or "libx265" in encoder_name:
        options["preset"] = "medium"
    return options


# ---------------------------------------------------------------------------
# Dimension helpers
# ---------------------------------------------------------------------------


def calculate_downscale_dimensions(width: int, height: int, max_dimension: int) -> tuple[int, int]:
    """Downscale dimensions to fit within max_dimension, preserving aspect ratio.

    Ensures both dimensions are even (required for yuv420p).
    """

    def ensure_even(value: int) -> int:
        return value if value % 2 == 0 else max(value - 1, 2)

    if width <= max_dimension and height <= max_dimension:
        return ensure_even(width), ensure_even(height)

    aspect_ratio = width / height
    if width > height:
        new_width = max_dimension
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_dimension
        new_width = int(new_height * aspect_ratio)

    return ensure_even(new_width), ensure_even(new_height)


# ---------------------------------------------------------------------------
# Encoder resolution
# ---------------------------------------------------------------------------

# Platform → hardware backend probe order.
_HW_PROBE_ORDER: dict[str, list[str]] = {
    "Darwin": ["videotoolbox"],
    "Linux": ["nvenc", "vaapi"],
}


def resolve_encoder(
    codec: str,
    *,
    test_fn: Callable[[str], bool],
    use_hardware: bool = True,
) -> str:
    """Pick the best available encoder for *codec*.

    Probes hardware encoders first (when *use_hardware* is True), then
    falls back to the software encoder. Uses *test_fn* to check whether
    a given encoder name is available on the system.

    Raises:
        ValueError: If no encoder is found for *codec*.
    """
    if use_hardware:
        hw = HARDWARE_CODEC_MAP.get(codec)
        if hw:
            for backend_name in _HW_PROBE_ORDER.get(platform.system(), []):
                encoder = hw.get(backend_name)
                if encoder and test_fn(encoder):
                    return encoder
    return get_software_encoder(codec)


def resolve_encoder_for_backend(
    codec: str,
    backend: str,
    *,
    test_fn: Callable[[str], bool],
) -> str:
    """Pick the encoder for *codec* using the specified *backend*.

    *backend* must be one of the ``EncoderBackend`` values: ``"auto"``,
    ``"software"``, ``"videotoolbox"``, ``"nvenc"``, or ``"vaapi"``.

    Raises:
        VideoEncoderError: If the encoder is unavailable or the backend is unknown.
    """
    if backend == "auto":
        try:
            return resolve_encoder(codec, test_fn=test_fn)
        except ValueError as exc:
            raise VideoEncoderError(str(exc)) from exc

    if backend == "software":
        try:
            return get_software_encoder(codec)
        except ValueError as exc:
            raise VideoEncoderError(str(exc)) from exc

    # Explicit hardware backend.
    hw = HARDWARE_CODEC_MAP.get(codec, {})
    encoder = hw.get(backend)
    if not encoder:
        raise VideoEncoderError(f"Hardware encoder '{backend}' not available for codec: {codec}")
    if not test_fn(encoder):
        raise VideoEncoderError(
            f"Hardware encoder '{encoder}' not available on this system. Try --encoder software."
        )
    return encoder
