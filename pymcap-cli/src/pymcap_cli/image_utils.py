"""Shared image decoding and encoder utilities for video/roscompress commands."""

import platform
import threading
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
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
    from pureini import CompressionOption, EncodingInfo, EncodingOptions

COMPRESSED_SCHEMAS = {"sensor_msgs/msg/CompressedImage", "sensor_msgs/CompressedImage"}
RAW_SCHEMAS = {"sensor_msgs/msg/Image", "sensor_msgs/Image"}
IMAGE_SCHEMAS = COMPRESSED_SCHEMAS | RAW_SCHEMAS
POINTCLOUD2_SCHEMAS = {"sensor_msgs/msg/PointCloud2", "sensor_msgs/PointCloud2"}

FOXGLOVE_COMPRESSED_VIDEO = """builtin_interfaces/Time timestamp
string frame_id
uint8[] data
string format

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""

COMPRESSED_POINTCLOUD2 = """\
std_msgs/Header header
uint32 height
uint32 width
sensor_msgs/PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] compressed_data
bool is_dense
string format

================================================================================
MSG: sensor_msgs/PointField
uint8 INT8    = 1
uint8 UINT8   = 2
uint8 INT16   = 3
uint8 UINT16  = 4
uint8 INT32   = 5
uint8 UINT32  = 6
uint8 FLOAT32 = 7
uint8 FLOAT64 = 8
string name
uint32 offset
uint8  datatype
uint32 count

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""


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


# ---------------------------------------------------------------------------
# Downscale / encoder detection
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
# Codec / encoder enums and mappings
# ---------------------------------------------------------------------------


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

# Platform → hardware backend probe order.
_HW_PROBE_ORDER: dict[str, list[str]] = {
    "Darwin": ["videotoolbox"],
    "Linux": ["nvenc", "vaapi"],
}


def _detect_best_hardware_encoder(codec: str) -> str | None:
    """Probe for the best available hardware encoder, or return None."""
    hw = HARDWARE_CODEC_MAP.get(codec)
    if not hw:
        return None
    for backend in _HW_PROBE_ORDER.get(platform.system(), []):
        encoder = hw.get(backend)
        if encoder and test_encoder(encoder):
            return encoder
    return None


def get_software_encoder(codec: str) -> str:
    """Return the software encoder name for *codec*.

    Raises:
        ValueError: If *codec* is not in SOFTWARE_CODEC_MAP.
    """
    sw = SOFTWARE_CODEC_MAP.get(codec)
    if not sw:
        raise ValueError(f"Unsupported codec '{codec}'. Supported: {', '.join(SOFTWARE_CODEC_MAP)}")
    return sw


def resolve_encoder(codec: str, *, use_hardware: bool = True) -> str:
    """Pick the best available encoder for *codec*.

    Attempts hardware detection first (when *use_hardware* is True and the
    codec has known hardware encoders), then falls back to the software
    encoder from SOFTWARE_CODEC_MAP.

    Raises:
        ValueError: If *codec* is not in SOFTWARE_CODEC_MAP.
    """
    if use_hardware:
        hw = _detect_best_hardware_encoder(codec)
        if hw:
            return hw
    return get_software_encoder(codec)


def resolve_encoder_for_backend(codec: str, backend: str) -> str:
    """Pick the encoder for *codec* using the specified *backend*.

    *backend* must be one of the ``EncoderBackend`` values: ``"auto"``,
    ``"software"``, ``"videotoolbox"``, ``"nvenc"``, or ``"vaapi"``.

    Raises:
        VideoEncoderError: If the encoder is unavailable or the backend is unknown.
    """
    if backend == "auto":
        try:
            return resolve_encoder(codec)
        except ValueError as exc:
            raise VideoEncoderError(str(exc)) from exc

    if backend == "software":
        try:
            return get_software_encoder(codec)
        except ValueError as exc:
            raise VideoEncoderError(str(exc)) from exc

    # Explicit hardware backend
    hw = HARDWARE_CODEC_MAP.get(codec, {})
    encoder = hw.get(backend)
    if not encoder:
        raise VideoEncoderError(f"Hardware encoder '{backend}' not available for codec: {codec}")
    if not test_encoder(encoder):
        raise VideoEncoderError(
            f"Hardware encoder '{encoder}' not available on this system. Try --encoder software."
        )
    return encoder


# ---------------------------------------------------------------------------
# Video encoder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Configuration for a video encoder."""

    width: int
    height: int
    codec_name: str


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
        # CRF -> bitrate: 5 Mbps baseline scaled by quality and resolution
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


# ---------------------------------------------------------------------------
# Point cloud compression
# ---------------------------------------------------------------------------


def _build_encoding_info(
    msg: object,
    encoding_opt: "EncodingOptions",
    compression_opt: "CompressionOption",
    resolution: float,
) -> "EncodingInfo":
    """Build pureini EncodingInfo from a decoded ROS2 PointCloud2 message."""
    from pureini import EncodingInfo, FieldType, PointField  # noqa: PLC0415

    info = EncodingInfo()
    info.width = msg.width  # type: ignore[attr-defined]
    info.height = msg.height  # type: ignore[attr-defined]
    info.point_step = msg.point_step  # type: ignore[attr-defined]
    info.encoding_opt = encoding_opt
    info.compression_opt = compression_opt

    info.fields = []
    for ros_field in msg.fields:  # type: ignore[attr-defined]
        field = PointField(
            name=ros_field.name,
            offset=ros_field.offset,
            type=FieldType(ros_field.datatype),
            resolution=resolution if ros_field.datatype == 7 else None,
        )
        info.fields.append(field)

    return info


class PointCloudCompressor:
    """Pureini-based point cloud compressor with encoder caching.

    Lazily imports pureini, maps string encoding/compression options to
    pureini enums, and caches the ``PointcloudEncoder`` per ``EncodingInfo``.
    """

    def __init__(
        self,
        encoding: str = "lossy",
        compression: str = "zstd",
        resolution: float = 0.01,
    ) -> None:
        from pureini import CompressionOption, EncodingOptions, PointcloudEncoder  # noqa: PLC0415

        encoding_map = {
            "lossy": EncodingOptions.LOSSY,
            "lossless": EncodingOptions.LOSSLESS,
            "none": EncodingOptions.NONE,
        }
        compression_map = {
            "zstd": CompressionOption.ZSTD,
            "lz4": CompressionOption.LZ4,
            "none": CompressionOption.NONE,
        }

        if encoding not in encoding_map:
            raise ValueError(
                f"Unknown encoding '{encoding}'. Choose from: {', '.join(encoding_map)}"
            )
        if compression not in compression_map:
            raise ValueError(
                f"Unknown compression '{compression}'. Choose from: {', '.join(compression_map)}"
            )

        self._encoding_opt = encoding_map[encoding]
        self._compression_opt = compression_map[compression]
        self._resolution = resolution
        self._PointcloudEncoder = PointcloudEncoder

        self._cached_info: EncodingInfo | None = None
        self._cached_encoder: PointcloudEncoder | None = None

    def compress(self, msg: object) -> bytes:
        """Compress a decoded ROS2 PointCloud2 message and return raw bytes."""
        info = _build_encoding_info(
            msg, self._encoding_opt, self._compression_opt, self._resolution
        )
        if self._cached_info != info:
            self._cached_info = info
            self._cached_encoder = self._PointcloudEncoder(info)
        return self._cached_encoder.encode(bytes(msg.data))  # type: ignore[attr-defined, union-attr]
