"""Decoder factories for decompressing compressed ROS topics.

Provides two independent factories for use with ``read_message_decoded``:

- ``VideoDecompressFactory`` — CompressedVideo → CompressedImage or Image (channel-aware)
- ``PointCloudDecompressFactory`` — CompressedPointCloud2 → PointCloud2 (schema-only)

Usage::

    from small_mcap import read_message_decoded
    from pymcap_cli.encoding.decompress import VideoDecompressFactory, PointCloudDecompressFactory

    with open("compressed.mcap", "rb") as f:
        factories = [VideoDecompressFactory(), PointCloudDecompressFactory()]
        for msg in read_message_decoded(f, decoder_factories=factories):
            print(msg.channel.topic, type(msg.decoded_message))
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

from pymcap_cli.encoding.encoder_common import EncoderMode

if TYPE_CHECKING:
    from collections.abc import Callable

    from av.video.frame import VideoFrame
    from small_mcap.records import Channel

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

_COMPRESSED_VIDEO_SCHEMA = "foxglove_msgs/msg/CompressedVideo"

COMPRESSED_IMAGE = """\
std_msgs/Header header
string format
uint8[] data

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""

IMAGE = """\
std_msgs/Header header
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""

POINTCLOUD2 = """\
std_msgs/Header header
uint32 height
uint32 width
sensor_msgs/PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] data
bool is_dense

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(obj: Any, key: str) -> Any:
    """Get a field from either a dict or an object with attributes."""
    try:
        return obj[key]
    except (TypeError, KeyError):
        return getattr(obj, key)


class _SchemaProtocol:
    id: int
    name: str
    encoding: str
    data: bytes


# ---------------------------------------------------------------------------
# Video decompression
# ---------------------------------------------------------------------------


class _VideoDecompressor:
    """Per-channel H.264/H.265 → CompressedImage or raw Image decoder.

    Maintains a persistent video decoder context for proper P-frame decoding.
    """

    def __init__(
        self,
        video_format: Literal["compressed", "raw"],
        jpeg_quality: int,
        backend: EncoderMode,
    ) -> None:
        self._video_format = video_format
        self._jpeg_quality = jpeg_quality
        self._backend = backend
        self._decoder: Any = None
        self._jpeg_encoder: Any = None

    def _ensure_decoder(self, codec_format: str) -> Any:
        if self._decoder is not None:
            return self._decoder
        from pymcap_cli.encoding.video_decoder import create_video_decoder  # noqa: PLC0415

        self._decoder = create_video_decoder(codec_format, mode=self._backend)
        return self._decoder

    def _ensure_jpeg_encoder(self, width: int, height: int) -> Any:
        if self._jpeg_encoder is not None:
            return self._jpeg_encoder
        import av  # noqa: PLC0415

        self._jpeg_encoder = av.CodecContext.create("mjpeg", "w")
        self._jpeg_encoder.width = width
        self._jpeg_encoder.height = height
        self._jpeg_encoder.pix_fmt = "yuvj420p"
        self._jpeg_encoder.time_base = Fraction(1, 1000)
        self._jpeg_encoder.options = {"q:v": str(max(1, 31 - self._jpeg_quality * 31 // 100))}
        self._jpeg_encoder.open()
        return self._jpeg_encoder

    def _frame_to_jpeg(self, frame: VideoFrame) -> bytes:
        encoder = self._ensure_jpeg_encoder(frame.width, frame.height)
        reformatted = frame.reformat(format="yuvj420p")
        reformatted.pts = 0
        packets = encoder.encode(reformatted)
        return b"".join(bytes(p) for p in packets)

    def _frame_to_raw(self, frame: VideoFrame) -> tuple[bytes, int, int, str, int]:
        rgb_frame = frame.reformat(format="rgb24")
        data = rgb_frame.to_ndarray().tobytes()
        return data, rgb_frame.height, rgb_frame.width, "rgb8", rgb_frame.width * 3

    def decode(self, msg: Any) -> dict[str, Any] | None:
        codec_format = _get(msg, "format")
        video_data = _get(msg, "data")
        if isinstance(video_data, memoryview):
            video_data = bytes(video_data)

        frame = self._ensure_decoder(codec_format).decode(video_data)
        if frame is None:
            return None

        timestamp = _get(msg, "timestamp")
        header = {
            "stamp": {"sec": _get(timestamp, "sec"), "nanosec": _get(timestamp, "nanosec")},
            "frame_id": _get(msg, "frame_id"),
        }

        if self._video_format == "compressed":
            return {"header": header, "format": "jpeg", "data": self._frame_to_jpeg(frame)}

        data, height, width, encoding, step = self._frame_to_raw(frame)
        return {
            "header": header,
            "height": height,
            "width": width,
            "encoding": encoding,
            "is_bigendian": 0,
            "step": step,
            "data": data,
        }


class VideoDecompressFactory:
    """Channel-aware decoder factory: CompressedVideo → CompressedImage or Image.

    Creates a separate video decoder per channel for proper P-frame handling.
    """

    channel_aware = True

    def __init__(
        self,
        *,
        video_format: Literal["compressed", "raw"] = "compressed",
        jpeg_quality: int = 90,
        backend: EncoderMode = EncoderMode.AUTO,
    ) -> None:
        self._video_format = video_format
        self._jpeg_quality = jpeg_quality
        self._backend = backend

        from mcap_ros2_support_fast.decoder import DecoderFactory  # noqa: PLC0415

        self._cdr_factory = DecoderFactory()
        self._decompressors: dict[int, _VideoDecompressor] = {}

    def decoder_for(
        self,
        message_encoding: str,
        schema: _SchemaProtocol | None,
        channel: Channel,
    ) -> Callable[[bytes | memoryview], Any] | None:
        if schema is None or schema.name != _COMPRESSED_VIDEO_SCHEMA:
            return None

        cdr_decoder = self._cdr_factory.decoder_for(message_encoding, schema)
        if cdr_decoder is None:
            return None

        if channel.id not in self._decompressors:
            self._decompressors[channel.id] = _VideoDecompressor(
                self._video_format, self._jpeg_quality, self._backend
            )
        video_dec = self._decompressors[channel.id]

        def _decode(data: bytes | memoryview) -> Any:
            return video_dec.decode(cdr_decoder(data))

        return _decode
