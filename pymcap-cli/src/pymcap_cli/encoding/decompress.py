"""Decoder factory for decompressing CompressedVideo topics.

Provides ``VideoDecompressFactory`` for use with ``read_message_decoded``.
Uses ``VideoDecompressorProtocol`` — no direct ``av`` or ``subprocess`` imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pymcap_cli.encoding.encoder_common import EncoderMode

if TYPE_CHECKING:
    from collections.abc import Callable

    from small_mcap.records import Channel

    from pymcap_cli.encoding.video_protocols import DecompressedFrame

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
# VideoDecompressFactory
# ---------------------------------------------------------------------------


class VideoDecompressFactory:
    """Channel-aware decoder factory: CompressedVideo → CompressedImage or Image.

    Creates a separate ``VideoDecompressorProtocol`` per channel for proper
    P-frame handling. No direct ``av`` or ``subprocess`` imports.
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
        self._decompressors: dict[int, Any] = {}

    def flush_all(self) -> list[Any]:
        """Flush all decompressors and return remaining frames."""
        frames: list[Any] = []
        for decompressor in self._decompressors.values():
            frames.extend(decompressor.flush())
        return frames

    def _get_decompressor(self, channel_id: int) -> Any:
        if channel_id not in self._decompressors:
            from pymcap_cli.encoding.video_factory import create_video_decompressor  # noqa: PLC0415

            self._decompressors[channel_id] = create_video_decompressor(
                video_format=self._video_format,
                jpeg_quality=self._jpeg_quality,
                mode=self._backend,
            )
        return self._decompressors[channel_id]

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

        decompressor = self._get_decompressor(channel.id)

        def _decode(data: bytes | memoryview) -> dict[str, Any] | None:
            decoded = cdr_decoder(data)
            codec = _get(decoded, "format")
            video_data = _get(decoded, "data")
            if isinstance(video_data, memoryview):
                video_data = bytes(video_data)

            frame: DecompressedFrame | None = decompressor.decompress(video_data, codec)
            if frame is None:
                return None

            timestamp = _get(decoded, "timestamp")
            header = {
                "stamp": {
                    "sec": _get(timestamp, "sec"),
                    "nanosec": _get(timestamp, "nanosec"),
                },
                "frame_id": _get(decoded, "frame_id"),
            }

            if frame.is_jpeg:
                return {"header": header, "format": "jpeg", "data": frame.data}

            return {
                "header": header,
                "height": frame.height,
                "width": frame.width,
                "encoding": "rgb8",
                "is_bigendian": 0,
                "step": frame.width * 3,
                "data": frame.data,
            }

        return _decode
