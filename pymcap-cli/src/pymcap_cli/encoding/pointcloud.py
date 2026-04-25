"""Pureini-based point cloud compression and decompression utilities.

Also exposes :class:`Pointcloud2DecoderFactory`, a small_mcap decoder
factory that decodes ``sensor_msgs/PointCloud2`` payloads directly into a
structured numpy array via :func:`pointcloud2.read_points`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from mcap_ros2_support_fast.decoder import DecoderFactory as Ros2DecoderFactory

if TYPE_CHECKING:
    from collections.abc import Callable

    from pointcloud2.messages import Pointcloud2Msg
    from pureini import CompressionOption, EncodingInfo, EncodingOptions

POINTCLOUD2_SCHEMAS = {"sensor_msgs/msg/PointCloud2", "sensor_msgs/PointCloud2"}

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


def _build_encoding_info(
    msg: Pointcloud2Msg,
    encoding_opt: EncodingOptions,
    compression_opt: CompressionOption,
    resolution: float,
) -> EncodingInfo:
    """Build pureini EncodingInfo from a decoded ROS2 PointCloud2 message."""
    from pureini import EncodingInfo, FieldType, PointField  # noqa: PLC0415

    info = EncodingInfo()
    info.width = msg.width
    info.height = msg.height
    info.point_step = msg.point_step
    info.encoding_opt = encoding_opt
    info.compression_opt = compression_opt

    info.fields = []
    for ros_field in msg.fields:
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

    def compress(self, msg: Pointcloud2Msg) -> bytes:
        """Compress a decoded ROS2 PointCloud2 message and return raw bytes."""
        info = _build_encoding_info(
            msg, self._encoding_opt, self._compression_opt, self._resolution
        )
        if self._cached_info != info:
            self._cached_info = info
            self._cached_encoder = self._PointcloudEncoder(info)
        return self._cached_encoder.encode(bytes(msg.data))  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Decompression
# ---------------------------------------------------------------------------

_COMPRESSED_POINTCLOUD2_SCHEMA = "point_cloud_interfaces/msg/CompressedPointCloud2"


def _get(obj: Any, key: str) -> Any:
    """Get a field from either a dict or an object with attributes."""
    try:
        return obj[key]
    except (TypeError, KeyError):
        return getattr(obj, key)


class PointCloudDecompressFactory:
    """Schema-only decoder factory: CompressedPointCloud2 → PointCloud2.

    Stateless — safe to share across all channels.
    """

    def __init__(self) -> None:
        from mcap_ros2_support_fast.decoder import DecoderFactory  # noqa: PLC0415
        from pureini import PointcloudDecoder  # noqa: PLC0415

        self._cdr_factory = DecoderFactory()
        self._pc_decoder: Any = PointcloudDecoder()

    def _decompress(self, msg: Any) -> dict[str, Any]:
        compressed_data = _get(msg, "compressed_data")
        if isinstance(compressed_data, memoryview):
            compressed_data = bytes(compressed_data)

        raw_bytes, _info = self._pc_decoder.decode(compressed_data)

        header_obj = _get(msg, "header")
        stamp_obj = _get(header_obj, "stamp")

        return {
            "header": {
                "stamp": {"sec": _get(stamp_obj, "sec"), "nanosec": _get(stamp_obj, "nanosec")},
                "frame_id": _get(header_obj, "frame_id"),
            },
            "height": _get(msg, "height"),
            "width": _get(msg, "width"),
            "fields": [
                {
                    "name": _get(f, "name"),
                    "offset": _get(f, "offset"),
                    "datatype": _get(f, "datatype"),
                    "count": _get(f, "count"),
                }
                for f in _get(msg, "fields")
            ],
            "is_bigendian": _get(msg, "is_bigendian"),
            "point_step": _get(msg, "point_step"),
            "row_step": _get(msg, "row_step"),
            "data": raw_bytes,
            "is_dense": _get(msg, "is_dense"),
        }

    def decoder_for(
        self,
        message_encoding: str,
        schema: Any | None,
    ) -> Callable[[bytes | memoryview], Any] | None:
        if schema is None or schema.name != _COMPRESSED_POINTCLOUD2_SCHEMA:
            return None

        cdr_decoder = self._cdr_factory.decoder_for(message_encoding, schema)
        if cdr_decoder is None:
            return None

        def _decode(data: bytes | memoryview) -> dict[str, Any]:
            return self._decompress(cdr_decoder(data))

        return _decode


class CompressedPointcloud2DecoderFactory:
    """Decode ``point_cloud_interfaces/CompressedPointCloud2`` → structured numpy array.

    Chains :class:`PointCloudDecompressFactory` (pureini decompression) with
    :func:`pointcloud2.read_points`, so the output has the same shape as the
    regular PointCloud2 path — a numpy structured array ready to be packed
    into an Arrow ``LIST<STRUCT<...>>``.

    Requires ``pureini``; construction raises ``ImportError`` if missing.
    """

    def __init__(self) -> None:
        self._decompress_factory = PointCloudDecompressFactory()

    def decoder_for(
        self,
        message_encoding: str,
        schema: Any | None,
    ) -> Callable[[bytes | memoryview], Any] | None:
        decoder = self._decompress_factory.decoder_for(message_encoding, schema)
        if decoder is None:
            return None

        from types import SimpleNamespace  # noqa: PLC0415

        from pointcloud2 import read_points  # noqa: PLC0415

        def _decode(data: bytes | memoryview) -> Any:
            cloud_dict = decoder(data)
            # read_points walks the message by attribute access.
            ns = SimpleNamespace(**{k: v for k, v in cloud_dict.items() if k != "header"})
            return read_points(ns, skip_nans=True)

        return _decode


_SCHEMA_ENCODING_ROS2 = "ros2msg"
_MESSAGE_ENCODING_CDR = "cdr"


class _SchemaProtocol(Protocol):
    id: int
    name: str
    encoding: str
    data: bytes


class Pointcloud2DecoderFactory:
    def __init__(self) -> None:
        self._ros2_decoder_factory = Ros2DecoderFactory()

    def decoder_for(
        self, message_encoding: str, schema: _SchemaProtocol | None
    ) -> Callable[[bytes | memoryview], Any] | None:
        if (
            message_encoding != _MESSAGE_ENCODING_CDR
            or schema is None
            or schema.encoding != _SCHEMA_ENCODING_ROS2
            or schema.name not in {"sensor_msgs/PointCloud2", "sensor_msgs/msg/PointCloud2"}
        ):
            return None

        if decoder := self._ros2_decoder_factory.decoder_for(message_encoding, schema):
            from pointcloud2 import read_points  # noqa: PLC0415

            return lambda data: read_points(decoder(data), skip_nans=True)
        return None
