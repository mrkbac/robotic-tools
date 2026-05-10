"""Point cloud decoder factories.

Also exposes :class:`Pointcloud2DecoderFactory`, a small_mcap decoder
factory that decodes ``sensor_msgs/PointCloud2`` payloads directly into a
structured numpy array via :func:`pointcloud2.read_points`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol

from mcap_codec_support.pointcloud.schemas import (
    COMPRESSED_POINTCLOUD2_SCHEMA,
    FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA,
    POINTCLOUD2_SCHEMAS,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from pointcloud2 import HeaderMsg, PointFieldDict, PointFieldMsg
    from pureini import PointcloudDecoder
    from small_mcap import Schema

    from mcap_codec_support._messages import Header, Stamp
    from mcap_codec_support.pointcloud._messages import Pointcloud2Dict


class _RosCompressedPointcloud2Msg(Protocol):
    """point_cloud_interfaces/msg/CompressedPointCloud2."""

    header: HeaderMsg
    height: int
    width: int
    fields: list[PointFieldMsg]
    is_bigendian: bool
    point_step: int
    row_step: int
    is_dense: bool
    format: str | bytes
    compressed_data: bytes


class _StampMsg(Protocol):
    sec: int
    nanosec: int


class _FoxgloveCompressedPointcloudMsg(Protocol):
    """foxglove_msgs/msg/CompressedPointCloud — flattened header."""

    timestamp: _StampMsg
    frame_id: str
    format: str | bytes
    data: bytes


_COMPRESSED_POINTCLOUD_SCHEMAS = {
    COMPRESSED_POINTCLOUD2_SCHEMA,
    FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA,
}


def is_compressed_codec_available() -> bool:
    """True if any compressed point-cloud codec backend (cloudini or draco) is importable."""
    try:
        import pureini  # noqa: F401, PLC0415
    except ImportError:
        pass
    else:
        return True
    try:
        import DracoPy  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def _pointcloud_dict_to_array(cloud_dict: Pointcloud2Dict) -> np.ndarray:
    from pointcloud2 import read_points  # noqa: PLC0415

    ns = SimpleNamespace(**{k: v for k, v in cloud_dict.items() if k != "header"})
    return read_points(ns, skip_nans=True)


def _decode_format(fmt: str | bytes) -> str:
    if isinstance(fmt, bytes):
        fmt = fmt.decode()
    return fmt.strip().lower()


def _as_bytes(payload: bytes | bytearray | memoryview) -> bytes:
    return payload if isinstance(payload, bytes) else bytes(payload)


def _stamp_from_msg(stamp: _StampMsg) -> Stamp:
    return {"sec": stamp.sec, "nanosec": stamp.nanosec}


def _header_from_ros_msg(msg: _RosCompressedPointcloud2Msg) -> Header:
    return {
        "stamp": _stamp_from_msg(msg.header.stamp),
        "frame_id": msg.header.frame_id,
    }


def _header_from_foxglove_msg(msg: _FoxgloveCompressedPointcloudMsg) -> Header:
    return {
        "stamp": _stamp_from_msg(msg.timestamp),
        "frame_id": msg.frame_id,
    }


def _fields_from_msg(fields: list[PointFieldMsg]) -> list[PointFieldDict]:
    return [
        {
            "name": field.name,
            "offset": field.offset,
            "datatype": field.datatype,
            "count": field.count,
        }
        for field in fields
    ]


def _fields_from_cloudini_info(info: Any) -> list[PointFieldDict]:
    return [
        {
            "name": field.name,
            "offset": field.offset,
            "datatype": int(field.type),
            "count": 1,
        }
        for field in info.fields
    ]


def _pointcloud2_from_cloudini_ros(
    msg: _RosCompressedPointcloud2Msg, pc_decoder: PointcloudDecoder
) -> Pointcloud2Dict:
    raw_bytes, _info = pc_decoder.decode(_as_bytes(msg.compressed_data))
    return {
        "header": _header_from_ros_msg(msg),
        "height": int(msg.height),
        "width": int(msg.width),
        "fields": _fields_from_msg(msg.fields),
        "is_bigendian": bool(msg.is_bigendian),
        "point_step": int(msg.point_step),
        "row_step": int(msg.row_step),
        "data": raw_bytes,
        "is_dense": bool(msg.is_dense),
    }


def _pointcloud2_from_cloudini_foxglove(
    msg: _FoxgloveCompressedPointcloudMsg, pc_decoder: PointcloudDecoder
) -> Pointcloud2Dict:
    raw_bytes, info = pc_decoder.decode(_as_bytes(msg.data))
    point_step = info.point_step
    width = info.width
    return {
        "header": _header_from_foxglove_msg(msg),
        "height": info.height,
        "width": width,
        "fields": _fields_from_cloudini_info(info),
        "is_bigendian": False,
        "point_step": point_step,
        "row_step": point_step * width,
        "data": raw_bytes,
        "is_dense": True,
    }


def _decode_draco_payload(payload: bytes, header: Header) -> Pointcloud2Dict:
    import DracoPy  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from pointcloud2 import fields_from_dtype  # noqa: PLC0415

    decoded = DracoPy.decode(_as_bytes(payload))
    positions = np.asarray(decoded.points, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("Draco point cloud does not contain at least two coordinate fields")

    point_count = int(positions.shape[0])
    columns: list[tuple[str, np.ndarray]] = [
        ("x", positions[:, 0].astype(np.float32, copy=False)),
        ("y", positions[:, 1].astype(np.float32, copy=False)),
    ]
    if positions.shape[1] >= 3:
        columns.append(("z", positions[:, 2].astype(np.float32, copy=False)))
    else:
        columns.append(("z", np.zeros(point_count, dtype=np.float32)))

    for index, attr in enumerate(decoded.attributes):
        if attr.get("attribute_type") == DracoPy.AttributeType.POSITION:
            continue
        name = attr.get("name") or f"attribute_{attr.get('unique_id', index)}"
        if name in {"x", "y", "z"}:
            continue
        values = attr.get("data")
        if values is None:
            continue
        arr = np.asarray(values)
        if arr.shape[0] != point_count:
            continue
        if arr.dtype.kind == "f":
            arr = arr.astype(np.float32, copy=False)
        elif arr.dtype == np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        elif arr.dtype == np.uint16:
            arr = arr.astype(np.uint16, copy=False)
        elif arr.dtype == np.uint32:
            arr = arr.astype(np.uint32, copy=False)
        else:
            continue

        if arr.ndim == 1 or arr.shape[1] == 1:
            columns.append((str(name), arr.reshape(point_count)))
        else:
            columns.extend(
                (f"{name}_{component}", arr[:, component]) for component in range(arr.shape[1])
            )

    dtype = np.dtype([(name, values.dtype) for name, values in columns])
    point_data = np.empty(point_count, dtype=dtype)
    for name, values in columns:
        point_data[name] = values

    fields: list[PointFieldDict] = [
        {"name": field.name, "offset": field.offset, "datatype": field.datatype, "count": 1}
        for field in fields_from_dtype(point_data.dtype)
    ]

    return {
        "header": header,
        "height": 1,
        "width": point_count,
        "fields": fields,
        "is_bigendian": False,
        "point_step": point_data.dtype.itemsize,
        "row_step": point_data.dtype.itemsize * point_count,
        "data": point_data.tobytes(),
        "is_dense": True,
    }


def _pointcloud2_from_draco_ros(msg: _RosCompressedPointcloud2Msg) -> Pointcloud2Dict:
    return _decode_draco_payload(msg.compressed_data, _header_from_ros_msg(msg))


def _pointcloud2_from_draco_foxglove(msg: _FoxgloveCompressedPointcloudMsg) -> Pointcloud2Dict:
    return _decode_draco_payload(msg.data, _header_from_foxglove_msg(msg))


# The two compressed-pointcloud schemas share ``format`` but differ in the
# payload field name (``compressed_data`` vs ``data``); branching on attribute
# presence is the only stable runtime discriminator.
def _is_ros_style_compressed_msg(
    msg: _RosCompressedPointcloud2Msg | _FoxgloveCompressedPointcloudMsg,
) -> bool:
    return hasattr(msg, "compressed_data")


class CloudiniPointCloudDecompressFactory:
    """Schema-only decoder factory: Cloudini compressed point cloud → PointCloud2.

    Stateless — safe to share across all channels.
    """

    def __init__(self) -> None:
        from mcap_ros2_support_fast.decoder import DecoderFactory  # noqa: PLC0415
        from pureini import PointcloudDecoder  # noqa: PLC0415

        self._cdr_factory = DecoderFactory()
        self._pc_decoder: PointcloudDecoder = PointcloudDecoder()

    def _decompress(
        self,
        msg: _RosCompressedPointcloud2Msg | _FoxgloveCompressedPointcloudMsg,
    ) -> Pointcloud2Dict:
        if _decode_format(msg.format) != "cloudini":
            raise ValueError(f"unsupported compressed point cloud format: {msg.format!r}")
        if _is_ros_style_compressed_msg(msg):
            return _pointcloud2_from_cloudini_ros(msg, self._pc_decoder)  # ty: ignore[invalid-argument-type]
        return _pointcloud2_from_cloudini_foxglove(msg, self._pc_decoder)  # ty: ignore[invalid-argument-type]

    def decoder_for(
        self,
        message_encoding: str,
        schema: Schema | None,
    ) -> Callable[[bytes | memoryview], Pointcloud2Dict] | None:
        if schema is None or schema.name not in _COMPRESSED_POINTCLOUD_SCHEMAS:
            return None
        cdr_decoder = self._cdr_factory.decoder_for(message_encoding, schema)
        if cdr_decoder is None:
            return None

        def _decode(data: bytes | memoryview) -> Pointcloud2Dict:
            return self._decompress(cdr_decoder(data))

        return _decode


class DracoPointCloudDecompressFactory:
    """Schema-only decoder factory: Draco compressed point cloud → PointCloud2."""

    def __init__(self) -> None:
        from mcap_ros2_support_fast.decoder import DecoderFactory  # noqa: PLC0415

        self._cdr_factory = DecoderFactory()

    def _decompress(
        self,
        msg: _RosCompressedPointcloud2Msg | _FoxgloveCompressedPointcloudMsg,
    ) -> Pointcloud2Dict:
        if _decode_format(msg.format) != "draco":
            raise ValueError(f"unsupported compressed point cloud format: {msg.format!r}")
        if _is_ros_style_compressed_msg(msg):
            return _pointcloud2_from_draco_ros(msg)  # ty: ignore[invalid-argument-type]
        return _pointcloud2_from_draco_foxglove(msg)  # ty: ignore[invalid-argument-type]

    def decoder_for(
        self,
        message_encoding: str,
        schema: Schema | None,
    ) -> Callable[[bytes | memoryview], Pointcloud2Dict] | None:
        if schema is None or schema.name not in _COMPRESSED_POINTCLOUD_SCHEMAS:
            return None
        cdr_decoder = self._cdr_factory.decoder_for(message_encoding, schema)
        if cdr_decoder is None:
            return None

        def _decode(data: bytes | memoryview) -> Pointcloud2Dict:
            return self._decompress(cdr_decoder(data))

        return _decode


class CompressedPointCloudDecompressFactory:
    """Decode either compressed point cloud schema by dispatching on ``format``."""

    def __init__(self) -> None:
        from mcap_ros2_support_fast.decoder import DecoderFactory  # noqa: PLC0415

        self._cdr_factory = DecoderFactory()
        self._cloudini_decoder: PointcloudDecoder | None = None

    def _ensure_cloudini_decoder(self) -> PointcloudDecoder:
        if self._cloudini_decoder is None:
            from pureini import PointcloudDecoder  # noqa: PLC0415

            self._cloudini_decoder = PointcloudDecoder()
        return self._cloudini_decoder

    def _decompress(
        self,
        msg: _RosCompressedPointcloud2Msg | _FoxgloveCompressedPointcloudMsg,
    ) -> Pointcloud2Dict:
        fmt = _decode_format(msg.format)
        is_ros = _is_ros_style_compressed_msg(msg)
        if fmt == "cloudini":
            decoder = self._ensure_cloudini_decoder()
            if is_ros:
                return _pointcloud2_from_cloudini_ros(msg, decoder)  # ty: ignore[invalid-argument-type]
            return _pointcloud2_from_cloudini_foxglove(msg, decoder)  # ty: ignore[invalid-argument-type]
        if fmt == "draco":
            if is_ros:
                return _pointcloud2_from_draco_ros(msg)  # ty: ignore[invalid-argument-type]
            return _pointcloud2_from_draco_foxglove(msg)  # ty: ignore[invalid-argument-type]
        raise ValueError(f"unsupported compressed point cloud format: {msg.format!r}")

    def decoder_for(
        self,
        message_encoding: str,
        schema: Schema | None,
    ) -> Callable[[bytes | memoryview], Pointcloud2Dict] | None:
        if schema is None or schema.name not in _COMPRESSED_POINTCLOUD_SCHEMAS:
            return None
        cdr_decoder = self._cdr_factory.decoder_for(message_encoding, schema)
        if cdr_decoder is None:
            return None

        def _decode(data: bytes | memoryview) -> Pointcloud2Dict:
            return self._decompress(cdr_decoder(data))

        return _decode


class CloudiniCompressedPointcloud2DecoderFactory:
    """Decode Cloudini compressed point clouds to a structured numpy array.

    Chains :class:`CloudiniPointCloudDecompressFactory` with
    :func:`pointcloud2.read_points`, so the output has the same shape as the
    regular PointCloud2 path — a numpy structured array ready to be packed
    into an Arrow ``LIST<STRUCT<...>>``.

    Requires ``pureini``; construction raises ``ImportError`` if missing.
    """

    def __init__(self) -> None:
        self._decompress_factory = CloudiniPointCloudDecompressFactory()

    def decoder_for(
        self,
        message_encoding: str,
        schema: Schema | None,
    ) -> Callable[[bytes | memoryview], np.ndarray] | None:
        decoder = self._decompress_factory.decoder_for(message_encoding, schema)
        if decoder is None:
            return None

        def _decode(data: bytes | memoryview) -> np.ndarray:
            return _pointcloud_dict_to_array(decoder(data))

        return _decode


class DracoCompressedPointcloudDecoderFactory:
    """Decode Draco compressed point clouds to a structured numpy array."""

    def __init__(self) -> None:
        self._decompress_factory = DracoPointCloudDecompressFactory()

    def decoder_for(
        self,
        message_encoding: str,
        schema: Schema | None,
    ) -> Callable[[bytes | memoryview], np.ndarray] | None:
        decoder = self._decompress_factory.decoder_for(message_encoding, schema)
        if decoder is None:
            return None

        def _decode(data: bytes | memoryview) -> np.ndarray:
            return _pointcloud_dict_to_array(decoder(data))

        return _decode


class CompressedPointCloudDecoderFactory:
    """Decode either compressed point cloud schema to a structured numpy array."""

    def __init__(self) -> None:
        self._decompress_factory = CompressedPointCloudDecompressFactory()

    def decoder_for(
        self,
        message_encoding: str,
        schema: Schema | None,
    ) -> Callable[[bytes | memoryview], np.ndarray] | None:
        decoder = self._decompress_factory.decoder_for(message_encoding, schema)
        if decoder is None:
            return None

        def _decode(data: bytes | memoryview) -> np.ndarray:
            return _pointcloud_dict_to_array(decoder(data))

        return _decode


PointCloudDecompressFactory = CompressedPointCloudDecompressFactory
CompressedPointcloud2DecoderFactory = CompressedPointCloudDecoderFactory


_SCHEMA_ENCODING_ROS2 = "ros2msg"
_MESSAGE_ENCODING_CDR = "cdr"


class Pointcloud2DecoderFactory:
    def __init__(self) -> None:
        from mcap_ros2_support_fast.decoder import DecoderFactory  # noqa: PLC0415

        self._ros2_decoder_factory = DecoderFactory()

    def decoder_for(
        self, message_encoding: str, schema: Schema | None
    ) -> Callable[[bytes | memoryview], np.ndarray] | None:
        if (
            message_encoding != _MESSAGE_ENCODING_CDR
            or schema is None
            or schema.encoding != _SCHEMA_ENCODING_ROS2
            or schema.name not in POINTCLOUD2_SCHEMAS
        ):
            return None

        decoder = self._ros2_decoder_factory.decoder_for(message_encoding, schema)
        if decoder is None:
            return None

        from pointcloud2 import read_points  # noqa: PLC0415

        def _decode(data: bytes | memoryview) -> np.ndarray:
            return read_points(decoder(data), skip_nans=True)

        return _decode
