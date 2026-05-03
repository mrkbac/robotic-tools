"""Point cloud decoder factories.

Also exposes :class:`Pointcloud2DecoderFactory`, a small_mcap decoder
factory that decodes ``sensor_msgs/PointCloud2`` payloads directly into a
structured numpy array via :func:`pointcloud2.read_points`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from mcap_codec_support.pointcloud.schemas import (
    COMPRESSED_POINTCLOUD2_SCHEMA,
    FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA,
    POINTCLOUD2_SCHEMAS,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pointcloud2 import PointFieldDict
    from small_mcap import Schema

    from mcap_codec_support._messages import Header, Pointcloud2Dict

_COMPRESSED_POINTCLOUD_SCHEMAS = {
    COMPRESSED_POINTCLOUD2_SCHEMA,
    FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA,
}


def _pointcloud_dict_to_array(cloud_dict: Pointcloud2Dict) -> Any:
    from pointcloud2 import read_points  # noqa: PLC0415

    ns = SimpleNamespace(**{k: v for k, v in cloud_dict.items() if k != "header"})
    return read_points(ns, skip_nans=True)


def _message_format(msg: Any) -> str:
    fmt = msg.format
    if isinstance(fmt, bytes):
        fmt = fmt.decode()
    return str(fmt).strip().lower()


def _compressed_payload(msg: Any) -> bytes:
    # CompressedPointCloud2 uses ``compressed_data``; foxglove uses ``data``.
    data = getattr(msg, "data", None)
    if data is None:
        data = msg.compressed_data
    return data if isinstance(data, bytes) else bytes(data)


def _header_from_compressed_msg(msg: Any) -> Header:
    header = getattr(msg, "header", None)
    if header is not None:
        stamp = header.stamp
        return {
            "stamp": {"sec": stamp.sec, "nanosec": stamp.nanosec},
            "frame_id": header.frame_id,
        }

    timestamp = msg.timestamp
    return {
        "stamp": {"sec": timestamp.sec, "nanosec": timestamp.nanosec},
        "frame_id": msg.frame_id,
    }


def _fields_from_msg(fields: Any) -> list[PointFieldDict]:
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


def _pointcloud2_from_cloudini(msg: Any, pc_decoder: Any | None = None) -> Pointcloud2Dict:
    if pc_decoder is None:
        from pureini import PointcloudDecoder  # noqa: PLC0415

        pc_decoder = PointcloudDecoder()

    raw_bytes, info = pc_decoder.decode(_compressed_payload(msg))
    fields = getattr(msg, "fields", None)
    point_step = int(getattr(msg, "point_step", info.point_step))
    width = int(getattr(msg, "width", info.width))
    fields_out = (
        _fields_from_msg(fields) if fields is not None else _fields_from_cloudini_info(info)
    )

    return {
        "header": _header_from_compressed_msg(msg),
        "height": int(getattr(msg, "height", info.height)),
        "width": width,
        "fields": fields_out,
        "is_bigendian": bool(getattr(msg, "is_bigendian", False)),
        "point_step": point_step,
        "row_step": int(getattr(msg, "row_step", point_step * width)),
        "data": raw_bytes,
        "is_dense": bool(getattr(msg, "is_dense", True)),
    }


class CloudiniPointCloudDecompressFactory:
    """Schema-only decoder factory: Cloudini compressed point cloud → PointCloud2.

    Stateless — safe to share across all channels.
    """

    def __init__(self) -> None:
        from mcap_ros2_support_fast.decoder import DecoderFactory  # noqa: PLC0415
        from pureini import PointcloudDecoder  # noqa: PLC0415

        self._cdr_factory = DecoderFactory()
        self._pc_decoder: Any = PointcloudDecoder()

    def _decompress(self, msg: Any) -> Pointcloud2Dict:
        if _message_format(msg) != "cloudini":
            raise ValueError(f"unsupported compressed point cloud format: {msg.format!r}")
        return _pointcloud2_from_cloudini(msg, self._pc_decoder)

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


def _pointcloud2_from_draco(msg: Any) -> Pointcloud2Dict:
    import DracoPy  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from pointcloud2 import fields_from_dtype  # noqa: PLC0415

    decoded = DracoPy.decode(_compressed_payload(msg))
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
        "header": _header_from_compressed_msg(msg),
        "height": 1,
        "width": point_count,
        "fields": fields,
        "is_bigendian": False,
        "point_step": point_data.dtype.itemsize,
        "row_step": point_data.dtype.itemsize * point_count,
        "data": point_data.tobytes(),
        "is_dense": True,
    }


class DracoPointCloudDecompressFactory:
    """Schema-only decoder factory: Draco compressed point cloud → PointCloud2."""

    def __init__(self) -> None:
        from mcap_ros2_support_fast.decoder import DecoderFactory  # noqa: PLC0415

        self._cdr_factory = DecoderFactory()

    def _decompress(self, msg: Any) -> Pointcloud2Dict:
        if _message_format(msg) != "draco":
            raise ValueError(f"unsupported compressed point cloud format: {msg.format!r}")
        return _pointcloud2_from_draco(msg)

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
        self._cloudini_decoder: Any | None = None

    def _decompress(self, msg: Any) -> Pointcloud2Dict:
        fmt = _message_format(msg)
        if fmt == "cloudini":
            if self._cloudini_decoder is None:
                from pureini import PointcloudDecoder  # noqa: PLC0415

                self._cloudini_decoder = PointcloudDecoder()
            return _pointcloud2_from_cloudini(msg, self._cloudini_decoder)
        if fmt == "draco":
            return _pointcloud2_from_draco(msg)
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
    ) -> Callable[[bytes | memoryview], Any] | None:
        decoder = self._decompress_factory.decoder_for(message_encoding, schema)
        if decoder is None:
            return None

        def _decode(data: bytes | memoryview) -> Any:
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
    ) -> Callable[[bytes | memoryview], Any] | None:
        decoder = self._decompress_factory.decoder_for(message_encoding, schema)
        if decoder is None:
            return None

        def _decode(data: bytes | memoryview) -> Any:
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
    ) -> Callable[[bytes | memoryview], Any] | None:
        decoder = self._decompress_factory.decoder_for(message_encoding, schema)
        if decoder is None:
            return None

        def _decode(data: bytes | memoryview) -> Any:
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
    ) -> Callable[[bytes | memoryview], Any] | None:
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

        def _decode(data: bytes | memoryview) -> Any:
            return read_points(decoder(data), skip_nans=True)

        return _decode
