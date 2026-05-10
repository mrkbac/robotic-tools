"""PointCloud2 compression helpers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from pointcloud2 import Pointcloud2Msg, PointFieldMsg
    from pureini import CompressionOption, EncodingInfo, EncodingOptions

DRACO_MAX_QUANTIZATION_BITS = 30


class PointCloudCompressionError(ValueError):
    """Raised when a point cloud cannot be compressed into the requested format."""


class PointCloudCompressorProtocol(Protocol):
    """Common interface implemented by Cloudini and Draco point cloud compressors."""

    def compress(self, msg: Pointcloud2Msg) -> bytes: ...


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


class CloudiniPointCloudCompressor:
    """Cloudini point cloud compressor with encoder caching."""

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


def _compute_position_quantization(
    positions: npt.NDArray[np.float32],
    resolution: float,
) -> tuple[int, float, npt.NDArray[np.float32]]:
    import numpy as np  # noqa: PLC0415

    origin = positions.min(axis=0).astype(np.float32, copy=False)
    spans = positions.max(axis=0) - origin
    quantization_range = max(float(np.max(spans)), resolution)
    bits = math.ceil(math.log2((quantization_range / resolution) + 1.0))
    bits = min(max(bits, 1), DRACO_MAX_QUANTIZATION_BITS)
    return bits, quantization_range, origin


def _native_numeric_array(values: npt.ArrayLike) -> npt.NDArray[np.number] | None:
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(values)
    if arr.dtype.fields is not None or arr.dtype.kind not in "iuf":
        return None
    if not arr.dtype.isnative:
        arr = arr.byteswap().view(arr.dtype.newbyteorder("="))

    if arr.dtype.kind == "f":
        arr = arr.astype(np.float32, copy=False)
    elif arr.dtype.kind == "u":
        if arr.dtype.itemsize <= 1:
            arr = arr.astype(np.uint8, copy=False)
        elif arr.dtype.itemsize <= 2:
            arr = arr.astype(np.uint16, copy=False)
        else:
            arr = arr.astype(np.uint32, copy=False)
    elif arr.size and int(arr.min()) >= 0:
        max_value = int(arr.max())
        if max_value <= np.iinfo(np.uint8).max:
            arr = arr.astype(np.uint8, copy=False)
        elif max_value <= np.iinfo(np.uint16).max:
            arr = arr.astype(np.uint16, copy=False)
        else:
            arr = arr.astype(np.uint32, copy=False)
    else:
        arr = arr.astype(np.float32, copy=False)

    arr = np.ascontiguousarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr.reshape(arr.shape[0], -1)


def _packed_color_attributes(
    values: npt.ArrayLike, *, include_alpha: bool
) -> dict[str, np.ndarray]:
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(values)
    if arr.dtype.kind == "f":
        packed = np.ascontiguousarray(arr.astype(np.float32, copy=False)).view(np.uint32)
    else:
        packed = arr.astype(np.uint32, copy=False)

    attrs = {
        "red": ((packed >> 16) & 0xFF).astype(np.uint8, copy=False).reshape(-1, 1),
        "green": ((packed >> 8) & 0xFF).astype(np.uint8, copy=False).reshape(-1, 1),
        "blue": (packed & 0xFF).astype(np.uint8, copy=False).reshape(-1, 1),
    }
    if include_alpha:
        attrs["alpha"] = ((packed >> 24) & 0xFF).astype(np.uint8, copy=False).reshape(-1, 1)
    return attrs


def _generic_attributes_from_points(
    points: np.ndarray, fields: list[PointFieldMsg]
) -> dict[str, np.ndarray]:
    attrs: dict[str, np.ndarray] = {}
    names = set(points.dtype.names or ())

    for field in fields:
        name = field.name
        if not name or name in {"x", "y", "z"}:
            continue

        count = int(field.count)
        if name in {"rgb", "rgba"}:
            if count == 1 and name in names:
                attrs.update(_packed_color_attributes(points[name], include_alpha=name == "rgba"))
                continue
            component_names = [f"{name}_{idx}" for idx in range(count)]
            if count in {3, 4} and all(component in names for component in component_names):
                for color, source in zip(
                    ("red", "green", "blue", "alpha"), component_names, strict=False
                ):
                    arr = _native_numeric_array(points[source])
                    if arr is not None:
                        attrs[color] = arr
                continue

        if count == 1:
            if name not in names:
                continue
            attr = _native_numeric_array(points[name])
        else:
            component_names = [f"{name}_{idx}" for idx in range(count)]
            if not all(component in names for component in component_names):
                continue
            import numpy as np  # noqa: PLC0415

            attr = _native_numeric_array(np.column_stack([points[c] for c in component_names]))

        if attr is not None:
            attrs[name] = attr

    return attrs


def build_foxglove_compressed_pointcloud_message(
    msg: Pointcloud2Msg,
    compressed_data: bytes,
    *,
    fmt: str = "draco",
) -> dict[str, Any]:
    """Build a Foxglove ``CompressedPointCloud`` ROS2 message dict."""
    stamp = msg.header.stamp
    return {
        "timestamp": {
            "sec": stamp.sec,
            "nanosec": stamp.nanosec,
        },
        "frame_id": msg.header.frame_id,
        "pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "data": compressed_data,
        "format": fmt,
    }


def build_compressed_pointcloud2_message(
    msg: Pointcloud2Msg,
    compressed_data: bytes,
    *,
    fmt: str,
) -> dict[str, Any]:
    """Build a ``CompressedPointCloud2`` ROS2 message dict."""
    stamp = msg.header.stamp
    return {
        "header": {
            "stamp": {
                "sec": stamp.sec,
                "nanosec": stamp.nanosec,
            },
            "frame_id": msg.header.frame_id,
        },
        "height": msg.height,
        "width": msg.width,
        "fields": [
            {
                "name": field.name,
                "offset": field.offset,
                "datatype": field.datatype,
                "count": field.count,
            }
            for field in msg.fields
        ],
        "is_bigendian": msg.is_bigendian,
        "point_step": msg.point_step,
        "row_step": msg.row_step,
        "compressed_data": compressed_data,
        "is_dense": msg.is_dense,
        "format": fmt,
    }


class DracoPointCloudCompressor:
    """Draco-based PointCloud2 compressor for Foxglove CompressedPointCloud."""

    def __init__(self, resolution: float = 0.01, compression_level: int = 7) -> None:
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        if not 0 <= compression_level <= 10:
            raise ValueError("compression_level must be in [0, 10]")
        self._resolution = resolution
        self._compression_level = compression_level

    def compress(self, msg: Pointcloud2Msg) -> bytes:
        """Compress a decoded ROS2 PointCloud2 message and return Draco bytes."""
        import DracoPy  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        from pointcloud2 import read_points  # noqa: PLC0415

        point_records = read_points(msg, skip_nans=False)
        names = set(point_records.dtype.names or ())
        missing = {"x", "y", "z"} - names
        if missing:
            missing_fields = ", ".join(sorted(missing))
            raise PointCloudCompressionError(
                f"PointCloud2 is missing required field(s): {missing_fields}"
            )

        positions = np.column_stack(
            [
                point_records["x"].astype(np.float32, copy=False),
                point_records["y"].astype(np.float32, copy=False),
                point_records["z"].astype(np.float32, copy=False),
            ]
        )
        finite_mask = np.isfinite(positions).all(axis=1)
        if not finite_mask.all():
            point_records = point_records[finite_mask]
            positions = positions[finite_mask]
        if positions.size == 0:
            raise PointCloudCompressionError("PointCloud2 has no finite XYZ points")

        quantization_bits, quantization_range, quantization_origin = _compute_position_quantization(
            positions, self._resolution
        )
        generic_attributes = _generic_attributes_from_points(point_records, list(msg.fields))

        return DracoPy.encode(
            np.ascontiguousarray(positions, dtype=np.float32),
            quantization_bits=quantization_bits,
            compression_level=self._compression_level,
            quantization_range=quantization_range,
            quantization_origin=quantization_origin,
            create_metadata=False,
            preserve_order=True,
            generic_attributes=generic_attributes,
        )

    def compress_message(self, msg: Pointcloud2Msg) -> dict[str, Any]:
        """Compress and wrap the payload in a Foxglove CompressedPointCloud message."""
        return build_foxglove_compressed_pointcloud_message(msg, self.compress(msg))


PointCloudCompressor = CloudiniPointCloudCompressor
