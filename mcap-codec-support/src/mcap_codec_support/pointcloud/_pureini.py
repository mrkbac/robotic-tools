"""PureINI adapters for decoded PointCloud2 messages."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pointcloud2 import Pointcloud2Msg
    from pureini import CompressionOption, EncodingInfo, EncodingOptions

_FIELD_TYPE_SIZES = {
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 4,
    6: 4,
    7: 4,
    8: 8,
}


def pointcloud2_to_encoding_info(
    msg: Pointcloud2Msg,
    *,
    encoding_opt: EncodingOptions | None = None,
    compression_opt: CompressionOption | None = None,
    resolution: float | None = None,
) -> EncodingInfo:
    from pureini import EncodingInfo, FieldType, PointField  # noqa: PLC0415

    fields = []
    for field in msg.fields:
        field_type = FieldType(field.datatype)
        field_size = _FIELD_TYPE_SIZES.get(field.datatype)
        if field_size is None:
            raise ValueError(f"Unsupported PointField datatype {field.datatype}")
        if field.count < 1:
            raise ValueError(f"PointField '{field.name}' count must be positive")
        fields.extend(
            (
                PointField(
                    name=field.name if index == 0 else f"{field.name}_{index}",
                    offset=field.offset + index * field_size,
                    type=field_type,
                    resolution=resolution
                    if resolution is not None and field.datatype == 7
                    else None,
                )
            )
            for index in range(field.count)
        )

    info = EncodingInfo(
        fields=fields,
        width=msg.width,
        height=msg.height,
        point_step=msg.point_step,
    )
    if encoding_opt is not None:
        info.encoding_opt = encoding_opt
    if compression_opt is not None:
        info.compression_opt = compression_opt
    return info


def pointcloud2_data(msg: Pointcloud2Msg) -> bytes:
    """Return tightly packed, little-endian point records."""
    width = int(msg.width)
    height = int(msg.height)
    point_step = int(msg.point_step)
    row_step = int(msg.row_step)
    if width < 0 or height < 0:
        raise ValueError("PointCloud2 dimensions must be non-negative")
    if point_step <= 0:
        raise ValueError("PointCloud2 point_step must be positive")

    packed_row_step = width * point_step
    if row_step < packed_row_step:
        raise ValueError(
            f"PointCloud2 row_step {row_step} is smaller than width * point_step "
            f"({packed_row_step})"
        )
    expected_size = row_step * height
    if len(msg.data) != expected_size:
        raise ValueError(
            f"PointCloud2 data has {len(msg.data)} bytes, expected row_step * height "
            f"({expected_size})"
        )
    if row_step == packed_row_step:
        packed = bytes(msg.data)
    else:
        packed = b"".join(
            msg.data[row_start : row_start + packed_row_step]
            for row_start in range(0, expected_size, row_step)
        )
    return packed
