"""Point-cloud preprocessing backed by PureINI's native Rust implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from mcap_codec_support.pointcloud._pureini import pointcloud2_data, pointcloud2_to_encoding_info

if TYPE_CHECKING:
    from pointcloud2 import Pointcloud2Msg


def drop_invalid_and_reorder(
    msg: Pointcloud2Msg,
    *,
    drop_invalid: bool = True,
    sort_field: str | None = "line",
) -> Pointcloud2Msg:
    """Drop invalid numeric XYZ points and stably group by ``sort_field``."""
    from pointcloud2 import PointCloud2  # noqa: PLC0415
    from pureini import PointcloudEncoder  # noqa: PLC0415

    point_count = msg.width * msg.height
    expected_size = point_count * msg.point_step
    if (
        point_count == 0
        or msg.point_step == 0
        or expected_size > len(msg.data)
        or (not drop_invalid and sort_field is None)
    ):
        return msg

    source = pointcloud2_data(msg)
    prepared, transformed_point_count, did_filter_invalid_xyz = PointcloudEncoder(
        pointcloud2_to_encoding_info(msg)
    ).preprocess(
        source,
        drop_invalid=drop_invalid,
        sort_field=sort_field,
        is_bigendian=msg.is_bigendian,
    )
    if prepared is source:
        return msg

    output_count = cast("int", transformed_point_count)

    return PointCloud2(
        header=msg.header,
        height=1,
        width=output_count,
        fields=msg.fields,
        is_bigendian=False,
        point_step=msg.point_step,
        row_step=output_count * msg.point_step,
        data=prepared,
        is_dense=bool(msg.is_dense) or did_filter_invalid_xyz,
    )
