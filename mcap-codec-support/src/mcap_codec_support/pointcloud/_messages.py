"""Pointcloud-specific TypedDict shapes."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pointcloud2 import PointFieldDict

    from mcap_codec_support._messages import Header


class Pointcloud2Dict(TypedDict):
    """Dict shape mirroring ``sensor_msgs/PointCloud2``.

    Compatible with ``pointcloud2.read_points`` once wrapped in a
    ``SimpleNamespace``.
    """

    header: Header
    height: int
    width: int
    fields: list[PointFieldDict]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: bytes
    is_dense: bool
