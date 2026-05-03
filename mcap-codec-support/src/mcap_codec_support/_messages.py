"""Shared TypedDict shapes for decoded ROS messages."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pointcloud2 import PointFieldDict


class Stamp(TypedDict):
    sec: int
    nanosec: int


class Header(TypedDict):
    stamp: Stamp
    frame_id: str


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


class CompressedImageDict(TypedDict):
    """Dict shape mirroring ``sensor_msgs/CompressedImage``."""

    header: Header
    format: str
    data: bytes


class ImageDict(TypedDict):
    """Dict shape mirroring ``sensor_msgs/Image`` for raw RGB frames."""

    header: Header
    height: int
    width: int
    encoding: str
    is_bigendian: int
    step: int
    data: bytes
