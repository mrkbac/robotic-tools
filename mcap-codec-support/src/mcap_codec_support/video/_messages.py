"""Video-specific TypedDict shapes."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from mcap_codec_support._messages import Header


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
