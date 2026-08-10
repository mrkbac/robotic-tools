"""ROS-like message classes produced by video decoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from mcap_codec_support._messages import Header, _RosMessage
from mcap_codec_support.video.schemas import COMPRESSED_IMAGE as COMPRESSED_IMAGE_SCHEMA_TEXT
from mcap_codec_support.video.schemas import IMAGE as IMAGE_SCHEMA_TEXT


@dataclass(slots=True)
class CompressedImage(_RosMessage):
    header: Header
    format: str
    data: bytes

    _type: ClassVar[str] = "sensor_msgs/msg/CompressedImage"
    _full_text: ClassVar[str] = COMPRESSED_IMAGE_SCHEMA_TEXT
    _fields_and_field_types: ClassVar[dict[str, str]] = {
        "header": "std_msgs/Header",
        "format": "string",
        "data": "sequence<uint8>",
    }


@dataclass(slots=True)
class Image(_RosMessage):
    header: Header
    height: int
    width: int
    encoding: str
    is_bigendian: int
    step: int
    data: bytes

    _type: ClassVar[str] = "sensor_msgs/msg/Image"
    _full_text: ClassVar[str] = IMAGE_SCHEMA_TEXT
    _fields_and_field_types: ClassVar[dict[str, str]] = {
        "header": "std_msgs/Header",
        "height": "uint32",
        "width": "uint32",
        "encoding": "string",
        "is_bigendian": "uint8",
        "step": "uint32",
        "data": "sequence<uint8>",
    }
